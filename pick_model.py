"""
pick_model.py — Automatic model size picker for GSM.

Generates configs starting from the absolute minimum viable model and scales
each dimension upward by a configurable factor until VRAM is exhausted.
Outputs the largest config that fits plus a ready-to-paste train command.

Usage:
    python pick_model.py
    python pick_model.py --vocab_path vocab.json --data_dir dataset_packed_128
    python pick_model.py --factor 1.5   # scale factor between configs (default 2.0)
    python pick_model.py --vram_budget 0.75 --seq_len 256
"""

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from model.gsm import GSM


# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_hardware():
    info = {}
    try:
        import psutil
        info["cpu_cores_physical"] = psutil.cpu_count(logical=False) or 4
        info["cpu_cores_logical"]  = psutil.cpu_count(logical=True) or 8
        info["ram_gb"]             = psutil.virtual_memory().total / 1e9
        info["ram_available_gb"]   = psutil.virtual_memory().available / 1e9
    except ImportError:
        import os
        info["cpu_cores_logical"]  = os.cpu_count() or 4
        info["cpu_cores_physical"] = info["cpu_cores_logical"] // 2
        info["ram_gb"]             = 16.0
        info["ram_available_gb"]   = 8.0

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"]      = props.name
        info["vram_total_gb"] = props.total_memory / 1e9
        info["vram_free_gb"]  = (props.total_memory - torch.cuda.memory_allocated()) / 1e9
        info["cuda"]          = True
        info["bf16"]          = torch.cuda.is_bf16_supported()
    else:
        info["gpu_name"]      = "CPU only"
        info["vram_total_gb"] = 0.0
        info["vram_free_gb"]  = 0.0
        info["cuda"]          = False
        info["bf16"]          = False

    return info


def print_hardware(hw):
    print(f"\n  GPU:   {hw['gpu_name']}")
    if hw["cuda"]:
        print(f"         {hw['vram_total_gb']:.1f} GB total  |  "
              f"{hw['vram_free_gb']:.1f} GB free  |  "
              f"bf16: {'yes' if hw['bf16'] else 'no'}")
    print(f"  CPU:   {hw['cpu_cores_physical']} physical / {hw['cpu_cores_logical']} logical cores")
    print(f"  RAM:   {hw['ram_gb']:.1f} GB total  |  {hw['ram_available_gb']:.1f} GB available")


# ─────────────────────────────────────────────────────────────────────────────
# Config generator — starts from minimum and scales by factor
# ─────────────────────────────────────────────────────────────────────────────

def snap(value, base):
    """Round value up to the nearest multiple of base."""
    return max(base, int(math.ceil(value / base) * base))


def generate_configs(vocab_size, factor=2.0, max_configs=32):
    """
    Generates GSM configs starting from the smallest viable model,
    scaling each dimension by `factor` on each step.

    Scaling strategy:
      - state_dim:  primary capacity axis, scales fastest
      - embed_dim:  scales with state_dim but capped (doesn't need to be huge)
      - hidden_dim: TransformNet width, scales with embed_dim
      - n_layers:   depth, increases slowly (every 2 doublings)
      - n_pairs:    rotation pairs, scales with state_dim / 32

    All values snapped to sensible multiples so they stay clean.
    """
    # Absolute minimum viable model
    min_state  = 64
    min_embed  = 64
    min_hidden = 128
    min_layers = 2
    min_pairs  = 8

    configs = []
    state = float(min_state)

    for i in range(max_configs):
        state_dim  = snap(state, 64)
        embed_dim  = snap(min(state_dim // 4, 1024), 64)   # embed grows with state but caps at 1024
        hidden_dim = snap(embed_dim * 2, 64)                # hidden = 2x embed
        n_layers   = min_layers + (i // 2)                  # add a layer every 2 steps
        n_layers   = min(n_layers, 12)                      # cap at 12
        n_pairs    = snap(max(state_dim // 32, min_pairs), 8)  # pairs scale with state

        # stop if state_dim is absurdly large (no GPU can hold it)
        if state_dim > 65536:
            break

        cfg = {
            "state_dim":  state_dim,
            "embed_dim":  embed_dim,
            "hidden_dim": hidden_dim,
            "n_layers":   n_layers,
            "n_pairs":    n_pairs,
            "vocab_size": vocab_size,
            "index":      i,
        }
        configs.append(cfg)
        state *= factor

    return configs


def count_params(cfg):
    model = GSM(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        state_dim=cfg["state_dim"],
        n_pairs=cfg["n_pairs"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        dropout=0.0,
    )
    n = model.count_parameters()
    del model
    return n


# ─────────────────────────────────────────────────────────────────────────────
# VRAM probe
# ─────────────────────────────────────────────────────────────────────────────

def probe_config(cfg, device, batch_size, seq_len, bf16, vram_budget_gb):
    """Returns (ok, peak_vram_mb, tok_per_sec, error_msg)"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    try:
        model = GSM(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            state_dim=cfg["state_dim"],
            n_pairs=cfg["n_pairs"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            dropout=0.1,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        x = torch.randint(0, cfg["vocab_size"], (batch_size, seq_len), device=device)
        y = torch.randint(0, cfg["vocab_size"], (batch_size, seq_len), device=device)

        dtype = torch.bfloat16 if bf16 and device.type == "cuda" else None

        t0 = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=dtype,
                                enabled=(dtype is not None and device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits.reshape(-1, cfg["vocab_size"]), y.reshape(-1))
        loss.backward()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
        peak_gb = peak_mb / 1024

        del model, x, y, logits, loss, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if vram_budget_gb > 0 and peak_gb > vram_budget_gb:
            return False, peak_mb, 0, f"VRAM {peak_gb:.2f}>{vram_budget_gb:.2f}GB"

        tok_per_sec = (batch_size * seq_len) / max(elapsed, 1e-9)
        return True, peak_mb, tok_per_sec, ""

    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return False, 0, 0, "OOM"
    except Exception as e:
        return False, 0, 0, str(e)[:60]


# ─────────────────────────────────────────────────────────────────────────────
# Max batch finder
# ─────────────────────────────────────────────────────────────────────────────

def find_max_batch(cfg, device, seq_len, bf16, vram_budget_gb):
    best = 1
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        ok, _, _, _ = probe_config(cfg, device, bs, seq_len, bf16, vram_budget_gb)
        if ok:
            best = bs
        else:
            break
    return best


def recommend_workers(hw):
    return min(max(hw.get("cpu_cores_logical", 4) // 3, 4), 20)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GSM automatic model size picker")
    parser.add_argument("--vocab_path",  default="vocab.json")
    parser.add_argument("--data_dir",    default="dataset_packed_128")
    parser.add_argument("--out_dir",     default="checkpoints")
    parser.add_argument("--seq_len",     type=int,   default=128)
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--factor",      type=float, default=2.0,
                        help="Scale factor between configs (default 2.0, try 1.5 for finer steps)")
    parser.add_argument("--vram_budget", type=float, default=0.80,
                        help="Fraction of free VRAM to use (default 0.80)")
    parser.add_argument("--vocab_size",  type=int,   default=373)
    parser.add_argument("--probe_batch", type=int,   default=32,
                        help="Batch size used during probing (default 32)")
    args = parser.parse_args()

    SEP = "=" * 70
    sep = "─" * 70

    print(f"\n{SEP}")
    print(f"  GSM Model Size Picker  (factor={args.factor})")
    print(SEP)

    hw = detect_hardware()
    print_hardware(hw)

    device = torch.device("cuda" if hw["cuda"] else "cpu")
    bf16   = hw["bf16"]

    vram_budget_gb = hw["vram_free_gb"] * args.vram_budget if hw["cuda"] else 0.0

    print(f"\n  VRAM budget: {vram_budget_gb:.2f} GB  ({args.vram_budget*100:.0f}% of {hw['vram_free_gb']:.2f} GB free)")
    print(f"  Factor:      {args.factor}x per step")
    print(f"  Probe batch: {args.probe_batch}  seq_len: {args.seq_len}")

    configs = generate_configs(args.vocab_size, factor=args.factor)

    print(f"\n  Generated {len(configs)} configs to sweep\n")
    print(f"  {'#':>3}  {'state':>6}  {'embed':>6}  {'hidden':>7}  {'layers':>6}  {'pairs':>6}  "
          f"{'params':>10}  {'VRAM MB':>8}  {'tok/s':>8}  status")
    print("  " + sep)

    best_cfg   = None
    best_tps   = 0
    best_peak  = 0

    for cfg in configs:
        params = count_params(cfg)

        # skip if weights alone won't fit
        weight_gb = params * 4 / 1e9
        if hw["cuda"] and weight_gb > vram_budget_gb * 0.6:
            print(f"  {cfg['index']:>3}  {cfg['state_dim']:>6}  {cfg['embed_dim']:>6}  "
                  f"{cfg['hidden_dim']:>7}  {cfg['n_layers']:>6}  {cfg['n_pairs']:>6}  "
                  f"{params/1e6:>8.2f}M  {'—':>8}  {'—':>8}  skip (weights {weight_gb:.2f}GB)")
            break

        ok, peak_mb, tps, err = probe_config(
            cfg, device, args.probe_batch, args.seq_len,
            bf16, vram_budget_gb
        )

        status = "✓" if ok else f"✗ {err}"
        print(f"  {cfg['index']:>3}  {cfg['state_dim']:>6}  {cfg['embed_dim']:>6}  "
              f"{cfg['hidden_dim']:>7}  {cfg['n_layers']:>6}  {cfg['n_pairs']:>6}  "
              f"{params/1e6:>8.2f}M  {peak_mb:>8.1f}  {tps:>8,.0f}  {status}")

        if ok:
            best_cfg  = cfg
            best_tps  = tps
            best_peak = peak_mb
        else:
            # first failure — done
            break

    print("  " + sep)

    if best_cfg is None:
        print("\n  ✗ No config fits. Try --vram_budget 0.5 or --seq_len 64.")
        sys.exit(1)

    # find max batch for winner
    print(f"\n  Finding max batch size for winning config (state={best_cfg['state_dim']})...")
    best_batch = find_max_batch(best_cfg, device, args.seq_len, bf16, vram_budget_gb)
    workers    = recommend_workers(hw)
    params     = count_params(best_cfg)

    print(f"  Max batch size: {best_batch}")

    cmd = (
        f"python -m train.train"
        f" --data_dir {args.data_dir}"
        f" --vocab_path {args.vocab_path}"
        f" --out_dir {args.out_dir}"
        f" --epochs {args.epochs}"
        f" --batch_size {best_batch}"
        f" --seq_len {args.seq_len}"
        f" --state_dim {best_cfg['state_dim']}"
        f" --embed_dim {best_cfg['embed_dim']}"
        f" --hidden_dim {best_cfg['hidden_dim']}"
        f" --n_layers {best_cfg['n_layers']}"
        f" --n_pairs {best_cfg['n_pairs']}"
        f" --workers {workers}"
    )

    print(f"\n{SEP}")
    print(f"  Winner: config #{best_cfg['index']}  |  {params/1e6:.2f}M params")
    print(SEP)
    print(f"  state_dim:   {best_cfg['state_dim']}")
    print(f"  embed_dim:   {best_cfg['embed_dim']}")
    print(f"  hidden_dim:  {best_cfg['hidden_dim']}")
    print(f"  n_layers:    {best_cfg['n_layers']}")
    print(f"  n_pairs:     {best_cfg['n_pairs']}")
    print(f"  batch_size:  {best_batch}")
    print(f"  workers:     {workers}")
    print(f"  VRAM peak:   {best_peak:.1f} MB  (at probe batch {args.probe_batch})")
    print(f"  Precision:   {'bf16' if bf16 else 'fp32'}")
    print(f"\n{SEP}")
    print(f"  Train command:")
    print(SEP)
    print(f"\n  {cmd}\n")
    print(SEP)


if __name__ == "__main__":
    main()
