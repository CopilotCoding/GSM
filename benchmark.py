"""
benchmark.py — GSM inference speed and memory profiling.

Tests:
  1. Inference throughput (tok/s) at multiple sequence lengths — proves O(1)
  2. Peak VRAM usage during inference and training forward pass
  3. Latency distribution (min/median/p95/max) per token
  4. Batch size scaling — throughput vs batch size
  5. Memory efficiency — MB per million parameters

Usage:
    python benchmark.py --checkpoint checkpoints/latest.pt --vocab_path vocab.json
    python benchmark.py --checkpoint checkpoints/latest.pt --vocab_path vocab.json --full
"""

import json
import time
import argparse
import sys
from pathlib import Path
from statistics import median, mean

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from model.gsm import GSM


# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GSM(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        state_dim=config["state_dim"],
        n_pairs=config["n_pairs"],
        hidden_dim=config.get("hidden_dim", 1024),
        n_layers=config.get("n_layers", 6),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def reset_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def current_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def fmt(n, unit=""):
    return f"{n:,.1f}{unit}"


def bar(value, max_value, width=30, char="█"):
    filled = int(round(value / max_value * width))
    return char * filled + "░" * (width - filled)


def separator(char="─", width=70):
    print(char * width)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Inference throughput at multiple lengths — O(1) proof
# ─────────────────────────────────────────────────────────────────────────────

def bench_throughput(model, config, device, lengths, warmup=3, runs=5):
    vocab_size = config["vocab_size"]
    results = []

    print("\n  Inference throughput vs sequence length")
    print("  (O(1) architecture: throughput should stay flat)\n")
    print(f"  {'Length':>8}  {'tok/s':>10}  {'ms/tok':>8}  {'VRAM MB':>9}  {'Bar':}")
    separator()

    max_tps = None

    for length in lengths:
        # warmup
        prompt = torch.randint(0, vocab_size, (1, 4), device=device)
        for _ in range(warmup):
            with torch.no_grad():
                _ = model.generate(prompt, max_new_tokens=length,
                                   temperature=1.0, top_k=50)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # timed runs
        reset_vram()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model.generate(prompt, max_new_tokens=length,
                                   temperature=1.0, top_k=50)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_sec   = mean(times)
        tps       = length / avg_sec
        ms_per_tok = avg_sec * 1000 / length
        vram      = peak_vram_mb()

        if max_tps is None:
            max_tps = tps
        max_tps = max(max_tps, tps)

        results.append({"length": length, "tok_per_sec": tps,
                        "ms_per_tok": ms_per_tok, "vram_mb": vram})

    # print table now that we know max_tps
    for r in results:
        b = bar(r["tok_per_sec"], max_tps)
        print(f"  {r['length']:>8,}  {r['tok_per_sec']:>10,.0f}  "
              f"{r['ms_per_tok']:>8.2f}  {r['vram_mb']:>9.1f}  {b}")

    separator()

    # O(1) check
    first_tps = results[0]["tok_per_sec"]
    last_tps  = results[-1]["tok_per_sec"]
    ratio     = last_tps / first_tps if first_tps > 0 else 0
    print(f"\n  Throughput ratio (length {lengths[-1]} / length {lengths[0]}): {ratio:.3f}")
    if ratio > 0.85:
        print("  ✓ O(1) confirmed — throughput stable across sequence lengths")
    else:
        print(f"  ⚠ Throughput degraded by {(1-ratio)*100:.1f}% — check architecture")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-token latency distribution
# ─────────────────────────────────────────────────────────────────────────────

def bench_latency(model, config, device, n_tokens=200, n_runs=50):
    vocab_size = config["vocab_size"]
    latencies  = []

    print("\n  Per-token latency distribution")
    print(f"  ({n_runs} single-token steps)\n")

    # build a state by running a short prompt
    prompt = torch.randint(0, vocab_size, (1, 4), device=device)

    for _ in range(n_runs):
        token = torch.randint(0, vocab_size, (1, 1), device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            emb = model.embedding(token).squeeze(1)
            # single step through the model
            carry = model.step.init_carry(model.S0, 1)
            _, S = model.step.step(carry, emb)
            _ = model.decoder(S)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p95_idx = int(len(latencies) * 0.95)

    print(f"  Min:    {min(latencies):.3f} ms")
    print(f"  Median: {median(latencies):.3f} ms")
    print(f"  Mean:   {mean(latencies):.3f} ms")
    print(f"  P95:    {latencies[p95_idx]:.3f} ms")
    print(f"  Max:    {max(latencies):.3f} ms")
    print(f"  Implied tok/s (median): {1000/median(latencies):,.0f}")

    return latencies


# ─────────────────────────────────────────────────────────────────────────────
# 3. Batch size scaling
# ─────────────────────────────────────────────────────────────────────────────

def bench_batch_scaling(model, config, device, batch_sizes, length=256, runs=3):
    vocab_size = config["vocab_size"]

    print("\n  Batch size scaling (inference throughput)\n")
    print(f"  {'Batch':>6}  {'tok/s':>10}  {'VRAM MB':>9}  {'MB/seq':>8}  {'Bar':}")
    separator()

    results = []
    max_tps = 1

    for bs in batch_sizes:
        try:
            reset_vram()
            prompt = torch.randint(0, vocab_size, (bs, 4), device=device)

            # warmup
            with torch.no_grad():
                _ = model.generate(prompt, max_new_tokens=length,
                                   temperature=1.0, top_k=50)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            reset_vram()
            t0 = time.perf_counter()
            for _ in range(runs):
                with torch.no_grad():
                    _ = model.generate(prompt, max_new_tokens=length,
                                       temperature=1.0, top_k=50)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / runs

            tps    = bs * length / elapsed
            vram   = peak_vram_mb()
            mb_seq = vram / bs

            max_tps = max(max_tps, tps)
            results.append({"batch": bs, "tps": tps, "vram": vram, "mb_seq": mb_seq})

        except torch.cuda.OutOfMemoryError:
            print(f"  {bs:>6}  OOM")
            break

    for r in results:
        b = bar(r["tps"], max_tps)
        print(f"  {r['batch']:>6}  {r['tps']:>10,.0f}  "
              f"{r['vram']:>9.1f}  {r['mb_seq']:>8.1f}  {b}")

    separator()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Memory profiling — forward + backward
# ─────────────────────────────────────────────────────────────────────────────

def bench_memory(model, config, device, batch_size=32, seq_len=128):
    vocab_size = config["vocab_size"]
    criterion  = nn.CrossEntropyLoss()

    print("\n  Memory profiling\n")

    # ── inference ──
    reset_vram()
    baseline = current_vram_mb()

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    with torch.no_grad():
        _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    inf_peak = peak_vram_mb()
    inf_used = inf_peak - baseline

    print(f"  Inference (batch={batch_size}, seq={seq_len}):")
    print(f"    Baseline VRAM:      {baseline:.1f} MB")
    print(f"    Peak VRAM:          {inf_peak:.1f} MB")
    print(f"    Delta (activations):{inf_used:.1f} MB")
    print(f"    MB per sequence:    {inf_used/batch_size:.2f} MB")
    print(f"    MB per token:       {inf_used/(batch_size*seq_len)*1000:.2f} KB")

    # ── training forward+backward ──
    model.train()
    reset_vram()
    baseline = current_vram_mb()

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    try:
        from torch import autocast
        with autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        train_peak = peak_vram_mb()
        train_used = train_peak - baseline

        print(f"\n  Training fwd+bwd (batch={batch_size}, seq={seq_len}, bf16):")
        print(f"    Baseline VRAM:      {baseline:.1f} MB")
        print(f"    Peak VRAM:          {train_peak:.1f} MB")
        print(f"    Delta (grad+acts):  {train_used:.1f} MB")
        print(f"    MB per sequence:    {train_used/batch_size:.2f} MB")

    except torch.cuda.OutOfMemoryError:
        print(f"\n  Training fwd+bwd: OOM at batch={batch_size}")

    model.eval()

    # ── model weights ──
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb    = param_bytes / 1e6
    params_m    = model.count_parameters() / 1e6

    print(f"\n  Model weights:")
    print(f"    Parameters:         {model.count_parameters():,}  ({params_m:.2f}M)")
    print(f"    Weight memory:      {param_mb:.1f} MB  (fp32)")
    print(f"    Weight memory:      {param_mb/2:.1f} MB  (bf16/fp16)")
    print(f"    MB per M params:    {param_mb/params_m:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary report
# ─────────────────────────────────────────────────────────────────────────────

def print_model_summary(model, config):
    params = model.count_parameters()
    print(f"\n  Model summary:")
    print(f"    Parameters:   {params:,}  ({params/1e6:.2f}M)")
    print(f"    State dim:    {config['state_dim']}")
    print(f"    Embed dim:    {config['embed_dim']}")
    print(f"    Layers:       {config.get('n_layers', 6)}")
    print(f"    Hidden dim:   {config.get('hidden_dim', 1024)}")
    print(f"    Rot pairs:    {config['n_pairs']}")
    print(f"    Vocab size:   {config['vocab_size']}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GSM benchmark suite")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab_path", default="vocab.json")
    parser.add_argument("--full", action="store_true",
                        help="Run full suite including batch scaling")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for memory profiling")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="Sequence length for memory profiling")
    parser.add_argument("--runs", type=int, default=5,
                        help="Timing runs per configuration")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bar_str = "=" * 70
    print(f"\n{bar_str}")
    print(f"  GSM Benchmark Suite")
    print(f"  Device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM:   {total_vram:.1f} GB total")
    print(f"{bar_str}")

    print(f"\n  Loading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    print_model_summary(model, config)

    # ── 1. throughput vs length ──────────────────────────────────────────────
    separator("─")
    print("  [1/4] Inference throughput vs sequence length")
    separator("─")
    lengths = [64, 128, 256, 512, 1024, 2048]
    bench_throughput(model, config, device, lengths, warmup=2, runs=args.runs)

    # ── 2. per-token latency ─────────────────────────────────────────────────
    separator("─")
    print("  [2/4] Per-token latency")
    separator("─")
    bench_latency(model, config, device, n_runs=100)

    # ── 3. memory profiling ──────────────────────────────────────────────────
    separator("─")
    print("  [3/4] Memory profiling")
    separator("─")
    bench_memory(model, config, device,
                 batch_size=args.batch_size, seq_len=args.seq_len)

    # ── 4. batch scaling (full only) ─────────────────────────────────────────
    if args.full:
        separator("─")
        print("  [4/4] Batch size scaling")
        separator("─")
        bench_batch_scaling(model, config, device,
                            batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
                            length=256, runs=args.runs)
    else:
        print("\n  [4/4] Batch scaling skipped (run with --full to include)")

    separator("=")
    print("  Benchmark complete.")
    separator("=")
    print()


if __name__ == "__main__":
    main()
