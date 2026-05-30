"""
benchmark_compare.py — Original vs Optimized GSM forward pass benchmark.

Measures:
  - Forward pass throughput (tok/s) across batch sizes
  - Forward + backward throughput (training throughput)
  - First-step latency (compile warmup)
  - Peak VRAM usage
  - Latency distribution (min / median / p95 / max)

Usage:
    python3 benchmark_compare.py
    python3 benchmark_compare.py --seq_len 512 --trials 50
"""

import argparse
import time
import sys
import copy
import gc
from pathlib import Path
from statistics import median, quantiles

import torch
import torch.nn as nn

# ── load both model versions ─────────────────────────────────────────────────

# Original (inline from zip — reproduced here to avoid file dependency)
class _RotaryOrig(nn.Module):
    def __init__(self, state_dim, n_pairs=128):
        super().__init__()
        self.state_dim = state_dim
        self.n_pairs = n_pairs
        idx = torch.randperm(state_dim)[:n_pairs * 2].reshape(n_pairs, 2)
        self.register_buffer("idx_a", idx[:, 0])
        self.register_buffer("idx_b", idx[:, 1])

    def forward(self, S, angles):
        cos_t = torch.cos(angles)
        sin_t = torch.sin(angles)
        a = S[:, self.idx_a]
        b = S[:, self.idx_b]
        new_a = cos_t * a - sin_t * b
        new_b = sin_t * a + cos_t * b
        S = S.clone()
        S.scatter_(1, self.idx_a.unsqueeze(0).expand_as(new_a), new_a)
        S.scatter_(1, self.idx_b.unsqueeze(0).expand_as(new_b), new_b)
        return S


class _TransformNet(nn.Module):
    def __init__(self, embed_dim, state_dim, n_pairs, hidden_dim=1024, n_layers=6):
        super().__init__()
        self.state_dim = state_dim
        self.n_pairs = n_pairs
        out_dim = state_dim * 3 + n_pairs
        self.input_proj = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.SiLU())
        self.res_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, hidden_dim))
            for _ in range(n_layers - 2)
        ])
        self.res_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers - 2)])
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, e):
        x = self.input_proj(e)
        for res_layer, norm in zip(self.res_layers, self.res_norms):
            x = norm(x + res_layer(x))
        out = self.output_proj(x)
        scale  = torch.sigmoid(out[:, :self.state_dim]) * 2.0
        shift  = out[:, self.state_dim:self.state_dim * 2] * 0.1
        gate   = torch.sigmoid(out[:, self.state_dim * 2:self.state_dim * 3])
        angles = out[:, self.state_dim * 3:] * 0.1
        return scale, shift, gate, angles


class _StepOrig(nn.Module):
    def __init__(self, embed_dim, state_dim, n_pairs, hidden_dim=1024, n_layers=6):
        super().__init__()
        self.transform_net = _TransformNet(embed_dim, state_dim, n_pairs, hidden_dim, n_layers)
        self.rotary = _RotaryOrig(state_dim, n_pairs)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, S, e):
        scale, shift, gate, angles = self.transform_net(e)
        S_t = scale * S + shift
        S_t = self.rotary(S_t, angles)
        S_new = gate * S_t + (1.0 - gate) * S
        return self.norm(S_new)


class GSM_Original(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, state_dim=4096,
                 n_pairs=128, hidden_dim=1024, n_layers=6, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.state_dim  = state_dim
        self.embed_dim  = embed_dim
        self.embedding  = nn.Embedding(vocab_size, embed_dim)
        self.embed_drop = nn.Dropout(dropout)
        self.S0         = nn.Parameter(torch.randn(state_dim) * 0.01)
        self.step       = _StepOrig(embed_dim, state_dim, n_pairs, hidden_dim, n_layers)
        self.decoder    = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2), nn.SiLU(),
            nn.Linear(state_dim // 2, state_dim // 4), nn.SiLU(),
            nn.Linear(state_dim // 4, vocab_size),
        )

    def forward(self, x):
        batch, seq_len = x.shape
        S = self.S0.unsqueeze(0).expand(batch, -1).clone()
        E = self.embed_drop(self.embedding(x))
        states = torch.empty(batch, seq_len, self.state_dim, device=x.device, dtype=S.dtype)
        for t in range(seq_len):
            S = self.step(S, E[:, t, :])
            states[:, t, :] = S
        return self.decoder(states.view(batch * seq_len, self.state_dim)).view(batch, seq_len, -1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── optimized version ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from model.gsm import GSM as GSM_Optimized


# ── helpers ──────────────────────────────────────────────────────────────────
def reset_vram():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def peak_vram_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0

def fmt(val, unit="", decimals=2):
    return f"{val:.{decimals}f}{unit}"

def hline(char="─", width=72):
    print(char * width)

def header(title):
    hline("═")
    print(f"  {title}")
    hline("═")

def section(title):
    print()
    hline()
    print(f"  {title}")
    hline()


# ── benchmark routines ───────────────────────────────────────────────────────
def bench_forward(model, device, dtype, batch, seq_len, vocab_size, trials, warmup):
    model.eval()
    x = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    ctx = torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype is not None))

    # warmup
    with torch.no_grad(), ctx:
        for _ in range(warmup):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad(), ctx:
        for _ in range(trials):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    tokens = batch * seq_len
    tok_per_sec = [tokens / t for t in times]
    return {
        "mean_ms":   sum(times) / len(times) * 1000,
        "min_ms":    min(times) * 1000,
        "median_ms": median(times) * 1000,
        "p95_ms":    quantiles(times, n=20)[18] * 1000,
        "max_ms":    max(times) * 1000,
        "tok_per_sec_mean": sum(tok_per_sec) / len(tok_per_sec),
        "tok_per_sec_max":  max(tok_per_sec),
    }


def bench_train_step(model, device, dtype, batch, seq_len, vocab_size, trials, warmup):
    model.train()
    x = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    criterion = nn.CrossEntropyLoss()
    ctx = torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype is not None))

    for _ in range(warmup):
        with ctx:
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        with ctx:
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    tokens = batch * seq_len
    tok_per_sec = [tokens / t for t in times]
    return {
        "mean_ms":   sum(times) / len(times) * 1000,
        "tok_per_sec_mean": sum(tok_per_sec) / len(tok_per_sec),
        "tok_per_sec_max":  max(tok_per_sec),
    }


def measure_first_step(model, device, dtype, batch, seq_len, vocab_size):
    """Time the very first forward pass — includes compile tracing if applicable."""
    model.eval()
    x = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    ctx = torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype is not None))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), ctx:
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def measure_vram(model, device, dtype, batch, seq_len, vocab_size):
    reset_vram()
    model.eval()
    x = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    ctx = torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype is not None))
    with torch.no_grad(), ctx:
        _ = model(x)
    return peak_vram_mb()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size",  type=int, default=373)
    parser.add_argument("--embed_dim",   type=int, default=512)
    parser.add_argument("--state_dim",   type=int, default=4096)
    parser.add_argument("--n_pairs",     type=int, default=128)
    parser.add_argument("--hidden_dim",  type=int, default=1024)
    parser.add_argument("--n_layers",    type=int, default=6)
    parser.add_argument("--seq_len",     type=int, default=256)
    parser.add_argument("--trials",      type=int, default=40)
    parser.add_argument("--warmup",      type=int, default=5)
    parser.add_argument("--no_compile",  action="store_true", help="Skip torch.compile on optimized model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    import platform
    can_compile = (
        not args.no_compile
        and hasattr(torch, "compile")
        and torch.__version__ >= "2.0"
        and device.type == "cuda"
        and platform.system() == "Linux"
    )

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    precision   = str(dtype).split(".")[-1] if dtype else "fp32"

    header("GSM Benchmark  —  Original vs Optimized")
    print(f"  Device:     {device_name}")
    print(f"  Precision:  {precision}")
    print(f"  seq_len:    {args.seq_len}  |  vocab: {args.vocab_size}")
    print(f"  state_dim:  {args.state_dim}  |  embed: {args.embed_dim}")
    print(f"  Trials:     {args.trials}  |  Warmup: {args.warmup}")
    print(f"  compile:    {'yes' if can_compile else 'no'}")

    shared_kwargs = dict(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        state_dim=args.state_dim,
        n_pairs=args.n_pairs,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=0.0,
    )

    print("\n  Building models...")
    orig  = GSM_Original(**shared_kwargs).to(device)
    opt   = GSM_Optimized(**shared_kwargs).to(device)

    # Share identical weights so any difference is purely architectural/compile
    opt.load_state_dict(orig.state_dict(), strict=False)

    if can_compile:
        print("  torch.compile: fusing TransformNet + decoder submodules...")
        opt.step.transform_net = torch.compile(opt.step.transform_net)
        opt.decoder            = torch.compile(opt.decoder)

    print(f"  Parameters: {sum(p.numel() for p in orig.parameters() if p.requires_grad):,}")

    # ── 1. First-step latency (compile warmup cost) ──────────────────────────
    section("1 · First-step latency  (includes compile tracing)")
    batch = 32
    orig_first = measure_first_step(orig, device, dtype, batch, args.seq_len, args.vocab_size)
    opt_first  = measure_first_step(opt,  device, dtype, batch, args.seq_len, args.vocab_size)
    speedup    = orig_first / opt_first if opt_first > 0 else float("inf")
    print(f"  {'':30s}  {'Original':>12}  {'Optimized':>12}  {'Δ':>10}")
    hline("·")
    print(f"  {'First step (ms)':30s}  {orig_first:>12.1f}  {opt_first:>12.1f}  {speedup:>9.2f}x")

    # ── 2. Forward throughput across batch sizes ─────────────────────────────
    section("2 · Forward pass throughput  (tok/s, no grad)")
    batch_sizes = [8, 32, 64, 128, 256]
    print(f"  {'batch':>6}  {'orig tok/s':>12}  {'opt tok/s':>12}  {'speedup':>9}  {'orig ms':>9}  {'opt ms':>9}")
    hline("·")
    fwd_results = {}
    for b in batch_sizes:
        ro = bench_forward(orig, device, dtype, b, args.seq_len, args.vocab_size, args.trials, args.warmup)
        rp = bench_forward(opt,  device, dtype, b, args.seq_len, args.vocab_size, args.trials, args.warmup)
        sp = rp["tok_per_sec_mean"] / ro["tok_per_sec_mean"]
        fwd_results[b] = (ro, rp, sp)
        print(f"  {b:>6}  {ro['tok_per_sec_mean']:>12,.0f}  {rp['tok_per_sec_mean']:>12,.0f}  {sp:>8.2f}x"
              f"  {ro['median_ms']:>9.1f}  {rp['median_ms']:>9.1f}")

    # ── 3. Latency distribution at batch=128 ─────────────────────────────────
    section("3 · Latency distribution  (batch=128, ms)")
    b = 128
    ro = fwd_results.get(b)
    if ro is None:
        ro = bench_forward(orig, device, dtype, b, args.seq_len, args.vocab_size, args.trials, args.warmup)
        rp = bench_forward(opt,  device, dtype, b, args.seq_len, args.vocab_size, args.trials, args.warmup)
        ro, rp, _ = ro, rp, None
    else:
        ro, rp, _ = ro
    print(f"  {'':12}  {'min':>8}  {'median':>8}  {'p95':>8}  {'max':>8}")
    hline("·")
    print(f"  {'Original':12}  {ro['min_ms']:>8.1f}  {ro['median_ms']:>8.1f}  {ro['p95_ms']:>8.1f}  {ro['max_ms']:>8.1f}")
    print(f"  {'Optimized':12}  {rp['min_ms']:>8.1f}  {rp['median_ms']:>8.1f}  {rp['p95_ms']:>8.1f}  {rp['max_ms']:>8.1f}")
    sp_median = ro["median_ms"] / rp["median_ms"]
    print(f"  {'Speedup':12}  {'':>8}  {sp_median:>7.2f}x  {'':>8}  {'':>8}")

    # ── 4. Training step throughput ───────────────────────────────────────────
    section("4 · Training step throughput  (forward + backward, tok/s)")
    train_batches = [32, 128]
    print(f"  {'batch':>6}  {'orig tok/s':>12}  {'opt tok/s':>12}  {'speedup':>9}  {'orig ms':>9}  {'opt ms':>9}")
    hline("·")
    for b in train_batches:
        ro = bench_train_step(orig, device, dtype, b, args.seq_len, args.vocab_size, args.trials, args.warmup)
        rp = bench_train_step(opt,  device, dtype, b, args.seq_len, args.vocab_size, args.trials, args.warmup)
        sp = rp["tok_per_sec_mean"] / ro["tok_per_sec_mean"]
        print(f"  {b:>6}  {ro['tok_per_sec_mean']:>12,.0f}  {rp['tok_per_sec_mean']:>12,.0f}  {sp:>8.2f}x"
              f"  {ro['mean_ms']:>9.1f}  {rp['mean_ms']:>9.1f}")

    # ── 5. Peak VRAM ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        section("5 · Peak VRAM usage  (batch=128, forward only)")
        vo = measure_vram(orig, device, dtype, 128, args.seq_len, args.vocab_size)
        vp = measure_vram(opt,  device, dtype, 128, args.seq_len, args.vocab_size)
        print(f"  {'':30s}  {'Original':>12}  {'Optimized':>12}  {'Δ':>10}")
        hline("·")
        print(f"  {'Peak VRAM (MB)':30s}  {vo:>12.1f}  {vp:>12.1f}  {(vp-vo):>+10.1f}")

    # ── summary ───────────────────────────────────────────────────────────────
    section("Summary")
    b128_fwd = fwd_results.get(128)
    if b128_fwd:
        ro, rp, sp = b128_fwd
        print(f"  Forward throughput (batch=128):  {sp:.2f}x  "
              f"({ro['tok_per_sec_mean']/1000:.1f}k → {rp['tok_per_sec_mean']/1000:.1f}k tok/s)")
    print(f"  First-step latency:              {speedup:.2f}x  "
          f"({orig_first:.0f}ms → {opt_first:.0f}ms)")
    print(f"  compile:  {'enabled' if can_compile else 'disabled (use Linux/WSL2 for full gains)'}")
    print()
    hline("═")


if __name__ == "__main__":
    main()
