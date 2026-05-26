"""
plot_training.py — Plot training metrics from training_log.csv.

Usage:
    python plot_training.py checkpoints/training_log.csv
    python plot_training.py checkpoints/training_log.csv --out training_plot.png
    python plot_training.py checkpoints/training_log.csv --smooth 200
"""

import argparse
import sys
from pathlib import Path

import csv
import math


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "step":         int(row["step"]),
                    "loss":         float(row["loss"]),
                    "avg_loss":     float(row["avg_loss"]),
                    "smooth_loss":  float(row.get("smooth_loss", row["avg_loss"])),
                    "lr":           float(row["lr"]),
                    "tok_per_sec":  float(row.get("tok_per_sec", 0)),
                    "vram_gb":      float(row.get("vram_reserved_gb") or row.get("vram_alloc_gb") or 0),
                    "elapsed_sec":  float(row.get("elapsed_sec", 0)),
                    "gpu_util":     float(row.get("gpu_util_pct", 0)),
                })
            except (ValueError, KeyError):
                continue
    return rows


def rolling_mean(values, window):
    out = []
    for i, v in enumerate(values):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot GSM training log")
    parser.add_argument("csv_path", help="Path to training_log.csv")
    parser.add_argument("--out", default=None, help="Output image path (default: show window)")
    parser.add_argument("--smooth", type=int, default=100, help="Smoothing window (default 100)")
    args = parser.parse_args()

    try:
        import matplotlib
        if args.out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    rows = load_csv(args.csv_path)
    if not rows:
        print(f"No data found in {args.csv_path}")
        sys.exit(1)

    steps       = [r["step"] for r in rows]
    loss        = [r["loss"] for r in rows]
    avg_loss    = [r["avg_loss"] for r in rows]
    smooth_loss = rolling_mean(loss, args.smooth)
    lr          = [r["lr"] for r in rows]
    tok_per_sec = [r["tok_per_sec"] / 1000 for r in rows]  # → k tok/s
    vram        = [r["vram_gb"] for r in rows]
    elapsed_h   = [r["elapsed_sec"] / 3600 for r in rows]
    gpu_util    = [r["gpu_util"] for r in rows]

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor="#0f0f0f")
    fig.suptitle(
        f"GSM Training  —  {Path(args.csv_path).parent.name}  "
        f"({steps[-1]:,} steps  |  best loss {min(avg_loss):.4f})",
        color="white", fontsize=13, y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    def ax(pos, title, ylabel, xlabel="Step"):
        a = fig.add_subplot(pos)
        a.set_facecolor("#1a1a1a")
        a.set_title(title, color="#aaaaaa", fontsize=10, pad=6)
        a.set_ylabel(ylabel, color="#666666", fontsize=9)
        a.set_xlabel(xlabel, color="#666666", fontsize=9)
        a.tick_params(colors="#555555", labelsize=8)
        for spine in a.spines.values():
            spine.set_edgecolor("#333333")
        a.grid(True, color="#2a2a2a", linewidth=0.5)
        return a

    # 1. Loss curve
    a1 = ax(gs[0, :], "Loss", "Loss")
    a1.plot(steps, loss, color="#333355", linewidth=0.4, alpha=0.5, label="per-step")
    a1.plot(steps, smooth_loss, color="#5588ff", linewidth=1.4, label=f"smooth ({args.smooth})")
    a1.plot(steps, avg_loss, color="#88aaff", linewidth=0.8, linestyle="--", alpha=0.6, label="epoch avg")
    a1.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white", edgecolor="#333333")
    best_step = steps[avg_loss.index(min(avg_loss))]
    a1.axvline(best_step, color="#ffaa44", linewidth=0.8, linestyle=":", alpha=0.7)
    a1.text(best_step, min(avg_loss), f" {min(avg_loss):.4f}", color="#ffaa44", fontsize=7, va="bottom")

    # 2. Learning rate
    a2 = ax(gs[1, 0], "Learning Rate", "LR")
    a2.plot(steps, lr, color="#44cc88", linewidth=1.0)
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))

    # 3. Throughput
    a3 = ax(gs[1, 1], "Throughput", "k tok/s")
    smooth_tps = rolling_mean(tok_per_sec, args.smooth)
    a3.plot(steps, tok_per_sec, color="#333333", linewidth=0.3, alpha=0.4)
    a3.plot(steps, smooth_tps, color="#ff8844", linewidth=1.2)
    a3.axhline(sum(tok_per_sec) / len(tok_per_sec), color="#ffaa66",
               linewidth=0.8, linestyle="--", alpha=0.6,
               label=f"avg {sum(tok_per_sec)/len(tok_per_sec):.1f}k")
    a3.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white", edgecolor="#333333")

    # 4. VRAM
    a4 = ax(gs[2, 0], "VRAM Reserved", "GB")
    a4.plot(steps, vram, color="#cc44aa", linewidth=0.8)
    if max(vram) > 0:
        a4.fill_between(steps, vram, alpha=0.15, color="#cc44aa")

    # 5. GPU utilization
    a5 = ax(gs[2, 1], "GPU Utilization", "%")
    smooth_util = rolling_mean(gpu_util, args.smooth)
    a5.plot(steps, gpu_util, color="#333333", linewidth=0.3, alpha=0.4)
    a5.plot(steps, smooth_util, color="#44cccc", linewidth=1.0)
    a5.set_ylim(0, 105)
    a5.axhline(sum(gpu_util) / len(gpu_util), color="#66dddd",
               linewidth=0.8, linestyle="--", alpha=0.6,
               label=f"avg {sum(gpu_util)/len(gpu_util):.0f}%")
    a5.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white", edgecolor="#333333")

    # ── stats footer ─────────────────────────────────────────────────────────
    total_tokens = rows[-1].get("tokens_total", 0) if "tokens_total" in rows[-1] else 0
    elapsed_h_val = elapsed_h[-1] if elapsed_h[-1] > 0 else 0
    stats = (
        f"Steps: {steps[-1]:,}  |  "
        f"Best loss: {min(avg_loss):.4f}  |  "
        f"Avg tok/s: {sum(tok_per_sec)/len(tok_per_sec)*1000:,.0f}  |  "
        f"Elapsed: {elapsed_h_val:.2f}h"
    )
    fig.text(0.5, 0.01, stats, ha="center", color="#555555", fontsize=8)

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
