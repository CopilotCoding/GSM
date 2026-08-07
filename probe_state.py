"""
probe_state.py — does the geometric state actually carry information?

A GSM can reach low loss two ways: by using the state to accumulate context, or
by ignoring the state and predicting from the current token alone. Loss cannot
distinguish these. This tool can.

Three measurements:

  1. Transform parameters. The folded multiplier `a = gate*scale + (1-gate)`
     controls how much of the previous state survives each step. Its cumulative
     product is the state's memory. If that product decays fast, the state is
     being discarded no matter what the architecture permits.

  2. State ablation. Corrupt S mid-sequence (zero / noise / shuffle) and compare
     subsequent predictions to the clean run. High agreement means the state was
     not being used -- the strongest available evidence that the geometry is
     decorative.

  3. Memory horizon. How many steps until a corrupted state stops affecting
     output, and how fast S0's contribution decays below numerical relevance.

Usage:
    python probe_state.py
    python probe_state.py --checkpoint checkpoints/epoch_010_loss0.8262.pt
    python probe_state.py --corrupt-at 128 --seq-len 512

Interpretation: agreement above ~95% after ablation means the state is nearly
irrelevant to prediction. A model relying on long context would diverge sharply
and never re-converge.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from play_live import load_model


def pick_sample(dataset_dir: Path, seq_len: int) -> list[int]:
    files = sorted(dataset_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No token files in {dataset_dir}; run data.pipeline first.")
    for f in files:
        toks = json.loads(f.read_text())
        if isinstance(toks, dict):
            toks = toks.get("ids") or toks.get("tokens") or []
        if isinstance(toks, list) and toks and isinstance(toks[0], list):
            toks = toks[0]
        if len(toks) >= seq_len:
            return [int(t) for t in toks[:seq_len]], f.stem
    toks = json.loads(files[0].read_text())
    return [int(t) for t in toks], files[0].stem


def main():
    ap = argparse.ArgumentParser(description="Measure whether the GSM state carries information.")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--dataset",    default="dataset")
    ap.add_argument("--seq-len",    type=int, default=512)
    ap.add_argument("--corrupt-at", type=int, default=None,
                    help="step at which to corrupt S (default: halfway)")
    ap.add_argument("--device",     default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(Path(args.checkpoint), device)
    toks, src = pick_sample(Path(args.dataset), args.seq_len)
    x = torch.tensor([toks], device=device)
    T = x.shape[1]
    k = args.corrupt_at if args.corrupt_at is not None else T // 2

    print(f"checkpoint : {args.checkpoint}")
    print(f"input      : {src} ({T} tokens)\n")

    with torch.no_grad():
        E = model.embedding(x)
        scale, shift, gate, angles = model.step.transform_net(E)
        a, b = model.step._affine(scale, shift, gate)

        print("=" * 72)
        print("1. LEARNED TRANSFORM PARAMETERS")
        print("=" * 72)
        print(f"  gate    mean={gate.mean():.4f}   min={gate.min():.4f}  max={gate.max():.4f}")
        print(f"  scale   mean={scale.mean():.4f}")
        print(f"  a       mean={a.mean():.6f}  min={a.min():.6f}  max={a.max():.6f}")
        print(f"  |shift| mean={shift.abs().mean():.4f}    |angles| mean={angles.abs().mean():.4f}")
        logprod = torch.log(a.clamp_min(1e-30)).sum(dim=1)
        print(f"\n  cumulative multiplier over {T} tokens: median e^{logprod.median():.1f}")
        print("  (this is how much of S0 survives to the end; e^-20 or lower means")
        print("   the initial state is numerically gone)")

        # decay curve
        cum = torch.log(a.clamp_min(1e-30)).cumsum(1)[0]
        med = cum.median(dim=1).values
        horizon = next((t for t in range(len(med)) if med[t] < math.log(1e-6)), None)
        print("\n  decay of S0's contribution:")
        for t in (1, 5, 10, 20, 50, 100):
            if t < len(med):
                print(f"    after {t:>3} tokens: e^{med[t]:>7.1f}")
        if horizon is not None:
            print(f"\n  EFFECTIVE MEMORY HORIZON: {horizon} tokens "
                  f"(S0 contribution < 1e-6 beyond this)")

        # --- ablation ---
        def replay(corrupt_at=None, mode=None, seed=0):
            S = model.S0.unsqueeze(0).expand(1, -1).contiguous()
            acc = torch.zeros(1, model.step.rotary.n_pairs, device=device)
            outs = []
            for t in range(T):
                S = a[:, t] * S + b[:, t]
                acc = acc + angles[:, t]
                if corrupt_at is not None and t == corrupt_at:
                    torch.manual_seed(seed)
                    if mode == "zero":
                        S = torch.zeros_like(S)
                    elif mode == "noise":
                        S = torch.randn_like(S) * S.abs().mean()
                    elif mode == "shuffle":
                        S = S[:, torch.randperm(S.shape[1], device=device)]
                outs.append(model.step.norm(model.step.rotary(S, acc)))
            return model.decoder(torch.cat(outs, 0)).argmax(-1)

        print("\n" + "=" * 72)
        print(f"2. STATE ABLATION (corrupt S at t={k})")
        print("=" * 72)
        clean = replay()
        for mode in ("zero", "noise", "shuffle"):
            cor = replay(corrupt_at=k, mode=mode)
            after_c, after_x = clean[k + 1:], cor[k + 1:]
            agree = (after_c == after_x).float().mean().item()
            diff = (after_c != after_x).nonzero().flatten()
            last = int(diff.max()) + 1 if len(diff) else 0
            print(f"  {mode:8s} → {agree*100:5.1f}% of later predictions unchanged"
                  f"   ({len(diff)} differ, last at +{last})")

        print("\n" + "=" * 72)
        print("VERDICT")
        print("=" * 72)
        cor = replay(corrupt_at=k, mode="zero")
        agree = (clean[k + 1:] == cor[k + 1:]).float().mean().item()
        if agree > 0.95:
            print(f"  State is NOT carrying information ({agree*100:.1f}% unchanged after zeroing).")
            print("  The model predicts from recent tokens, not accumulated context.")
            print("  Low loss here does not indicate the geometry is doing work.")
        elif agree > 0.75:
            print(f"  State is weakly used ({agree*100:.1f}% unchanged after zeroing).")
        else:
            print(f"  State is materially used ({agree*100:.1f}% unchanged after zeroing).")
        print("=" * 72)


if __name__ == "__main__":
    main()
