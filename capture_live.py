"""
capture_live.py — run play_live.py's generation path headlessly and save tokens.

`play_live.py` streams notes straight to a MIDI output device, so there is no way
to inspect what it produces without listening to it. This runs the identical
generation path -- same checkpoint loader, same carry-based incremental step,
same sampling -- but writes token streams to disk instead of sending them to an
audio device.

Why the live path specifically: it uses `GeometricStateStep.step()`, the
one-token-at-a-time path, which is different code from the parallel scan used in
training and from `GSM.generate()`. Checking only the scan would leave the
incremental path unverified, and that is the path a live session actually runs.

Pairs with check_memorization.py:

    python capture_live.py --checkpoint checkpoints/best.pt
    python check_memorization.py --generated generated_live

Defaults mirror play_live.py (temperature 0.9, top_k 50, random 4-token prompt)
so the captured output reflects what a live session would actually play. Override
them to test other settings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from play_live import load_model  # reuse the real loader, not a copy of it


def generate_tokens(model, vocab_size, device, *, length, temperature, top_k,
                    prompt_ids=None, seed=None):
    """
    One sample via the incremental step path.

    This mirrors play_live.generation_worker's inner loop exactly. If that
    function changes, this should be updated to match -- the whole point is that
    the captured tokens are what a live session would play.
    """
    if seed is not None:
        torch.manual_seed(seed)

    ids = list(prompt_ids) if prompt_ids else torch.randint(0, vocab_size, (4,)).tolist()
    prompt = torch.tensor(ids, dtype=torch.long, device=device)

    with torch.no_grad():
        # Ingest the prompt through the carry-based step.
        carry = model.step.init_carry(model.S0, 1)
        E = model.embedding(prompt.unsqueeze(0))
        for t in range(prompt.shape[0]):
            carry, S = model.step.step(carry, E[:, t])

        for _ in range(length):
            e        = model.embedding(torch.tensor([[ids[-1]]], device=device)).squeeze(1)
            carry, S = model.step.step(carry, e)
            logits   = model.decoder(S) / temperature
            if top_k > 0:
                top_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_vals[:, -1:]] = float("-inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            ids.append(next_id)

    return ids


def main():
    ap = argparse.ArgumentParser(
        description="Capture play_live.py's generation output as token files.")
    ap.add_argument("--checkpoint",  default="checkpoints/best.pt")
    ap.add_argument("--out_dir",     default="generated_live")
    ap.add_argument("--n_samples",   type=int,   default=5)
    ap.add_argument("--length",      type=int,   default=512)
    ap.add_argument("--temperature", type=float, default=0.9,  help="play_live default")
    ap.add_argument("--top_k",       type=int,   default=50,   help="play_live default")
    ap.add_argument("--seed",        type=int,   default=1234,
                    help="base seed; sample i uses seed+i (reproducible)")
    ap.add_argument("--prompt_tokens", default=None,
                    help="comma-separated seed ids; default is a random 4-token prompt")
    ap.add_argument("--device",      default=None, help="cuda or cpu (default: auto)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    model, cfg = load_model(ckpt, device)
    vocab_size = cfg["vocab_size"]
    print(f"Loaded {ckpt}  |  state_dim={cfg['state_dim']}  vocab={vocab_size}  device={device}")
    print(f"Sampling: temperature={args.temperature}  top_k={args.top_k}  "
          f"length={args.length}\n")

    prompt_ids = ([int(x) for x in args.prompt_tokens.split(",")]
                  if args.prompt_tokens else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_samples):
        ids = generate_tokens(
            model, vocab_size, device,
            length=args.length, temperature=args.temperature, top_k=args.top_k,
            prompt_ids=prompt_ids,
            seed=(args.seed + i) if args.seed is not None else None,
        )
        # *_tokens.json is the suffix check_memorization.py reads directly,
        # skipping a MIDI round-trip that would distort the comparison.
        path = out_dir / f"live_{i+1:03d}_tokens.json"
        path.write_text(json.dumps(ids))
        print(f"  live_{i+1:03d}: {len(ids)} tokens  prompt={ids[:4]}  → {path.name}")

    print(f"\nWrote {args.n_samples} samples → {out_dir}/")
    print(f"Check them with:\n  python check_memorization.py --generated {out_dir}")


if __name__ == "__main__":
    main()
