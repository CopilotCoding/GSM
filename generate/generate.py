"""
Generate MIDI from a trained GSM checkpoint.
"""

import json
import argparse
from pathlib import Path

import torch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.gsm import GSM


def load_tokenizer(vocab_path):
    from miditok import REMI, TokenizerConfig
    import os
    config = TokenizerConfig(
        num_velocities=16, use_chords=False, use_programs=False,
        use_tempos=True, use_time_signatures=True,
    )
    if os.path.exists(vocab_path):
        try:
            return REMI(params=vocab_path)
        except Exception:
            pass
        try:
            tok = REMI(config)
            tok.load_params(vocab_path)
            return tok
        except Exception:
            pass
    return REMI(config)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
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


def tokens_to_midi(token_ids, vocab_path, out_path):
    tok = load_tokenizer(vocab_path)
    try:
        midi = tok.decode(token_ids)
        midi.dump_midi(out_path)
        print(f"Saved: {out_path}")
        return
    except Exception:
        pass
    try:
        from miditok import TokSequence
        seq = TokSequence(ids=token_ids)
        midi = tok.tokens_to_midi([seq])
        midi.dump_midi(out_path)
        print(f"Saved: {out_path}")
        return
    except Exception as e:
        print(f"MIDI conversion failed: {e}")
        fallback = out_path.replace(".mid", "_tokens.json")
        with open(fallback, "w") as f:
            json.dump(token_ids, f)
        print(f"Saved raw tokens: {fallback}")


def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device)
    print(f"Loaded | state_dim={config['state_dim']} | embed_dim={config['embed_dim']} | "
          f"layers={config.get('n_layers',6)} | params={model.count_parameters():,}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for i in range(args.n_samples):
        print(f"\nGenerating {i+1}/{args.n_samples}...")
        if args.prompt_tokens:
            ids = [int(x) for x in args.prompt_tokens.split(",")]
            prompt = torch.tensor([ids], dtype=torch.long, device=device)
        else:
            prompt = torch.randint(0, config["vocab_size"], (1, 4), device=device)

        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=args.length,
                                        temperature=args.temperature, top_k=args.top_k)

        token_ids = generated[0].cpu().tolist()
        print(f"Generated {len(token_ids)} tokens")
        tokens_to_midi(token_ids, args.vocab_path,
                       str(Path(args.out_dir) / f"sample_{i+1:03d}.mid"))

    print(f"\nDone. {args.n_samples} samples → {args.out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab_path", default="vocab.json")
    parser.add_argument("--out_dir", default="generated")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--prompt_tokens", type=str, default=None)
    args = parser.parse_args()
    generate(args)
