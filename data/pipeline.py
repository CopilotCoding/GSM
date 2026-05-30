"""
MIDI Data Pipeline for GSM (Geometric State Machine)
Tokenizes MIDI files into integer sequences using MidiTok REMI.
"""

import os
import json
import argparse
import multiprocessing
from pathlib import Path
from tqdm import tqdm


def build_tokenizer(vocab_path: str = None):
    from miditok import REMI, TokenizerConfig
    config = TokenizerConfig(
        num_velocities=16,
        use_chords=False,
        use_programs=False,
        use_tempos=True,
        use_time_signatures=True,
    )
    tok = REMI(config)
    if vocab_path and os.path.exists(vocab_path):
        try:
            tok.load_params(vocab_path)
        except Exception:
            pass  # use default vocab, no HuggingFace calls
    return tok


def tokenize_midi_path(tok, path):
    """Try both miditok v2 and v3 APIs."""
    # v3 API: encode from file path directly
    try:
        result = tok.encode(path)
        if result and hasattr(result[0], 'ids'):
            return result[0].ids
        if result and isinstance(result[0], list):
            return result[0]
    except Exception:
        pass

    # v2 API: pass pretty_midi object
    try:
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(str(path))
        result = tok(midi)
        if result and hasattr(result[0], 'ids'):
            return result[0].ids
        if result and isinstance(result[0], list):
            return result[0]
    except Exception:
        pass

    return None


# Module-level tokenizer for pool workers — built once per worker, not once per file.
_worker_tok = None
_worker_out_dir = None

def _pool_worker_init(vocab_path, out_dir):
    global _worker_tok, _worker_out_dir
    _worker_tok = build_tokenizer(vocab_path)
    _worker_out_dir = out_dir


def process_file(path):
    """Worker function — uses module-level tokenizer set by _pool_worker_init."""
    try:
        ids = tokenize_midi_path(_worker_tok, path)

        if ids is None or len(ids) < 16:
            return None

        stem = Path(path).stem
        out_path = Path(_worker_out_dir) / f"{stem}.json"
        with open(out_path, "w") as f:
            json.dump(ids, f)

        return len(ids)
    except Exception:
        return None


def run_pipeline(midi_dir: str, out_dir: str, vocab_path: str, workers: int = 8):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    paths = list(Path(midi_dir).rglob("*.mid")) + list(Path(midi_dir).rglob("*.midi"))
    print(f"Found {len(paths)} MIDI files")

    if not paths:
        print("ERROR: No MIDI files found. Check your --midi_dir path.")
        return 0

    # Test single file first
    print("Testing single file before full run...")
    test_result = process_file((paths[0], vocab_path, out_dir))
    if test_result is None:
        print("Single file test failed. Diagnosing...")
        try:
            tok = build_tokenizer(None)
            ids = tokenize_midi_path(tok, paths[0])
            print(f"  Without vocab: ids={ids is not None}, len={len(ids) if ids else 0}")
        except Exception as e:
            print(f"  Tokenizer error: {e}")
        try:
            import pretty_midi
            midi = pretty_midi.PrettyMIDI(str(paths[0]))
            print(f"  pretty_midi OK: {len(midi.instruments)} instruments")
        except Exception as e:
            print(f"  pretty_midi error: {e}")
        vocab_path_used = None
    else:
        print(f"  OK — {test_result} tokens from {paths[0].name}")
        vocab_path_used = vocab_path

    # Build vocab in single process
    print("Building tokenizer vocab...")
    tok = build_tokenizer(vocab_path_used)
    for p in tqdm(paths[:min(500, len(paths))], desc="Vocab sample"):
        try:
            tokenize_midi_path(tok, p)
        except Exception:
            pass

    # Save vocab — try new API then old
    try:
        tok.save(vocab_path)
    except Exception:
        try:
            tok.save_params(vocab_path)
        except Exception as e:
            print(f"Warning: could not save vocab: {e}")

    vocab_size = len(tok)
    print(f"Vocab size: {vocab_size}")

    # Process all files
    success = 0
    errors = 0

    # Initializer builds the tokenizer once per worker instead of once per file.
    with multiprocessing.Pool(
        workers,
        initializer=_pool_worker_init,
        initargs=(vocab_path_used, out_dir),
    ) as pool:
        for result in tqdm(pool.imap_unordered(process_file, [str(p) for p in paths]),
                           total=len(paths), desc="Processing"):
            if result is not None:
                success += 1
            else:
                errors += 1

    print(f"Done. {success}/{len(paths)} files processed → {out_dir}  ({errors} skipped)")
    return vocab_size


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", required=True)
    parser.add_argument("--out_dir", default="dataset")
    parser.add_argument("--vocab_path", default="vocab.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_pipeline(args.midi_dir, args.out_dir, args.vocab_path, args.workers)
