"""
pack.py — Convert JSON dataset to memory-mapped binary.
Run once after pipeline.py. Training will load instantly after.

Usage:
    python -m data.pack --data_dir dataset --out_dir dataset_packed --seq_len 256
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def pack(data_dir: str, out_dir: str, seq_len: int = 256, vocab_size: int = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = list(Path(data_dir).glob("*.json"))
    print(f"Found {len(paths)} JSON files")

    # First pass: count total chunks so we can allocate exactly
    print("Counting chunks...")
    total_chunks = 0
    stride = seq_len // 2
    chunk_len = seq_len + 1  # x + y

    for p in tqdm(paths, desc="Counting"):
        try:
            with open(p) as f:
                ids = json.load(f)
            if vocab_size:
                ids = [i for i in ids if i < vocab_size]
            n = max(0, (len(ids) - chunk_len) // stride + 1)
            total_chunks += sum(
                1 for start in range(0, len(ids) - chunk_len + 1, stride)
                if len(ids[start:start + chunk_len]) == chunk_len
            )
        except Exception:
            continue

    print(f"Total chunks: {total_chunks:,}")

    # Allocate memory-mapped array
    mmap_path = out / "sequences.bin"
    data = np.memmap(mmap_path, dtype=np.uint16, mode='w+', shape=(total_chunks, chunk_len))

    # Second pass: write chunks
    print("Writing packed binary...")
    idx = 0
    for p in tqdm(paths, desc="Packing"):
        try:
            with open(p) as f:
                ids = json.load(f)
            if vocab_size:
                ids = [i for i in ids if i < vocab_size]
            for start in range(0, len(ids) - chunk_len + 1, stride):
                chunk = ids[start:start + chunk_len]
                if len(chunk) == chunk_len:
                    data[idx] = chunk
                    idx += 1
        except Exception:
            continue

    data.flush()

    # Save metadata
    meta = {
        "total_chunks": idx,
        "seq_len": seq_len,
        "chunk_len": chunk_len,
        "dtype": "uint16",
    }
    import json as _json
    with open(out / "meta.json", "w") as f:
        _json.dump(meta, f, indent=2)

    print(f"Done. {idx:,} chunks → {mmap_path}")
    print(f"File size: {mmap_path.stat().st_size / 1e9:.2f} GB")
    print(f"\nNow train with: --data_dir {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory of JSON files from pipeline.py")
    parser.add_argument("--out_dir", required=True, help="Output directory for packed binary")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=None)
    args = parser.parse_args()
    pack(args.data_dir, args.out_dir, args.seq_len, args.vocab_size)
