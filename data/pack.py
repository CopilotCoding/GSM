"""
pack.py — Convert JSON dataset to memory-mapped binary.
Uses multiprocessing to parse JSON files in parallel.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp


def process_file(args):
    path, seq_len, vocab_size = args
    stride = seq_len // 2
    chunk_len = seq_len + 1
    try:
        with open(path) as f:
            ids = json.load(f)
        if vocab_size:
            ids = [i for i in ids if i < vocab_size]
        arr = np.array(ids, dtype=np.uint16)
        chunks = []
        for start in range(0, len(arr) - chunk_len + 1, stride):
            chunks.append(arr[start:start + chunk_len])
        return chunks
    except Exception:
        return []


def pack(data_dir: str, out_dir: str, seq_len: int = 256, vocab_size: int = None, workers: int = 8):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = list(Path(data_dir).glob("*.json"))
    print(f"Found {len(paths)} JSON files")

    chunk_len = seq_len + 1
    bin_path = out / "sequences.bin"
    idx = 0

    args_list = [(str(p), seq_len, vocab_size) for p in paths]

    with open(bin_path, "wb", buffering=64 * 1024 * 1024) as f:
        with mp.Pool(workers) as pool:
            for chunks in tqdm(pool.imap(process_file, args_list, chunksize=64), 
                               total=len(paths), desc="Packing"):
                for chunk in chunks:
                    f.write(chunk.tobytes())
                    idx += 1

    total_chunks = idx
    print(f"Total chunks: {total_chunks:,}")
    print(f"File size: {bin_path.stat().st_size / 1e9:.2f} GB")

    meta = {
        "total_chunks": total_chunks,
        "seq_len": seq_len,
        "chunk_len": chunk_len,
        "dtype": "uint16",
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. {total_chunks:,} chunks -> {bin_path}")
    print(f"\nNow train with: --data_dir {out_dir}")


if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    pack(args.data_dir, args.out_dir, args.seq_len, args.vocab_size, args.workers)
