"""
Dataset for GSM training.
Reads from memory-mapped binary (fast) OR falls back to JSON files (slow).
Use data/pack.py to convert JSON dataset to binary first.
"""

import json
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch


class MIDIDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = 256, vocab_size: int = None):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        data_path = Path(data_dir)
        bin_path = data_path / "sequences.bin"
        meta_path = data_path / "meta.json"

        if bin_path.exists() and meta_path.exists():
            # Fast path: memory-mapped binary
            with open(meta_path) as f:
                meta = json.load(f)
            self.total = meta["total_chunks"]
            self.chunk_len = meta["chunk_len"]
            self.data = np.memmap(bin_path, dtype=np.uint16, mode='r',
                                  shape=(self.total, self.chunk_len))
            self._mode = "mmap"
            print(f"Loaded {self.total:,} chunks from packed binary (instant)")
        else:
            # Slow fallback: JSON files
            print("No packed binary found — loading from JSON (slow). Run data/pack.py first.")
            self.chunks = []
            paths = list(data_path.glob("*.json"))
            random.shuffle(paths)
            for p in paths:
                try:
                    with open(p) as f:
                        ids = json.load(f)
                    if vocab_size:
                        ids = [i for i in ids if i < vocab_size]
                    chunk_len = seq_len + 1
                    stride = seq_len // 2
                    for start in range(0, len(ids) - chunk_len + 1, stride):
                        chunk = ids[start:start + chunk_len]
                        if len(chunk) == chunk_len:
                            self.chunks.append(chunk)
                except Exception:
                    continue
            self.total = len(self.chunks)
            self.chunk_len = seq_len + 1
            self._mode = "json"
            print(f"Loaded {self.total:,} sequence chunks from {len(paths)} files")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if self._mode == "mmap":
            chunk = self.data[idx].astype(np.int64)
        else:
            chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
