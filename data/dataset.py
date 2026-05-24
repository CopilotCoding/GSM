"""
Dataset for GSM training.
Loads tokenized MIDI sequences from JSON files and yields fixed-length chunks.
"""

import json
import random
from pathlib import Path
from torch.utils.data import Dataset
import torch


class MIDIDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = 256, vocab_size: int = None):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.chunks = []

        paths = list(Path(data_dir).glob("*.json"))
        random.shuffle(paths)

        for p in paths:
            try:
                with open(p) as f:
                    ids = json.load(f)
                if vocab_size:
                    ids = [i for i in ids if i < vocab_size]
                # Slice into overlapping chunks
                for start in range(0, len(ids) - seq_len, seq_len // 2):
                    chunk = ids[start:start + seq_len + 1]
                    if len(chunk) == seq_len + 1:
                        self.chunks.append(chunk)
            except Exception:
                continue

        print(f"Loaded {len(self.chunks)} sequence chunks from {len(paths)} files")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
