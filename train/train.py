"""
Training script for GSM.
bf16 mixed precision + large batch for maximum throughput.
"""

import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import MIDIDataset
from model.gsm import GSM


def load_tokenizer(vocab_path):
    from miditok import REMI, TokenizerConfig
    config = TokenizerConfig(
        num_velocities=16,
        use_chords=False,
        use_programs=False,
        use_tempos=True,
        use_time_signatures=True,
    )
    tok = REMI(config)
    if os.path.exists(vocab_path):
        try:
            tok = REMI(params=vocab_path)
        except Exception:
            try:
                tok.load_params(vocab_path)
            except Exception:
                pass
    return tok


def fmt_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Precision: bf16")
        else:
            dtype = torch.float16
            print("Precision: fp16")
    else:
        dtype = None
        print("Precision: fp32 (CPU)")

    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(f"Vocab not found at {args.vocab_path}. Run data/pipeline.py first.")

    tok = load_tokenizer(args.vocab_path)
    vocab_size = len(tok)
    print(f"Vocab size: {vocab_size}")

    dataset = MIDIDataset(args.data_dir, seq_len=args.seq_len, vocab_size=vocab_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = GSM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        state_dim=args.state_dim,
        n_pairs=args.n_pairs,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    print(f"Parameters:   {model.count_parameters():,}")
    print(f"State dim:    {args.state_dim}  |  Embed dim: {args.embed_dim}")
    print(f"TransformNet: {args.n_layers} layers, hidden {args.hidden_dim}")
    print(f"Batch size:   {args.batch_size}  |  Batches/epoch: {len(loader)}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    criterion = nn.CrossEntropyLoss()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    ckpt_path = Path(args.out_dir) / "latest.pt"
    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    config_out = {
        "vocab_size": vocab_size,
        "embed_dim": args.embed_dim,
        "state_dim": args.state_dim,
        "n_pairs": args.n_pairs,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
    }
    with open(Path(args.out_dir) / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training GSM  |  {args.epochs} epochs  |  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    global_step = start_epoch * len(loader)
    run_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()

        bar = tqdm(
            loader,
            desc=f"Epoch {epoch+1:>3}/{args.epochs}",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        for step, (x, y) in enumerate(bar):
            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast('cuda', dtype=dtype, enabled=(dtype is not None)):
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            if dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            global_step += 1

            avg_loss = total_loss / (step + 1)
            lr_now = scheduler.get_last_lr()[0]
            steps_done = global_step - start_epoch * len(loader)
            steps_total = (args.epochs - start_epoch) * len(loader)
            elapsed_run = time.time() - run_start
            eta_run = (elapsed_run / max(steps_done, 1)) * (steps_total - steps_done)

            bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{lr_now:.1e}",
                "time": datetime.now().strftime("%H:%M:%S"),
                "ETA": fmt_time(eta_run),
            }, refresh=True)

        avg_loss = total_loss / len(loader)
        epoch_time = time.time() - epoch_start
        epochs_left = args.epochs - epoch - 1
        eta_finish = datetime.now() + timedelta(seconds=(epoch_time * epochs_left))

        print(f"\n  ✓ Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | "
              f"took={fmt_time(epoch_time)} | total={fmt_time(time.time()-run_start)} | "
              f"finish≈{eta_finish.strftime('%H:%M:%S')}\n")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss,
            "config": config_out,
        }, ckpt_path)

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "config": config_out,
        }, Path(args.out_dir) / f"epoch_{epoch+1:03d}_loss{avg_loss:.4f}.pt")

    total_time = time.time() - run_start
    print(f"\n{'='*60}")
    print(f"  Training complete in {fmt_time(total_time)}")
    print(f"  Checkpoints saved to {args.out_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--vocab_path", default="vocab.json")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--state_dim", type=int, default=4096)
    parser.add_argument("--n_pairs", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    train(args)
