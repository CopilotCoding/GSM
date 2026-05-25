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


def save_checkpoint(path, model, optimizer, epoch, step, loss, config):
    torch.save({
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
        "config": config,
    }, path)


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
        pin_memory=False,  # already pinned in dataset
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
    print(f"Checkpoint every {args.save_steps} steps")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    criterion = nn.CrossEntropyLoss()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    start_step = 0
    ckpt_path = Path(args.out_dir) / "latest.pt"

    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        start_step = ckpt.get("step", 0)

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
    epoch_times = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()
        steps_this_epoch = 0

        for step, (x, y) in enumerate(loader):
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
            steps_this_epoch += 1

            avg_loss = total_loss / steps_this_epoch
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - epoch_start
            rate = steps_this_epoch / max(elapsed, 1e-6)
            steps_left_epoch = len(loader) - step - 1
            epochs_left = args.epochs - epoch - 1
            eta_epoch = steps_left_epoch / max(rate, 1e-6)
            avg_epoch_t = elapsed / steps_this_epoch * len(loader)
            eta_run = eta_epoch + avg_epoch_t * epochs_left

            # Print every 100 steps
            if global_step % 10 == 0:
                print(f"  step {global_step:>8} | epoch {epoch+1}/{args.epochs} | "
                      f"loss {avg_loss:.4f} | lr {lr_now:.1e} | "
                      f"{rate:.2f}it/s | ETA epoch {fmt_time(eta_epoch)} | "
                      f"ETA run {fmt_time(eta_run)}")

            # Step checkpoint
            if global_step % args.save_steps == 0:
                save_checkpoint(ckpt_path, model, optimizer, epoch, global_step, avg_loss, config_out)
                print(f"  >>> Checkpoint saved at step {global_step} | loss {avg_loss:.4f}")

        avg_loss = total_loss / len(loader)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        epochs_left = args.epochs - epoch - 1
        avg_epoch_t = sum(epoch_times[-3:]) / len(epoch_times[-3:]) if epoch_times else epoch_time
        eta_finish = datetime.now() + timedelta(seconds=(avg_epoch_t * epochs_left))

        print(f"\n  ✓ Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | "
              f"took={fmt_time(epoch_time)} | total={fmt_time(time.time()-run_start)} | "
              f"finish≈{eta_finish.strftime('%H:%M:%S')}\n")

        save_checkpoint(ckpt_path, model, optimizer, epoch + 1, global_step, avg_loss, config_out)
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
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every N steps")
    args = parser.parse_args()
    train(args)
