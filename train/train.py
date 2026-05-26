"""
Training script for GSM.
bf16 mixed precision + large batch for maximum throughput.
Full statistics, benchmarking, periodic disk saves, and CSV logging.
"""

import os
import csv
import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

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


def fmt_bytes(n):
    if n >= 1 << 30: return f"{n/(1<<30):.2f} GB"
    if n >= 1 << 20: return f"{n/(1<<20):.1f} MB"
    return f"{n/(1<<10):.0f} KB"


def gpu_stats():
    if not torch.cuda.is_available():
        return {}
    d = torch.cuda.current_device()
    alloc   = torch.cuda.memory_allocated(d)
    reserved = torch.cuda.memory_reserved(d)
    total   = torch.cuda.get_device_properties(d).total_memory
    util    = torch.cuda.utilization(d) if hasattr(torch.cuda, "utilization") else -1
    return {
        "vram_alloc_gb": round(reserved / 1e9, 3),
        "vram_reserved_gb": round(reserved / 1e9, 3),
        "vram_total_gb": round(total / 1e9, 3),
        "gpu_util_pct": util,
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, loss, config):
    torch.save({
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }, path)


def write_csv_row(csv_path, row: dict):
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def print_header(args, model, loader, vocab_size, device, dtype):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  GSM Training Run  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(bar)
    print(f"  Device:       {device}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Precision:    {str(dtype).split('.')[-1] if dtype else 'fp32'}")
    print(f"  Parameters:   {model.count_parameters():,}")
    print(f"  Vocab size:   {vocab_size}")
    print(f"  State dim:    {args.state_dim}  |  Embed dim: {args.embed_dim}")
    print(f"  TransformNet: {args.n_layers} layers, hidden {args.hidden_dim}, {args.n_pairs} rot pairs")
    print(f"  Dataset:      {len(loader.dataset):,} chunks  |  {len(loader):,} batches/epoch")
    print(f"  Batch size:   {args.batch_size}  |  Seq len: {args.seq_len}")
    print(f"  Tokens/batch: {args.batch_size * args.seq_len:,}")
    print(f"  Epochs:       {args.epochs}  |  LR: {args.lr:.1e} → {args.lr*0.1:.1e}")
    print(f"  Save every:   {args.save_steps} steps  |  Timed save every: {args.save_minutes} min")
    if torch.cuda.is_available():
        g = gpu_stats()
        print(f"  VRAM:         {g['vram_alloc_gb']:.2f} GB alloc / {g['vram_total_gb']:.1f} GB total")
    print(bar)
    print()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = None

    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(f"Vocab not found at {args.vocab_path}. Run data/pipeline.py first.")

    tok = load_tokenizer(args.vocab_path)
    vocab_size = len(tok)

    dataset = MIDIDataset(args.data_dir, seq_len=args.seq_len, vocab_size=vocab_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
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

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    criterion = nn.CrossEntropyLoss()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path     = out / "latest.pt"
    csv_path      = out / "training_log.csv"
    stats_path    = out / "run_stats.json"

    start_epoch = 0
    global_step = 0

    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt.get("step", 0)
        print(f"  Resumed at epoch {start_epoch}, step {global_step}, loss {ckpt['loss']:.4f}")

    config_out = {
        "vocab_size": vocab_size,
        "embed_dim": args.embed_dim,
        "state_dim": args.state_dim,
        "n_pairs": args.n_pairs,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "parameters": model.count_parameters(),
    }
    with open(out / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    print_header(args, model, loader, vocab_size, device, dtype)

    run_start      = time.time()
    last_timed_save = time.time()
    epoch_times    = []
    tokens_total   = 0
    recent_losses  = deque(maxlen=100)   # rolling window for smoothed loss
    best_loss      = float("inf")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss      = 0.0
        epoch_start     = time.time()
        steps_this_epoch = 0
        epoch_tokens    = 0

        for step, (x, y) in enumerate(loader):
            step_start = time.time()
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

            step_time    = time.time() - step_start
            loss_val     = loss.item()
            tokens_step  = args.batch_size * args.seq_len
            tok_per_sec  = tokens_step / max(step_time, 1e-6)

            total_loss       += loss_val
            global_step      += 1
            steps_this_epoch += 1
            epoch_tokens     += tokens_step
            tokens_total     += tokens_step
            recent_losses.append(loss_val)

            avg_loss     = total_loss / steps_this_epoch
            smooth_loss  = sum(recent_losses) / len(recent_losses)
            lr_now       = scheduler.get_last_lr()[0]
            elapsed      = time.time() - epoch_start
            rate         = steps_this_epoch / max(elapsed, 1e-6)
            eta_epoch    = (len(loader) - step - 1) / max(rate, 1e-6)
            eta_run      = eta_epoch + (elapsed / steps_this_epoch * len(loader)) * (args.epochs - epoch - 1)
            run_elapsed  = time.time() - run_start

            # ── print every N steps ──────────────────────────────────────────
            if global_step % args.print_steps == 0:
                g = gpu_stats()
                vram_str = f" | vram {g['vram_alloc_gb']:.2f}/{g['vram_total_gb']:.1f}GB" if g else ""
                print(
                    f"  step {global_step:>7} | ep {epoch+1}/{args.epochs} "
                    f"| loss {avg_loss:.4f} (smooth {smooth_loss:.4f}) "
                    f"| lr {lr_now:.2e} "
                    f"| {rate:.2f}it/s | {tok_per_sec/1000:.1f}k tok/s "
                    f"| ETA ep {fmt_time(eta_epoch)} | ETA run {fmt_time(eta_run)}"
                    f"{vram_str}"
                )

            # ── CSV log every step ───────────────────────────────────────────
            g = gpu_stats()
            write_csv_row(csv_path, {
                "step": global_step,
                "epoch": epoch + 1,
                "loss": round(loss_val, 6),
                "avg_loss": round(avg_loss, 6),
                "smooth_loss": round(smooth_loss, 6),
                "lr": round(lr_now, 8),
                "it_per_sec": round(rate, 3),
                "tok_per_sec": round(tok_per_sec, 0),
                "tokens_total": tokens_total,
                "vram_alloc_gb": g.get("vram_alloc_gb", ""),
                "vram_reserved_gb": g.get("vram_reserved_gb", ""),
                "gpu_util_pct": g.get("gpu_util_pct", ""),
                "elapsed_sec": round(time.time() - run_start, 1),
                "timestamp": datetime.now().isoformat(),
            })

            # ── step checkpoint ──────────────────────────────────────────────
            if global_step % args.save_steps == 0:
                save_checkpoint(ckpt_path, model, optimizer, scheduler,
                                epoch, global_step, avg_loss, config_out)
                print(f"  >>> [step ckpt] step {global_step} | loss {avg_loss:.4f} | {fmt_time(run_elapsed)} elapsed")

            # ── timed checkpoint ─────────────────────────────────────────────
            if (time.time() - last_timed_save) >= args.save_minutes * 60:
                timed_path = out / f"timed_{datetime.now().strftime('%Y%m%d_%H%M%S')}_step{global_step}.pt"
                save_checkpoint(timed_path, model, optimizer, scheduler,
                                epoch, global_step, avg_loss, config_out)
                last_timed_save = time.time()
                print(f"  >>> [timed ckpt] {timed_path.name} | loss {avg_loss:.4f}")

        # ── end of epoch ─────────────────────────────────────────────────────
        avg_loss    = total_loss / len(loader)
        epoch_time  = time.time() - epoch_start
        epoch_times.append(epoch_time)
        epoch_tok_per_sec = epoch_tokens / max(epoch_time, 1e-6)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = out / "best.pt"
            save_checkpoint(best_path, model, optimizer, scheduler,
                            epoch + 1, global_step, avg_loss, config_out)
            best_tag = "  ★ new best"
        else:
            best_tag = ""

        epochs_left = args.epochs - epoch - 1
        avg_epoch_t = sum(epoch_times[-3:]) / len(epoch_times[-3:]) if epoch_times else epoch_time
        eta_finish  = datetime.now() + timedelta(seconds=(avg_epoch_t * epochs_left))

        print(f"\n  {'='*66}")
        print(f"  Epoch {epoch+1}/{args.epochs} complete{best_tag}")
        print(f"    Loss:          {avg_loss:.4f}  (best: {best_loss:.4f})")
        print(f"    Epoch time:    {fmt_time(epoch_time)}")
        print(f"    Total elapsed: {fmt_time(time.time() - run_start)}")
        print(f"    Tokens seen:   {tokens_total:,}  ({epoch_tokens:,} this epoch)")
        print(f"    Tok/sec:       {epoch_tok_per_sec/1000:.1f}k (epoch avg)")
        print(f"    ETA finish:    {eta_finish.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  {'='*66}\n")

        save_checkpoint(ckpt_path, model, optimizer, scheduler,
                        epoch + 1, global_step, avg_loss, config_out)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "config": config_out,
            "loss": avg_loss,
        }, out / f"epoch_{epoch+1:03d}_loss{avg_loss:.4f}.pt")

    # ── final stats ───────────────────────────────────────────────────────────
    total_time = time.time() - run_start
    final_stats = {
        "total_time_sec": round(total_time, 1),
        "total_tokens": tokens_total,
        "avg_tok_per_sec": round(tokens_total / total_time, 0),
        "best_loss": round(best_loss, 6),
        "epochs": args.epochs,
        "parameters": model.count_parameters(),
        "state_dim": args.state_dim,
        "embed_dim": args.embed_dim,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "completed": datetime.now().isoformat(),
    }
    with open(stats_path, "w") as f:
        json.dump(final_stats, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Training complete in {fmt_time(total_time)}")
    print(f"  Total tokens processed: {tokens_total:,}")
    print(f"  Average throughput:     {tokens_total/total_time/1000:.1f}k tok/sec")
    print(f"  Best loss:              {best_loss:.4f}")
    print(f"  Stats saved to:         {stats_path}")
    print(f"  Log saved to:           {csv_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--vocab_path",  default="vocab.json")
    parser.add_argument("--out_dir",     default="checkpoints")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--seq_len",     type=int,   default=256)
    parser.add_argument("--embed_dim",   type=int,   default=512)
    parser.add_argument("--state_dim",   type=int,   default=4096)
    parser.add_argument("--n_pairs",     type=int,   default=128)
    parser.add_argument("--hidden_dim",  type=int,   default=1024)
    parser.add_argument("--n_layers",    type=int,   default=6)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--workers",     type=int,   default=0)
    parser.add_argument("--save_steps",  type=int,   default=2000,
                        help="Save latest.pt every N steps")
    parser.add_argument("--save_minutes",type=int,   default=30,
                        help="Save a timestamped checkpoint every N minutes")
    parser.add_argument("--print_steps", type=int,   default=10,
                        help="Print stats every N steps")
    args = parser.parse_args()
    train(args)
