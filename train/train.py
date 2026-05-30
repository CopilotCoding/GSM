"""
Training script for GSM.
bf16 mixed precision + large batch for maximum throughput.
Full statistics, benchmarking, periodic disk saves, and CSV logging.
Rich terminal UI with live progress bars and stats panels.
"""

import os
import csv
import json
import argparse
import time
import queue
import threading
import platform
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn, TaskProgressColumn
)
from rich.columns import Columns
from rich.text import Text
from rich import box
from rich.rule import Rule

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import MIDIDataset
from model.gsm import GSM

console = Console()


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
    reserved = torch.cuda.memory_reserved(d)
    total    = torch.cuda.get_device_properties(d).total_memory
    util     = torch.cuda.utilization(d) if hasattr(torch.cuda, "utilization") else -1
    return {
        "vram_alloc_gb":    round(reserved / 1e9, 3),
        "vram_reserved_gb": round(reserved / 1e9, 3),
        "vram_total_gb":    round(total / 1e9, 3),
        "gpu_util_pct":     util,
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, loss, config):
    torch.save({
        "epoch":     epoch,
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss":      loss,
        "config":    config,
        "timestamp": datetime.now().isoformat(),
    }, path)


def flush_csv_buffer(csv_path, buffer: list):
    if not buffer:
        return
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(buffer[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(buffer)
    buffer.clear()


class PrefetchLoader:
    """
    Wraps a DataLoader and pushes batches to a background thread queue.
    Overlaps CPU->GPU transfer with GPU compute. No spawn overhead (Windows-safe).
    """
    def __init__(self, loader, device, queue_size=2):
        self.loader     = loader
        self.device     = device
        self.queue_size = queue_size

    def __iter__(self):
        q        = queue.Queue(maxsize=self.queue_size)
        sentinel = object()

        def producer():
            for batch in self.loader:
                q.put(batch)
            q.put(sentinel)

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            x, y = item
            yield x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset


def make_stats_table(step, epoch, n_epochs, loss, smooth, lr, rate, tok_per_sec,
                     tokens_total, run_elapsed, eta_epoch, eta_run, g):
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column(style="bold cyan", no_wrap=True)
    t.add_column(style="white", min_width=12, no_wrap=True)
    t.add_column(style="bold cyan", no_wrap=True)
    t.add_column(style="white", min_width=12, no_wrap=True)

    t.add_row("Step",        f"{step:,}",
              "Epoch",       f"{epoch}/{n_epochs}")
    t.add_row("Loss",        f"{loss:.4f}",
              "Smooth",      f"{smooth:.4f}")
    t.add_row("LR",          f"{lr:.3e}",
              "it/s",        f"{rate:.2f}")
    t.add_row("Tok/s",       f"{tok_per_sec/1000:.1f}k",
              "Tokens",      f"{tokens_total/1e6:.2f}M")
    t.add_row("Elapsed",     fmt_time(run_elapsed),
              "ETA epoch",   fmt_time(eta_epoch))
    t.add_row("ETA run",     fmt_time(eta_run),
              "",            "")

    if g:
        util_str = f"{g['gpu_util_pct']}%" if g.get("gpu_util_pct", -1) >= 0 else "n/a"
        t.add_row("VRAM",    f"{g['vram_alloc_gb']:.1f}/{g['vram_total_gb']:.0f}GB",
                  "GPU util", util_str)
    return t


def print_header(args, model, loader, vocab_size, device, dtype, compiled):
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    precision   = str(dtype).split(".")[-1] if dtype else "fp32"

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column(style="bold cyan", no_wrap=True)
    t.add_column(style="white")
    t.add_column(style="bold cyan", no_wrap=True)
    t.add_column(style="white")

    t.add_row("Device",      f"{device_name}",      "Precision",   precision)
    t.add_row("Parameters",  f"{model.count_parameters():,}", "Vocab",  f"{vocab_size}")
    t.add_row("State dim",   f"{args.state_dim}",   "Embed dim",   f"{args.embed_dim}")
    t.add_row("Layers",      f"{args.n_layers}",    "Hidden",      f"{args.hidden_dim}")
    t.add_row("Rot pairs",   f"{args.n_pairs}",     "Dropout",     f"{args.dropout}")
    t.add_row("Chunks",      f"{len(loader.dataset):,}", "Batches/ep", f"{len(loader):,}")
    t.add_row("Batch size",  f"{args.batch_size}",  "Seq len",     f"{args.seq_len}")
    t.add_row("Tok/batch",   f"{args.batch_size * args.seq_len:,}", "Epochs", f"{args.epochs}")
    t.add_row("LR",          f"{args.lr:.1e} → {args.lr*0.1:.1e}", "compile", "yes" if compiled else "no")
    t.add_row("Save steps",  f"{args.save_steps}",  "Save mins",   f"{args.save_minutes}")

    if torch.cuda.is_available():
        g = gpu_stats()
        t.add_row("VRAM",    f"{g['vram_alloc_gb']:.2f} / {g['vram_total_gb']:.1f} GB", "", "")

    console.print()
    console.print(Panel(t, title=f"[bold magenta]GSM Training Run[/]  [dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]",
                        border_style="magenta"))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = None

    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(f"Vocab not found at {args.vocab_path}. Run data/pipeline.py first.")

    with console.status("[cyan]Loading tokenizer...[/]"):
        tok        = load_tokenizer(args.vocab_path)
        vocab_size = len(tok)

    with console.status("[cyan]Loading dataset...[/]"):
        dataset = MIDIDataset(args.data_dir, seq_len=args.seq_len, vocab_size=vocab_size)

    effective_workers = 0 if platform.system() == "Windows" else args.workers
    _loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    loader = PrefetchLoader(_loader, device, queue_size=2)

    with console.status("[cyan]Building model...[/]"):
        model = GSM(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            state_dim=args.state_dim,
            n_pairs=args.n_pairs,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
        ).to(device)

    # torch.compile with gradient checkpointing causes backward deadlocks.
    # Compile only the submodules that are purely feedforward (no checkpoint wrapper):
    # TransformNet and decoder. The recurrence is already compile-disabled.
    compiled = False
    if (
        hasattr(torch, "compile")
        and torch.__version__ >= "2.0"
        and device.type == "cuda"
        and platform.system() == "Linux"
    ):
        with console.status("[cyan]torch.compile — fusing TransformNet + decoder...[/]"):
            model.step.transform_net = torch.compile(model.step.transform_net)
            model.decoder            = torch.compile(model.decoder)
            compiled = True
        console.print("[green]✓[/] torch.compile enabled (TransformNet + decoder)")
    else:
        console.print("[yellow]·[/] torch.compile disabled (not Linux or PyTorch < 2.0)")

    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    criterion   = nn.CrossEntropyLoss()

    out       = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path  = out / "latest.pt"
    csv_path   = out / "training_log.csv"
    stats_path = out / "run_stats.json"

    start_epoch = 0
    global_step = 0

    if ckpt_path.exists():
        console.print(f"[cyan]Resuming from[/] {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt.get("step", 0)
        console.print(f"  [dim]epoch {start_epoch}, step {global_step}, loss {ckpt['loss']:.4f}[/]")

    config_out = {
        "vocab_size": vocab_size,
        "embed_dim":  args.embed_dim,
        "state_dim":  args.state_dim,
        "n_pairs":    args.n_pairs,
        "hidden_dim": args.hidden_dim,
        "n_layers":   args.n_layers,
        "dropout":    args.dropout,
        "parameters": model.count_parameters(),
    }
    with open(out / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    print_header(args, model, loader, vocab_size, device, dtype, compiled)

    run_start       = time.time()
    last_timed_save = time.time()
    epoch_times     = []
    _csv_buffer     = []
    tokens_total    = 0
    recent_losses   = deque(maxlen=100)
    best_loss       = float("inf")

    # ── progress bars ────────────────────────────────────────────────────────
    run_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Run[/]"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("epochs"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    epoch_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Epoch {task.fields[epoch_label]}[/]"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("·"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )

    run_task   = run_progress.add_task("run",   total=args.epochs - start_epoch)
    epoch_task = epoch_progress.add_task("epoch", total=len(loader), epoch_label=f"?/{args.epochs}")

    # live stats table updated every print_steps
    stats_renderable = {"table": Text("initializing...")}

    def make_layout():
        return Columns([
            Panel(run_progress,   title="[magenta]Overall[/]",  border_style="magenta", width=52),
            Panel(epoch_progress, title="[green]Epoch[/]",      border_style="green",   width=60),
        ])

    with Live(make_layout(), console=console, refresh_per_second=4) as live:

        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_loss       = 0.0
            epoch_start      = time.time()
            steps_this_epoch = 0
            epoch_tokens     = 0

            epoch_progress.reset(epoch_task, total=len(loader),
                                 epoch_label=f"{epoch+1}/{args.epochs}")

            for step, (x, y) in enumerate(loader):
                step_start = time.time()

                with torch.amp.autocast('cuda', dtype=dtype, enabled=(dtype is not None)):
                    logits = model(x)
                    loss   = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

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

                step_time        = time.time() - step_start
                loss_val         = loss.item()
                tokens_step      = args.batch_size * args.seq_len
                tok_per_sec      = tokens_step / max(step_time, 1e-6)
                total_loss       += loss_val
                global_step      += 1
                steps_this_epoch += 1
                epoch_tokens     += tokens_step
                tokens_total     += tokens_step
                recent_losses.append(loss_val)

                avg_loss    = total_loss / steps_this_epoch
                smooth_loss = sum(recent_losses) / len(recent_losses)
                lr_now      = scheduler.get_last_lr()[0]
                elapsed     = time.time() - epoch_start
                rate        = steps_this_epoch / max(elapsed, 1e-6)
                eta_epoch   = (len(loader) - step - 1) / max(rate, 1e-6)
                eta_run     = eta_epoch + (elapsed / steps_this_epoch * len(loader)) * (args.epochs - epoch - 1)
                run_elapsed = time.time() - run_start

                epoch_progress.advance(epoch_task)

                if global_step % args.print_steps == 0:
                    g = gpu_stats()
                    stats_table = make_stats_table(
                        global_step, epoch+1, args.epochs,
                        avg_loss, smooth_loss, lr_now, rate, tok_per_sec,
                        tokens_total, run_elapsed, eta_epoch, eta_run, g
                    )
                    live.update(Columns([
                        Panel(run_progress,   title="[magenta]Overall[/]",  border_style="magenta", width=54),
                        Panel(epoch_progress, title="[green]Epoch[/]",      border_style="green",   width=62),
                        Panel(stats_table,    title="[yellow]Stats[/]",     border_style="yellow",  width=56),
                    ]))
                    flush_csv_buffer(csv_path, _csv_buffer)

                # ── CSV buffer ───────────────────────────────────────────────
                g = gpu_stats()
                _csv_buffer.append({
                    "step":             global_step,
                    "epoch":            epoch + 1,
                    "loss":             round(loss_val, 6),
                    "avg_loss":         round(avg_loss, 6),
                    "smooth_loss":      round(smooth_loss, 6),
                    "lr":               round(lr_now, 8),
                    "it_per_sec":       round(rate, 3),
                    "tok_per_sec":      round(tok_per_sec, 0),
                    "tokens_total":     tokens_total,
                    "vram_alloc_gb":    g.get("vram_alloc_gb", ""),
                    "vram_reserved_gb": g.get("vram_reserved_gb", ""),
                    "gpu_util_pct":     g.get("gpu_util_pct", ""),
                    "elapsed_sec":      round(run_elapsed, 1),
                    "timestamp":        datetime.now().isoformat(),
                })

                # ── step checkpoint ──────────────────────────────────────────
                if global_step % args.save_steps == 0:
                    save_checkpoint(ckpt_path, model, optimizer, scheduler,
                                    epoch, global_step, avg_loss, config_out)
                    console.log(f"[cyan]▸ step ckpt[/] step {global_step} | loss {avg_loss:.4f} | {fmt_time(run_elapsed)} elapsed")

                # ── timed checkpoint ─────────────────────────────────────────
                if (time.time() - last_timed_save) >= args.save_minutes * 60:
                    timed_path = out / f"timed_{datetime.now().strftime('%Y%m%d_%H%M%S')}_step{global_step}.pt"
                    save_checkpoint(timed_path, model, optimizer, scheduler,
                                    epoch, global_step, avg_loss, config_out)
                    last_timed_save = time.time()
                    console.log(f"[cyan]▸ timed ckpt[/] {timed_path.name} | loss {avg_loss:.4f}")

            # ── end of epoch ─────────────────────────────────────────────────
            flush_csv_buffer(csv_path, _csv_buffer)
            avg_loss         = total_loss / len(loader)
            epoch_time       = time.time() - epoch_start
            epoch_times.append(epoch_time)
            epoch_tok_per_sec = epoch_tokens / max(epoch_time, 1e-6)

            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                save_checkpoint(out / "best.pt", model, optimizer, scheduler,
                                epoch + 1, global_step, avg_loss, config_out)

            epochs_left = args.epochs - epoch - 1
            avg_epoch_t = sum(epoch_times[-3:]) / len(epoch_times[-3:]) if epoch_times else epoch_time
            eta_finish  = datetime.now() + timedelta(seconds=(avg_epoch_t * epochs_left))

            run_progress.advance(run_task)

            # epoch summary printed below the live display
            star = " [bold yellow]★ new best[/]" if is_best else ""
            console.print(Rule(f"[bold]Epoch {epoch+1}/{args.epochs}[/]{star}", style="dim"))
            summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            summary.add_column(style="cyan")
            summary.add_column(style="white")
            summary.add_column(style="cyan")
            summary.add_column(style="white")
            summary.add_row("Loss",          f"{avg_loss:.4f}",
                            "Best",          f"{best_loss:.4f}")
            summary.add_row("Epoch time",    fmt_time(epoch_time),
                            "Total elapsed", fmt_time(time.time() - run_start))
            summary.add_row("Tokens (ep)",   f"{epoch_tokens/1e6:.2f}M",
                            "Tok/s",         f"{epoch_tok_per_sec/1000:.1f}k")
            summary.add_row("ETA finish",    eta_finish.strftime("%Y-%m-%d %H:%M:%S"), "", "")
            console.print(summary)

            save_checkpoint(ckpt_path, model, optimizer, scheduler,
                            epoch + 1, global_step, avg_loss, config_out)
            torch.save({
                "epoch":  epoch,
                "model":  model.state_dict(),
                "config": config_out,
                "loss":   avg_loss,
            }, out / f"epoch_{epoch+1:03d}_loss{avg_loss:.4f}.pt")

    # ── final stats ───────────────────────────────────────────────────────────
    total_time = time.time() - run_start
    final_stats = {
        "total_time_sec":  round(total_time, 1),
        "total_tokens":    tokens_total,
        "avg_tok_per_sec": round(tokens_total / total_time, 0),
        "best_loss":       round(best_loss, 6),
        "epochs":          args.epochs,
        "parameters":      model.count_parameters(),
        "state_dim":       args.state_dim,
        "embed_dim":       args.embed_dim,
        "batch_size":      args.batch_size,
        "seq_len":         args.seq_len,
        "completed":       datetime.now().isoformat(),
    }
    with open(stats_path, "w") as f:
        json.dump(final_stats, f, indent=2)

    console.print()
    console.print(Panel(
        f"[bold green]Training complete[/] in {fmt_time(total_time)}\n"
        f"  Total tokens:  [white]{tokens_total:,}[/]\n"
        f"  Throughput:    [white]{tokens_total/total_time/1000:.1f}k tok/sec[/]\n"
        f"  Best loss:     [white]{best_loss:.4f}[/]\n"
        f"  Stats:         [dim]{stats_path}[/]\n"
        f"  Log:           [dim]{csv_path}[/]",
        border_style="green", title="[bold green]Done[/]"
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     required=True)
    parser.add_argument("--vocab_path",   default="vocab.json")
    parser.add_argument("--out_dir",      default="checkpoints")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--seq_len",      type=int,   default=256)
    parser.add_argument("--embed_dim",    type=int,   default=512)
    parser.add_argument("--state_dim",    type=int,   default=4096)
    parser.add_argument("--n_pairs",      type=int,   default=128)
    parser.add_argument("--hidden_dim",   type=int,   default=1024)
    parser.add_argument("--n_layers",     type=int,   default=6)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--workers",      type=int,   default=0)
    parser.add_argument("--save_steps",   type=int,   default=2000,
                        help="Save latest.pt every N steps")
    parser.add_argument("--save_minutes", type=int,   default=30,
                        help="Save a timestamped checkpoint every N minutes")
    parser.add_argument("--print_steps",  type=int,   default=10,
                        help="Refresh stats panel every N steps")
    args = parser.parse_args()
    train(args)
