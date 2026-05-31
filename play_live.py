"""
play_live.py — stream tokens from a trained GSM checkpoint and play them live.

Usage:
    python play_live.py --checkpoint checkpoints/best.pt --vocab_path vocab.json
    python play_live.py --checkpoint checkpoints/best.pt --buffer_secs 4 --temperature 0.85
"""

import argparse
import sys
import time
import threading
import queue
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from model.gsm import GSM


DEFAULT_CHANNEL = 0


@dataclass
class Note:
    pitch:       int
    velocity:    int
    duration_ms: float
    onset_ms:    float


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    model = GSM(
        vocab_size = cfg["vocab_size"],
        embed_dim  = cfg["embed_dim"],
        state_dim  = cfg["state_dim"],
        n_pairs    = cfg["n_pairs"],
        hidden_dim = cfg.get("hidden_dim", 1024),
        n_layers   = cfg.get("n_layers", 6),
        dropout    = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_tokenizer(vocab_path):
    from miditok import REMI, TokenizerConfig
    cfg = TokenizerConfig(
        num_velocities=16, use_chords=False, use_programs=False,
        use_tempos=True, use_time_signatures=True,
    )
    if Path(vocab_path).exists():
        try:
            return REMI(params=vocab_path)
        except Exception:
            pass
    return REMI(cfg)


# ── Token → Note decoder ──────────────────────────────────────────────────────

class REMIDecoder:
    """
    Accumulates REMI tokens and emits Note objects.

    Vocab layout (confirmed from live output):
      Pitch_N       — MIDI pitch value directly
      Velocity_N    — MIDI velocity value directly
      Duration_b.p.r — beat.position.resolution  (ms = (b + p/r) * ms_per_beat)
      TimeShift_b.p.r — same encoding, advances the cursor
    """

    def __init__(self, id_to_str: dict, bpm: float = 120.0):
        self.id_to_str   = id_to_str
        self.ms_per_beat = 60_000.0 / bpm
        self.cursor_ms   = 0.0

        self._pitch:        Optional[int]   = None
        self._velocity:     Optional[int]   = None
        self._last_dur_ms:  float           = 0.0

    def _bpr_to_ms(self, token_str: str) -> float:
        """'Duration_b.p.r' or 'TimeShift_b.p.r' → milliseconds."""
        try:
            parts = token_str.split("_", 1)[1].split(".")
            b, p, r = int(parts[0]), int(parts[1]), int(parts[2])
            return (b + p / r) * self.ms_per_beat
        except Exception:
            return 0.0

    def push(self, token_id: int) -> Optional[Note]:
        s = self.id_to_str.get(token_id, "")
        if not s:
            return None

        kind = s.split("_")[0]

        if kind == "Pitch":
            # No TimeShift between notes — advance cursor by previous note's duration
            if self._last_dur_ms:
                self.cursor_ms += self._last_dur_ms
                self._last_dur_ms = 0.0
            self._pitch    = int(s.split("_")[1])
            self._velocity = None

        elif kind == "Velocity" and self._pitch is not None:
            self._velocity = int(s.split("_")[1])

        elif kind == "Duration" and self._pitch is not None and self._velocity is not None:
            dur_ms = self._bpr_to_ms(s)
            note = Note(
                pitch       = self._pitch,
                velocity    = self._velocity,
                duration_ms = dur_ms,
                onset_ms    = self.cursor_ms,
            )
            self._pitch = self._velocity = None
            self._last_dur_ms = dur_ms
            return note

        elif kind == "TimeShift":
            self.cursor_ms += self._bpr_to_ms(s)
            self._last_dur_ms = 0.0

        elif kind == "Pitch":
            # No TimeShift between notes — advance cursor by previous note's duration
            if self._last_dur_ms:
                self.cursor_ms += self._last_dur_ms
                self._last_dur_ms = 0.0
            self._pitch    = int(s.split("_")[1])
            self._velocity = None
            return None

        return None


# ── Generation thread ─────────────────────────────────────────────────────────

def generation_worker(model, id_to_str, device, cfg, args, note_queue, stop_event):
    decoder    = REMIDecoder(id_to_str, bpm=args.bpm)
    vocab_size = cfg["vocab_size"]

    if args.prompt_tokens:
        ids = [int(x) for x in args.prompt_tokens.split(",")]
    else:
        ids = torch.randint(0, vocab_size, (4,)).tolist()

    prompt = torch.tensor(ids, dtype=torch.long, device=device)

    with torch.no_grad():
        S = model.S0.unsqueeze(0).clone()
        E = model.embedding(prompt.unsqueeze(0))
        for t in range(prompt.shape[0]):
            S = model.step(S, E[:, t])

    print(f"[gen] Prompt: {ids}  →  generating…")

    generated = 0
    with torch.no_grad():
        while not stop_event.is_set():
            if args.max_tokens and generated >= args.max_tokens:
                print("[gen] Reached max_tokens.")
                break

            e      = model.embedding(torch.tensor([[ids[-1]]], device=device)).squeeze(1)
            S      = model.step(S, e)
            logits = model.decoder(S) / args.temperature

            if args.top_k > 0:
                top_vals, _ = torch.topk(logits, args.top_k)
                logits[logits < top_vals[:, -1:]] = float("-inf")

            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            ids.append(next_id)
            generated += 1

            note = decoder.push(next_id)
            if note is not None:
                note_queue.put(note)

    note_queue.put(None)


# ── Playback thread ───────────────────────────────────────────────────────────

def playback_worker(note_queue, buffer_secs, stop_event):
    try:
        import pygame.midi
    except ImportError:
        print("[play] pygame not found:  pip install pygame")
        stop_event.set()
        return

    pygame.midi.init()
    device_id = pygame.midi.get_default_output_id()
    if device_id < 0:
        print("[play] No MIDI output device found.")
        stop_event.set()
        pygame.midi.quit()
        return

    out = pygame.midi.Output(device_id, latency=0)
    out.set_instrument(0, DEFAULT_CHANNEL)
    print(f"[play] MIDI output: {pygame.midi.get_device_info(device_id)}")
    print(f"[play] Buffering {buffer_secs}s before playback…")

    # Pre-fill buffer
    buffered: list[Note] = []
    buffer_ms = buffer_secs * 1000.0

    while True:
        try:
            note = note_queue.get(timeout=30)
        except queue.Empty:
            print("[play] Timed out waiting for notes — no notes generated.")
            stop_event.set()
            out.close()
            pygame.midi.quit()
            return
        if note is None:
            break
        buffered.append(note)
        if (note.onset_ms + note.duration_ms) >= buffer_ms:
            break

    if not buffered:
        print("[play] No notes decoded.")
        stop_event.set()
        out.close()
        pygame.midi.quit()
        return

    print(f"[play] Buffer ready ({len(buffered)} notes). Starting playback.")

    play_start  = time.perf_counter()
    pending     = list(buffered)
    active: list[tuple[float, int]] = []   # (note_off wall time, pitch)

    while not stop_event.is_set():
        now     = time.perf_counter()
        play_ms = (now - play_start) * 1000.0

        # Pull more notes from queue (non-blocking)
        try:
            while True:
                note = note_queue.get_nowait()
                if note is None:
                    stop_event.set()
                    break
                pending.append(note)
        except queue.Empty:
            pass

        # Fire note-ons that are due (onset is within lookahead)
        still_pending = []
        for note in pending:
            if note.onset_ms <= play_ms + buffer_ms:
                wall_on = play_start + note.onset_ms / 1000.0
                sleep   = wall_on - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                out.note_on(note.pitch, note.velocity, DEFAULT_CHANNEL)
                active.append((play_start + (note.onset_ms + note.duration_ms) / 1000.0, note.pitch))
            else:
                still_pending.append(note)
        pending = still_pending

        # Fire note-offs that are due
        still_active = []
        for (off_time, pitch) in active:
            if time.perf_counter() >= off_time:
                out.note_off(pitch, 0, DEFAULT_CHANNEL)
            else:
                still_active.append((off_time, pitch))
        active = still_active

        if not pending and not active and stop_event.is_set():
            break

        time.sleep(0.005)

    for (_, pitch) in active:
        out.note_off(pitch, 0, DEFAULT_CHANNEL)
    out.close()
    pygame.midi.quit()
    print("[play] Done.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--vocab_path",    default="vocab.json")
    parser.add_argument("--buffer_secs",   type=float, default=3.0)
    parser.add_argument("--temperature",   type=float, default=0.9)
    parser.add_argument("--top_k",         type=int,   default=50)
    parser.add_argument("--bpm",           type=float, default=120.0)
    parser.add_argument("--max_tokens",    type=int,   default=None)
    parser.add_argument("--prompt_tokens", type=str,   default=None)
    parser.add_argument("--device",        type=str,   default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[main] Device: {device}")

    model, cfg = load_model(args.checkpoint, device)
    print(f"[main] Model loaded | params={model.count_parameters():,} | state_dim={cfg['state_dim']}")

    tokenizer  = load_tokenizer(args.vocab_path)
    id_to_str  = {v: k for k, v in tokenizer.vocab.items()}
    print(f"[main] Tokenizer loaded | vocab_size={len(id_to_str)}")

    note_queue  = queue.Queue(maxsize=4096)
    stop_event  = threading.Event()

    gen_thread  = threading.Thread(
        target=generation_worker,
        args=(model, id_to_str, device, cfg, args, note_queue, stop_event),
        daemon=True,
    )
    play_thread = threading.Thread(
        target=playback_worker,
        args=(note_queue, args.buffer_secs, stop_event),
        daemon=True,
    )

    gen_thread.start()
    play_thread.start()

    try:
        while gen_thread.is_alive() or play_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[main] Stopping…")
        stop_event.set()

    gen_thread.join(timeout=2)
    play_thread.join(timeout=2)
    print("[main] Exited.")


if __name__ == "__main__":
    main()
