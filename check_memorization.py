"""
Memorization check — how much of the generated output is copied from training data?

A very low training loss is ambiguous: the model may have learned the style, or it
may have memorized the corpus. This tool distinguishes the two by measuring, for
each generated sample, how much of it appears verbatim in the training set.

The primary metric is the **longest common token substring (LCS)** between a
generated sample and any training file. A model that has learned style produces
sequences whose longest verbatim overlap with any single training piece is short
-- a few tokens of shared idiom. A model that has memorized reproduces long runs.

Secondary metrics:

    coverage    fraction of the sample covered by matches of >= --min-run tokens
                found anywhere in training data. Catches collage-style copying
                that a single long LCS would miss.
    novel       fraction of the sample's n-grams never seen in training.
    self-sim    LCS between generated samples. High values mean mode collapse:
                the model emits the same material regardless of the prompt.

Interpretation is deliberately conservative. Music is repetitive and shares
idiom, so short matches are expected and meaningless. The judgement thresholds
below flag only the ranges that indicate genuine copying.

Usage:
    python check_memorization.py
    python check_memorization.py --generated generated --dataset dataset
    python check_memorization.py --min-run 12 --ngram 8 --json report.json

Compares in token space (the model's own output alphabet) rather than MIDI, so
results are unaffected by decode/encode round-trip differences.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ── token loading ─────────────────────────────────────────────────────────────

def load_training_tokens(dataset_dir: Path) -> dict[str, list[int]]:
    """Load tokenized training corpus: {name: [token ids]}."""
    files = sorted(dataset_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No .json token files in {dataset_dir}. "
                         f"Run: python -m data.pipeline --midi_dir <midi> --out_dir {dataset_dir}")
    out = {}
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"  warning: could not read {f.name}: {e}", file=sys.stderr)
            continue
        # pipeline writes a flat list of ids; tolerate {"ids": [...]} too
        if isinstance(data, dict):
            data = data.get("ids") or data.get("tokens") or []
        if isinstance(data, list) and data and isinstance(data[0], list):
            data = data[0]  # single-track nesting
        if isinstance(data, list) and data:
            out[f.stem] = [int(t) for t in data]
    return out


def load_generated_tokens(generated_dir: Path, vocab_path: Path) -> dict[str, list[int]]:
    """
    Re-tokenize generated MIDI back to ids using the same tokenizer.

    Generation saves .mid files; if it fell back to raw token json (it does that
    when MIDI conversion fails) those are used directly, which is more faithful.
    """
    out = {}

    for f in sorted(generated_dir.glob("*_tokens.json")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list) and data:
                out[f.stem] = [int(t) for t in data]
        except Exception:
            pass

    mids = sorted(generated_dir.glob("*.mid"))
    if mids:
        try:
            from data.pipeline import build_tokenizer, tokenize_midi_path
            tok = build_tokenizer(str(vocab_path) if vocab_path.exists() else None)
            for f in mids:
                if f.stem in out:
                    continue
                try:
                    ids = tokenize_midi_path(tok, f)
                    if ids:
                        out[f.stem] = [int(t) for t in ids]
                except Exception as e:
                    print(f"  warning: could not tokenize {f.name}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  warning: tokenizer unavailable ({e}); "
                  f"only *_tokens.json files can be checked", file=sys.stderr)

    if not out:
        raise SystemExit(f"No usable generated samples in {generated_dir}.")
    return out


# ── matching primitives ───────────────────────────────────────────────────────

def longest_common_substring(a: list[int], b: list[int]) -> tuple[int, int, int]:
    """
    Length and positions of the longest contiguous common run.
    Returns (length, index_in_a, index_in_b). O(len(a) * len(b)) time,
    O(len(b)) space -- rolling two rows rather than a full DP table.
    """
    if not a or not b:
        return 0, 0, 0
    best = best_i = best_j = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                v = prev[j - 1] + 1
                cur[j] = v
                if v > best:
                    best, best_i, best_j = v, i - v, j - v
        prev = cur
    return best, best_i, best_j


def build_ngram_index(corpus: dict[str, list[int]], n: int) -> set[tuple]:
    """All n-grams appearing anywhere in the training corpus."""
    idx = set()
    for toks in corpus.values():
        for i in range(len(toks) - n + 1):
            idx.add(tuple(toks[i:i + n]))
    return idx


def covered_fraction(sample: list[int], ngrams: set[tuple], n: int) -> float:
    """
    Fraction of sample positions inside at least one training n-gram.
    Detects collage copying that a single longest-run metric would miss.
    """
    if len(sample) < n:
        return 0.0
    covered = bytearray(len(sample))
    for i in range(len(sample) - n + 1):
        if tuple(sample[i:i + n]) in ngrams:
            for k in range(i, i + n):
                covered[k] = 1
    return sum(covered) / len(sample)


def novel_ngram_fraction(sample: list[int], ngrams: set[tuple], n: int) -> float:
    """Fraction of the sample's n-grams that never occur in training."""
    if len(sample) < n:
        return 1.0
    total = len(sample) - n + 1
    unseen = sum(1 for i in range(total) if tuple(sample[i:i + n]) not in ngrams)
    return unseen / total


# ── analysis ──────────────────────────────────────────────────────────────────

@dataclass
class SampleReport:
    name: str
    length: int
    lcs_len: int = 0
    lcs_source: str = ""
    lcs_frac: float = 0.0
    lcs_tokens: list[int] = field(default_factory=list)
    coverage: float = 0.0
    novel: float = 0.0
    verdict: str = ""


def judge(lcs_frac: float, lcs_len: int, coverage: float, min_run: int) -> str:
    """
    Conservative thresholds. Music shares idiom, so short runs are expected;
    only sustained verbatim reproduction counts as memorization.
    """
    if lcs_frac >= 0.50 or lcs_len >= 200:
        return "COPIED"
    if lcs_frac >= 0.25 or lcs_len >= 100:
        return "HEAVY OVERLAP"
    if coverage >= 0.90:
        return "COLLAGE"
    if lcs_frac >= 0.10 or lcs_len >= 40:
        return "SOME OVERLAP"
    return "OK"


def analyse(generated, training, ngram: int, min_run: int) -> list[SampleReport]:
    print(f"Indexing {len(training)} training files ({ngram}-grams)...")
    ngrams_small = build_ngram_index(training, ngram)
    ngrams_run = build_ngram_index(training, min_run) if min_run != ngram else ngrams_small

    reports = []
    for i, (name, sample) in enumerate(generated.items(), 1):
        print(f"[{i}/{len(generated)}] {name} ({len(sample)} tokens)...", flush=True)
        rep = SampleReport(name=name, length=len(sample))

        best = (0, "", 0, 0)
        for src, toks in training.items():
            ln, ia, _ = longest_common_substring(sample, toks)
            if ln > best[0]:
                best = (ln, src, ia, 0)

        rep.lcs_len = best[0]
        rep.lcs_source = best[1]
        rep.lcs_frac = best[0] / len(sample) if sample else 0.0
        rep.lcs_tokens = sample[best[2]:best[2] + min(best[0], 24)]
        rep.coverage = covered_fraction(sample, ngrams_run, min_run)
        rep.novel = novel_ngram_fraction(sample, ngrams_small, ngram)
        rep.verdict = judge(rep.lcs_frac, rep.lcs_len, rep.coverage, min_run)
        reports.append(rep)
    return reports


def self_similarity(generated: dict[str, list[int]]) -> list[tuple[str, str, int, float]]:
    """Pairwise LCS between generated samples -- detects mode collapse."""
    names = list(generated)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = generated[names[i]], generated[names[j]]
            ln, _, _ = longest_common_substring(a, b)
            out.append((names[i], names[j], ln, ln / min(len(a), len(b))))
    return out


# ── reporting ─────────────────────────────────────────────────────────────────

BAR = "=" * 78

def report(reports, selfsim, ngram, min_run):
    print(f"\n{BAR}\nMEMORIZATION REPORT\n{BAR}\n")
    print(f"{'sample':<22}{'len':>6}{'LCS':>7}{'LCS%':>8}{'cover':>8}{'novel':>8}  {'verdict':<14}")
    print("-" * 78)
    for r in reports:
        print(f"{r.name[:21]:<22}{r.length:>6}{r.lcs_len:>7}{r.lcs_frac*100:>7.1f}%"
              f"{r.coverage*100:>7.1f}%{r.novel*100:>7.1f}%  {r.verdict:<14}")

    print(f"\n  LCS    longest run of tokens identical to some training file")
    print(f"  LCS%   that run as a fraction of the sample")
    print(f"  cover  fraction of sample inside any training {min_run}-gram")
    print(f"  novel  fraction of {ngram}-grams never seen in training")

    print("\nClosest training source per sample:")
    for r in reports:
        if r.lcs_source:
            print(f"  {r.name[:24]:<26} → {r.lcs_source:<24} ({r.lcs_len} tokens)")

    if selfsim:
        print("\nSelf-similarity between generated samples (mode-collapse check):")
        for a, b, ln, frac in sorted(selfsim, key=lambda x: -x[2]):
            note = "  ← near-identical" if frac >= 0.5 else ""
            print(f"  {a[:20]:<22} vs {b[:20]:<22} {ln:>5} tokens ({frac*100:.1f}%){note}")

    print(f"\n{BAR}")
    worst = [r for r in reports if r.verdict in ("COPIED", "HEAVY OVERLAP", "COLLAGE")]
    mild = [r for r in reports if r.verdict == "SOME OVERLAP"]
    mean_novel = sum(r.novel for r in reports) / len(reports)
    mean_lcs = sum(r.lcs_frac for r in reports) / len(reports)

    if worst:
        print(f"MEMORIZATION DETECTED — {len(worst)}/{len(reports)} samples reproduce "
              f"training data.\nThe low training loss reflects copying, not learned style.")
    elif mild:
        print(f"BORDERLINE — {len(mild)}/{len(reports)} samples share notable runs with "
              f"training data.\nShort shared runs are normal in music; inspect the "
              f"matched spans before concluding.")
    else:
        print("NO MEMORIZATION DETECTED — no sample reproduces a long verbatim run.")
    print(f"Mean novel {ngram}-grams: {mean_novel*100:.1f}%   "
          f"mean longest copied run: {mean_lcs*100:.1f}% of sample")
    if selfsim and max(s[3] for s in selfsim) >= 0.5:
        print("WARNING: generated samples are highly similar to each other "
              "(possible mode collapse).")
    print(BAR)


def main():
    ap = argparse.ArgumentParser(description="Check generated output for memorized training data.")
    ap.add_argument("--generated", default="generated", help="dir of generated .mid / *_tokens.json")
    ap.add_argument("--dataset", default="dataset", help="dir of tokenized training .json")
    ap.add_argument("--vocab_path", default="vocab.json")
    ap.add_argument("--ngram", type=int, default=8, help="n-gram size for novelty (default 8)")
    ap.add_argument("--min-run", type=int, default=12, help="run length for coverage (default 12)")
    ap.add_argument("--json", default=None, help="also write the report to this JSON path")
    args = ap.parse_args()

    gen_dir, ds_dir = Path(args.generated), Path(args.dataset)
    if not gen_dir.is_dir():
        raise SystemExit(f"Generated dir not found: {gen_dir}")
    if not ds_dir.is_dir():
        raise SystemExit(f"Dataset dir not found: {ds_dir}")

    training = load_training_tokens(ds_dir)
    generated = load_generated_tokens(gen_dir, Path(args.vocab_path))
    print(f"Loaded {len(generated)} generated samples, {len(training)} training files.\n")

    reports = analyse(generated, training, args.ngram, args.min_run)
    selfsim = self_similarity(generated)
    report(reports, selfsim, args.ngram, args.min_run)

    if args.json:
        Path(args.json).write_text(json.dumps({
            "reports": [asdict(r) for r in reports],
            "self_similarity": [
                {"a": a, "b": b, "lcs": ln, "frac": fr} for a, b, ln, fr in selfsim
            ],
            "params": {"ngram": args.ngram, "min_run": args.min_run},
        }, indent=2))
        print(f"\nJSON report → {args.json}")


if __name__ == "__main__":
    main()
