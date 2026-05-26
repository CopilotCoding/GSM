# GSM — Geometric State Machine

> No attention. No KV cache. No quadratic scaling. A fixed point in R^N being continuously deformed by a learned algebra of transformations.

Trained on 228 Bach MIDI files in 54 minutes on a single consumer GPU. Final loss 0.1196. Generates convincing baroque piano music. Scales to 179k+ file datasets with memory-mapped binary packing — no architecture changes required.

---

## What Is This

Most sequence models treat context as something to *store* — transformers cache every previous token's keys and values, RNNs overwrite a memory buffer. Both approaches scale poorly: transformers pay quadratic cost in sequence length, RNNs struggle with long-range dependencies.

GSM treats context as something to *accumulate geometrically*.

The model maintains a single fixed-size point `S ∈ R^N` — a position in a high-dimensional geometric space. Each token is not a data point to store but a **transformation operator** that deforms that geometry. The state is never a memory buffer being overwritten. It's a manifold position being continuously reshaped by a learned transformation algebra.

Knowledge isn't stored. It's shaped into the geometry.

---

## Why It's Different From An RNN

This is the question worth answering carefully, because the surface structure looks similar — both update a hidden state per token with fixed compute.

| Property | RNN / LSTM / GRU | GSM |
|----------|-----------------|-----|
| State update | `W_hh × h + W_xh × x` | `Transform(S, params(x))` |
| State semantics | Memory buffer | Manifold position |
| Transformation | Fixed recurrent weight matrix | Input-parameterized field |
| Geometric op | None | Vectorized subspace rotations |
| Inductive bias | Sequential memory compression | Geometric deformation |
| Long-range | Vanishing gradient problem | Gate controls deformation magnitude |

The critical difference: RNNs have a **fixed recurrent weight matrix W_hh** that maps state to state regardless of input. In GSM, the transformation of the state is **entirely parameterized by the input token**. The state has no direct path to itself — it only moves when a token moves it, and how it moves depends entirely on what the token is.

The subspace rotation component has no RNN analogue at all. It applies learned rotations in random fixed dimension pairs across R^N — pure geometric deformation with no equivalent in any classical sequence model.

---

## Architecture

```
Token t
  └─ Embedding lookup → e_t ∈ R^{embed_dim}
       └─ TransformNet (6-layer MLP with residual connections)
            ├─ scale  ∈ R^{state_dim}    multiplicative field
            ├─ shift  ∈ R^{state_dim}    additive perturbation
            ├─ gate   ∈ R^{state_dim}    geometric mixing coefficient
            └─ angles ∈ R^{n_pairs}      subspace rotation angles
                 └─ RotarySubspaceTransform (vectorized, all pairs parallel)
                      └─ S' = gate ⊙ Rotate(scale ⊙ S + shift) + (1 - gate) ⊙ S
                           └─ LayerNorm → bounded manifold position
                                └─ Decoder (3-layer MLP) → logits ∈ R^{vocab_size}
```

### The High-Dimensional Plane

The state lives in flat R^N, not on a curved manifold. This is intentional.

In sufficiently high dimensions, flat space functionally folds — two points can be geometrically distant in every low-dimensional projection yet adjacent along some dimension the model has learned to use. The richness comes not from topology but from the transformation algebra carving semantic structure into the geometry through training. Recurring patterns deepen into stable attractors. Noise washes out. The manifold learns to fold itself.

### TransformNet

A 6-layer MLP with residual connections that maps a token embedding to transformation parameters. Fixed depth means fixed compute — O(1) per token regardless of sequence length or corpus size. Initialized to near-identity so training starts from a stable geometric configuration.

### RotarySubspaceTransform

The geometrically novel component. A fixed set of random dimension pairs `(i, j)` in R^N. For each pair, the model produces a rotation angle and applies a 2D rotation in that subspace. All pairs computed simultaneously via gather/scatter — no Python loops, fully vectorized on GPU.

There is no classical sequence model operation that corresponds to input-parameterized subspace rotations on a fixed geometric object.

---

## Complexity

| Property | Transformer | RNN/LSTM | GSM |
|----------|------------|---------|-----|
| Memory per token | O(n) KV cache | O(1) | O(1) |
| Compute per token | O(n) attention | O(1) | O(1) |
| State size | Grows with context | Fixed | Fixed |
| Scales to any corpus | ✓ | ✓ | ✓ |
| Long context cost | Quadratic | Linear | **O(1)** |

---

## Performance TEST on 170K corpus

Tested on a single RTX 5060 Ti (16GB VRAM), Windows 11, pure PyTorch — no custom CUDA kernels, no `torch.compile`, no Triton:

- **~29,000 tokens/second** sustained training throughput (batch 128, seq 128, bf16)
- **18M parameter model** fits in under 6GB dedicated VRAM
- **1.77 it/s** at batch 128 — stable across the full epoch
- Faster wall-clock per epoch than a scalar Mamba variant trained on the same corpus at comparable parameter count
- Loss descending from 5.15 → sub-2.0 in under 3 hours on 179k files, generating audible music before the first epoch completes

The architecture is genuinely lightweight. Inference is O(1) — fixed compute and memory per token regardless of sequence length. Token 1 and token 100,000 cost exactly the same.

---

## Training Tradeoffs

GSM's O(1) inference property comes with a training cost worth understanding before committing to a large run.

The state update is strictly sequential — each step depends on the previous one, so the forward pass is a Python loop over sequence length regardless of batch size. This means:

- **Small datasets (<10k sequences):** Fast. The Bach corpus trained in 54 minutes.
- **Large datasets (millions of sequences):** Slower. Each batch requires `seq_len` sequential GPU dispatches, and with millions of batches per epoch this compounds significantly.
- **`torch.compile`** would fuse these kernel launches and largely solve the problem, but is not supported on Windows as of PyTorch 2.x.
- **Custom CUDA kernels** could parallelize across the sequence dimension but defeat the goal of single-developer simplicity.

The fundamental tradeoff: **training speed scales with dataset size; inference speed does not.** For deployment on constrained hardware, streaming, or edge devices, GSM remains attractive. For large-scale training on a single consumer GPU without compile support, expect slower throughput than a transformer of equivalent parameter count.

One area where GSM has a clear advantage: **tiny datasets.** The geometric state appears to generalize from very little data. GSM trained on 228 Bach MIDI files produces convincing baroque music. If your dataset is small, GSM is likely capable of learning from it.

---

## Results

**Hardware**: RTX 5060 Ti (16GB VRAM)
**Dataset**: 228 Bach MIDI files (217 processed, 11 skipped), 3,357 training sequences
**Model**: 32,731,125 parameters
**Total training time**: 54 minutes 12 seconds

| Epoch | Loss | Note |
|-------|------|------|
| 1 | 4.3802 | Random baseline ~5.92 |
| 3 | 2.8804 | Steep structural drop |
| 5 | 2.0017 | Structural learning established |
| 10 | 1.3773 | Where the 6M param model finished after 30 epochs |
| 20 | 1.0132 | Sub-1.0 |
| 30 | 0.8131 | |
| 47 | 0.5119 | **"Sounds like Bach"** — listeners confirmed |
| 60 | 0.3211 | |
| 80 | 0.1683 | |
| 100 | 0.1196 | Final — curve still falling |

At temperature 0.75 after epoch 47: generates convincing baroque piano music. Not "vaguely melodic" — actual baroque phrasing and harmonic structure.

A smaller 6M parameter GSM trained on the same data reached a best loss of 1.3768 after 30 epochs (~9 minutes). The 32M model passed that at epoch 10 and reached 0.1196 by epoch 100.

---

## Installation

```cmd
pip install torch pretty_midi miditok tqdm psutil
```

Requires Python 3.10+. GPU strongly recommended (CUDA). Tested on Windows with RTX 5060 Ti.

`psutil` is optional but enables full hardware detection in `pick_model.py` and `benchmark.py`.

---

## Usage

### 0. Pick the right model size (recommended first step)

Automatically detects your hardware, sweeps model sizes from smallest possible upward by a configurable factor, and outputs a ready-to-paste train command:

```cmd
python pick_model.py --data_dir dataset_packed_128 --vocab_path vocab.json
```

Options:

| Arg | Default | Notes |
|-----|---------|-------|
| `--factor` | 2.0 | Scale factor between configs. Use 1.5 for finer steps |
| `--vram_budget` | 0.80 | Fraction of free VRAM to use |
| `--seq_len` | 128 | Sequence length used during probing |
| `--probe_batch` | 32 | Batch size used during probing |

Example output:
```
  Winner: config #5  |  18.37M params
  state_dim:   2048
  embed_dim:   512
  ...
  python -m train.train --data_dir dataset_packed_128 ...
```

### 1. Process MIDI Dataset

```cmd
python -m data.pipeline --midi_dir C:\path\to\midi\files --out_dir dataset --vocab_path vocab.json --workers 8
```

Works with any MIDI dataset. Tested with Bach MIDI corpus and LMD (178k files).

### 2. Pack Dataset (recommended for large datasets)

For datasets over ~1k files, convert to a memory-mapped binary before training. Loads instantly into RAM regardless of dataset size.

```cmd
python -m data.pack --data_dir dataset --out_dir dataset_packed --seq_len 256 --workers 20
```

Run once after pipeline. The packed binary is read fully into pinned RAM at training startup for maximum GPU throughput. For 179k files expect ~2GB on disk.

Skip for small datasets (<1k files) — the JSON fallback is fast enough.

### 3. Train

```cmd
python -m train.train --data_dir dataset_packed --vocab_path vocab.json --out_dir checkpoints --epochs 100 --workers 8
```

Training produces:
- `checkpoints/latest.pt` — saved every `--save_steps` steps (default 2000)
- `checkpoints/best.pt` — saved whenever a new best epoch loss is reached
- `checkpoints/timed_*.pt` — timestamped saves every `--save_minutes` minutes (default 30)
- `checkpoints/training_log.csv` — per-step log: loss, lr, tok/s, VRAM, timestamp
- `checkpoints/run_stats.json` — end-of-run summary

Key training flags:

| Arg | Default | Notes |
|-----|---------|-------|
| `--state_dim` | 4096 | Size of geometric object. Bigger = richer geometry |
| `--embed_dim` | 512 | Token embedding / transformation operator size |
| `--n_pairs` | 128 | Subspace rotation pairs |
| `--hidden_dim` | 1024 | TransformNet hidden size |
| `--n_layers` | 6 | TransformNet depth |
| `--batch_size` | 128 | Reduce to 64 if OOM |
| `--epochs` | 100 | Loss still falling at 100, more is fine |
| `--lr` | 3e-4 | Cosine annealed to 3e-5 |
| `--save_steps` | 2000 | Save latest.pt every N steps |
| `--save_minutes` | 30 | Also save a timestamped checkpoint every N minutes |
| `--print_steps` | 10 | Print stats every N steps (includes elapsed time) |

### 4. Generate

```cmd
python -m generate.generate --checkpoint checkpoints\latest.pt --vocab_path vocab.json --out_dir generated --n_samples 5 --length 512 --temperature 0.75
```

| Arg | Notes |
|-----|-------|
| `--temperature` | Lower = more conservative. 0.75 sounds best for Bach |
| `--top_k` | Vocabulary cutoff per step (default 50) |
| `--length` | Tokens to generate (512 ≈ 30-60 seconds of music) |
| `--n_samples` | Number of MIDI files to generate |

Output is `.mid` files. Open in MuseScore, FL Studio, Reaper, or drag into **midi.city** in browser to listen instantly.

### 5. Benchmark

Inference speed and memory profiling suite. Proves O(1) throughput empirically.

```cmd
python benchmark.py --checkpoint checkpoints\latest.pt --vocab_path vocab.json
```

With full batch scaling sweep:

```cmd
python benchmark.py --checkpoint checkpoints\latest.pt --vocab_path vocab.json --full
```

The benchmark runs four tests: throughput vs sequence length (with O(1) confirmation), per-token latency distribution (min/median/p95/max), memory profiling (inference and training forward+backward), and batch size scaling.

### 6. Plot Training

Plot loss, throughput, VRAM, LR, and GPU utilization from the training CSV:

```cmd
python plot_training.py checkpoints\training_log.csv
python plot_training.py checkpoints\training_log.csv --out plot.png --smooth 200
```

Requires `matplotlib` (`pip install matplotlib`). Outputs a dark-theme five-panel figure. Use `--out` to save to disk instead of opening a window.

### 7. Sanity Check

```cmd
python test.py
```

Verifies O(1) property, shape correctness, forward/backward pass.

---

## Hyperparameter Guide

**Don't know where to start? Run `pick_model.py` first.** It probes your hardware and outputs the exact command to run.

**Smaller dataset (<500 files):**
```cmd
--state_dim 2048 --epochs 100 --batch_size 128
```

**Larger dataset (LMD 178k files):**
```cmd
python -m data.pack --data_dir dataset --out_dir dataset_packed --workers 20
python -m train.train --data_dir dataset_packed --vocab_path vocab.json --out_dir checkpoints --state_dim 4096 --epochs 30 --batch_size 128 --workers 8
```

**Low VRAM (<8GB):**
```cmd
--state_dim 1024 --embed_dim 256 --batch_size 32 --n_pairs 64
```

---

## The Geometric Intuition

High-dimensional flat space isn't actually flat in any meaningful experiential sense. With N=4096 dimensions there are 4096 independent directions to move through. Two concepts that seem distant in any 2D or 3D projection can be adjacent along dimension 3847. The learned transformation algebra carves semantic structure into this geometry — recurring patterns deepen into stable attractors, noise washes out, and the manifold learns to fold itself around the structure of the training data.

You don't ask what the model *remembers*. You ask what shape the training data left behind.

---

## Further Reading

See `GSM_paper.md` in this repo for a full technical writeup covering the formal motivation, complexity analysis, comparison to RNNs and transformers, and the substantial unexplored potential of the architecture (infinite context, streaming, genomics, continual learning, interpretability).

---

## License

MIT
