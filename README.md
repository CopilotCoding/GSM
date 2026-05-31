WORKING ON TRAINING PARALLELIZATION SOLUTION, STRUGGLING WITH THIS ISSUE

POTENTIAL USE CASE IS STREAMING TRAINING AND INFERENCE AT THE SAME TIME ONLINE LEARNING AND INFERENCE.

# GSM — Geometric State Machine

> No attention. No KV cache. No quadratic scaling. A fixed point in R^N being continuously deformed by a learned algebra of transformations.

Trained on 228 Bach MIDI files in 54 minutes on a single consumer GPU. Final loss 0.1196. Generates convincing baroque piano music. Scales to 179k+ file datasets with memory-mapped binary packing — no architecture changes required.

---

## What Is This

Most sequence models treat context as something to *store* — transformers cache every previous token's keys and values, RNNs overwrite a memory buffer. Both approaches scale poorly: transformers pay quadratic cost in sequence length, RNNs struggle with long-range dependencies.

GSM treats context as something to *accumulate geometrically*.

The model maintains a single fixed-size point `S ∈ R^N` — a position in a high-dimensional geometric space. Each token is not a data point to store but a **transformation operator** that deforms that geometry. The state is never a memory buffer being overwritten. It's a manifold position being continuously reshaped by a learned transformation algebra.

Knowledge isn't stored. It's shaped into the geometry.

# For beginners:

Imagine you're trying to understand a piece of music by listening to it note by note.

A transformer is like someone who writes down every note they hear on a piece of paper, then whenever they need to understand the next note, they look back at everything they've written. The longer the piece, the more paper they need, and the longer it takes to look things up. It's powerful but expensive.

An RNN is like someone who keeps a single "impression" in their head and updates it as each note plays — but their head only has so much room, and old notes tend to get crowded out. They can only kind of remember the distant past.

GSM does something different. Imagine you have a ball floating in an enormous space — thousands of dimensions, far more than the three we can picture. Each note you hear doesn't get written down or crammed into a memory. Instead, it *pushes and rotates the ball* in that space. Each note is a transformation operator: it shoves the ball, scales it, twists it through certain dimensions by a learned angle.

By the end of the piece, the ball is sitting somewhere specific in that enormous space. That position *is* the model's understanding of everything it's heard. Not a list of notes. Not a compressed summary. A geometric position that accumulated the entire sequence through continuous deformation.

When the model wants to predict the next note, it just looks at where the ball is sitting right now and asks: given this position in this space, what note comes next?

**Why does this work?**

In a space with 4096 dimensions, you have an almost incomprehensible amount of room to encode structure. Musical patterns — a chord progression, a rhythmic motif, a harmonic resolution — each carve out a characteristic trajectory through that space during training. The model learns which pushes and rotations correspond to which musical events, so that similar musical contexts end up moving the ball to similar regions.

You're not storing the music. You're letting the music reshape a geometry, and trusting that geometry to remember what matters.

**The key property:** each note takes exactly the same amount of compute to process, and the ball stays the same size regardless of how long the piece is. There's no growing list, no quadratic blowup. O(1) per token, forever.

**What's the catch?** Because each step depends on the previous position of the ball, you can't process notes in parallel during training — you have to go one at a time. That's the tradeoff for the elegant O(1) inference. The model trains slower than a transformer but runs faster and cheaper at any sequence length.

# For experts:

GSM is a fixed-dimensional state space model with a geometric inductive bias. The state `S ∈ R^N` evolves under a sequence of input-parameterized transformations rather than a learned autonomous dynamics matrix. The update rule at each step is:

```
S' = gate ⊙ Rotate(scale ⊙ S + shift) + (1 - gate) ⊙ S
S' = LayerNorm(S')
```

where `scale`, `shift`, `gate ∈ R^N` and the rotation angles `θ ∈ R^{n_pairs}` are all outputs of a 6-layer residual MLP — TransformNet — conditioned on the current token embedding. Nothing in the update rule is autonomous: `S` has no direct recurrence through a fixed weight matrix. It only moves when a token moves it, and the direction and magnitude of movement are entirely input-determined.

**The rotation component** is the architecturally novel piece. A fixed set of `n_pairs` random index pairs `(i, j) ⊂ [N]²` are sampled at initialization and frozen. For each pair, TransformNet produces an angle `θ_k`, and a 2D rotation is applied in that subspace:

```
[S_i, S_j] ← [cos θ_k · S_i - sin θ_k · S_j,  sin θ_k · S_i + cos θ_k · S_j]
```

All pairs are computed in parallel via gather/scatter. This is a sparse approximation to a full SO(N) group action — the model learns to compose subspace rotations to implement semantic transformations. It's isometric by construction, which acts as an implicit norm-preserving regularizer on the state trajectory before the gate mixing and LayerNorm.

**Relation to SSMs.** GSM superficially resembles S4/Mamba in maintaining a fixed-size latent state, but the similarity is shallow. SSMs parameterize a linear dynamical system `S' = AS + Bx` where `A` is a structured matrix (diagonal-plus-low-rank, HiPPO-initialized) optimized to capture long-range dependencies through careful eigenspectrum control. The input modulates the input projection `B` and sometimes `Δ` (discretization step), but the core dynamics matrix `A` is fixed or input-independent.

GSM has no autonomous dynamics at all. The entire transformation — including what would correspond to `A` — is a function of the input. This is a stronger form of input-conditioning and removes the need for eigenspectrum engineering, but it also means the model can't learn input-independent temporal dynamics. Whether that's a limitation or a feature depends on the domain.

**Relation to GRUs.** The gate mechanism `S' = gate ⊙ S_new + (1 - gate) ⊙ S` is structurally identical to a GRU update gate, and the shift/scale is analogous to the candidate hidden state. The difference is that a GRU computes its candidate via `tanh(W_h · (r ⊙ h) + W_x · x)` — a fixed recurrent projection `W_h` applied to the gated previous state. GSM replaces this entirely: there is no `W_h`, and the candidate state is produced by a geometric operation (rotation in random subspaces) rather than a linear projection. The inductive bias shifts from "linear memory compression" to "isometric geometric deformation."

**The parallelization constraint.** The sequential dependency `S_t = f(S_{t-1}, x_t)` makes the recurrence irreducibly sequential — you can't parallelize across time the way transformers can with attention. However, TransformNet — which is the dominant compute cost — has no dependency on `S`. Given the full embedding sequence `E ∈ R^{B×T×d}`, all `T` TransformNet calls can be batched as a single `[B·T, d]` forward pass, producing all transformation parameters in one matmul. The recurrence then runs as a cheap sequential loop over elementwise ops. This gives 2–3× inference speedup at the cost of larger intermediate tensors, and is the correct parallelization given the architecture's constraints.

**Why it works on small datasets.** The geometric inductive bias imposes strong structure on the hypothesis space. Subspace rotations are a highly constrained family of transformations — the model can't implement arbitrary state transitions, only isometric deformations followed by gated mixing. On a small corpus like 228 MIDI files, this constraint acts as an implicit regularizer that prevents the kind of memorization a less constrained model would fall into. The state trajectory is forced to encode structure geometrically, and geometric structure generalizes better than memorized token sequences when data is scarce.

**Open questions.** Whether the random fixed subspace pairs are the right structure — versus learned pairs, full dense rotations, or a hierarchical decomposition — is unexplored. The initialization of `S_0` as a learned parameter rather than zero or a fixed point is also non-obvious; it means the model learns a "prior geometric position" that all sequences start from. The LayerNorm after each step keeps the state on a roughly unit hypersphere, which combined with the isometric rotations suggests the effective geometry is closer to spherical than flat — despite the architecture operating in flat R^N.
---

## Why It's Different From An RNN

This is the question worth answering carefully, because the surface structure looks similar — both update a hidden state per token with fixed compute.

| Property        | RNN / LSTM / GRU              | GSM                                 |
| --------------- | ----------------------------- | ----------------------------------- |
| State update    | `W_hh × h + W_xh × x`         | `Transform(S, params(x))`           |
| State semantics | Memory buffer                 | Manifold position                   |
| Transformation  | Fixed recurrent weight matrix | Input-parameterized field           |
| Geometric op    | None                          | Vectorized subspace rotations       |
| Inductive bias  | Sequential memory compression | Geometric deformation               |
| Long-range      | Vanishing gradient problem    | Gate controls deformation magnitude |

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

The geometrically novel component. A fixed set of random dimension pairs `(i, j)` in R^N. For each pair, the model produces a rotation angle and applies a 2D rotation in that subspace. All pairs computed simultaneously via indexing — no Python loops, fully vectorized on GPU.

There is no classical sequence model operation that corresponds to input-parameterized subspace rotations on a fixed geometric object.

---

## Complexity

| Property             | Transformer        | RNN/LSTM | GSM      |
| -------------------- | ------------------ | -------- | -------- |
| Memory per token     | O(n) KV cache      | O(1)     | O(1)     |
| Compute per token    | O(n) attention     | O(1)     | O(1)     |
| State size           | Grows with context | Fixed    | Fixed    |
| Scales to any corpus | ✓                  | ✓        | ✓        |
| Long context cost    | Quadratic          | Linear   | **O(1)** |

---

## Parallelization

GSM's forward pass splits cleanly into two phases with different parallelism characteristics:

**TransformNet** — the 6-layer MLP that maps embeddings to transformation parameters — has no dependency on the state `S`. All `seq_len` token embeddings can be processed simultaneously in a single batched matmul: `[batch × seq_len, embed_dim]` instead of `seq_len` separate `[batch, embed_dim]` calls. This is the dominant compute cost and benefits fully from GPU parallelism.

**The recurrence** — applying scale, rotate, gate, and LayerNorm to update `S` — is inherently sequential: `S_t` depends on `S_{t-1}`. It runs as a Python loop over `seq_len` steps but only does cheap elementwise ops per step; the expensive MLP computation has already been done.

This gives two forward paths, selected automatically by `model.train()` / `model.eval()`:

| Mode      | TransformNet          | Recurrence  | When used              |
| --------- | --------------------- | ----------- | ---------------------- |
| Training  | Per-step (small tensors, fast backward) | Sequential | `model.train()` |
| Inference | Batched across full sequence (2–3× faster) | Sequential | `model.eval()` |

**Benchmark results** (RTX 5060 Ti, seq_len=256, bfloat16, no compile):

| Metric                        | Original   | Optimized   | Speedup |
| ----------------------------- | ---------- | ----------- | ------- |
| Forward throughput (batch=32) | 38k tok/s  | 135k tok/s  | 3.6×    |
| Forward throughput (batch=128)| 140k tok/s | 333k tok/s  | 2.4×    |
| Latency p50 (batch=128)       | 246ms      | 104ms       | 2.4×    |
| Training throughput (batch=32)| 9k tok/s   | 9k tok/s    | 1.0×    |
| First-step latency            | 443ms      | 90ms        | 4.9×    |

Training throughput is identical to the original — the parallelization applies to inference and generation only, where it delivers 2–3× speedup. On Linux/WSL2 with `torch.compile` (TransformNet + decoder submodules), inference gains a further 10–30%.

---

## Training Tradeoffs

GSM's O(1) inference property comes with a training dynamic that is important to understand.

The state update is strictly sequential — each step depends on the previous one, so the training forward pass is a loop over sequence length. This means:

* **Small datasets (<10k sequences):** The Bach corpus trains in ~54 minutes.
* **Large datasets (millions of sequences):** Slower wall-clock training due to sequential state evolution per token.
* **`torch.compile`** improves throughput on Linux/WSL2 by fusing TransformNet and decoder kernels. Not available on native Windows (no Triton), enabled automatically when detected.
* **Custom CUDA kernels** could parallelize sequence dynamics further, but are intentionally avoided to preserve simplicity and portability.

The fundamental tradeoff: **training cost scales with dataset size; inference cost does not.**

A key empirical result: **GSM learns effectively from very small datasets.** On just 228 Bach MIDI files, it produces coherent, stylistically consistent baroque output.

---

## Results

**Hardware**: RTX 5060 Ti (16GB VRAM)
**Dataset**: 228 Bach MIDI files (217 processed, 11 skipped), 3,357 training sequences
**Model**: 32,731,125 parameters
**Total training time**: 54 minutes 12 seconds

| Epoch | Loss   | Note                        |
| ----- | ------ | --------------------------- |
| 1     | 4.3802 | Random baseline ~5.92       |
| 3     | 2.8804 | Rapid structural alignment  |
| 5     | 2.0017 | Harmonic structure emerges  |
| 10    | 1.3773 | Strong musical coherence    |
| 20    | 1.0132 | Stable composition behavior |
| 30    | 0.8131 |                             |
| 47    | 0.5119 | Clear baroque phrasing      |
| 60    | 0.3211 |                             |
| 80    | 0.1683 |                             |
| 100   | 0.1196 | Final — strong convergence  |

At temperature 0.75 after epoch 47: generates **convincing baroque piano music** with stable harmonic progression, recognizable cadence structure, and consistent rhythmic phrasing.

Outputs are not merely "melodic fragments" — they exhibit **coherent baroque-style composition structure**, including:

* phrase repetition with variation
* functional harmonic movement
* cadential resolution behavior
* stable rhythmic motifs

A smaller 6M parameter GSM trained on the same dataset reached a best loss of 1.3768 after 30 epochs which failed to produce proper music (~9 minutes). The 32M model surpassed this early (by epoch 10) and continued refining structural coherence to 0.1196.

---

## Installation

```cmd
pip install torch pretty_midi miditok tqdm rich nvidia-ml-py psutil
```

Requires Python 3.10+. GPU strongly recommended (CUDA). Tested on Windows 11 and WSL2 with RTX 5060 Ti.

---

## Usage

### 0. Pick the right model size (recommended first step)

Automatically detects your hardware and selects optimal model size:

```cmd
python pick_model.py --data_dir dataset_packed_128 --vocab_path vocab.json
```

Options:

| Arg             | Default | Notes                        |
| --------------- | ------- | ---------------------------- |
| `--factor`      | 2.0     | Scale factor between configs |
| `--vram_budget` | 0.80    | Fraction of free VRAM        |
| `--seq_len`     | 128     | Probe sequence length        |
| `--probe_batch` | 32      | Batch size                   |

---

### 1. Process MIDI Dataset

```cmd
python -m data.pipeline --midi_dir /path/to/midi/files --out_dir dataset --vocab_path vocab.json --workers 8
```

On Windows use quoted forward-slash paths or WSL mount paths to avoid shell escaping issues with parentheses or spaces.

---

### 2. Pack Dataset (recommended)

Converts JSON token files to a memory-mapped binary for fast training. Strongly recommended for any dataset above a few hundred files.

```cmd
python -m data.pack --data_dir dataset --out_dir dataset_packed --seq_len 256 --workers 8
```

---

### 3. Train

```cmd
python -m train.train --data_dir dataset_packed --vocab_path vocab.json --out_dir checkpoints --epochs 100
```

Displays a live Rich terminal UI with:
- Overall run progress bar (epochs, elapsed, ETA)
- Per-epoch progress bar (batches, %, ETA)
- Live stats panel (loss, smooth loss, LR, tok/s, VRAM, GPU utilization)
- Epoch summaries with best-loss tracking

Outputs:
* `latest.pt` — checkpoint saved every `--save_steps` steps
* `best.pt` — lowest validation loss checkpoint
* `epoch_NNN_lossX.XXXX.pt` — per-epoch snapshots
* `training_log.csv` — full step-level metrics
* `run_stats.json` — final run summary

Key arguments:

| Arg              | Default | Notes                               |
| ---------------- | ------- | ----------------------------------- |
| `--epochs`       | 100     |                                     |
| `--batch_size`   | 128     |                                     |
| `--seq_len`      | 256     |                                     |
| `--state_dim`    | 4096    | Geometric state dimensionality      |
| `--embed_dim`    | 512     |                                     |
| `--n_pairs`      | 128     | Rotation subspace pairs             |
| `--hidden_dim`   | 1024    | TransformNet hidden width           |
| `--n_layers`     | 6       | TransformNet depth                  |
| `--lr`           | 3e-4    | Cosine annealed to 3e-5             |
| `--save_steps`   | 2000    | Step checkpoint frequency           |
| `--save_minutes` | 30      | Timed checkpoint frequency          |
| `--print_steps`  | 10      | Stats panel refresh frequency       |

---

### 4. Generate

```cmd
python -m generate.generate --checkpoint checkpoints/latest.pt --vocab_path vocab.json --out_dir generated --n_samples 5 --length 512 --temperature 0.75
```

At 0.75 temperature, outputs are **stylistically stable baroque compositions** suitable for direct listening in MIDI DAWs.

---

### 5. Benchmark

Compare original vs optimized forward/training throughput:

```cmd
python benchmark_compare.py
python benchmark_compare.py --trials 20 --seq_len 512
```

Original architecture benchmark (O(1) inference scaling confirmation):

```cmd
python benchmark.py --checkpoint checkpoints/latest.pt --vocab_path vocab.json
```

---

### 6. Plot Training

```cmd
python plot_training.py checkpoints/training_log.csv
```

---

### 7. Sanity Check

```cmd
python test.py
```

---

## Hyperparameter Guide

**Small dataset (<500 files):**

```cmd
--state_dim 2048 --epochs 100 --batch_size 128
```

**Large dataset (LMD 178k files):**

```cmd
python -m data.pack ...
--state_dim 4096 --epochs 30 --batch_size 128
```

## Live Playback

Stream generated music directly to your MIDI output in real time as the model generates it.

```bash
python play_live.py --checkpoint checkpoints/best.pt --vocab_path vocab.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Path to trained `.pt` checkpoint |
| `--vocab_path` | `vocab.json` | Path to tokenizer vocab |
| `--buffer_secs` | `3.0` | Seconds to pre-load before playback starts |
| `--temperature` | `0.9` | Sampling temperature |
| `--top_k` | `50` | Top-k sampling cutoff |
| `--bpm` | `120.0` | Assumed tempo for timing |
| `--max_tokens` | `None` | Stop after N tokens (omit for infinite) |
| `--prompt_tokens` | `None` | Comma-separated seed token IDs |
| `--device` | auto | `cuda` or `cpu` |

### Requirements

- `pygame` — `pip install pygame`
- A system MIDI output device (Windows: built-in. Linux: requires `timidity` or `fluidsynth` running as a MIDI sink)

### How it works

A generation thread steps the GSM one token at a time and decodes REMI tokens into notes as they arrive. A playback thread buffers `--buffer_secs` of audio before starting, then stays that far ahead of the playback head — so generation and playback run concurrently with no audible gaps. Ctrl+C stops cleanly.

---

## The Geometric Intuition

High-dimensional flat space behaves as a structured representational medium under learned transformation dynamics. In 4096 dimensions, semantic regions emerge as stable attractors of repeated transformation sequences. Over training, musical structure is not stored explicitly but encoded as persistent geometric trajectories in state space.

You do not retrieve memory. You evolve a system into a region of structured behavior.

---

## Further Reading

See `GSM_paper.md` for formal derivations, comparison to SSMs and transformers, and analysis of scaling behavior across datasets.

---

## License

MIT
