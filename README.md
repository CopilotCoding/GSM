WORKING ON TRAINING PARALLELIZATION SOLUTION, CURRENTLY SEEING GOOD RESULTS


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

The geometrically novel component. A fixed set of random dimension pairs `(i, j)` in R^N. For each pair, the model produces a rotation angle and applies a 2D rotation in that subspace. All pairs computed simultaneously via gather/scatter — no Python loops, fully vectorized on GPU.

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

## Training Tradeoffs

GSM’s O(1) inference property comes with a training dynamic that is important to understand.

The state update is strictly sequential — each step depends on the previous one, so the forward pass is a loop over sequence length regardless of batch size. This means:

* **Small datasets (<10k sequences):** The Bach corpus trains in ~54 minutes.
* **Large datasets (millions of sequences):** Slower wall-clock training due to sequential state evolution per token.
* **`torch.compile`** would significantly improve throughput by fusing step execution, but is not available in all environments.
* **Custom CUDA kernels** could parallelize sequence dynamics, but are intentionally avoided to preserve simplicity and portability.

The fundamental tradeoff: **training cost scales with dataset size; inference cost does not.**

For large datasets, it remains viable but benefits strongly from optimized compilation paths.

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

Outputs are not merely “melodic fragments” — they exhibit **coherent baroque-style composition structure**, including:

* phrase repetition with variation
* functional harmonic movement
* cadential resolution behavior
* stable rhythmic motifs

A smaller 6M parameter GSM trained on the same dataset reached a best loss of 1.3768 after 30 epochs which failed to produce proper music (~9 minutes). The 32M model surpassed this early (by epoch 10) and continued refining structural coherence to 0.1196.

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

Example output:

```
Winner: config #5 | 18.37M params
...
```

---

### 1. Process MIDI Dataset

```cmd
python -m data.pipeline --midi_dir C:\path\to\midi\files --out_dir dataset --vocab_path vocab.json --workers 8
```

---

### 2. Pack Dataset (recommended for large datasets)

```cmd
python -m data.pack --data_dir dataset --out_dir dataset_packed --seq_len 256 --workers 20
```

---

### 3. Train

```cmd
python -m train.train --data_dir dataset_packed --vocab_path vocab.json --out_dir checkpoints --epochs 100 --workers 8
```

Outputs:

* `latest.pt`
* `best.pt`
* logs + samples + metrics

---

### 4. Generate

```cmd
python -m generate.generate --checkpoint checkpoints\latest.pt --vocab_path vocab.json --out_dir generated --n_samples 5 --length 512 --temperature 0.75
```

At 0.75 temperature, outputs are **stylistically stable baroque compositions** suitable for direct listening in MIDI DAWs.

---

### 5. Benchmark

```cmd
python benchmark.py --checkpoint checkpoints\latest.pt --vocab_path vocab.json
```

Confirms:

* O(1) inference scaling
* constant memory usage
* stable throughput across sequence lengths

---

### 6. Plot Training

```cmd
python plot_training.py checkpoints\training_log.csv
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
