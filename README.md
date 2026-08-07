POTENTIAL USE CASE IS STREAMING TRAINING AND INFERENCE AT THE SAME TIME ONLINE LEARNING AND INFERENCE.

# GSM — Geometric State Machine

> No attention. No KV cache. No quadratic scaling. A fixed point in R^N being continuously deformed by a learned algebra of transformations.

**Training parallelization is solved.** The recurrence is now an associative scan that resolves a whole sequence in O(log T) depth instead of a T-step loop — verified exact to 5.1e-13 in float64 at T=512, with training cost roughly flat in sequence length. See [Parallelization](#parallelization).

Scales to 179k+ file datasets with memory-mapped binary packing — no architecture changes required.

---

## What Is Actually Verified

Separating what has been measured from what is asserted. Every claim below is reproducible with a script in this repo.

### Verified true

| Claim | Evidence |
| ----- | -------- |
| The scan is algebraically exact | 5.1e-13 max deviation from step-by-step reference in float64 at T=512; verified at T = 1…1024 including non-powers-of-two |
| Training is ~6× faster | 54k tok/s at seq_len 512 vs. 9k tok/s recorded for the sequential version at seq_len 256 |
| Cost is near-flat in sequence length | 8× longer sequences cost 19% more time (5.8ms → 6.9ms) |
| Stable without gradient clipping | 70 epochs at seq_len 512 under fp16 AMP, no NaN/inf, gradient norm ~2 |
| Generation matches training semantics | Incremental `step()` agrees with the parallel scan to fp32 rounding |

### Verified false, or unsupported

| Previous claim | What measurement shows |
| -------------- | ---------------------- |
| "Generates convincing baroque piano music" | Generated samples are **60–96% verbatim copies** of single training pieces (`check_memorization.py`, 8/8 samples flagged across two independent generation paths) |
| "Knowledge is shaped into the geometry" | The state is a **leaky integrator over ~19 tokens**, not an accumulated manifold position. Zeroing it changes 0.8% of later predictions — it rebuilds within ~19 steps. |
| "Accumulates context geometrically" / long-range memory | Predictions match full-history at a **32-token** truncated context (100% agreement; 85% at 16, 65% at 8). It is a short-window model. |
| **Subspace rotation is the novel core** | **Removing rotation entirely changes 0.00% of predictions.** It touches 128 pairs = 6.2% of 4096 dims, and cumulative angle reaches ~63 rad (≈10 full turns). The decoder learned to ignore those dimensions. |
| "Geometric bias prevents memorization on small data" | The opposite was observed on exactly the dataset used to support the claim. |
| "GSM learns effectively from very small datasets" | It memorized 202 files. Whether it *learns* from them is untested. |

### Correction to an earlier version of this section

A previous revision stated the state "decays to `e^-358`" and was therefore "decorative." **That was wrong**, and the error is worth recording: `e^-358` is the decay of `S₀`'s contribution — the *initial condition* — not of the state. The shift term `b` reinjects signal every step, so `|S|` stays around 0.1 for all 512 tokens. The state is real and carries information; it just only carries the recent past. Measuring decay of the initial condition and reporting it as decay of the state conflated two different quantities.

### What the model actually is

With rotation contributing nothing and the state's horizon at ~19 tokens, stripping the geometric framing leaves:

> a 6-layer MLP mapping each token to `scale`/`shift`/`gate`, composed into a **leaky-integrator running average** (`S ← a⊙S + b`, `a ≈ 0.71`) over roughly the last 19 tokens, decoded by a 3-layer MLP.

That is a gated linear RNN with diagonal transitions — structurally close to a simplified Mamba/RWKV minus the selective mechanism, plus a rotation that measurement shows is inert. The leaky integrator is a legitimate architecture. The parts that made GSM *novel* — the manifold framing and the subspace rotations — are the parts that do nothing here.

### Not yet tested

Generalization of any kind. There is **no held-out validation split** in `train.py`, so every loss number in this README is training loss and none of them measure learning. An n-gram baseline comparison was attempted and is **unusable**: the "held-out" files were in GSM's training set. (The 4-gram baseline itself, 40.8% held-out next-token accuracy, is valid; GSM's 96.3% is memorization, not skill.)

Whether the architecture can generalize with early stopping, more data, or regularization is genuinely open. These results describe one training run, not a proof the approach cannot work.

Reproduce with:

```cmd
python check_memorization.py                        # copying vs. training data
python probe_state.py                               # is the state used?
python capture_live.py && python check_memorization.py --generated generated_live
```

⚠️ **This architecture has not been shown to work as described.** A trained checkpoint (loss 0.0121, 70 epochs on 228 Bach files) reproduces training data verbatim, uses only ~19 tokens of context, and is bit-identical in its predictions with the subspace rotations removed.

What *is* verified is the scan reformulation: algebraically exact (5.1e-13 in float64 at T=512), ~6× faster training, numerically stable without gradient clipping. Those are claims about the *implementation*, not about the architecture learning anything.

The Bach results this README previously led with — 54 minutes, loss 0.1196, "convincing baroque piano music" — came from the older sequential recurrence and were never tested for memorization. Given that the current model memorizes on the same corpus, **those results should be assumed to reflect copying as well** until someone checks. Preserved in [Results](#results) as a historical record only.

---

## What Is This

Most sequence models treat context as something to *store* — transformers cache every previous token's keys and values, RNNs overwrite a memory buffer. Both approaches scale poorly: transformers pay quadratic cost in sequence length, RNNs struggle with long-range dependencies.

GSM is an attempt to treat context as something to *accumulate geometrically*.

The model maintains a single fixed-size point `S ∈ R^N` — a position in a high-dimensional geometric space. Each token is not a data point to store but a **transformation operator** that deforms that geometry. The intent is that the state is never a memory buffer being overwritten, but a manifold position continuously reshaped by a learned transformation algebra.

That is the design intent. In the one trained model measured so far, it did not happen: the state's contribution decays to nothing within ~19 tokens, and the model reproduces training data rather than composing. See [What Is Actually Verified](#what-is-actually-verified). Read the rest of this section as a description of the architecture's design, not of demonstrated behavior.

# For beginners:

Imagine you're trying to understand a piece of music by listening to it note by note.

A transformer is like someone who writes down every note they hear on a piece of paper, then whenever they need to understand the next note, they look back at everything they've written. The longer the piece, the more paper they need, and the longer it takes to look things up. It's powerful but expensive.

An RNN is like someone who keeps a single "impression" in their head and updates it as each note plays — but their head only has so much room, and old notes tend to get crowded out. They can only kind of remember the distant past.

GSM does something different. Imagine you have a ball floating in an enormous space — thousands of dimensions, far more than the three we can picture. Each note you hear doesn't get written down or crammed into a memory. Instead, it *pushes and rotates the ball* in that space. Each note is a transformation operator: it shoves the ball, scales it, twists it through certain dimensions by a learned angle.

By the end of the piece, the ball is sitting somewhere specific in that enormous space. That position *is* the model's understanding of everything it's heard. Not a list of notes. Not a compressed summary. A geometric position that accumulated the entire sequence through continuous deformation.

When the model wants to predict the next note, it just looks at where the ball is sitting right now and asks: given this position in this space, what note comes next?

**Why might this work?**

In a space with 4096 dimensions, there is an enormous amount of room to encode structure. The hope is that musical patterns — a chord progression, a rhythmic motif, a harmonic resolution — each carve out a characteristic trajectory through that space during training, so that similar musical contexts move the ball to similar regions.

The idea is to not store the music, but let the music reshape a geometry, and trust that geometry to remember what matters.

**Whether it does is a separate question, and so far the answer is no.** In the trained model measured here, the ball only remembers about the last 19 notes. Each note nudges it, and each nudge fades to nothing within roughly 19 more notes — so it works like a running average of the recent past, not an accumulation of the whole piece. You can pick the ball up and move it somewhere completely different mid-piece, and within 19 notes the model has recovered and carries on as if nothing happened.

The twisting — the rotation, the part that made this design unlike anything else — turned out to do **literally nothing**. Switch it off entirely and the model produces the exact same notes. It only ever twisted 6% of the dimensions, and it twisted them so far (about ten full turns by the end of a piece) that the result was scrambled noise the rest of the model learned to ignore.

So what remained was a short-memory model with a very large memory of *pieces*, reciting what it had seen. The architecture permits the behavior described above; it does not compel it, and nothing has yet made it happen.

**The compute property is real and unaffected:** each note takes the same amount of compute, and the ball stays the same size regardless of how long the piece is. No growing list, no quadratic blowup. That much holds regardless of how well the model learns.

**What about training speed?** This used to be the catch. Because each step depends on the previous position of the ball, the obvious way to train is one note at a time — no parallelism, slow on a GPU.

The fix is a trick about *order of operations*. If each note's effect on the ball can be written as "stretch it by this much, then nudge it by that much," then two notes in a row can be collapsed into a single combined stretch-and-nudge before either is applied to the ball. Do that pairwise, over and over, and a thousand notes fold down in about ten rounds instead of a thousand steps — and you get exactly the same answer, because you never changed *what* is computed, only *when*. Nothing is approximated.

So the ball still moves note by note in principle, but the GPU works out where it lands without walking each step.

# For experts:

GSM is a fixed-dimensional state space model with a geometric inductive bias. The state `S ∈ R^N` evolves under a sequence of input-parameterized transformations rather than a learned autonomous dynamics matrix. The update rule at each step is:

```
a_t = gate ⊙ scale + (1 - gate)        folded affine multiplier
b_t = gate ⊙ shift                     folded affine offset

S_t = a_t ⊙ S_{t-1} + b_t              associative affine map
readout_t = LayerNorm(Rotate(S_t, Σ_{i≤t} θ_i))
```

where `scale`, `shift`, `gate ∈ R^N` and the rotation angles `θ ∈ R^{n_pairs}` are all outputs of a 6-layer residual MLP — TransformNet — conditioned on the current token embedding.

This factoring is what makes the model parallelizable, and it is a deliberate departure from the earlier formulation `S' = LayerNorm(gate ⊙ Rotate(scale ⊙ S + shift) + (1-gate) ⊙ S)`. Two changes: LayerNorm and rotation are moved *out* of the recurrence and applied to a readout snapshot, so the quantity actually threaded through time is the bare affine state. Composition of affine maps is associative, so the whole sequence resolves by scan (see [Parallelization](#parallelization)); LayerNorm inside the loop would destroy that, since it is nonlinear and does not compose.

Rotation can be lifted out because it is applied *after* the affine part rather than interleaved with it, and rotations in a fixed 2-D subspace compose additively in angle — so the cumulative rotation at step `t` is just the prefix sum of angles. Nothing in the update rule is autonomous: `S` has no direct recurrence through a fixed weight matrix. It only moves when a token moves it, and the direction and magnitude of movement are entirely input-determined.

**The rotation component** is the architecturally novel piece. A fixed set of `n_pairs` random index pairs `(i, j) ⊂ [N]²` are sampled at initialization and frozen. For each pair, TransformNet produces an angle `θ_k`, and a 2D rotation is applied in that subspace:

```
[S_i, S_j] ← [cos θ_k · S_i - sin θ_k · S_j,  sin θ_k · S_i + cos θ_k · S_j]
```

All pairs are computed in parallel via gather/scatter. This is a sparse approximation to a full SO(N) group action — the model learns to compose subspace rotations to implement semantic transformations. It's isometric by construction, which acts as an implicit norm-preserving regularizer on the state trajectory before the gate mixing and LayerNorm.

**Stability without the per-step norm.** Removing LayerNorm from the recurrence removes a safety net, and this has a concrete consequence: `scale` is bounded to `(0, 1)`, not `(0, 2)` as in the earlier formulation. With the per-step norm gone, the folded multiplier `a_t` compounds multiplicatively across the entire sequence, so any `a_t > 1` grows without bound. Constraining `scale < 1` makes `a_t = gate·scale + (1-gate)` a convex combination of `scale` and `1`, hence in `(0, 1]` — the composed multiplier over any span is a product of terms `≤ 1` and cannot expand. Measured with a `(0, 2)` range, the state reaches ~`e^69` by 100 tokens and overflows fp32 well before a 512-token sequence finishes. This bound is not a tuning choice; it is what keeps the no-clipping stability property true.

**Relation to SSMs.** GSM superficially resembles S4/Mamba in maintaining a fixed-size latent state, but the similarity is shallow. SSMs parameterize a linear dynamical system `S' = AS + Bx` where `A` is a structured matrix (diagonal-plus-low-rank, HiPPO-initialized) optimized to capture long-range dependencies through careful eigenspectrum control. The input modulates the input projection `B` and sometimes `Δ` (discretization step), but the core dynamics matrix `A` is fixed or input-independent.

GSM has no autonomous dynamics at all. The entire transformation — including what would correspond to `A` — is a function of the input. This is a stronger form of input-conditioning and removes the need for eigenspectrum engineering, but it also means the model can't learn input-independent temporal dynamics. Whether that's a limitation or a feature depends on the domain.

**In practice the similarity turned out to be much less shallow than claimed.** With the scan reformulation, the recurrence *is* `S ← a⊙S + b` with input-dependent diagonal `a` — a selective diagonal linear recurrence, which is precisely Mamba's core operation. Since measurement shows the rotation contributes nothing, what remains is a diagonal SSM whose transition is input-conditioned, without HiPPO initialization or a principled eigenspectrum. The learned `a ≈ 0.71` gives an effective horizon of ~19 tokens, where structured SSM initializations exist specifically to avoid that kind of fast forgetting. The honest positioning is "a diagonal selective SSM that omits the initialization theory," not "a fundamentally different paradigm."

**Relation to GRUs.** The gate mechanism `S' = gate ⊙ S_new + (1 - gate) ⊙ S` is structurally identical to a GRU update gate, and the shift/scale is analogous to the candidate hidden state. The difference is that a GRU computes its candidate via `tanh(W_h · (r ⊙ h) + W_x · x)` — a fixed recurrent projection `W_h` applied to the gated previous state. GSM replaces this entirely: there is no `W_h`, and the candidate state is produced by a geometric operation (rotation in random subspaces) rather than a linear projection. The inductive bias shifts from "linear memory compression" to "isometric geometric deformation."

**Parallelization.** Two independent facts combine here. First, TransformNet — the dominant compute cost — has no dependency on `S`, so all `T` calls batch into a single `[B·T, d]` matmul. Second, and this is the part that used to be missing, the state recurrence itself is no longer sequential: with the update written as an affine map, composition is associative and the whole sequence resolves by parallel scan in O(log T) depth. There is no Python loop over time in the forward pass at all.

The earlier claim that the recurrence was "irreducibly sequential" was wrong — it was a property of the chosen formulation (per-step LayerNorm, interleaved rotation), not of the architecture. Refactoring the update to expose an associative operator removes the constraint entirely.

**Small datasets — a hypothesis that failed its first test.** The argument was: subspace rotations are a highly constrained family of transformations, so the model can't implement arbitrary state transitions, only isometric deformations followed by gated mixing. On a small corpus this constraint should act as an implicit regularizer preventing the memorization a less constrained model would fall into.

Measurement contradicts this. On 228 Bach files the model memorized thoroughly — 60–96% verbatim reproduction. The rotation constraint never bound, for two independent reasons: rotation touches only 6.2% of state dimensions and ablates to no effect at all, and the state's own horizon is ~19 tokens, so the model is effectively a short-window token map with 32M parameters of lookup behind it. A constraint on an unused component cannot regularize anything.

The flaw in the original argument is that it constrains *how the state evolves* while saying nothing about capacity. The state is a pointer, not the store; 32M parameters in TransformNet are where a corpus this size actually fits. A fixed-size state bounds how much context can be live at once — not how much the model can memorize. Those are independent quantities, and this README previously conflated them.

**The state was never used, at any point in training.** Probing checkpoints across the run:

| Epoch | Loss | gate | Horizon | Zeroing S changes | Copying |
| ----- | ---- | ---- | ------- | ----------------- | ------- |
| 1 | 3.7235 | 0.548 | 30 tok | 1.6% | — |
| 3 | 1.6455 | 0.507 | 34 tok | 6.7% | 11.0% |
| 10 | 0.8262 | 0.433 | 27 tok | 3.5% | 5.2% |
| 20 | 0.3188 | 0.396 | 22 tok | 0.8% | 6.0% |
| 30 | 0.0871 | 0.390 | 21 tok | 1.2% | 18.3% |
| 70 | 0.0121 | 0.389 | 19 tok | 0.8% | **63.4%** |

There is no collapse to find — state usage peaks around 7% at epoch 3 and drifts down. Copying is a *separate, later* phenomenon that tracks loss, not state usage, and only explodes after epoch 30.

**A likely cause is the initialization.** `TransformNet.output_proj` is zero-initialized so training starts at the identity transform (`a ≡ 1`, `b ≡ 0`, `gate = 0.5`). At that point the state is a frozen constant contributing nothing, so *every* early gradient comes through the token-conditioned path. By the time the state could become useful, the token path already explains most of the loss. The zero-init that guarantees a stable start also makes state-independence the path of least resistance. This is a hypothesis suggested by the sweep, not a tested claim.

**The usable window is roughly epochs 10–30**: 40–60% novel 8-grams with modest copying. `epoch_010_loss0.8262.pt` is the best available checkpoint at 5.2% copying — not a good model, but not a lookup table. (3 samples per epoch; the epoch-30 uptick could be noise.)

**Open questions.** Whether the random fixed subspace pairs are the right structure — versus learned pairs, full dense rotations, or a hierarchical decomposition — is unexplored. The initialization of `S_0` as a learned parameter rather than zero or a fixed point is also non-obvious; it means the model learns a "prior geometric position" that all sequences start from.

Moving LayerNorm out of the recurrence changes the state geometry in a way that has not been characterized empirically. Previously the per-step norm held the state on a roughly unit hypersphere, which combined with the isometric rotations made the effective geometry closer to spherical than flat. Now the threaded state is bare affine and contracting (`a_t ≤ 1`), with normalization applied only at readout — so the internal trajectory is free to shrink toward the origin over long sequences even though the readout stays normalized. Whether that contraction costs long-range expressivity in practice, and whether the readout norm fully compensates, is the main open question introduced by the scan reformulation.
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
| Long-range      | Vanishing gradient problem    | Gate *can* control deformation magnitude — in the trained model it learned to discard state within ~19 tokens |
| Training depth  | O(T) sequential               | O(log T) associative scan           |

The critical difference: RNNs have a **fixed recurrent weight matrix W_hh** that maps state to state regardless of input. In GSM, the transformation of the state is **entirely parameterized by the input token**. The state has no direct path to itself — it only moves when a token moves it, and how it moves depends entirely on what the token is.

The subspace rotation component has no RNN analogue at all. It applies learned rotations in random fixed dimension pairs across R^N — pure geometric deformation with no equivalent in any classical sequence model.

---

## Architecture

```
All tokens t = 1..T, in parallel
  └─ Embedding lookup → E ∈ R^{B×T×embed_dim}
       └─ TransformNet (6-layer MLP, one batched [B·T, d] matmul)
            ├─ scale  ∈ R^{state_dim}    multiplicative field, bounded (0,1)
            ├─ shift  ∈ R^{state_dim}    additive perturbation
            ├─ gate   ∈ R^{state_dim}    geometric mixing coefficient
            └─ angles ∈ R^{n_pairs}      subspace rotation angles
                 │
                 ├─ fold → a = gate⊙scale + (1-gate),  b = gate⊙shift
                 │    └─ associative scan over (a, b)      O(log T) depth
                 │         (a₂,b₂)∘(a₁,b₁) = (a₂a₁, a₂b₁+b₂)
                 │         └─ S = a ⊙ S₀ + b                all t at once
                 │
                 └─ cumsum(angles) → RotarySubspaceTransform (all pairs parallel)
                      └─ LayerNorm → bounded manifold position
                           └─ Decoder (3-layer MLP) → logits ∈ R^{vocab_size}
```

Generation uses the same math one token at a time: `GeometricStateStep.step()` carries `(S_raw, angle_acc)` — the pre-rotation, pre-norm state plus accumulated angle — and applies rotation and LayerNorm to a snapshot for readout without feeding them back. This is what keeps sampling consistent with training.

### The High-Dimensional Plane

The state lives in flat R^N, not on a curved manifold. This is intentional.

In sufficiently high dimensions, flat space functionally folds — two points can be geometrically distant in every low-dimensional projection yet adjacent along some dimension the model has learned to use. The richness comes not from topology but from the transformation algebra carving semantic structure into the geometry through training. Recurring patterns deepen into stable attractors. Noise washes out. The manifold learns to fold itself.

### TransformNet

A 6-layer MLP with residual connections that maps a token embedding to transformation parameters. Fixed depth means fixed compute — O(1) per token regardless of sequence length or corpus size. Initialized to near-identity so training starts from a stable geometric configuration.

### RotarySubspaceTransform

The geometrically novel component, and **the one measurement shows is doing nothing.**

A fixed set of random dimension pairs `(i, j)` in R^N. For each pair, the model produces a rotation angle and applies a 2D rotation in that subspace. All pairs computed simultaneously via indexing — no Python loops, fully vectorized on GPU.

There is no classical sequence model operation that corresponds to input-parameterized subspace rotations on a fixed geometric object. That novelty is real. Its contribution to this model is not:

```
rotation ON vs OFF, 512 tokens:  100.00% identical predictions
```

Two structural reasons, both fixable:

* **Coverage.** `n_pairs=128` rotates 256 of 4096 dimensions — **6.2%** of the state. The other 93.8% never rotate. Scaling `n_pairs` toward `state_dim / 2` would make rotation act on the whole state.
* **Angle wrap.** Angles accumulate additively with no bound, reaching **~63 radians** (≈10 full revolutions) by t=512. At that magnitude the rotation is an arbitrary scramble of those dimensions, uncorrelated with anything, so the decoder learns to ignore them. Bounding cumulative angle — or applying rotation per-step rather than cumulatively — would keep it in an informative range.

The logits are not bit-identical (max difference 9.03), so rotation is not mathematically inert. The decoder simply routed around it. Anyone continuing this work should treat these two fixes as the first experiment, not the geometric framing as settled.

---

## Complexity

| Property             | Transformer        | RNN/LSTM | GSM      |
| -------------------- | ------------------ | -------- | -------- |
| Memory per token     | O(n) KV cache      | O(1)     | O(1)     |
| Compute per token    | O(n) attention     | O(1)     | O(1)     |
| State size           | Grows with context | Fixed    | Fixed    |
| Scales to any corpus | ✓                  | ✓        | ✓        |
| Long context cost    | Quadratic          | Linear   | **O(1)** |

These are costs, not capabilities. O(1) memory per token says nothing about how much context the model actually *uses*: in the trained checkpoint measured here the effective horizon was ~19 tokens, so the fixed-size state was cheap precisely because it was carrying almost nothing. A model that ignores its state has excellent memory complexity and no memory.

---

## Parallelization

There is a single forward path, and it has no loop over time. `model.train()` and `model.eval()` no longer select different implementations.

**TransformNet** — the 6-layer MLP that maps embeddings to transformation parameters — has no dependency on the state `S`. All `seq_len` token embeddings process simultaneously in one batched matmul. This is the dominant compute cost and benefits fully from GPU parallelism.

**The recurrence** resolves by associative scan. Each token's update is folded into an affine map `S → a⊙S + b`, and two such maps compose as `(a₂,b₂)∘(a₁,b₁) = (a₂a₁, a₂b₁+b₂)` — associative, so a Hillis-Steele scan computes all `T` prefixes in `⌈log₂ T⌉` rounds of elementwise ops. Rotations compose additively in angle and are applied once via `cumsum`. The scan is exact, not an approximation.

**Exactness.** Verified against a step-by-step reference at `T = 1, 2, 3, 7, 16, 17, 64, 100, 257, 512, 1024` (non-powers-of-two included, since the scan pads with the identity operator). In float64 the maximum disagreement is **5.1e-13 at T=512** and 7.1e-13 at T=1024 — algebraically exact. In float32 the same comparison drifts to ~4e-4 at T=512, which is accumulation rounding, not a logic difference.

**Measured cost** (RTX 5060 Ti, forward+backward, batch 8, state_dim 256, n_pairs 32, fp32):

| seq_len | Time per fwd+bwd |
| ------- | ---------------- |
| 64      | 5.8 ms           |
| 128     | 5.9 ms           |
| 256     | 6.3 ms           |
| 512     | 6.9 ms           |

An 8× increase in sequence length costs 19% more time — the log-depth showing up in wall clock. The old sequential loop scaled linearly.

**Stability check.** 60 training steps at `seq_len=512` under fp16 AMP: loss 4.16 → 3.76, gradient norm ~2 with **no clipping applied**, state finite throughout (`max|S| = 4.14`). The contraction bound on `scale` is doing its job.

> The earlier benchmark table in this section compared two variants of the sequential implementation (per-step vs. batched TransformNet, 2–3× on inference, 1.0× on training). Both are gone, so those numbers no longer describe any code path in the repo and have been removed rather than left to mislead.

On Linux/WSL2, `torch.compile` on TransformNet and the decoder gives a further 10–30%.

---

## Training Tradeoffs

The training forward pass is no longer a loop over sequence length, so the main tradeoff described here previously — slow wall-clock training from sequential state evolution — no longer applies. Training cost is now roughly flat in sequence length (see [Parallelization](#parallelization)).

What remains:

* **`torch.compile`** improves throughput on Linux/WSL2 by fusing TransformNet and decoder kernels. Not available on native Windows (no Triton), enabled automatically when detected.
* **Custom CUDA kernels** would still help — the scan is written in pure PyTorch and materializes `⌈log₂ T⌉` intermediate tensors. A fused kernel would cut memory traffic. Avoided so far to preserve simplicity and portability.
* **Memory** is the new axis to watch: the scan holds `[B, T, state_dim]` tensors rather than one `[B, state_dim]` state, so a step that was O(1) in memory during the recurrence is now O(T). At `state_dim=4096` and long sequences this can bind before compute does.

Training cost still scales with dataset size; inference cost does not.

This README previously claimed GSM "learns effectively from very small datasets," citing coherent baroque output from 228 files. That claim is withdrawn. The scan model trained on the same corpus reproduces training pieces verbatim rather than composing, and the sequential result it was based on was never checked for memorization — the tooling to check ([check_memorization.py](check_memorization.py)) did not exist until after that claim was written.

---

## Results

> ⚠️ **These numbers do not demonstrate that the model learned to compose.** They predate the scan reformulation and were produced by the older sequential recurrence (per-step LayerNorm, interleaved rotation, `scale ∈ (0,2)`). More importantly, they are **training** losses with no held-out split, and the output was never checked for memorization. When the current model was trained on this same corpus and checked, it reproduced training pieces verbatim. The qualitative descriptions below ("convincing baroque piano music", "coherent composition structure") describe output that was listened to but not tested for copying, and should be read with that in mind. Kept as a historical record.

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

### Existing checkpoints

⚠️ **Checkpoints trained before the scan reformulation will load without error but will not work correctly.** Parameter shapes are unchanged, so `load_state_dict` accepts them silently, but they were trained under a different recurrence (per-step LayerNorm, interleaved rotation, `scale ∈ (0,2)`). Nothing in a checkpoint file records which update rule produced it.

If a loaded model produces garbage for no apparent reason, this is the first thing to check. Retrain rather than attempting to port — there is no weight transformation that converts between the two recurrences.

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

⚠️ Check generated output before treating it as composition. At temperature 0.75 the current checkpoint emits near-verbatim copies of training pieces (mean 90.7% contiguous overlap). Raising temperature to 0.9 reduces but does not eliminate this (mean 63.4%). Always run:

```cmd
python check_memorization.py --generated generated
```

---

### 5. Benchmark

Compare the scan model against the older sequential recurrence, which `benchmark_compare.py` keeps its own self-contained copy of as a baseline:

```cmd
python benchmark_compare.py
python benchmark_compare.py --trials 20 --seq_len 512
```

O(1) inference scaling confirmation:

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

### 8. Validate what the model actually learned

Loss cannot tell you whether a model learned or memorized, and it cannot tell you whether the geometric state is being used. These two tools can. **Run both before believing any result from this repo.**

```cmd
python check_memorization.py                     # is output copied from training data?
python probe_state.py                            # does the state carry information?
```

`check_memorization.py` measures, per generated sample, the longest verbatim token run shared with any training file, plus n-gram coverage and novelty, plus self-similarity between samples (mode collapse). Compares in token space so MIDI round-tripping cannot distort it.

`probe_state.py` reports the learned gate/multiplier statistics, the effective memory horizon, and — the decisive test — what fraction of predictions change when the state is zeroed, noised, or shuffled mid-sequence. High agreement means the state is decorative.

To check the live-playback path (which uses the incremental `step()`, different code from training's scan):

```cmd
python capture_live.py                                          # headless play_live
python check_memorization.py --generated generated_live
```

Useful options:

| Flag | Tool | Notes |
| ---- | ---- | ----- |
| `--generated`, `--dataset` | check_memorization | directories to compare |
| `--ngram`, `--min-run` | check_memorization | match lengths (default 8, 12) |
| `--json` | check_memorization | write a machine-readable report |
| `--checkpoint` | probe_state, capture_live | probe an earlier epoch to find where copying begins |
| `--corrupt-at` | probe_state | step at which to ablate the state |
| `--temperature`, `--top_k` | capture_live | defaults mirror `play_live.py` |

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

The intuition motivating the design: high-dimensional flat space behaves as a structured representational medium under learned transformation dynamics. In 4096 dimensions, semantic regions might emerge as stable attractors of repeated transformation sequences, so that musical structure is encoded as persistent geometric trajectories rather than stored explicitly.

The aspiration is to evolve a system into a region of structured behavior rather than retrieve memory.

What was measured instead is retrieval: a random 4-token prompt selects a memorized piece, which is then replayed near-verbatim, while the state itself decays to irrelevance within ~19 tokens. The intuition above remains a hypothesis. It has been tested once and it did not hold.

---

## Further Reading

See `GSM_paper.md` for formal derivations, comparison to SSMs and transformers, and analysis of scaling behavior across datasets.

---

## License

MIT
