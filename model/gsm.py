"""
GSM — Geometric State Machine
==============================
Fixed-geometry state manifold architecture.
Each token is a transformation operator acting on a point in R^N.
Fixed compute per token, fixed state size, scales to any corpus.

Parallel associative scan
-------------------------
The recurrence is expressed as an associative operator so the whole sequence
resolves in O(log T) depth instead of a T-step Python loop.

Each token contributes an affine map on the state, `S -> a * S + b`, with the
gate folded into the affine part:

    a = gate * scale + (1 - gate)
    b = gate * shift

Two such maps compose as

    (a2, b2) o (a1, b1) = (a2 * a1,  a2 * b1 + b2)

which is associative, so a Hillis-Steele scan applies. Rotations are
norm-preserving and compose additively in angle within each 2-D subspace, so
the cumulative rotation at step t is the prefix sum of angles and is applied
once after the affine scan resolves.

Stability comes from geometry rather than clipping: rotation preserves length,
and the affine part is non-expanding by construction. `scale` is bounded to
(0, 1) so that `a` is a convex combination of `scale` and 1 and therefore lies
in (0, 1] -- the composed multiplier over any span is a product of terms <= 1
and cannot grow. This bound is load-bearing: dropping the per-step LayerNorm is
what buys associativity, and without it a scale range of (0, 2) compounds
multiplicatively and overflows fp32 within a few hundred tokens.

This is a different recurrence from the earlier sequential GSM, which applied
LayerNorm inside the loop and interleaved rotation with each affine step.
Neither is associative, so neither could be scanned.

WARNING: checkpoints trained under that sequential recurrence have identical
parameter shapes, so `load_state_dict` accepts them without complaint, but they
were trained under different dynamics and will not reproduce their old outputs.
Nothing in a checkpoint marks which recurrence produced it -- if a loaded model
behaves badly for no visible reason, this is the first thing to check. Retrain
rather than trying to port.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotarySubspaceTransform(nn.Module):
    """
    Vectorized rotations of `n_pairs` disjoint 2-D subspaces.
    Norm-preserving by construction. Operates on any leading batch shape.
    """
    def __init__(self, state_dim: int, n_pairs: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_pairs = n_pairs
        idx = torch.randperm(state_dim)[:n_pairs * 2].reshape(n_pairs, 2)
        self.register_buffer("idx_a", idx[:, 0].contiguous())
        self.register_buffer("idx_b", idx[:, 1].contiguous())

    def forward(self, S: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        cos_t = torch.cos(angles)
        sin_t = torch.sin(angles)
        a = S[..., self.idx_a]
        b = S[..., self.idx_b]
        new_a = cos_t * a - sin_t * b
        new_b = sin_t * a + cos_t * b
        S_new = S.clone()
        S_new[..., self.idx_a] = new_a
        S_new[..., self.idx_b] = new_b
        return S_new


class TransformNet(nn.Module):
    """
    Maps token embeddings to transformation parameters.
    Deep MLP with residual connections for gradient flow.
    Produces: scale, shift, gate (state_dim each), angles (n_pairs).

    Accepts any leading batch shape, so it can be called on [B, T, E] directly.
    """
    def __init__(self, embed_dim: int, state_dim: int, n_pairs: int, hidden_dim: int = 1024, n_layers: int = 6):
        super().__init__()
        self.state_dim = state_dim
        self.n_pairs = n_pairs
        out_dim = state_dim * 3 + n_pairs

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
        )

        # Deep residual middle layers
        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_layers - 2)
        ])
        self.res_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers - 2)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

        # Init output to near-identity
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, e: torch.Tensor):
        x = self.input_proj(e)

        for res_layer, norm in zip(self.res_layers, self.res_norms):
            x = norm(x + res_layer(x))

        out = self.output_proj(x)
        d = self.state_dim
        # scale is bounded to (0, 1), NOT (0, 2) as in the earlier sequential
        # GSM. There, an in-loop LayerNorm renormalized the state every step,
        # so scale > 1 could not compound. The scan has no per-step norm --
        # that is precisely what makes it associative -- so the affine
        # multiplier composes multiplicatively across the whole sequence.
        # With scale < 1 the folded multiplier a = gate*scale + (1-gate) is a
        # convex combination of scale and 1, hence in (0, 1]: the cumulative
        # product is non-expanding and the state cannot blow up. Allowing
        # scale up to 2 overflows fp32 within a few hundred tokens.
        scale  = torch.sigmoid(out[..., :d])
        shift  = out[..., d:d * 2] * 0.1
        gate   = torch.sigmoid(out[..., d * 2:d * 3])
        angles = out[..., d * 3:] * 0.1
        return scale, shift, gate, angles


class GeometricStateStep(nn.Module):
    """
    The scan-form state update, in both parallel and incremental flavours.

    Per token the state map is the affine operator

        S -> a * S + b,   a = gate * scale + (1 - gate),  b = gate * shift

    followed, once the affine part has resolved, by a rotation through the
    accumulated angle. `scan` applies this to a whole sequence in O(log T)
    depth; `step` advances a single token for generation using exactly the same
    math, so sampling matches training.
    """
    def __init__(self, embed_dim: int, state_dim: int, n_pairs: int, hidden_dim: int = 1024, n_layers: int = 6):
        super().__init__()
        self.transform_net = TransformNet(embed_dim, state_dim, n_pairs, hidden_dim, n_layers)
        self.rotary = RotarySubspaceTransform(state_dim, n_pairs)
        self.norm = nn.LayerNorm(state_dim)

    @staticmethod
    def _affine(scale, shift, gate):
        """Fold scale/shift/gate into a single affine operator (a, b)."""
        a = gate * scale + (1.0 - gate)
        b = gate * shift
        return a, b

    def scan(self, E: torch.Tensor, S0: torch.Tensor) -> torch.Tensor:
        """
        Resolve a whole sequence. E: [B, T, embed_dim], S0: [state_dim].
        Returns normalized states [B, T, state_dim].
        """
        scale, shift, gate, angles = self.transform_net(E)
        a, b = self._affine(scale, shift, gate)

        batch, seq_len, state_dim = a.shape

        # Hillis-Steele inclusive scan over the affine operators. Each round
        # composes every element with the one `sh` positions back; identity
        # (a=1, b=0) pads the front so early positions compose with a no-op.
        steps = math.ceil(math.log2(seq_len)) if seq_len > 1 else 0
        for i in range(steps):
            sh = 1 << i
            a_prev = F.pad(a[:, :-sh], (0, 0, sh, 0), value=1.0)
            b_prev = F.pad(b[:, :-sh], (0, 0, sh, 0), value=0.0)
            b = a * b_prev + b
            a = a * a_prev

        states = a * S0.view(1, 1, -1) + b
        # Rotation composes additively in angle -> prefix sum.
        states = self.rotary(states, torch.cumsum(angles, dim=1))
        return self.norm(states)

    def step(self, carry: tuple, e: torch.Tensor) -> tuple:
        """
        Advance one token, scan-consistent.

        `carry` is (S_raw, angle_acc): the *pre-rotation, pre-norm* state and
        the accumulated rotation angle. Keeping the raw state is what makes
        this match the scan -- rotation and normalization are applied to a
        snapshot for readout, never fed back into the recurrence.

        Returns (new_carry, S_out) where S_out is the normalized readout state.
        """
        S_raw, angle_acc = carry
        scale, shift, gate, angles = self.transform_net(e)
        a, b = self._affine(scale, shift, gate)

        S_raw = a * S_raw + b
        angle_acc = angle_acc + angles

        S_out = self.norm(self.rotary(S_raw, angle_acc))
        return (S_raw, angle_acc), S_out

    def init_carry(self, S0: torch.Tensor, batch: int) -> tuple:
        """Fresh carry for `step`: raw state at S0, zero accumulated angle."""
        S_raw = S0.unsqueeze(0).expand(batch, -1).contiguous()
        angle_acc = torch.zeros(batch, self.rotary.n_pairs,
                                device=S0.device, dtype=S0.dtype)
        return (S_raw, angle_acc)


class GSM(nn.Module):
    """
    Geometric State Machine.
    S ∈ R^N fixed-size manifold point, resolved by parallel associative scan.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 512,
                 state_dim: int = 4096, n_pairs: int = 128,
                 hidden_dim: int = 1024, n_layers: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_drop = nn.Dropout(dropout)
        self.S0 = nn.Parameter(torch.randn(state_dim) * 0.01)
        self.step = GeometricStateStep(embed_dim, state_dim, n_pairs, hidden_dim, n_layers)

        # Deeper decoder too
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.SiLU(),
            nn.Linear(state_dim // 2, state_dim // 4),
            nn.SiLU(),
            nn.Linear(state_dim // 4, vocab_size),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        E = self.embed_drop(self.embedding(x))
        states = self.step.scan(E, self.S0)
        return self.decoder(states.view(batch * seq_len, self.state_dim)).view(batch, seq_len, -1)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 256,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        self.eval()
        batch = prompt.shape[0]

        # Ingest the prompt with one scan, then hand off to the incremental
        # step. The scan does not expose the raw carry, so replay the prompt's
        # affine composition directly -- same operators, same order.
        carry = self.step.init_carry(self.S0, batch)
        E = self.embedding(prompt)
        for t in range(prompt.shape[1]):
            carry, S_out = self.step.step(carry, E[:, t, :])

        generated = prompt.clone()
        for _ in range(max_new_tokens):
            logits = self.decoder(S_out) / temperature
            if top_k > 0:
                top_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_vals[:, -1:]] = float('-inf')
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            e = self.embedding(next_token).squeeze(1)
            carry, S_out = self.step.step(carry, e)

        return generated

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
