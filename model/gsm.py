"""
GSM — Geometric State Machine
==============================
Fixed-geometry state manifold architecture.
Each token is a transformation operator acting on a point in R^N.
O(1) per token: fixed compute, fixed state size, scales to any corpus.

v2 improvements:
- Deeper TransformNet (6 layers instead of 3)
- Larger default state_dim (4096)
- Larger default embed_dim (512)
- Residual connections in TransformNet for gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotarySubspaceTransform(nn.Module):
    """
    Vectorized subspace rotations. All pairs computed in parallel.
    """
    def __init__(self, state_dim: int, n_pairs: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_pairs = n_pairs
        idx = torch.randperm(state_dim)[:n_pairs * 2].reshape(n_pairs, 2)
        self.register_buffer("idx_a", idx[:, 0])
        self.register_buffer("idx_b", idx[:, 1])

    def forward(self, S: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        cos_t = torch.cos(angles)
        sin_t = torch.sin(angles)
        a = S[:, self.idx_a]
        b = S[:, self.idx_b]
        new_a = cos_t * a - sin_t * b
        new_b = sin_t * a + cos_t * b
        S = S.clone()
        S.scatter_(1, self.idx_a.unsqueeze(0).expand_as(new_a), new_a)
        S.scatter_(1, self.idx_b.unsqueeze(0).expand_as(new_b), new_b)
        return S


class TransformNet(nn.Module):
    """
    Maps token embeddings to transformation parameters.
    6-layer deep MLP with residual connections for gradient flow.
    Produces: scale, shift, gate (state_dim each), angles (n_pairs).
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
        scale  = torch.sigmoid(out[:, :self.state_dim]) * 2.0
        shift  = out[:, self.state_dim:self.state_dim * 2] * 0.1
        gate   = torch.sigmoid(out[:, self.state_dim * 2:self.state_dim * 3])
        angles = out[:, self.state_dim * 3:] * 0.1
        return scale, shift, gate, angles


class GeometricStateStep(nn.Module):
    """
    Single O(1) step: S' = gate * Rotate(scale * S + shift) + (1 - gate) * S
    """
    def __init__(self, embed_dim: int, state_dim: int, n_pairs: int, hidden_dim: int = 1024, n_layers: int = 6):
        super().__init__()
        self.transform_net = TransformNet(embed_dim, state_dim, n_pairs, hidden_dim, n_layers)
        self.rotary = RotarySubspaceTransform(state_dim, n_pairs)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, S: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        scale, shift, gate, angles = self.transform_net(e)
        S_t = scale * S + shift
        S_t = self.rotary(S_t, angles)
        S_new = gate * S_t + (1.0 - gate) * S
        return self.norm(S_new)


class GSM(nn.Module):
    """
    Geometric State Machine.
    S ∈ R^N fixed-size manifold point, updated O(1) per token.
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
        S = self.S0.unsqueeze(0).expand(batch, -1).clone()
        E = self.embed_drop(self.embedding(x))

        logits_list = []
        for t in range(seq_len):
            S = self.step(S, E[:, t, :])
            logits_list.append(self.decoder(S))

        return torch.stack(logits_list, dim=1)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 256,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        self.eval()
        batch = prompt.shape[0]
        S = self.S0.unsqueeze(0).expand(batch, -1).clone()

        E = self.embedding(prompt)
        for t in range(prompt.shape[1]):
            S = self.step(S, E[:, t, :])

        generated = prompt.clone()
        for _ in range(max_new_tokens):
            e = self.embedding(generated[:, -1:]).squeeze(1)
            S = self.step(S, e)
            logits = self.decoder(S) / temperature
            if top_k > 0:
                top_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_vals[:, -1:]] = float('-inf')
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
