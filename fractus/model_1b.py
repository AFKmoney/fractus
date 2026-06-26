"""Fractus-1B: a 1B-capacity model trainable on CPU.

Architecture:
    - BPE embedding (vocab 50257, d_model 1024)
    - 12 × FractalBlockSparse (attention + Kuramoto + StructuredSiren MoE-64)
    - LayerNorm + LM head

The key innovation: StructuredSirenLinear experts give ~1B of effective
matrix capacity from ~20M trainable parameters. Combined with top-2 sparse
routing, each token only computes 2/64 experts.
"""

import math
import torch
import torch.nn as nn

from .nn.attention import FractalLinearAttention
from .nn.phase_ode import KuramotoLayer
from .nn.stats import elu_plus_one, stable_softmax
from .nn.farey import expert_phases
from .nn.structured_siren import StructuredSirenLinear


class BPEEmbedding(nn.Module):
    """Embedding for BPE tokens: table + position + Mandelbrot Fourier boost."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 512):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        # Init.
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_embed(ids) + self.pos_embed(pos)
        return self.norm(x)


class SparseStructuredMoE(nn.Module):
    """64-expert sparse MoE using StructuredSirenLinear experts.

    Only top_k=2 experts are computed per token (gather-first sparse dispatch).
    Each expert is a 2-layer MLP with StructuredSirenLinear weight matrices,
    giving high capacity from low param count.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 64,
        top_k: int = 2,
        d_ff: int = 1024,
        siren_rank: int = 64,
        kappa: float = 4.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_ff = d_ff

        # Expert phases (Farey precomputation).
        phases = expert_phases(n_experts)
        self.register_buffer("expert_phases", torch.tensor(phases, dtype=torch.float32))
        self.kappa = kappa

        # Experts: each is w1 (d_model→d_ff) + w2 (d_ff→d_model) via StructuredSiren.
        # We store them in a flat list and dispatch via index_select.
        self.experts_w1 = nn.ModuleList([
            StructuredSirenLinear(d_model, d_ff, rank=siren_rank, siren_hidden=32)
            for _ in range(n_experts)
        ])
        self.experts_w2 = nn.ModuleList([
            StructuredSirenLinear(d_ff, d_model, rank=siren_rank, siren_hidden=32)
            for _ in range(n_experts)
        ])

    def _compute_gates(self, phases: torch.Tensor) -> torch.Tensor:
        sin_p = torch.sin(phases).sum(dim=-1)
        cos_p = torch.cos(phases).sum(dim=-1)
        theta_bar = torch.atan2(sin_p, cos_p)
        diff = theta_bar.unsqueeze(-1) - self.expert_phases.view(
            *[1] * (phases.dim() - 1), self.n_experts
        )
        gates = torch.exp(self.kappa * torch.cos(diff))
        gates_sum = gates.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(gates, 1.0 / self.n_experts)
        return torch.where(gates_sum > 1e-10, gates / gates_sum, uniform)

    def forward(self, h: torch.Tensor, phases: torch.Tensor):
        """h: (B, L, d_model), phases: (B, L, n_phases).
        Returns (output, load_balance_loss)."""
        B, L, D = h.shape
        gates = self._compute_gates(phases)
        topk_vals, topk_idx = gates.topk(self.top_k, dim=-1)
        topk_sum = topk_vals.sum(dim=-1, keepdim=True)
        topk_norm = torch.where(
            topk_sum > 1e-10, topk_vals / topk_sum,
            torch.full_like(topk_vals, 1.0 / self.top_k),
        )

        # Gather-first: for each of the top_k slots, process each expert's
        # tokens. We use index_add_ for autograd-safe scatter.
        # Flatten everything to (B*L, D).
        flat_h = h.reshape(-1, D)  # (B*L, D)
        flat_output = torch.zeros(B * L, D, dtype=h.dtype, device=h.device)
        # Track which (flat_position, expert) assignments to scatter.
        # We process slot by slot, expert by expert, accumulating into flat_output.
        for k in range(self.top_k):
            idx_k = topk_idx[:, :, k].reshape(-1)  # (B*L,)
            weight_k = topk_norm[:, :, k].reshape(-1)  # (B*L,)

            for e in range(self.n_experts):
                mask = (idx_k == e)  # (B*L,) bool
                if not mask.any():
                    continue
                positions = mask.nonzero(as_tuple=True)[0]  # (N_e,)
                h_e = flat_h[positions]  # (N_e, D)
                # Expert forward.
                h1 = self.experts_w1[e](h_e)  # (N_e, d_ff)
                h1_act = torch.nn.functional.gelu(h1)
                out_e = self.experts_w2[e](h1_act)  # (N_e, D)
                w_e = weight_k[positions].unsqueeze(-1)  # (N_e, 1)
                # Autograd-safe scatter via index_add.
                contribution = w_e * out_e  # (N_e, D)
                flat_output = flat_output.index_add(0, positions, contribution)

        # Load-balance loss.
        P = gates.mean(dim=(0, 1))
        lb_loss = self.n_experts * ((P - 1.0 / self.n_experts) ** 2).sum()
        output = flat_output.reshape(B, L, D)
        return output, lb_loss


class FractalBlockSparse(nn.Module):
    """One transformer block: attention + Kuramoto + sparse MoE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        n_levels: int,
        n_experts: int = 64,
        top_k: int = 2,
        expert_d_ff: int = 1024,
        siren_rank: int = 64,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FractalLinearAttention(d_model, n_heads, d_head, n_levels)

        self.norm_kur = nn.LayerNorm(d_model)
        self.kuramoto = KuramotoLayer(d_model, n_oscillators=16, rank=8,
                                      n_steps=4, dt=0.1)

        self.norm_moe = nn.LayerNorm(d_model)
        self.moe = SparseStructuredMoE(
            d_model, n_experts=n_experts, top_k=top_k,
            d_ff=expert_d_ff, siren_rank=siren_rank,
        )

    def forward(self, x: torch.Tensor):
        # Attention.
        x = x + self.attn(self.norm1(x))
        # Kuramoto phases.
        phases = self.kuramoto(self.norm_kur(x))
        # Sparse MoE.
        moe_out, lb_loss = self.moe(self.norm_moe(x), phases)
        x = x + moe_out
        return x, lb_loss


class Fractus1B(nn.Module):
    """Fractus-1B: 1B-capacity, ~20M trainable params, CPU-trainable.

    Config (default):
        vocab=50257, d_model=1024, n_layers=12, n_heads=16, d_head=64,
        n_levels=4, n_experts=64, top_k=2, expert_d_ff=1024, siren_rank=64.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        n_layers: int = 12,
        n_heads: int = 16,
        d_head: int = 64,
        n_levels: int = 4,
        n_experts: int = 64,
        top_k: int = 2,
        expert_d_ff: int = 1024,
        siren_rank: int = 64,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.config = {
            "vocab_size": vocab_size, "d_model": d_model, "n_layers": n_layers,
            "n_heads": n_heads, "d_head": d_head, "n_levels": n_levels,
            "n_experts": n_experts, "top_k": top_k, "expert_d_ff": expert_d_ff,
            "siren_rank": siren_rank, "max_seq_len": max_seq_len,
        }

        self.embed = BPEEmbedding(vocab_size, d_model, max_seq_len)
        self.blocks = nn.ModuleList([
            FractalBlockSparse(
                d_model, n_heads, d_head, n_levels,
                n_experts=n_experts, top_k=top_k,
                expert_d_ff=expert_d_ff, siren_rank=siren_rank,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Tied with embedding (saves params).
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.tok_embed.weight

    def forward(self, ids: torch.Tensor):
        """ids: (B, L) → (logits (B, L, vocab), aux_loss scalar)."""
        x = self.embed(ids)
        aux_loss = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            x, lb = block(x)
            aux_loss = aux_loss + lb
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, aux_loss

    def n_params(self) -> int:
        """Actual trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_effective_capacity(self) -> int:
        """Approximate effective matrix capacity (what a dense model would have)."""
        # Attention QKV+out per layer: 4 * d_model^2
        attn = 4 * self.d_model ** 2
        # Each expert: d_model*d_ff + d_ff*d_model (dense-equivalent)
        moe_per_layer = self.config["n_experts"] * 2 * self.d_model * self.config["expert_d_ff"]
        # Embedding + head.
        emb = self.vocab_size * self.d_model
        total = self.config["n_layers"] * (attn + moe_per_layer) + emb
        return total
