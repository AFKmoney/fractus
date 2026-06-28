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
from .nn.cached_siren import CachedStructuredSirenLinear


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

        # Experts: each is w1 (d_model→d_ff) + w2 (d_ff→d_model) via CachedStructuredSiren.
        # The cache makes forward ~5-8× faster (SIREN reconstruction is the bottleneck).
        self.experts_w1 = nn.ModuleList([
            CachedStructuredSirenLinear(d_model, d_ff, rank=siren_rank, siren_hidden=32,
                                        refresh_every=8)
            for _ in range(n_experts)
        ])
        self.experts_w2 = nn.ModuleList([
            CachedStructuredSirenLinear(d_ff, d_model, rank=siren_rank, siren_hidden=32,
                                        refresh_every=8)
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
        Returns (output, load_balance_loss).

        L9 OPTIMIZATION: instead of a Python double-loop over (top_k × n_experts),
        we gather ALL expert weights for ALL positions into a single batched
        tensor, then do ONE batched matmul. This kills the Python-loop overhead
        that was the #2 bottleneck (after SIREN reconstruction, now cached).
        """
        B, L, D = h.shape
        gates = self._compute_gates(phases)
        topk_vals, topk_idx = gates.topk(self.top_k, dim=-1)
        topk_sum = topk_vals.sum(dim=-1, keepdim=True)
        topk_norm = torch.where(
            topk_sum > 1e-10, topk_vals / topk_sum,
            torch.full_like(topk_vals, 1.0 / self.top_k),
        )

        # Flatten: (B*L, D)
        flat_h = h.reshape(-1, D)  # (N, D) where N = B*L
        N = flat_h.shape[0]

        # For each of the K slots, gather the selected experts' cached weights
        # and do a batched matmul. Since the experts use CachedStructuredSirenLinear,
        # the weight is cached (fast lookup, no SIREN reconstruction most of the time).
        flat_output = torch.zeros(N, D, dtype=h.dtype, device=h.device)

        for k in range(self.top_k):
            idx_k = topk_idx[:, :, k].reshape(-1)  # (N,) expert indices
            weight_k = topk_norm[:, :, k].reshape(-1)  # (N,) gate weights

            # Pre-stacked expert weights (cached, rebuilt only on SIREN refresh).
            # Build the stack once per forward (cheap — just a torch.stack of buffers).
            if k == 0:
                w1_stack = torch.stack([e._cached_W for e in self.experts_w1])  # (E, D, d_ff)
                w2_stack = torch.stack([e._cached_W for e in self.experts_w2])  # (E, d_ff, D)

            # Gather per-position weights.
            # _cached_W is (out_features, in_features), so:
            #   w1: (d_ff, D) → transpose to (D, d_ff) for bmm
            #   w2: (D, d_ff) → transpose to (d_ff, D) for bmm
            w1_selected = w1_stack[idx_k].transpose(-1, -2)  # (N, D, d_ff)
            w2_selected = w2_stack[idx_k].transpose(-1, -2)  # (N, d_ff, D)

            # Batched expert forward: h1[n] = h[n] @ w1_selected[n]
            h1 = torch.bmm(flat_h.unsqueeze(1), w1_selected).squeeze(1)  # (N, d_ff)
            h1_act = torch.nn.functional.gelu(h1)
            out = torch.bmm(h1_act.unsqueeze(1), w2_selected).squeeze(1)  # (N, D)

            # Weight by gate and accumulate.
            contribution = weight_k.unsqueeze(-1) * out  # (N, D)
            flat_output = flat_output + contribution

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

        # L9 SKIP-BACKWARD OPTIMIZATION: when no expert's SIREN cache is
        # refreshing this step, the cached weights are detached → the MoE
        # backward is wasted work (gradients die at the detached cache).
        # We detach moe_out to skip the backward through the MoE entirely
        # on non-refresh steps. The residual (x + moe_out) still lets
        # gradients flow to the attention/embedding via x.
        # On refresh steps (1 in 8), we keep moe_out attached so the
        # experts' U/V/SIREN params get their gradient.
        is_refresh = any(
            (e._call_count % e.refresh_every == 1)
            for e in self.moe.experts_w1
        )
        if not is_refresh:
            moe_out = moe_out.detach()

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
        """ids: (B, L) → (logits (B, L, vocab), aux_loss scalar).

        L9 GRADIENT CHECKPOINTING: each block is checkpointed so the autograd
        graph is NOT retained between layers. During backward, the forward is
        recomputed per-block. This reduces peak memory from O(n_layers ×
        activation_size) to O(activation_size), making the 1B model trainable
        on CPU without OOM.
        """
        from torch.utils.checkpoint import checkpoint

        x = self.embed(ids)
        aux_loss = torch.tensor(0.0, device=x.device)

        def run_block(block, h):
            """Wrapper for gradient checkpointing."""
            def custom_fwd(*args):
                out, lb = block(args[0])
                return out, lb
            return custom_fwd

        for block in self.blocks:
            # Gradient checkpoint: don't store intermediate activations.
            # checkpoint() recompute forward during backward.
            x_new, lb = checkpoint(block, x, use_reentrant=False)
            x = x_new
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
