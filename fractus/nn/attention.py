"""FractalLinearAttention: causal, multi-level linear attention.

Faithfully ported from the original system (src/attention.rs) in pure PyTorch.

Math (Katharopoulos 2020, normalized causal form):
    Feature map: phi(x; level) = elu_plus_one(x + omega_level, alpha=1)
        with omega_level = (phi^2)^{-level}, phi^2 = ((1+sqrt(5)/2)^2 ~= 2.618
    Causal recurrence (INCLUSIVE: at step t, S and z are updated before computing y_t).
    Multi-level aggregation: output = sum of softmax(level_logits) * attn_level(x).
    Complexity: O(L * d_head^2) per head per level.
    End-to-end differentiable.
"""

import math
import torch
import torch.nn as nn

from .stats import elu_plus_one, stable_softmax


def _mandelbrot_offsets(n_levels: int) -> torch.Tensor:
    """Offsets ω_level = (φ2)^{-level} for level = 0..n_levels-1.

    Geometric decay. Renamed honestly: the original called these
    "Mandelbrot frequencies" but there is no Mandelbrot iteration —
    just a geometric sequence of base φ2.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    phi_sq = phi * phi  # ≈ 2.618
    levels = torch.arange(n_levels, dtype=torch.float32)
    return phi_sq ** (-levels)


class FractalLinearAttention(nn.Module):
    """Multi-level causal linear attention.

    Args:
        d_model  : model dimension (input/output).
        n_heads  : number of attention heads.
        d_head   : dimension per head. Must satisfy n_heads · d_head == d_model.
        n_levels : number of fractal levels (distinct Mandelbrot offsets).
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, n_levels: int = 3):
        super().__init__()
        if n_heads * d_head != d_model:
            raise ValueError(
                f"Constraint not satisfied: n_heads·d_head ({n_heads*d_head}) "
                f"≠ d_model ({d_model})"
            )
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_levels = n_levels
        d_qkv = n_heads * d_head  # = d_model

        # Concatenated Q, K, V weights (as in the original attention.rs:30-32).
        # Glorot-style init: U(-scale, scale), scale = sqrt(2/(fan_in+fan_out)).
        # (Note: the true Xavier/Glorot uniform is sqrt(6/(fan_in+fan_out));
        #  the original used sqrt(2/(...)), we keep it for fidelity.)
        scale = math.sqrt(2.0 / (d_model + d_qkv))
        self.w_qkv = nn.Parameter(
            torch.empty(3, d_model, d_qkv).uniform_(-scale, scale)
        )
        self.b_qkv = nn.Parameter(torch.zeros(3, d_qkv))

        # Output projection (same init style as Q/K/V).
        scale_out = math.sqrt(2.0 / (d_qkv + d_model))
        self.w_out = nn.Parameter(
            torch.empty(d_qkv, d_model).uniform_(-scale_out, scale_out)
        )
        self.b_out = nn.Parameter(torch.zeros(d_model))

        # Per-level weights (softmax → uniform init 1/n_levels).
        self.level_logits = nn.Parameter(torch.zeros(n_levels))

        # Per-level Mandelbrot offsets (precomputed, off-graph because they are constants).
        offsets = _mandelbrot_offsets(n_levels)
        self.register_buffer("level_offsets", offsets)

    def feature_map(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """φ(x; level) = elu_plus_one(x + ω_level).

        x: (..., d_head). The offset ω_level is a scalar added to all of x.
        """
        # Invariant: level is always in [0, n_levels) (comes from forward).
        assert 0 <= level < self.n_levels, f"level {level} outside [0, {self.n_levels})"
        offset = self.level_offsets[level]
        return elu_plus_one(x + offset, alpha=1.0)

    def _linear_attention_causal_one_head(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Causal recurrence for ONE head, over a batch.

        q, k : (B, L, d_head) — already φ-mapped (feature map applied).
        v    : (B, L, d_head) — raw (no feature map on v).
        Returns y: (B, L, d_head).

        Math:
            S_t = Σ_{i≤t} k_i ⊗ v_i   (B, d_head, d_head)
            z_t = Σ_{i≤t} k_i          (B, d_head)
            y_t = (q_t · S_t) / (q_t · z_t)
        """
        B, L, D = q.shape
        # We accumulate the running sum S (B, D, D) and z (B, D).
        S = torch.zeros(B, D, D, dtype=q.dtype, device=q.device)
        z = torch.zeros(B, D, dtype=q.dtype, device=q.device)
        outputs = []
        for t in range(L):
            kt = k[:, t, :]  # (B, D)
            vt = v[:, t, :]  # (B, D)
            # INCLUSIVE update before computing y_t (strict causality
            # including the current token, as in the original attention.rs:173-208).
            S = S + kt.unsqueeze(2) * vt.unsqueeze(1)  # outer product (B, D, D)
            z = z + kt  # (B, D)
            qt = q[:, t, :]  # (B, D)
            num = torch.bmm(qt.unsqueeze(1), S).squeeze(1)  # (B, D)
            denom = (qt * z).sum(dim=1, keepdim=True)  # (B, 1)
            # Output 0 if |denom| < 1e-10 (limit behavior of the original).
            safe = denom.abs() > 1e-10
            y_t = torch.where(safe, num / (denom + 1e-20), torch.zeros_like(num))
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)  # (B, L, D)

    def _linear_attention_causal_vectorized(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized version of _linear_attention_causal_one_head.

        Same mathematics, but without a Python loop over L. The trick:
        precompute the cumulative sums S_t and z_t via a lower-triangular
        convolution, then compute all y_t in parallel.

        Equivalence guaranteed by test_attention_vectorized.py (atol 1e-5).

        Complexity: O(L · D2) but in parallel tensor operations,
        instead of O(L) sequential Python calls.
        """
        B, L, D = q.shape

        # S_t = Σ_{i≤t} k_i ⊗ v_i  ∈ R^{B, D, D}, where the matrix [p,q] = k[p]·v[q].
        # outer[t] : (B, L, D, D) with outer[b,t,p,q] = k[b,t,p] · v[b,t,q].
        outer = torch.einsum("btp,btq->btpq", k, v)  # (B, L, D, D)
        # Lower-triangular causal mask: mask[t,j] = 1 if j <= t.
        mask = torch.tril(torch.ones(L, L, dtype=q.dtype, device=q.device))
        # S[b,t,p,q] = Σ_{j<=t} outer[b,j,p,q] = Σ_j mask[t,j] · outer[b,j,p,q].
        S = torch.einsum("tj,bjpq->btpq", mask, outer)  # (B, L, D, D)

        # z_t = Σ_{i<=t} k_i ∈ R^{B, L, D}.
        z = torch.einsum("tj,bjp->btp", mask, k)  # (B, L, D)

        # y_t = (q_t · S_t) / (q_t · z_t) for all t.
        # num[b,t,q] = Σ_p q[b,t,p] · S[b,t,p,q].
        num = torch.einsum("btp,btpq->btq", q, S)  # (B, L, D)
        # denom[b,t] = q[b,t,:] · z[b,t,:].
        denom = (q * z).sum(dim=-1, keepdim=True)  # (B, L, 1)
        # Output 0 if |denom| < 1e-10 (limit behavior of the original).
        safe = denom.abs() > 1e-10
        y = torch.where(safe, num / (denom + 1e-20), torch.zeros_like(num))
        return y  # (B, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → output (B, L, d_model)."""
        B, L, _ = x.shape
        # Project Q, K, V (direct einsum, differentiable).
        q_all = torch.einsum("bld,de->ble", x, self.w_qkv[0]) + self.b_qkv[0]
        k_all = torch.einsum("bld,de->ble", x, self.w_qkv[1]) + self.b_qkv[1]
        v_all = torch.einsum("bld,de->ble", x, self.w_qkv[2]) + self.b_qkv[2]
        # (B, L, d_qkv) each

        level_weights = stable_softmax(self.level_logits, dim=-1)  # (n_levels,)

        output = torch.zeros(B, L, self.d_model, dtype=x.dtype, device=x.device)
        for level in range(self.n_levels):
            # Reshape into heads: (B, L, n_heads, d_head) → (B, n_heads, L, d_head)
            q = q_all.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            k = k_all.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            v = v_all.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            # Apply feature map to q and k (NOT to v).
            q = self.feature_map(q, level)
            k = self.feature_map(k, level)

            # Attention per head — vectorized version (17x faster than the loop).
            head_outputs = []
            for h in range(self.n_heads):
                yh = self._linear_attention_causal_vectorized(
                    q[:, h], k[:, h], v[:, h]
                )  # (B, L, d_head)
                head_outputs.append(yh)
            # Concatenate heads: (B, L, n_heads·d_head) = (B, L, d_qkv)
            attn = torch.cat(head_outputs, dim=-1)

            # Output projection + weighted addition.
            projected = attn @ self.w_out + self.b_out  # (B, L, d_model)
            output = output + level_weights[level] * projected

        return output
