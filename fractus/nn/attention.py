"""FractalLinearAttention: causal, multi-level linear attention.

Faithfully portedd from the original architecture (src/attention.rs) in pure PyTorch.

Math (Katharopoulos 2020, normalized causal form):
    Feature map: phi(x; level) = elu_more_one(x + omega_level, alpha=1)
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

    Decay geometric. Rename honesty : the original called these
    "Mandelbrot frequencies" but il n'y a not d'iteration of Mandelbrot —
    just a sequence geometrique of base φ2.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    phi_sq = phi * phi  # ≈ 2.618
    levels = torch.arange(n_levels, dtype=torch.float32)
    return phi_sq ** (-levels)


class FractalLinearAttention(nn.Module):
    """Attention lineaire causale multi-niveaux.

    Args:
        d_model  : dimension modele (entree/sortie).
        n_heads  : number of tetes d'attention.
        d_head   : dimension per head. Must satisfy n_heads · d_head == d_model.
        n_levels : number of niveaux fractals (offsets Mandelbrot distincts).
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, n_levels: int = 3):
        super().__init__()
        if n_heads * d_head != d_model:
            raise ValueError(
                f"Contrainte non respectee : n_heads·d_head ({n_heads*d_head}) "
                f"≠ d_model ({d_model})"
            )
        if n_levels < 1:
            raise ValueError("n_levels must etre >= 1")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_levels = n_levels
        d_qkv = n_heads * d_head  # = d_model

        # Poids Q, K, V concatenes (comme the original attention.rs:30-32).
        # Init of type Glorot : U(-scale, scale), scale = sqrt(2/(fan_in+fan_out)).
        # (Note : the true Xavier/Glorot uniform is sqrt(6/(fan_in+fan_out)) ;
        #  the original utilisait sqrt(2/(...)), on conserve for fidelite.)
        scale = math.sqrt(2.0 / (d_model + d_qkv))
        self.w_qkv = nn.Parameter(
            torch.empty(3, d_model, d_qkv).uniform_(-scale, scale)
        )
        self.b_qkv = nn.Parameter(torch.zeros(3, d_qkv))

        # Output projection (same style d'init that Q/K/V).
        scale_out = math.sqrt(2.0 / (d_qkv + d_model))
        self.w_out = nn.Parameter(
            torch.empty(d_qkv, d_model).uniform_(-scale_out, scale_out)
        )
        self.b_out = nn.Parameter(torch.zeros(d_model))

        # Poids per level (softmax → init uniforme 1/n_levels).
        self.level_logits = nn.Parameter(torch.zeros(n_levels))

        # Offsets Mandelbrot per level (precomputationes, hors-graphe because constants).
        offsets = _mandelbrot_offsets(n_levels)
        self.register_buffer("level_offsets", offsets)

    def feature_map(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """φ(x; level) = elu_more_one(x + ω_level).

        x : (..., d_head). L'offset ω_level est a scalar ajoute a all x.
        """
        # Invariant : level est always in [0, n_levels) (venu of forward).
        assert 0 <= level < self.n_levels, f"level {level} hors [0, {self.n_levels})"
        offset = self.level_offsets[level]
        return elu_plus_one(x + offset, alpha=1.0)

    def _linear_attention_causal_one_head(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Causal recurrence for UNE tete, on a batch.

        q, k : (B, L, d_head) — already φ-mappes (feature map appliquee).
        v    : (B, L, d_head) — brut (no feature map on v).
        Retourne y : (B, L, d_head).

        Math :
            S_t = Σ_{i≤t} k_i ⊗ v_i   (B, d_head, d_head)
            z_t = Σ_{i≤t} k_i          (B, d_head)
            y_t = (q_t · S_t) / (q_t · z_t)
        """
        B, L, D = q.shape
        # On accumule the running sum S (B, D, D) and z (B, D).
        S = torch.zeros(B, D, D, dtype=q.dtype, device=q.device)
        z = torch.zeros(B, D, dtype=q.dtype, device=q.device)
        outputs = []
        for t in range(L):
            kt = k[:, t, :]  # (B, D)
            vt = v[:, t, :]  # (B, D)
            # Mise a jour INCLUSIVE before computation of y_t (causalite stricte
            # incluant the present token, comme the original attention.rs:173-208).
            S = S + kt.unsqueeze(2) * vt.unsqueeze(1)  # outer product (B, D, D)
            z = z + kt  # (B, D)
            qt = q[:, t, :]  # (B, D)
            num = torch.bmm(qt.unsqueeze(1), S).squeeze(1)  # (B, D)
            denom = (qt * z).sum(dim=1, keepdim=True)  # (B, 1)
            # Sortie 0 si |denom| < 1e-10 (comporteddment aux limites of the original).
            safe = denom.abs() > 1e-10
            y_t = torch.where(safe, num / (denom + 1e-20), torch.zeros_like(num))
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)  # (B, L, D)

    def _linear_attention_causal_vectorized(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Version VECTORISEE of _linear_attention_causal_one_head.

        Meme mathematical, but without boucle Python on L. Astuce :
        precomputationer the sommes cumulees S_t and z_t via a convolution
        triangulaire inferieure, then computationer all the y_t en parallele.

        Equivalence guaranteed by test_attention_vectorized.py (atol 1e-5).

        Complexite : O(L · D2) but en operations tensorielles paralleles,
        instead of O(L) appels Python sequentiels.
        """
        B, L, D = q.shape

        # S_t = Σ_{i≤t} k_i ⊗ v_i  ∈ R^{B, D, D}, or the matrix [p,q] = k[p]·v[q].
        # outer[t] : (B, L, D, D) with outer[b,t,p,q] = k[b,t,p] · v[b,t,q].
        outer = torch.einsum("btp,btq->btpq", k, v)  # (B, L, D, D)
        # Masque causal triangulaire inferieur : mask[t,j] = 1 si j <= t.
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
        # Sortie 0 si |denom| < 1e-10 (comporteddment aux limites the original).
        safe = denom.abs() > 1e-10
        y = torch.where(safe, num / (denom + 1e-20), torch.zeros_like(num))
        return y  # (B, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, d_model) → sortie (B, L, d_model)."""
        B, L, _ = x.shape
        # Projeter Q, K, V (einsum direct, differentiable).
        q_all = torch.einsum("bld,de->ble", x, self.w_qkv[0]) + self.b_qkv[0]
        k_all = torch.einsum("bld,de->ble", x, self.w_qkv[1]) + self.b_qkv[1]
        v_all = torch.einsum("bld,de->ble", x, self.w_qkv[2]) + self.b_qkv[2]
        # (B, L, d_qkv) chacun

        level_weights = stable_softmax(self.level_logits, dim=-1)  # (n_levels,)

        output = torch.zeros(B, L, self.d_model, dtype=x.dtype, device=x.device)
        for level in range(self.n_levels):
            # Reshape en tetes : (B, L, n_heads, d_head) → (B, n_heads, L, d_head)
            q = q_all.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            k = k_all.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            v = v_all.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            # Appliquer feature map a q and k (PAS a v).
            q = self.feature_map(q, level)
            k = self.feature_map(k, level)

            # Attention per head — version vectorisee (17x more rapide that the boucle).
            head_outputs = []
            for h in range(self.n_heads):
                yh = self._linear_attention_causal_vectorized(
                    q[:, h], k[:, h], v[:, h]
                )  # (B, L, d_head)
                head_outputs.append(yh)
            # Concatener tetes : (B, L, n_heads·d_head) = (B, L, d_qkv)
            attn = torch.cat(head_outputs, dim=-1)

            # Output projection + ajout pondere.
            projected = attn @ self.w_out + self.b_out  # (B, L, d_model)
            output = output + level_weights[level] * projected

        return output
