"""FractalLinearAttention : attention linéaire causale multi-niveaux.

Portée fidèlement depuis FNN v5.0 (src/attention.rs) en PyTorch pur.

Mathématique (Katharopoulos 2020, forme causale normalisée) :

    Feature map : φ(x; level) = elu_plus_one(x + ω_level, α=1)
        avec ω_level = (φ²)^{-level}, φ² = ((1+√5)/2)² ≈ 2.618
        (offset Mandelbrot-décroissant, renommé honnêtement).

    Récurrence causale (INCLUSIVE : à l'instant t, S et z sont mis à jour
    avec k_t, v_t AVANT de calculer y_t) :
        S_t = Σ_{i≤t} φ(k_i) ⊗ v_i   ∈ R^{d_head × d_head}
        z_t = Σ_{i≤t} φ(k_i)          ∈ R^{d_head}
        y_t = (φ(q_t)ᵀ S_t) / (φ(q_t)ᵀ z_t)
        (sortie 0 si |dénom| < 1e-10)

    Agrégation multi-niveaux : output = Σ_level w_level · attn_level(x)
        avec w = softmax(level_logits) (init uniforme 1/n_levels).

Complexité : O(L · d_head²) par tête par niveau, vs O(L² · d_head) pour
l'attention softmax classique. C'est l'intérêt.

Différentiable de bout en bout (tous les poids sont des nn.Parameter).
"""

import math
import torch
import torch.nn as nn

from .stats import elu_plus_one, stable_softmax


def _mandelbrot_offsets(n_levels: int) -> torch.Tensor:
    """Offsets ω_level = (φ²)^{-level} pour level = 0..n_levels-1.

    Décroissance géométrique dorée. Renommée honnête : FNN appelait ça
    "Mandelbrot frequencies" mais il n'y a pas d'itération de Mandelbrot —
    juste une suite géométrique de base φ².
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    phi_sq = phi * phi  # ≈ 2.618
    levels = torch.arange(n_levels, dtype=torch.float32)
    return phi_sq ** (-levels)


class FractalLinearAttention(nn.Module):
    """Attention linéaire causale multi-niveaux.

    Args:
        d_model  : dimension du modèle (entrée/sortie).
        n_heads  : nombre de têtes d'attention.
        d_head   : dimension par tête. Doit satisfaire n_heads · d_head == d_model.
        n_levels : nombre de niveaux fractals (offsets Mandelbrot distincts).
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, n_levels: int = 3):
        super().__init__()
        if n_heads * d_head != d_model:
            raise ValueError(
                f"Contrainte non respectée : n_heads·d_head ({n_heads*d_head}) "
                f"≠ d_model ({d_model})"
            )
        if n_levels < 1:
            raise ValueError("n_levels doit être >= 1")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_levels = n_levels
        d_qkv = n_heads * d_head  # = d_model

        # Poids Q, K, V concaténés (comme FNN attention.rs:30-32).
        # Init Xavier uniforme : U(-scale, scale), scale = sqrt(2/(fan_in+fan_out)).
        scale = math.sqrt(2.0 / (d_model + d_qkv))
        self.w_qkv = nn.Parameter(
            torch.empty(3, d_model, d_qkv).uniform_(-scale, scale)
        )
        self.b_qkv = nn.Parameter(torch.zeros(3, d_qkv))

        # Projection de sortie.
        scale_out = math.sqrt(2.0 / (d_qkv + d_model))
        self.w_out = nn.Parameter(
            torch.empty(d_qkv, d_model).uniform_(-scale_out, scale_out)
        )
        self.b_out = nn.Parameter(torch.zeros(d_model))

        # Poids par niveau (softmax → init uniforme 1/n_levels).
        self.level_logits = nn.Parameter(torch.zeros(n_levels))

        # Offsets Mandelbrot par niveau (précalculés, hors-graphe car constants).
        offsets = _mandelbrot_offsets(n_levels)
        self.register_buffer("level_offsets", offsets)

    def feature_map(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """φ(x; level) = elu_plus_one(x + ω_level).

        x : (..., d_head). L'offset ω_level est un scalaire ajouté à tout x.
        """
        offset = self.level_offsets[level] if level < self.n_levels else 0.0
        return elu_plus_one(x + offset, alpha=1.0)

    def _linear_attention_causal_one_head(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Récurrence causale pour UNE tête, sur un batch.

        q, k : (B, L, d_head) — déjà φ-mappés (feature map appliquée).
        v    : (B, L, d_head) — brut (pas de feature map sur v).
        Retourne y : (B, L, d_head).

        Math :
            S_t = Σ_{i≤t} k_i ⊗ v_i   (B, d_head, d_head)
            z_t = Σ_{i≤t} k_i          (B, d_head)
            y_t = (q_t · S_t) / (q_t · z_t)
        """
        B, L, D = q.shape
        # On accumule la running sum S (B, D, D) et z (B, D).
        S = torch.zeros(B, D, D, dtype=q.dtype, device=q.device)
        z = torch.zeros(B, D, dtype=q.dtype, device=q.device)
        outputs = []
        for t in range(L):
            kt = k[:, t, :]  # (B, D)
            vt = v[:, t, :]  # (B, D)
            # Mise à jour INCLUSIVE avant calcul de y_t (causalité stricte
            # incluant le présent token, comme FNN attention.rs:173-208).
            S = S + kt.unsqueeze(2) * vt.unsqueeze(1)  # outer product (B, D, D)
            z = z + kt  # (B, D)
            qt = q[:, t, :]  # (B, D)
            num = torch.bmm(qt.unsqueeze(1), S).squeeze(1)  # (B, D)
            denom = (qt * z).sum(dim=1, keepdim=True)  # (B, 1)
            # Sortie 0 si |dénom| < 1e-10 (comportement aux limites de FNN).
            safe = denom.abs() > 1e-10
            y_t = torch.where(safe, num / (denom + 1e-20), torch.zeros_like(num))
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)  # (B, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, d_model) → sortie (B, L, d_model)."""
        B, L, _ = x.shape
        # Projeter Q, K, V (einsum direct, différentiable).
        q_all = torch.einsum("bld,de->ble", x, self.w_qkv[0]) + self.b_qkv[0]
        k_all = torch.einsum("bld,de->ble", x, self.w_qkv[1]) + self.b_qkv[1]
        v_all = torch.einsum("bld,de->ble", x, self.w_qkv[2]) + self.b_qkv[2]
        # (B, L, d_qkv) chacun

        level_weights = stable_softmax(self.level_logits, dim=-1)  # (n_levels,)

        output = torch.zeros(B, L, self.d_model, dtype=x.dtype, device=x.device)
        for level in range(self.n_levels):
            # Reshape en têtes : (B, L, n_heads, d_head) → (B, n_heads, L, d_head)
            q = q_all.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            k = k_all.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            v = v_all.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            # Appliquer feature map à q et k (PAS à v).
            q = self.feature_map(q, level)
            k = self.feature_map(k, level)

            # Attention par tête (boucle ; petit n_heads donc OK).
            head_outputs = []
            for h in range(self.n_heads):
                yh = self._linear_attention_causal_one_head(
                    q[:, h], k[:, h], v[:, h]
                )  # (B, L, d_head)
                head_outputs.append(yh)
            # Concaténer têtes : (B, L, n_heads·d_head) = (B, L, d_qkv)
            attn = torch.cat(head_outputs, dim=-1)

            # Projection de sortie + ajout pondéré.
            projected = attn @ self.w_out + self.b_out  # (B, L, d_model)
            output = output + level_weights[level] * projected

        return output
