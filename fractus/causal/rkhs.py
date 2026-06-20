"""RKHSCausalOperator : operateur causal L: H_X → H_Y in un RKHS.

CORRECTION DU FAUX RKHS D'OMNI :
- OMNI (rkhs_causal.py) n'avait PAS de noyau — juste x @ U @ Vᵀ, une projection
  bas-rang nue. Pas de RKHS, pas de Hilbert, pas de RFF malgre le docstring.
- Ici : VRAI RKHS via Random Fourier Features (Rahimi-Recht 2007).

Math (Rahimi-Recht 2007) :
    Noyau gaussien : k(x, y) = exp(-||x-y||² / (2σ²))
    Approximation : k(x, y) ≈ φ(x) · φ(y)
    ou φ(x) = [cos(ω_1·x), sin(ω_1·x), ..., cos(ω_K·x), sin(ω_K·x)] / √K
    with ω_k ~ N(0, 1/σ²) (features aleatoires, figees une fois tirees).

Operateur causal L in le RKHS :
    features = φ(x)         # (N, 2K), fige
    transformed = features @ (U @ Vᵀ)  # (N, 2K), U,V ∈ R^{2K × rank}
    y = decode(transformed) # (N, d), decode est une Linear entrainable

Les ω_k (W_rff) sont FIGES (non entraines) — c'est la methode Rahimi-Recht.
Seuls U, V, decode sont entraines.
"""

import torch
import torch.nn as nn


class RKHSCausalOperator(nn.Module):
    """Operateur causal in un RKHS approxime par Random Fourier Features.

    Args:
        dim    : dimension d'entree/sortie (espace original).
        rank   : rang de la decomposition bas-rang A = U @ Vᵀ in le RKHS.
        n_rff  : number de features aleatoires K (plus = meilleure approximation).
        sigma  : width de bande du noyau gaussien (1.0 par defaut).
    """

    def __init__(
        self,
        dim: int,
        rank: int = 16,
        n_rff: int = 64,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.n_rff = n_rff
        self.sigma = sigma
        self.feature_dim = 2 * n_rff

        # Features aleatoires RFF : ω_k ~ N(0, 1/σ²). FIGEES.
        W_rff = torch.randn(dim, n_rff) / sigma
        self.register_buffer("W_rff", W_rff)
        b_rff = torch.rand(n_rff) * 2 * 3.141592653589793
        self.register_buffer("b_rff", b_rff)

        # Operateur bas-rang A = U @ Vᵀ in le RKHS. ENTRAINABLE.
        scale = 0.02
        self.U = nn.Parameter(torch.randn(self.feature_dim, rank) * scale)
        self.V = nn.Parameter(torch.randn(self.feature_dim, rank) * scale)

        # Decodeur : ramene de l'espace des features vers dim. ENTRAINABLE.
        self.decode = nn.Linear(self.feature_dim, dim, bias=False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """φ(x) = [cos(ω·x + b), sin(ω·x + b)] / √K. Shape (N, 2K)."""
        proj = x @ self.W_rff + self.b_rff  # (N, K)
        sqrt_K = (self.n_rff ** 0.5)
        cos_part = torch.cos(proj) / sqrt_K
        sin_part = torch.sin(proj) / sqrt_K
        return torch.cat([cos_part, sin_part], dim=-1)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Noyau gaussien approxime : k(x, y) ≈ φ(x) · φ(y). Shape (N_x, N_y)."""
        return self.features(x) @ self.features(y).T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, dim) → y : (N, dim)."""
        phi = self.features(x)
        low_rank = phi @ self.U
        transformed = low_rank @ self.V.T
        y = self.decode(transformed)
        return y
