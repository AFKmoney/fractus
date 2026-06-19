"""FractalBlock : bloc transformer fractal minimal (L2a).

Architecture (L2a, sans Kuramoto/MoE — ceux-ci viennent en L2b) :

    x → LayerNorm → FractalLinearAttention → Dropout → + x (résiduelle)

C'est le pré-bloc : on aura un transformer fonctionnel après L2a. En L2b on
étendra ce bloc pour intégrer PhaseSoliton, KuramotoODE et PhaseRoutedMoE.

La connexion résiduelle (output = x + attn(LN(x))) garantit la stabilité et
permet l'empilement de plusieurs blocs.
"""

import torch
import torch.nn as nn

from .attention import FractalLinearAttention


class FractalBlock(nn.Module):
    """Bloc transformer fractal minimal (L2a).

    Args:
        d_model  : dimension du modèle.
        n_heads  : nombre de têtes d'attention.
        d_head   : dimension par tête (n_heads·d_head == d_model requis).
        n_levels : niveaux fractals de l'attention.
        dropout  : taux de dropout (0 par défaut en L2a, on ajoutera en L7).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        n_levels: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = FractalLinearAttention(d_model, n_heads, d_head, n_levels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, d_model) → (B, L, d_model).

        Connexion résiduelle : out = x + dropout(attn(norm(x))).
        """
        return x + self.dropout(self.attn(self.norm(x)))
