"""FractalBlock : bloc transformer fractal (L2a minimal + L2b complete).

L2a (FractalBlock minimal) :
    x → LayerNorm → FractalLinearAttention → Dropout → + x (residuelle)

L2b (FractalBlockFull) :
    x → LN → FractalLinearAttention → + x (residuelle 1)
        → LN → KuramotoLayer → phases
        → LN → PhaseRoutedMoE(hidden, phases) → + x (residuelle 2)

La connexion residuelle guaranteedt the stabilite and permet l'empilement.
"""

import torch
import torch.nn as nn

from .attention import FractalLinearAttention
from .phase_ode import KuramotoLayer
from .moe import PhaseRoutedMoE


class FractalBlock(nn.Module):
    """Bloc transformer fractal minimal (L2a).

    Args:
        d_model  : dimension modele.
        n_heads  : number of tetes d'attention.
        d_head   : dimension by tete (n_heads·d_head == d_model requis).
        n_levels : niveaux fractals of l'attention.
        dropout  : taux of dropout (0 by defaut en L2a, on ajoutera en L7).
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

        Connexion residuelle : out = x + dropout(attn(norm(x))).
        """
        return x + self.dropout(self.attn(self.norm(x)))


class FractalBlockFull(nn.Module):
    """Bloc transformer fractal complete (L2b) : integre Kuramoto + MoE.

    Architecture :
        x → LN → FractalLinearAttention → + x (residuelle 1)
              → LN → KuramotoLayer → phases
              → LN → PhaseRoutedMoE(hidden, phases) → + x (residuelle 2)

    Retourne (output, loss_aux) or loss_aux est the load_balance_loss MoE
    (a ajouter a the loss principale by the caller).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        n_levels: int,
        n_oscillators: int,
        coupling_rank: int,
        n_experts: int,
        top_k: int,
        kappa: float = 4.0,
        kuramoto_steps: int = 4,
        kuramoto_dt: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Sous-bloc attention.
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FractalLinearAttention(d_model, n_heads, d_head, n_levels)
        # Kuramoto + MoE.
        self.norm_kur = nn.LayerNorm(d_model)
        self.kuramoto = KuramotoLayer(d_model, n_oscillators, coupling_rank,
                                      n_steps=kuramoto_steps, dt=kuramoto_dt)
        self.norm_moe = nn.LayerNorm(d_model)
        self.moe = PhaseRoutedMoE(d_model, n_experts, top_k, kappa=kappa)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        """x : (B, L, d_model) → (output (B, L, d_model), loss_aux scalar)."""
        # Residuelle 1 : attention.
        x = x + self.dropout(self.attn(self.norm1(x)))
        # Kuramoto : phases depuis hidden normalise.
        phases = self.kuramoto(self.norm_kur(x))  # (B, L, N)
        # MoE : routing by phases.
        moe_out, lb_loss = self.moe(self.norm_moe(x), phases)
        x = x + self.dropout(moe_out)
        return x, lb_loss
