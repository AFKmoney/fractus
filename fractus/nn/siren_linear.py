"""SirenLinear : couche nn.Linear-like dont la matrix de poids est produite
par une SIREN.

CORRECTION vs OMNI : in OMNI, la matrix decompressee W was calculee then
JETEE (training_loop.py:30-37 appliquait mirror a W then tournait sur l'entree
brute). Ici, la SIREN EST la matrix : on evalue la SIREN a each forward for
obtenir W, then on fait y = x @ W + b. Tout est in le graphe autodiff.

Usage : remplacer certaines nn.Linear par SirenLinear for compresser leurs
poids via SIREN. Le trade-off : moins de params (compression) but un forward
plus cher (evaluation SIREN a each appel) et une expressivite potentiellement
reduite (les poids SIREN sont lisses, pas denses — voir demo L3).
"""

import torch
import torch.nn as nn

from .siren import TorusSirenWeight


class SirenLinear(nn.Module):
    """Couche lineaire dont la matrix W = SIREN(grid).

    Args:
        in_features, out_features : dimensions (comme nn.Linear).
        hidden : width de la SIREN qui produit W.
        omega0 : frequence SIREN.
        bias   : si True, ajoute un biais entrainable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: int = 32,
        omega0: float = 30.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # La matrix de poids vient d'une SIREN evaluee sur une grille
        # (in_features, out_features).
        self.siren = TorusSirenWeight(
            out_h=in_features, out_w=out_features, hidden=hidden, omega0=omega0
        )
        # Biais entrainable separe (pas compresse — c'est un vector, pas une matrix).
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (..., in_features) → (..., out_features).

        W = self.siren() : (in_features, out_features), in le graphe autodiff.
        y = x @ W + bias.
        """
        W = self.siren()  # (in_features, out_features), differentiable
        y = x @ W
        if self.bias is not None:
            y = y + self.bias
        return y
