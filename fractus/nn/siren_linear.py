"""SirenLinear : couche nn.Linear-like dont the matrix of poids est produite
par a SIREN.

CORRECTION vs the original : in the original, the matrix decompressee W was computationee then
JETEE (training_loop.py:30-37 appliquait mirror a W then tournait on l'entree
brute). Ici, the SIREN EST the matrix : on evalue the SIREN a each forward for
obtenir W, then on does y = x @ W + b. Tout est in the graphe autodiff.

Usage : remplacer certaines nn.Linear by SirenLinear for compresser leurs
poids via SIREN. Le trade-off : less of params (compression) but a forward
plus cher (evaluation SIREN a each appel) and a expressivite potentiellement
reduite (les poids SIREN are lisses, not denses — voir demo L3).
"""

import torch
import torch.nn as nn

from .siren import TorusSirenWeight


class SirenLinear(nn.Module):
    """Couche lineaire dont the matrix W = SIREN(grid).

    Args:
        in_features, out_features : dimensions (comme nn.Linear).
        hidden : width of the SIREN which produit W.
        omega0 : frequence SIREN.
        bias   : si True, ajoute a biais entrainable.
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
        # La matrix of poids vient d'une SIREN evaluee on a grille
        # (in_features, out_features).
        self.siren = TorusSirenWeight(
            out_h=in_features, out_w=out_features, hidden=hidden, omega0=omega0
        )
        # Biais entrainable separe (pas compresse — this is a vector, not a matrix).
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (..., in_features) → (..., out_features).

        W = self.siren() : (in_features, out_features), in the graphe autodiff.
        y = x @ W + bias.
        """
        W = self.siren()  # (in_features, out_features), differentiable
        y = x @ W
        if self.bias is not None:
            y = y + self.bias
        return y
