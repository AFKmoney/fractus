"""SirenLinear : couche nn.Linear-like dont la matrice de poids est produite
par une SIREN.

CORRECTION vs OMNI : dans OMNI, la matrice décompressée W était calculée puis
JETÉE (training_loop.py:30-37 appliquait mirror à W puis tournait sur l'entrée
brute). Ici, la SIREN EST la matrice : on évalue la SIREN à chaque forward pour
obtenir W, puis on fait y = x @ W + b. Tout est dans le graphe autodiff.

Usage : remplacer certaines nn.Linear par SirenLinear pour compresser leurs
poids via SIREN. Le trade-off : moins de params (compression) mais un forward
plus cher (évaluation SIREN à chaque appel) et une expressivité potentiellement
réduite (les poids SIREN sont lisses, pas denses — voir démo L3).
"""

import torch
import torch.nn as nn

from .siren import TorusSirenWeight


class SirenLinear(nn.Module):
    """Couche linéaire dont la matrice W = SIREN(grid).

    Args:
        in_features, out_features : dimensions (comme nn.Linear).
        hidden : largeur de la SIREN qui produit W.
        omega0 : fréquence SIREN.
        bias   : si True, ajoute un biais entraînable.
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
        # La matrice de poids vient d'une SIREN évaluée sur une grille
        # (in_features, out_features).
        self.siren = TorusSirenWeight(
            out_h=in_features, out_w=out_features, hidden=hidden, omega0=omega0
        )
        # Biais entraînable séparé (pas compressé — c'est un vecteur, pas une matrice).
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (..., in_features) → (..., out_features).

        W = self.siren() : (in_features, out_features), dans le graphe autodiff.
        y = x @ W + bias.
        """
        W = self.siren()  # (in_features, out_features), différentiable
        y = x @ W
        if self.bias is not None:
            y = y + self.bias
        return y
