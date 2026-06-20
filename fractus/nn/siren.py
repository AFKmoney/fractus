"""TorusSirenWeight : vraie SIREN sin(ω0·) for representer a matrix of poids.

CORRECTION DU MENSONGE D'the original design :
- the original utilisait nn.SiLU (torus_siren.py:15,17) → ici on utilise torch.sin(ω0·(Wx+b)),
  the VRAIE non-linearite SIREN (Sitzmann and al. 2020).
- the original utilisait ω0=56 non justifie → ici ω0=30.0 (value empirique papier SIREN,
  which montre that ω0≈30 est optimal for the representation of functions continues).
- the original commentait "Simple reconstruction: sum of harmonics (real implementation uses
  Fourier)" (torus_siren.py:39) → ici the reconstruction est REELLE (forward SIREN
  on grille 2D).

POSITION SCIENTIFIQUE HONNETE :
Une SIREN represente well functions lisses (images, champs scalars).
Les poids d'un reseau entraine are essentiellement bruit structure dense.
On s'attend therefore a a ratio of compression FAIBLE (~1× a 3×), PAS 20.4×.
The ratio is MEASURED (metrics/compression.py), never hardcoded.

Math (Sitzmann 2020) :
    Non-linearite : sin(ω0 · (Wx + b)) for each couche cachee.
    Couche of sortie : lineaire (pas of sin).
    Init : premiere couche U(-1/ω0, 1/ω0) ; suivantes U(-√(6/(ω02·fan_in)), ...).

La SIREN prend en entree coords (u,v) ∈ [0,1)2 on the tore T2 and produit
un scalar W[u,v]. Evaluee on a grille h×w, elle regenerated the matrix W.
"""

import math
import torch
import torch.nn as nn


class TorusSirenWeight(nn.Module):
    """SIREN which represente a matrix of poids W[out_h, out_w] comme a champ
    scalar on the tore T2 = [0,1)2.

    Args:
        out_h, out_w : dimensions of the matrix a regenerate.
        hidden       : width couches cachees of the SIREN.
        omega0       : frequence fondamentale (30.0 by defaut, Sitzmann 2020).
    """

    def __init__(
        self,
        out_h: int,
        out_w: int,
        hidden: int = 32,
        omega0: float = 30.0,
    ):
        super().__init__()
        if out_h < 1 or out_w < 1 or hidden < 1:
            raise ValueError("out_h, out_w, hidden must etre >= 1")
        self.out_h = out_h
        self.out_w = out_w
        self.hidden = hidden
        self.omega0 = omega0

        # Trois couches (comme SIREN papier for champs scalars).
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self._init_siren_weights()

        # Grille precomputationee (hors-graphe because constant).
        grid = self._build_grid(out_h, out_w)
        self.register_buffer("grid", grid)

    def _init_siren_weights(self):
        """Init SIREN specifique (Sitzmann 2020, section 3.2)."""
        with torch.no_grad():
            # Premiere couche : U(-1/ω0, 1/ω0).
            nn.init.uniform_(self.fc1.weight, -1.0 / self.omega0, 1.0 / self.omega0)
            nn.init.zeros_(self.fc1.bias)
            # Couches suivantes : U(-√(6/(ω02·fan_in)), √(6/(ω02·fan_in))).
            for layer in [self.fc2, self.fc3]:
                fan_in = layer.weight.shape[1]
                bound = math.sqrt(6.0 / (self.omega0 ** 2 * fan_in))
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _build_grid(h: int, w: int) -> torch.Tensor:
        """Grille of coords (u,v) ∈ [0,1)2 on the tore, shape (h·w, 2).

        Sur the tore T2=[0,1)2, the point 1 ≡ 0 (identification). On exclut therefore
        l'extremite 1 for eviter the duplication bord (i/h for i=0..h-1).
        """
        u = torch.arange(h, dtype=torch.float32) / h
        v = torch.arange(w, dtype=torch.float32) / w
        grid = torch.stack(torch.meshgrid(u, v, indexing="ij"), dim=-1)  # (h, w, 2)
        return grid.reshape(-1, 2)  # (h·w, 2)

    def forward(self) -> torch.Tensor:
        """Evalue the SIREN on the grille → matrix W[out_h, out_w].

        This is the 'decompression' : on regenerated W depuis the params SIREN.
        """
        x = self.grid  # (h·w, 2)
        # Couche 1 + sin(ω0·).
        x = torch.sin(self.omega0 * self.fc1(x))
        # Couche 2 + sin(ω0·).
        x = torch.sin(self.omega0 * self.fc2(x))
        # Couche of sortie : lineaire (pas of sin).
        x = self.fc3(x)  # (h·w, 1)
        return x.squeeze(-1).reshape(self.out_h, self.out_w)
