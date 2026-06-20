"""TorusSirenWeight : vraie SIREN sin(ω₀·) for representer une matrix de poids.

CORRECTION DU MENSONGE D'the original design :
- OMNI utilisait nn.SiLU (torus_siren.py:15,17) → ici on utilise torch.sin(ω₀·(Wx+b)),
  la VRAIE non-linearite SIREN (Sitzmann et al. 2020).
- OMNI utilisait ω₀=56 non justifie → ici ω₀=30.0 (valeur empirique du papier SIREN,
  qui montre que ω₀≈30 est optimal for la representation de functions continues).
- OMNI commentait "Simple reconstruction: sum of harmonics (real implementation uses
  Fourier)" (torus_siren.py:39) → ici la reconstruction est REELLE (forward SIREN
  sur grille 2D).

POSITION SCIENTIFIQUE HONNETE :
Une SIREN represente bien des functions lisses (images, champs scalaires).
Les poids d'un reseau entraine sont essentiellement du bruit structure dense.
On s'attend therefore a un ratio de compression FAIBLE (~1× a 3×), PAS 20.4×.
The ratio is MEASURED (metrics/compression.py), never hardcoded.

Math (Sitzmann 2020) :
    Non-linearite : sin(ω₀ · (Wx + b)) for each couche cachee.
    Couche de sortie : lineaire (pas de sin).
    Init : premiere couche U(-1/ω₀, 1/ω₀) ; suivantes U(-√(6/(ω₀²·fan_in)), ...).

La SIREN prend en entree des coords (u,v) ∈ [0,1)² sur le tore T² et produit
un scalar W[u,v]. Evaluee sur une grille h×w, elle regenere la matrix W.
"""

import math
import torch
import torch.nn as nn


class TorusSirenWeight(nn.Module):
    """SIREN qui represente une matrix de poids W[out_h, out_w] comme un champ
    scalar sur le tore T² = [0,1)².

    Args:
        out_h, out_w : dimensions de la matrix a regenerer.
        hidden       : width des couches cachees de la SIREN.
        omega0       : frequence fondamentale (30.0 par defaut, Sitzmann 2020).
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

        # Trois couches (comme SIREN papier for champs scalaires).
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self._init_siren_weights()

        # Grille precalculee (hors-graphe because constant).
        grid = self._build_grid(out_h, out_w)
        self.register_buffer("grid", grid)

    def _init_siren_weights(self):
        """Init SIREN specifique (Sitzmann 2020, section 3.2)."""
        with torch.no_grad():
            # Premiere couche : U(-1/ω₀, 1/ω₀).
            nn.init.uniform_(self.fc1.weight, -1.0 / self.omega0, 1.0 / self.omega0)
            nn.init.zeros_(self.fc1.bias)
            # Couches suivantes : U(-√(6/(ω₀²·fan_in)), √(6/(ω₀²·fan_in))).
            for layer in [self.fc2, self.fc3]:
                fan_in = layer.weight.shape[1]
                bound = math.sqrt(6.0 / (self.omega0 ** 2 * fan_in))
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _build_grid(h: int, w: int) -> torch.Tensor:
        """Grille de coords (u,v) ∈ [0,1)² sur le tore, shape (h·w, 2).

        Sur le tore T²=[0,1)², le point 1 ≡ 0 (identification). On exclut therefore
        l'extremite 1 for eviter la duplication du bord (i/h for i=0..h-1).
        """
        u = torch.arange(h, dtype=torch.float32) / h
        v = torch.arange(w, dtype=torch.float32) / w
        grid = torch.stack(torch.meshgrid(u, v, indexing="ij"), dim=-1)  # (h, w, 2)
        return grid.reshape(-1, 2)  # (h·w, 2)

    def forward(self) -> torch.Tensor:
        """Evalue la SIREN sur la grille → matrix W[out_h, out_w].

        C'est la 'decompression' : on regenere W depuis les params SIREN.
        """
        x = self.grid  # (h·w, 2)
        # Couche 1 + sin(ω₀·).
        x = torch.sin(self.omega0 * self.fc1(x))
        # Couche 2 + sin(ω₀·).
        x = torch.sin(self.omega0 * self.fc2(x))
        # Couche de sortie : lineaire (pas de sin).
        x = self.fc3(x)  # (h·w, 1)
        return x.squeeze(-1).reshape(self.out_h, self.out_w)
