"""TorusSirenWeight : vraie SIREN sin(ω₀·) pour représenter une matrice de poids.

CORRECTION DU MENSONGE D'OMNI-FRACTAL :
- OMNI utilisait nn.SiLU (torus_siren.py:15,17) → ici on utilise torch.sin(ω₀·(Wx+b)),
  la VRAIE non-linéarité SIREN (Sitzmann et al. 2020).
- OMNI utilisait ω₀=56 non justifié → ici ω₀=30.0 (valeur empirique du papier SIREN,
  qui montre que ω₀≈30 est optimal pour la représentation de fonctions continues).
- OMNI commentait "Simple reconstruction: sum of harmonics (real implementation uses
  Fourier)" (torus_siren.py:39) → ici la reconstruction est RÉELLE (forward SIREN
  sur grille 2D).

POSITION SCIENTIFIQUE HONNÊTE :
Une SIREN représente bien des fonctions lisses (images, champs scalaires).
Les poids d'un réseau entraîné sont essentiellement du bruit structuré dense.
On s'attend donc à un ratio de compression FAIBLE (~1× à 3×), PAS 20.4×.
Le ratio est MESURÉ (metrics/compression.py), jamais hardcodé.

Math (Sitzmann 2020) :
    Non-linéarité : sin(ω₀ · (Wx + b)) pour chaque couche cachée.
    Couche de sortie : linéaire (pas de sin).
    Init : première couche U(-1/ω₀, 1/ω₀) ; suivantes U(-√(6/(ω₀²·fan_in)), ...).

La SIREN prend en entrée des coords (u,v) ∈ [0,1)² sur le tore T² et produit
un scalaire W[u,v]. Évaluée sur une grille h×w, elle régénère la matrice W.
"""

import math
import torch
import torch.nn as nn


class TorusSirenWeight(nn.Module):
    """SIREN qui représente une matrice de poids W[out_h, out_w] comme un champ
    scalaire sur le tore T² = [0,1)².

    Args:
        out_h, out_w : dimensions de la matrice à régénérer.
        hidden       : largeur des couches cachées de la SIREN.
        omega0       : fréquence fondamentale (30.0 par défaut, Sitzmann 2020).
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
            raise ValueError("out_h, out_w, hidden doivent être >= 1")
        self.out_h = out_h
        self.out_w = out_w
        self.hidden = hidden
        self.omega0 = omega0

        # Trois couches (comme SIREN papier pour champs scalaires).
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self._init_siren_weights()

        # Grille précalculée (hors-graphe car constante).
        grid = self._build_grid(out_h, out_w)
        self.register_buffer("grid", grid)

    def _init_siren_weights(self):
        """Init SIREN spécifique (Sitzmann 2020, section 3.2)."""
        with torch.no_grad():
            # Première couche : U(-1/ω₀, 1/ω₀).
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
        """Grille de coords (u,v) ∈ [0,1)² sur le tore, shape (h·w, 2)."""
        u = torch.linspace(0, 1, h, dtype=torch.float32)
        v = torch.linspace(0, 1, w, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(u, v, indexing="ij"), dim=-1)  # (h, w, 2)
        return grid.reshape(-1, 2)  # (h·w, 2)

    def forward(self) -> torch.Tensor:
        """Évalue la SIREN sur la grille → matrice W[out_h, out_w].

        C'est la 'décompression' : on régénère W depuis les params SIREN.
        """
        x = self.grid  # (h·w, 2)
        # Couche 1 + sin(ω₀·).
        x = torch.sin(self.omega0 * self.fc1(x))
        # Couche 2 + sin(ω₀·).
        x = torch.sin(self.omega0 * self.fc2(x))
        # Couche de sortie : linéaire (pas de sin).
        x = self.fc3(x)  # (h·w, 1)
        return x.squeeze(-1).reshape(self.out_h, self.out_w)
