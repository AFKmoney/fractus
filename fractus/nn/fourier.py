"""Base de Fourier à décroissance Mandelbrot pour l'embedding fractal.

Inspiré de FNN v5.0 (src/math/mandelbrot.rs + src/embedding.rs) mais renommé
honnêtement : FNN appelait ça "Mandelbrot frequencies" en référence à l'ensemble
de Mandelbrot, alors qu'il s'agit juste d'une décroissance géométrique de base
φ² (le carré du nombre d'or). On appelle donc ça "Mandelbrot-decayed Fourier
basis" — la décroissance est réelle et justifiée (séparation d'échelles
multi-niveaux), mais le lien à l'ensemble de Mandelbrot est nul.

Mathématique :
    φ = (1 + √5) / 2  ≈ 1.618
    φ² ≈ 2.618
    ω_k = (φ²)^{-k}    pour k = 0, 1, ..., n_freq-1

La base de Fourier associe à chaque token id t et chaque fréquence k la paire
(sin, cos) de ω_k · t :
    M[t, 2k]   = sin(ω_k · t)
    M[t, 2k+1] = cos(ω_k · t)

On stocke n_freq fréquences ; la matrice produite a 2·n_freq colonnes
(sin+cos par fréquence). Le caller (FractalEmbedding) gère la projection finale.

AUCUN paramètre entraînable ici : tout est déterministe, précalculé une fois.
"""

import math
import torch


class MandelbrotFourierBasis:
    """Base de Fourier déterministe avec décroissance (φ²)^{-k}.

    Attributs :
        vocab_size   : nombre de token ids couverts (0 .. vocab_size-1)
        n_frequencies : nombre de fréquences ω_k
        frequencies  : tenseur (n_frequencies,) des ω_k, en float32
    """

    def __init__(self, vocab_size: int, n_frequencies: int):
        if vocab_size <= 0 or n_frequencies <= 0:
            raise ValueError("vocab_size et n_frequencies doivent être > 0")
        self.vocab_size = vocab_size
        self.n_frequencies = n_frequencies

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        phi_sq = phi * phi  # ≈ 2.618
        ks = torch.arange(n_frequencies, dtype=torch.float32)
        # ω_k = (φ²)^{-k}
        self.frequencies = phi_sq ** (-ks)

        # Précalcul de la matrice (vocab_size, 2·n_frequencies).
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> torch.Tensor:
        """Construit la matrice M[t, :] = [sin(ω_k·t), cos(ω_k·t)] pour tout k."""
        t = torch.arange(self.vocab_size, dtype=torch.float32).unsqueeze(1)  # (V, 1)
        omega = self.frequencies.unsqueeze(0)  # (1, K)
        phases = omega * t  # (V, K) broadcast
        sin_part = torch.sin(phases)  # (V, K)
        cos_part = torch.cos(phases)  # (V, K)
        # Interleave sin/cos : colonnes 0,2,4,... = sin ; 1,3,5,... = cos
        M = torch.empty(self.vocab_size, 2 * self.n_frequencies, dtype=torch.float32)
        M[:, 0::2] = sin_part
        M[:, 1::2] = cos_part
        return M

    def matrix(self) -> torch.Tensor:
        """Retourne la matrice précalculée (vocab_size, 2·n_frequencies)."""
        return self._matrix

    def dim_output(self) -> int:
        """Dimension de sortie (nombre de colonnes de la matrice)."""
        return 2 * self.n_frequencies
