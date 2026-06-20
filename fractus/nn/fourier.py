"""Base de Fourier a decroissance Mandelbrot for l'embedding fractal.

Inspire de the original architecture (src/math/mandelbrot.rs + src/embedding.rs) but renomme
honnetement : FNN appelait ca "Mandelbrot frequencies" en reference a l'ensemble
de Mandelbrot, alors qu'il s'agit juste d'une decroissance geometrique de base
φ² (le carre du number d'or). On appelle therefore ca "Mandelbrot-decayed Fourier
basis" — la decroissance est real et justifiee (separation d'echelles
multi-niveaux), but le lien a l'ensemble de Mandelbrot est nul.

Mathematique :
    φ = (1 + √5) / 2  ≈ 1.618
    φ² ≈ 2.618
    ω_k = (φ²)^{-k}    for k = 0, 1, ..., n_freq-1

La base de Fourier associe a each token id t et each frequence k la paire
(sin, cos) de ω_k · t :
    M[t, 2k]   = sin(ω_k · t)
    M[t, 2k+1] = cos(ω_k · t)

On stocke n_freq frequences ; la matrix produite a 2·n_freq colonnes
(sin+cos par frequence). Le caller (FractalEmbedding) gere la projection finale.

AUCUN parameter entrainable ici : tout est deterministe, precalcule une fois.
"""

import math
import torch


class MandelbrotFourierBasis:
    """Base de Fourier deterministe with decroissance (φ²)^{-k}.

    Attributs :
        vocab_size   : number de token ids couverts (0 .. vocab_size-1)
        n_frequencies : number de frequences ω_k
        frequencies  : tenseur (n_frequencies,) des ω_k, en float32
    """

    def __init__(self, vocab_size: int, n_frequencies: int):
        if vocab_size <= 0 or n_frequencies <= 0:
            raise ValueError("vocab_size et n_frequencies must etre > 0")
        self.vocab_size = vocab_size
        self.n_frequencies = n_frequencies

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        phi_sq = phi * phi  # ≈ 2.618
        ks = torch.arange(n_frequencies, dtype=torch.float32)
        # ω_k = (φ²)^{-k}
        self.frequencies = phi_sq ** (-ks)

        # Precalcul de la matrix (vocab_size, 2·n_frequencies).
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> torch.Tensor:
        """Construit la matrix M[t, :] = [sin(ω_k·t), cos(ω_k·t)] for tout k."""
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
        """Retourne la matrix precalculee (vocab_size, 2·n_frequencies)."""
        return self._matrix
