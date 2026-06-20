"""Base of Fourier a Mandelbrot decay for l'embedding fractal.

Inspire of the original architecture (src/math/mandelbrot.rs + src/embedding.rs) but renomme
honestetement : the original appelait ca "Mandelbrot frequencies" en reference a l'ensemble
de Mandelbrot, alors qu'il s'agit juste d'une decroissance geometrique of base
φ2 (le carre number d'or). On appelle therefore ca "Mandelbrot-decayed Fourier
basis" — the decroissance est real and justifiee (separation d'echelles
multi-niveaux), but the lien a l'ensemble of Mandelbrot est nul.

Mathematique :
    φ = (1 + √5) / 2  ≈ 1.618
    φ2 ≈ 2.618
    ω_k = (φ2)^{-k}    for k = 0, 1, ..., n_freq-1

La Fourier basis associe a each token id t and each frequence k the paire
(sin, cos) of ω_k · t :
    M[t, 2k]   = sin(ω_k · t)
    M[t, 2k+1] = cos(ω_k · t)

On stocke n_freq frequences ; the matrix produite a 2·n_freq colonnes
(sin+cos by frequence). Le caller (FractalEmbedding) gere the projection finale.

AUCUN parameter entrainable ici : all est deterministic, precomputatione a fois.
"""

import math
import torch


class MandelbrotFourierBasis:
    """Base of Fourier deterministic with decroissance (φ2)^{-k}.

    Attributs :
        vocab_size   : number of token ids couverts (0 .. vocab_size-1)
        n_frequencies : number of frequences ω_k
        frequencies  : tenseur (n_frequencies,) ω_k, en float32
    """

    def __init__(self, vocab_size: int, n_frequencies: int):
        if vocab_size <= 0 or n_frequencies <= 0:
            raise ValueError("vocab_size et n_frequencies must etre > 0")
        self.vocab_size = vocab_size
        self.n_frequencies = n_frequencies

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        phi_sq = phi * phi  # ≈ 2.618
        ks = torch.arange(n_frequencies, dtype=torch.float32)
        # ω_k = (φ2)^{-k}
        self.frequencies = phi_sq ** (-ks)

        # Precomputation of the matrix (vocab_size, 2·n_frequencies).
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> torch.Tensor:
        """Construit the matrix M[t, :] = [sin(ω_k·t), cos(ω_k·t)] for all k."""
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
        """Retourne the matrix precomputationee (vocab_size, 2·n_frequencies)."""
        return self._matrix
