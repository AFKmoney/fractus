"""FractalEmbedding : embedding de codepoint fractal entrainable.

Assemblage de trois sources de features for each token id t :

    (A) 16 features morphologiques deterministes (CharClassFeatures)
    (B) base de Fourier a decroissance Mandelbrot (MandelbrotFourierBasis)
    (C) conditionnement vortex : un hash 2-adique (Collatz, calcule en Rust,
        hors-graphe autodiff) est projete en phases via un MLP entrainable
        (PyTorch, in le graphe). C'est l'option B du spec L1 : le vortex
        2-adique influence l'apprentissage without pretendre etre differentiable.

La projection finale vers d_model est un nn.Linear entrainable. Toute la forward
est differentiable de bout en bout — les parties deterministes (A, B, et le hash
de C) sont precalculees en buffers hors-graphe ; seul le MLP de C et la
projection finale portent des parameters.

Corrections vs systems originaux :
- FNN n'apprenait pas (training.rs:399 = bruit) → ici backward() marche (test).
- OMNI : the 2-adic vortex was orphaned (never imported by Python) →
  ici il conditionne reellement l'embedding.
- OMNI : les « Mandelbrot frequencies » etaient mal nommees → ici on dit
  « Mandelbrot-decayed Fourier basis » (voir fourier.py).
"""

import torch
import torch.nn as nn

from .char_features import CharClassFeatures
from .fourier import MandelbrotFourierBasis


class FractalEmbedding(nn.Module):
    """Embedding fractal entrainable.

    Args :
        vocab_size     : number de token ids couverts.
        d_model        : dimension de sortie.
        n_frequencies  : number de frequences ω_k for la base de Fourier.
        vortex_hidden  : width du MLP qui projette le hash Collatz en phases.
        collatz_steps  : number d'iterations Collatz for le hash (deterministe).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_frequencies: int = 16,
        vortex_hidden: int = 32,
        collatz_steps: int = 7,
    ):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("vocab_size et d_model must etre > 0")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.collatz_steps = collatz_steps

        # (A) Features morphologiques : precalcul deterministe, hors-graphe.
        char_matrix = torch.stack(
            [CharClassFeatures.extract(t) for t in range(vocab_size)], dim=0
        )  # (vocab, 16)
        self.register_buffer("char_features", char_matrix)

        # (B) Base de Fourier Mandelbrot-decroissante : precalcul deterministe.
        self.fourier = MandelbrotFourierBasis(vocab_size, n_frequencies)
        fourier_matrix = self.fourier.matrix()  # (vocab, 2·n_freq)
        self.register_buffer("fourier_features", fourier_matrix)

        # (C) Conditionnement vortex : hash Collatz precalcule (hors-graphe),
        # then projete par un MLP entrainable (in le graphe).
        # On importe le hash depuis le module natif Rust.
        try:
            from fractus import _core
        except ImportError as e:
            raise ImportError(
                "fractus._core introuvable. Lance `maturin develop`."
            ) from e
        hashes = torch.tensor(
            [_core.collatz_hash(t, collatz_steps) for t in range(vocab_size)],
            dtype=torch.float32,
        )  # (vocab,)
        # Normalisation douce : on ramene in [0, 1) via / (max+1) for stabilite.
        max_h = hashes.max().item() + 1.0
        hashes_norm = hashes / max_h
        self.register_buffer("vortex_hashes", hashes_norm)  # (vocab,)

        # MLP entrainable : projette le scalar hash (1) vers un vector de
        # dimension vortex_phase_dim. C'est ici que le vortex « conditionne »
        # le reseau : le MLP apprend a interpreter le hash 2-adique.
        self.vortex_phase_dim = vortex_hidden
        self.vortex_mlp = nn.Sequential(
            nn.Linear(1, vortex_hidden),
            nn.Tanh(),
            nn.Linear(vortex_hidden, vortex_hidden),
        )

        # Projection finale entrainable vers d_model.
        # dim d'entree = 16 (char) + 2·n_freq (fourier) + vortex_hidden
        in_dim = 16 + fourier_matrix.shape[1] + vortex_hidden
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids : (N,) ou (N, L) d'integers in [0, vocab_size).

        Retourne (N, d_model) ou (N, L, d_model).
        """
        if token_ids.max() >= self.vocab_size or token_ids.min() < 0:
            raise IndexError(
                f"token_id hors [0, {self.vocab_size}) : "
                f"min={int(token_ids.min())}, max={int(token_ids.max())}"
            )

        original_shape = token_ids.shape
        flat = token_ids.reshape(-1)  # (M,)

        # (A) + (B) : lookup in les buffers precalcules (hors-graphe, but le
        # result alimente la projection entrainable, therefore le graphe traverse).
        char = self.char_features[flat]      # (M, 16)
        fourier = self.fourier_features[flat]  # (M, 2·n_freq)

        # (C) : hash precalcule → reshape (M, 1) → MLP entrainable (in le graphe).
        h = self.vortex_hashes[flat].unsqueeze(1)  # (M, 1)
        vortex_phases = self.vortex_mlp(h)         # (M, vortex_hidden)

        # Concat et projection.
        x = torch.cat([char, fourier, vortex_phases], dim=1)  # (M, in_dim)
        out = self.proj(x)  # (M, d_model)

        return out.reshape(*original_shape, self.d_model)
