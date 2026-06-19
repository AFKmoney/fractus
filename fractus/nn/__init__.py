"""Sous-package nn — modules de réseau de neurones (PyTorch).

L1 : embedding fractal (FractalEmbedding).
L2a : attention linéaire causale (FractalLinearAttention) + bloc minimal (FractalBlock).
"""

from .char_features import CharClassFeatures
from .fourier import MandelbrotFourierBasis
from .embedding import FractalEmbedding
from .stats import elu_plus_one, stable_softmax
from .attention import FractalLinearAttention
from .block import FractalBlock

__all__ = [
    "CharClassFeatures",
    "MandelbrotFourierBasis",
    "FractalEmbedding",
    "elu_plus_one",
    "stable_softmax",
    "FractalLinearAttention",
    "FractalBlock",
]
