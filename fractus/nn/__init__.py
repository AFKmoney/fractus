"""Sous-package nn — modules de réseau de neurones (PyTorch).

L1 : embedding fractal entraînable (FractalEmbedding).
"""

from .char_features import CharClassFeatures
from .fourier import MandelbrotFourierBasis
from .embedding import FractalEmbedding

__all__ = ["CharClassFeatures", "MandelbrotFourierBasis", "FractalEmbedding"]
