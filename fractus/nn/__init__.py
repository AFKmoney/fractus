"""Sous-package nn — modules of reseau of neurones (PyTorch).

L1 : embedding fractal (FractalEmbedding).
L2a : attention lineaire causale (FractalLinearAttention) + bloc minimal (FractalBlock).
L2b : Kuramoto (KuramotoLayer) + MoE von Mises/Farey (PhaseRoutedMoE) + bloc complete (FractalBlockFull).
"""

from .char_features import CharClassFeatures
from .fourier import MandelbrotFourierBasis
from .embedding import FractalEmbedding
from .stats import elu_plus_one, stable_softmax
from .attention import FractalLinearAttention
from .farey import farey_sequence, expert_phases
from .phase_ode import KuramotoLayer
from .moe import PhaseRoutedMoE
from .block import FractalBlock, FractalBlockFull
from .siren import TorusSirenWeight
from .siren_linear import SirenLinear

__all__ = [
    "CharClassFeatures",
    "MandelbrotFourierBasis",
    "FractalEmbedding",
    "elu_plus_one",
    "stable_softmax",
    "FractalLinearAttention",
    "farey_sequence",
    "expert_phases",
    "KuramotoLayer",
    "PhaseRoutedMoE",
    "FractalBlock",
    "FractalBlockFull",
    "TorusSirenWeight",
    "SirenLinear",
]
