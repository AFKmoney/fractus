"""Sous-package metrics : mesures honnêtes (compression, causal, perplexité).

L3 : compression (mesure réelle, pas de hardcode).
L4 : causal (SHD, causal accuracy, pas de clamp).
L6 : perplexity (vraie exp(val_loss), pas proxy norme-embedding).
"""

from .compression import measure_compression_ratio
from .causal import structural_hamming_distance, causal_accuracy
from .perplexity import honest_perplexity

__all__ = [
    "measure_compression_ratio",
    "structural_hamming_distance",
    "causal_accuracy",
    "honest_perplexity",
]
