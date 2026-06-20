"""Sous-package metrics : mesures honestetes (compression, causal, perplexite).

L3 : compression (mesure real, not of hardcode).
L4 : causal (SHD, causal accuracy, not of clamp).
L6 : perplexity (vraie exp(val_loss), not proxy norme-embedding).
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
