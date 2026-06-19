"""Sous-package math : utilitaires mathématiques purs.

L5 : PrimeSieve (crible d'Ératosthène), FibonacciSequence, cosine_similarity, sigmoid.
"""

from .primes import PrimeSieve
from .fibonacci import FibonacciSequence
from .stats import sigmoid, cosine_similarity

__all__ = ["PrimeSieve", "FibonacciSequence", "sigmoid", "cosine_similarity"]
