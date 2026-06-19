"""Sous-package stability : Lyapunov honnête (sous-système Kuramoto).

L6 : fonction de Lyapunov du sous-système Kuramoto (le seul vrai système
dynamique du modèle). Corrige le faux Lyapunov d'OMNI (lyapunov_shield.py
trackait ||y||² sans système dynamique défini).
"""

from .lyapunov import KuramotoLyapunov

__all__ = ["KuramotoLyapunov"]
