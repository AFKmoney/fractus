"""Sous-package stability : Lyapunov honnete (under-system Kuramoto).

L6 : function de Lyapunov du under-system Kuramoto (le seul true system
dynamique du modele). Corrige le false Lyapunov d'OMNI (lyapunov_shield.py
trackait ||y||² without system dynamique defini).
"""

from .lyapunov import KuramotoLyapunov

__all__ = ["KuramotoLyapunov"]
