"""Sous-package stability : Lyapunov honestete (under-system Kuramoto).

L6 : function of Lyapunov under-system Kuramoto (le seul true system
dynamique modele). Corrige the false Lyapunov d'the original (lyapunov_shield.py
trackait ||y||2 without system dynamique defini).
"""

from .lyapunov import KuramotoLyapunov

__all__ = ["KuramotoLyapunov"]
