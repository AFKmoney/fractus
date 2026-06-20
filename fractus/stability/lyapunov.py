"""KuramotoLyapunov : function of Lyapunov under-system Kuramoto.

CORRECTION DU FAUX LYAPUNOV D'the original :
- the original (lyapunov_shield.py) trackait ||y||2 (norme of sortie reseau) et
  l'appelait "Lyapunov Shield". Mais il n'y avait AUCUN system dynamique defini
  — a transformer n'est not naturellement a system dynamique.
  Donc V = ||y||2 n'est PAS a function of Lyapunov au sens mathematical.
- Ici : VRAIE function of Lyapunov on the under-system Kuramoto, which EST un
  true system dynamique (dθ/dt = f(θ)).

Math : a function of Lyapunov V(x) for a system dx/dt = f(x) must satisfaire :
    1. V(0) = 0, V(x) > 0 for x != 0  (definie positive)
    2. dV/dt = ∇V · f(x) <= 0 the long trajectoires  (non-croissante)

Pour Kuramoto : V(θ) = 1⁄2·Σi (θi − θ*)2 with θ* = phase synchronisee target.
dV/dt = Σi (θi − θ*) · dθi/dt, or dθi/dt vient of the derivee Kuramoto.
Pour Kuramoto with courange attractif (Λ > 0), V decroit toward the synchronisation.

On SIMULE a trajectoire Kuramoto and we measure V(t) — must etre monotone
decroissante si the system est stable (courange attractif).
"""

import math
import torch
import torch.nn as nn

from ..nn.phase_ode import KuramotoLayer


class KuramotoLyapunov(nn.Module):
    """Fonction of Lyapunov under-system Kuramoto.

    Args:
        kuramoto : a KuramotoLayer (dont on veut mesurer the stabilite).
        target_phase : phase synchronisee target θ* (0.0 by defaut).
    """

    def __init__(self, kuramoto: KuramotoLayer, target_phase: float = 0.0):
        super().__init__()
        self.kuramoto = kuramoto
        self.target_phase = target_phase

    def V(self, phases: torch.Tensor) -> torch.Tensor:
        """V(θ) = 1⁄2·Σi (θi − θ*)2 (definie positive, = 0 si synchronise).

        phases : (..., N). Retourne a scalar.
        """
        # Distance circulaire : min(|θ-θ*|, 2π - |θ-θ*|) for gerer the wrap.
        diff = phases - self.target_phase
        # Wrap in [-π, π].
        diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
        return 0.5 * (diff ** 2).sum(dim=-1)

    def is_stable_trajectory(self, phases_trajectory: list) -> bool:
        """Verifie that V decroit the long of the trajectoire.

        phases_trajectory : liste of tenseurs (..., N), a by not of temps.
        Retourne True si V est monotone non-croissante (a epsilon pres).
        """
        if len(phases_trajectory) < 2:
            return True
        Vs = [self.V(p).mean().item() for p in phases_trajectory]
        eps = 1e-6
        for i in range(len(Vs) - 1):
            if Vs[i + 1] > Vs[i] + eps:
                return False
        return True
