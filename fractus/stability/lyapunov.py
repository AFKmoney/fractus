"""KuramotoLyapunov : function de Lyapunov du under-system Kuramoto.

CORRECTION DU FAUX LYAPUNOV D'OMNI :
- OMNI (lyapunov_shield.py) trackait ||y||² (norme de sortie du reseau) et
  l'appelait "Lyapunov Shield". Mais il n'y avait AUCUN system dynamique defini
  — un transformer n'est pas naturellement un system dynamique.
  Donc V = ||y||² n'est PAS une function de Lyapunov au sens mathematical.
- Ici : VRAIE function de Lyapunov sur le under-system Kuramoto, qui EST un
  true system dynamique (dθ/dt = f(θ)).

Math : une function de Lyapunov V(x) for un system dx/dt = f(x) must satisfaire :
    1. V(0) = 0, V(x) > 0 for x != 0  (definie positive)
    2. dV/dt = ∇V · f(x) <= 0 le long des trajectoires  (non-croissante)

Pour Kuramoto : V(θ) = ½·Σᵢ (θᵢ − θ*)² with θ* = phase synchronisee cible.
dV/dt = Σᵢ (θᵢ − θ*) · dθᵢ/dt, ou dθᵢ/dt vient de la derivee Kuramoto.
Pour Kuramoto with courange attractif (Λ > 0), V decroit vers la synchronisation.

On SIMULE une trajectoire Kuramoto et on mesure V(t) — must etre monotone
decroissante si le system est stable (courange attractif).
"""

import math
import torch
import torch.nn as nn

from ..nn.phase_ode import KuramotoLayer


class KuramotoLyapunov(nn.Module):
    """Fonction de Lyapunov du under-system Kuramoto.

    Args:
        kuramoto : un KuramotoLayer (dont on veut mesurer la stabilite).
        target_phase : phase synchronisee cible θ* (0.0 par defaut).
    """

    def __init__(self, kuramoto: KuramotoLayer, target_phase: float = 0.0):
        super().__init__()
        self.kuramoto = kuramoto
        self.target_phase = target_phase

    def V(self, phases: torch.Tensor) -> torch.Tensor:
        """V(θ) = ½·Σᵢ (θᵢ − θ*)² (definie positive, = 0 si synchronise).

        phases : (..., N). Retourne un scalar.
        """
        # Distance circulaire : min(|θ-θ*|, 2π - |θ-θ*|) for gerer le wrap.
        diff = phases - self.target_phase
        # Wrap in [-π, π].
        diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
        return 0.5 * (diff ** 2).sum(dim=-1)

    def is_stable_trajectory(self, phases_trajectory: list) -> bool:
        """Verifie que V decroit le long de la trajectoire.

        phases_trajectory : liste de tenseurs (..., N), un par pas de temps.
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
