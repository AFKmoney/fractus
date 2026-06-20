"""KuramotoLyapunov: Lyapunov function of the Kuramoto subsystem.

TRUE Lyapunov function V(theta) = 0.5 * sum(theta_i - theta*)^2 on the Kuramoto
subsystem (the only true dynamical system in the model). Not ||y||^2.
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
        Returns True si V est monotone non-croissante (a epsilon pres).
        """
        if len(phases_trajectory) < 2:
            return True
        Vs = [self.V(p).mean().item() for p in phases_trajectory]
        eps = 1e-6
        for i in range(len(Vs) - 1):
            if Vs[i + 1] > Vs[i] + eps:
                return False
        return True
