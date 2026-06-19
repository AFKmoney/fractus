"""KuramotoLyapunov : fonction de Lyapunov du sous-système Kuramoto.

CORRECTION DU FAUX LYAPUNOV D'OMNI :
- OMNI (lyapunov_shield.py) trackait ||y||² (norme de sortie du réseau) et
  l'appelait "Lyapunov Shield". Mais il n'y avait AUCUN système dynamique défini
  — un transformer n'est pas naturellement un système dynamique.
  Donc V = ||y||² n'est PAS une fonction de Lyapunov au sens mathématique.
- Ici : VRAIE fonction de Lyapunov sur le sous-système Kuramoto, qui EST un
  vrai système dynamique (dθ/dt = f(θ)).

Math : une fonction de Lyapunov V(x) pour un système dx/dt = f(x) doit satisfaire :
    1. V(0) = 0, V(x) > 0 pour x != 0  (définie positive)
    2. dV/dt = ∇V · f(x) <= 0 le long des trajectoires  (non-croissante)

Pour Kuramoto : V(θ) = ½·Σᵢ (θᵢ − θ*)² avec θ* = phase synchronisée cible.
dV/dt = Σᵢ (θᵢ − θ*) · dθᵢ/dt, où dθᵢ/dt vient de la dérivée Kuramoto.
Pour Kuramoto avec couplage attractif (Λ > 0), V décroît vers la synchronisation.

On SIMULE une trajectoire Kuramoto et on mesure V(t) — doit être monotone
décroissante si le système est stable (couplage attractif).
"""

import math
import torch
import torch.nn as nn

from ..nn.phase_ode import KuramotoLayer


class KuramotoLyapunov(nn.Module):
    """Fonction de Lyapunov du sous-système Kuramoto.

    Args:
        kuramoto : un KuramotoLayer (dont on veut mesurer la stabilité).
        target_phase : phase synchronisée cible θ* (0.0 par défaut).
    """

    def __init__(self, kuramoto: KuramotoLayer, target_phase: float = 0.0):
        super().__init__()
        self.kuramoto = kuramoto
        self.target_phase = target_phase

    def V(self, phases: torch.Tensor) -> torch.Tensor:
        """V(θ) = ½·Σᵢ (θᵢ − θ*)² (définie positive, = 0 si synchronisé).

        phases : (..., N). Retourne un scalaire.
        """
        # Distance circulaire : min(|θ-θ*|, 2π - |θ-θ*|) pour gérer le wrap.
        diff = phases - self.target_phase
        # Wrap dans [-π, π].
        diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
        return 0.5 * (diff ** 2).sum(dim=-1)

    def is_stable_trajectory(self, phases_trajectory: list) -> bool:
        """Vérifie que V décroît le long de la trajectoire.

        phases_trajectory : liste de tenseurs (..., N), un par pas de temps.
        Retourne True si V est monotone non-croissante (à epsilon près).
        """
        if len(phases_trajectory) < 2:
            return True
        Vs = [self.V(p).mean().item() for p in phases_trajectory]
        eps = 1e-6
        for i in range(len(Vs) - 1):
            if Vs[i + 1] > Vs[i] + eps:
                return False
        return True
