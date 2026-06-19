"""RecursiveReasoner : Adaptive Computation Time (Graves 2016).

Porté depuis FNN v5.0 (src/reasoning.rs:15-187) en PyTorch pur.

ACT : le modèle "réfléchit" un nombre variable de steps par position. À chaque
step, une probabilité d'arrêt p_t est calculée. L'output est une moyenne pondérée
des états cachés successifs, jusqu'à épuisement du budget (1.0 par position).

Algorithme exact (FNN reasoning.rs:78-176) :
    output = 0
    remaining = 1.0  (budget par position)
    for step in range(max_steps):
        halt_p = σ(w_halt · h + b_halt)   # b_halt init à 1.0 (favorise l'arrêt)
        for positions non encore arrêtées :
            p_actual = min(halt_p, remaining)   # ne pas dépasser le budget
            output += p_actual · h
            remaining -= p_actual
        si toutes arrêtées : break
        h = block_fn(h)
    # Verser le reliquat pour les positions non arrêtées.
    for positions non arrêtées :
        output += remaining · h
    ponder_loss = moyenne du nombre de pas par position.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn

from ..math.stats import sigmoid


class RecursiveReasoner(nn.Module):
    """Adaptive Computation Time.

    Args:
        d_model    : dimension du modèle.
        epsilon    : seuil d'arrêt (0.01 par défaut, FNN act_epsilon).
        max_steps  : nombre max de steps (6 par défaut, FNN max_act_steps).
    """

    def __init__(self, d_model: int, epsilon: float = 0.01, max_steps: int = 6):
        super().__init__()
        self.d_model = d_model
        self.epsilon = epsilon
        self.max_steps = max_steps
        # w_halt, b_halt : b_halt init à 1.0 (favorise l'arrêt, σ(1)≈0.73).
        scale = (1.0 / d_model) ** 0.5
        self.w_halt = nn.Parameter(torch.empty(d_model).uniform_(-scale, scale))
        self.b_halt = nn.Parameter(torch.tensor(1.0))
        # steps_taken : pour logging (moyenne sur batch par position).
        self.steps_taken: Optional[torch.Tensor] = None

    def halt_probability(self, h: torch.Tensor) -> torch.Tensor:
        """h : (..., d_model) → probs (...) = σ(w_halt · h + b_halt)."""
        logits = (h * self.w_halt).sum(dim=-1) + self.b_halt
        return sigmoid(logits)

    def forward(
        self,
        hidden: torch.Tensor,
        block_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, float]:
        """hidden : (B, L, d_model). block_fn : (B,L,d) → (B,L,d).

        Retourne (output (B,L,d), ponder_loss float).
        """
        B, L, D = hidden.shape
        device = hidden.device
        output = torch.zeros_like(hidden)
        remaining = torch.ones(B, L, device=device)
        steps = torch.zeros(B, L, device=device)
        h = hidden

        for _step in range(self.max_steps):
            halt_probs = self.halt_probability(h)  # (B, L)
            all_halted = True
            for b in range(B):
                for t in range(L):
                    if remaining[b, t] < self.epsilon:
                        continue
                    p = halt_probs[b, t]
                    # Si ce pas épuise le budget, prend le reste exact.
                    if remaining[b, t] - p < self.epsilon:
                        p_actual = remaining[b, t]
                    else:
                        p_actual = p
                    output[b, t, :] = output[b, t, :] + p_actual * h[b, t, :]
                    remaining[b, t] = remaining[b, t] - p_actual
                    steps[b, t] = steps[b, t] + 1
                    if remaining[b, t] >= self.epsilon:
                        all_halted = False
            if all_halted:
                break
            h = block_fn(h)

        # Verser le reliquat pour les positions non arrêtées.
        for b in range(B):
            for t in range(L):
                if remaining[b, t] >= self.epsilon:
                    output[b, t, :] = output[b, t, :] + remaining[b, t] * h[b, t, :]
                    steps[b, t] = steps[b, t] + 1

        # Moyenne sur batch pour logging.
        self.steps_taken = steps.mean(dim=0)  # (L,)
        ponder_loss = float(steps.mean().item())
        return output, ponder_loss
