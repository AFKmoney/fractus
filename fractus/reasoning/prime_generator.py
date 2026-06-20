"""PrimeGenerator : MLP qui apprend à produire des nombres premiers.

REDESIGN L5+ v2 : la tâche 'converger vers une target à 1e-3' (FNN proof.rs)
était inatteignable avec l'architecture GRU+EMA (prouvé par 2 tentatives :
REINFORCE pur, puis curriculum+shaping+baseline, toutes deux ~0% de validité).

NOUVELLE TÂCHE : produire des entiers qui sont PREMIERS. Le vérificateur exact
(PrimeSieve) teste la primalité — c'est binaire (vrai/faux sur entiers concrets),
pas une convergence numérique.

Pourquoi cette tâche est apprenable :
    - Densité de premiers dans [2,100] = 25%. Un générateur aléatoire réussit
      déjà 1 fois sur 4 → REINFORCE a un signal constant non-nul.
    - L'objectif (maximiser la fraction de n premiers) est clair et mesurable.
    - Le vérificateur est SOUND : tout n accepté est mathématiquement premier.

Architecture : MLP simple. Entrée = un contexte (vecteur aléatoire ou cible).
Sortie = logits sur N classes (n ∈ [2, N]). argmax = n prédit.

Note honnête : c'est une tâche SYMBOLIQUE simple (produire un premier), pas
une preuve logique structurée. C'est le niveau atteignable avec REINFORCE sur
un vérificateur exact. Une vraie "preuve" au sens dérivation logique nécessite
un arbre de recherche + vérification structurelle (future work).
"""

import torch
import torch.nn as nn

from ..math.primes import PrimeSieve


class PrimeGenerator(nn.Module):
    """MLP qui apprend à produire des nombres premiers.

    Args:
        max_n    : les entiers prédits sont dans [2, max_n].
        context_dim : dimension du vecteur de contexte (entrée).
        hidden   : largeur du MLP.
    """

    def __init__(self, max_n: int = 100, context_dim: int = 16, hidden: int = 64):
        super().__init__()
        if max_n < 2:
            raise ValueError("max_n doit être >= 2")
        self.max_n = max_n
        self.context_dim = context_dim
        # Classes : indices 0..max_n-2 correspondent à n = 2..max_n.
        self.n_classes = max_n - 1
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.n_classes),
        )
        self.sieve = PrimeSieve(max(max_n, 1000))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context : (B, context_dim) → logits (B, n_classes).

        logits[b, i] correspond à n = i + 2.
        """
        return self.mlp(context)

    def predict(self, context: torch.Tensor) -> torch.Tensor:
        """Retourne les n prédits (argmax), shape (B,). n ∈ [2, max_n]."""
        logits = self.forward(context)
        indices = logits.argmax(dim=-1)  # (B,) ∈ [0, n_classes-1]
        return indices + 2  # n = index + 2

    def is_prime_pred(self, n: torch.Tensor) -> torch.Tensor:
        """Vérifie la primalité des n prédits. Retourne bool tensor (B,)."""
        # n est un tensor d'entiers ; on vérifie via le crible.
        results = []
        for ni in n.tolist():
            results.append(self.sieve.verify_prime(int(ni)))
        return torch.tensor(results, dtype=torch.bool)
