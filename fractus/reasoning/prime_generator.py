"""PrimeGenerator : MLP qui apprend a produire des numbers premiers.

REDESIGN L5+ v2 : la tâche 'converger vers une target a 1e-3' (FNN proof.rs)
was inatteignable with l'architecture GRU+EMA (prouve par 2 tentatives :
REINFORCE pur, then curriculum+shaping+baseline, toutes deux ~0% de validite).

NOUVELLE TÂCHE : produire des integers qui sont PREMIERS. Le verify exact
(PrimeSieve) teste la primalite — c'est binary (true/false sur integers concrets),
pas une convergence numerique.

Pourquoi cette tâche est apprenable :
    - Densite de premiers in [2,100] = 25%. Un generateur aleatoire reussit
      deja 1 fois sur 4 → REINFORCE a un signal constant non-nul.
    - L'objective (maximiser la fraction de n premiers) est clair et mesurable.
    - Le verify est SOUND : tout n accepte est mathematiquement premier.

Architecture : MLP simple. Entree = un contexte (vector aleatoire ou cible).
Sortie = logits sur N classes (n ∈ [2, N]). argmax = n predit.

Note honnete : c'est une tâche SYMBOLIQUE simple (produire un premier), pas
une preuve logical structuree. C'est le niveau atteignable with REINFORCE sur
un verify exact. Une vraie "preuve" au sens derivation logical necessite
un arbre de recherche + verification structurelle (future work).
"""

import torch
import torch.nn as nn

from ..math.primes import PrimeSieve


class PrimeGenerator(nn.Module):
    """MLP qui apprend a produire des numbers premiers.

    Args:
        max_n    : les integers predits sont in [2, max_n].
        context_dim : dimension du vector de contexte (entree).
        hidden   : width du MLP.
    """

    def __init__(self, max_n: int = 100, context_dim: int = 16, hidden: int = 64):
        super().__init__()
        if max_n < 2:
            raise ValueError("max_n must etre >= 2")
        self.max_n = max_n
        self.context_dim = context_dim
        # Classes : indices 0..max_n-2 correspondent a n = 2..max_n.
        self.n_classes = max_n - 1
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.n_classes),
        )
        self.sieve = PrimeSieve(max(max_n, 1000))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context : (B, context_dim) → logits (B, n_classes).

        logits[b, i] correspond a n = i + 2.
        """
        return self.mlp(context)

    def predict(self, context: torch.Tensor) -> torch.Tensor:
        """Retourne les n predits (argmax), shape (B,). n ∈ [2, max_n]."""
        logits = self.forward(context)
        indices = logits.argmax(dim=-1)  # (B,) ∈ [0, n_classes-1]
        return indices + 2  # n = index + 2

    def is_prime_pred(self, n: torch.Tensor) -> torch.Tensor:
        """Verifie la primalite des n predits. Retourne bool tensor (B,)."""
        # n est un tensor d'integers ; on verifies via le crible.
        results = []
        for ni in n.tolist():
            results.append(self.sieve.verify_prime(int(ni)))
        return torch.tensor(results, dtype=torch.bool)
