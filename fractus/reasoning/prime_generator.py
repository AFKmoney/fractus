"""PrimeGenerator : MLP which apprend a produire numbers premiers.

REDESIGN L5+ v2 : the tache 'convergesr toward a target a 1e-3' (the original proof.rs)
was inatteignable with l'architecture GRU+EMA (prouve by 2 tentatives :
REINFORCE pur, then curriculum+shaping+baseline, all deux ~0% of validity).

NOUVELLE TACHE : produire integers which are PREMIERS. Le verify exact
(PrimeSieve) teste the primality — this is binary (true/false on integers concrets),
pas a convergesnce numerique.

Pourquoi cette tache est apprenable :
    - Densite of premiers in [2,100] = 25%. Un generateur aleatoire reussit
      already 1 fois on 4 → REINFORCE a a signal constant non-nul.
    - L'objective (maximiser the fraction of n premiers) est clair and mesurable.
    - Le verify est SOUND : all n acceptedd est mathematicalment premier.

Architecture : MLP simple. Entree = a contexte (vector aleatoire or target).
Sortie = logits on N classes (n ∈ [2, N]). argmax = n predit.

Note honesty : this is a tache SYMBOLIQUE simple (produire a premier), pas
une proof logical structuree. This is the niveau atteignable with REINFORCE sur
un verify exact. Une vraie "proof" au sens derivation logical necessite
un arbre of recherche + verification structurelle (future work).
"""

import torch
import torch.nn as nn

from ..math.primes import PrimeSieve


class PrimeGenerator(nn.Module):
    """MLP which apprend a produire numbers premiers.

    Args:
        max_n    : the integers predits are in [2, max_n].
        context_dim : dimension vector of contexte (input).
        hidden   : width MLP.
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
        """Retourne the n predits (argmax), shape (B,). n ∈ [2, max_n]."""
        logits = self.forward(context)
        indices = logits.argmax(dim=-1)  # (B,) ∈ [0, n_classes-1]
        return indices + 2  # n = index + 2

    def is_prime_pred(self, n: torch.Tensor) -> torch.Tensor:
        """Verifie the primality n predits. Retourne bool tensor (B,)."""
        # n est a tensor d'integers ; on verifiesss via the sieve.
        results = []
        for ni in n.tolist():
            results.append(self.sieve.verify_prime(int(ni)))
        return torch.tensor(results, dtype=torch.bool)
