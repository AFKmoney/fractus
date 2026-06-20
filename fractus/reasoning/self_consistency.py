"""SelfConsistencyCheck : debat interne by bruit and vote.

Porte depuis the original architecture (src/reasoning.rs:194-279) en PyTorch pur.

Algorithme :
    generate_candidates(h, n, noise_scale) : produit n versions bruitees of h.
        Bruit uniforme U(-noise_scale, +noise_scale) (the original dit "Gaussian-like"
        but utilise uniforme — on est faithful a l'implementation).
    score_candidates(cands, ref) : for each candidat, moyenne of the similarite
        cosinus with the reference on all the (batch, position).
    select_best : argmax scores.
"""

from typing import List

import torch
import torch.nn as nn

from ..math.stats import cosine_similarity


class SelfConsistencyCheck(nn.Module):
    """Debat interne : generated candidats bruites and vote the more coherent.

    Args:
        n_candidates : number of candidats bruites.
        noise_scale  : amplitude bruit uniforme.
    """

    def __init__(self, n_candidates: int = 5, noise_scale: float = 0.1):
        super().__init__()
        self.n_candidates = n_candidates
        self.noise_scale = noise_scale

    def generate_candidates(self, h: torch.Tensor) -> List[torch.Tensor]:
        """Produit n_candidates versions bruitees of h.

        Bruit uniforme U(-noise_scale, +noise_scale), comme the original reasoning.rs:211-220.
        """
        candidates = []
        for _ in range(self.n_candidates):
            noise = (torch.rand_like(h) - 0.5) * 2.0 * self.noise_scale
            candidates.append(h + noise)
        return candidates

    def score_candidates(
        self, candidates: List[torch.Tensor], reference: torch.Tensor
    ) -> List[float]:
        """Score = moyenne of cosine_similarity(cand, ref) on (batch, position)."""
        B, L, _D = reference.shape
        scores = []
        for c in candidates:
            total_sim = 0.0
            count = 0
            for b in range(B):
                for t in range(L):
                    total_sim += float(cosine_similarity(reference[b, t], c[b, t]).item())
                    count += 1
            scores.append(total_sim / count if count > 0 else 0.0)
        return scores

    def select_best(
        self, candidates: List[torch.Tensor], reference: torch.Tensor
    ) -> tuple[int, float]:
        """Retourne (best_idx, best_score)."""
        if not candidates:
            return 0, 0.0
        scores = self.score_candidates(candidates, reference)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return best_idx, scores[best_idx]
