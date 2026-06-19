"""Métriques causales honnêtes : Structural Hamming Distance, causal accuracy.

CORRECTION DU MENSONGE D'OMNI :
- OMNI (benchmarks.py:43-46) calculait 'causal_acc = max(0, 1 - pehe/2)' puis
  'min(causal_acc, 0.98)' — plafonnait artificiellement à 0.98. Rigged.
- Ici : SHD et causal_accuracy MESURÉES sur un vrai DAG, sans clamp.
"""

import torch


def structural_hamming_distance(
    true_W: torch.Tensor,
    pred_W: torch.Tensor,
    threshold: float = 0.3,
) -> int:
    """SHD : nombre d'arêtes mal prédites après binarisation.

    Args:
        true_W : vraie matrice d'adjacence (n, n).
        pred_W : matrice prédite (n, n).
        threshold : seuil de binarisation (|W_ij| > threshold → arête présente).
    Returns:
        shd : entier >= 0. 0 = prédiction parfaite.
    """
    true_bin = (true_W.abs() > threshold).float()
    pred_bin = (pred_W.abs() > threshold).float()
    diff = (true_bin != pred_bin).sum().item()
    return int(diff)


def causal_accuracy(
    true_W: torch.Tensor,
    pred_W: torch.Tensor,
    threshold: float = 0.3,
) -> float:
    """Fraction d'entrées correctement prédites. PAS de clamp (à la différence
    d'OMNI qui plafonnait à 0.98).

    Args:
        true_W, pred_W : matrices (n, n).
        threshold : seuil de binarisation.
    Returns:
        accuracy ∈ [0, 1].
    """
    true_bin = (true_W.abs() > threshold).float()
    pred_bin = (pred_W.abs() > threshold).float()
    correct = (true_bin == pred_bin).float().mean().item()
    return correct
