"""Metriques causales honestetes : Structural Hamming Distance, causal accuracy.

CORRECTION DU MENSONGE D'the original :
- the original (benchmarks.py:43-46) computationait 'causal_acc = max(0, 1 - pehe/2)' then
  'min(causal_acc, 0.98)' — plafonnait artificiellement a 0.98. Rigged.
- Ici : SHD and causal_accuracy MESUREES on a true DAG, without clamp.
"""

import torch


def structural_hamming_distance(
    true_W: torch.Tensor,
    pred_W: torch.Tensor,
    threshold: float = 0.3,
) -> int:
    """SHD : number d'aretes mal predites after binarisation.

    Args:
        true_W : vraie matrix d'adjacence (n, n).
        pred_W : matrix predite (n, n).
        threshold : seuil of binarisation (|W_ij| > threshold → arete presente).
    Returns:
        shd : integer >= 0. 0 = prediction parfaite.
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
    """Fraction d'entrees correctement predites. PAS of clamp (a the difference
    d'the original which plafonnait a 0.98).

    Args:
        true_W, pred_W : matrixs (n, n).
        threshold : seuil of binarisation.
    Returns:
        accuracy ∈ [0, 1].
    """
    true_bin = (true_W.abs() > threshold).float()
    pred_bin = (pred_W.abs() > threshold).float()
    correct = (true_bin == pred_bin).float().mean().item()
    return correct
