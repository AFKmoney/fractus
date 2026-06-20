"""Tests de notears_penalty : differentiable, =0 for DAG, >0 for cycle."""

import torch


def test_notears_zero_for_dag():
    """h(W) ≈ 0 si W est un DAG evident (triangulaire inferieur strict)."""
    from fractus.causal.notears import notears_penalty
    W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.4, 0.0],
    ])
    h = notears_penalty(W)
    assert abs(h.item()) < 1e-3, f"h(DAG) should etre ~0, eu {h.item()}"


def test_notears_positive_for_cycle():
    """h(W) > 0 si W contient un cycle."""
    from fractus.causal.notears import notears_penalty
    W = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    h = notears_penalty(W)
    assert h.item() > 0.5, f"h(cycle) should etre > 0.5, eu {h.item()}"


def test_notears_zero_for_zero_matrix():
    """h(0) = 0 (matrix nulle trivialement acyclique)."""
    from fractus.causal.notears import notears_penalty
    W = torch.zeros(4, 4)
    h = notears_penalty(W)
    assert abs(h.item()) < 1e-6


def test_notears_is_differentiable():
    """h(W) must etre differentiable."""
    from fractus.causal.notears import notears_penalty
    W = torch.randn(3, 3, requires_grad=True)
    h = notears_penalty(W)
    h.backward()
    assert W.grad is not None
    assert torch.isfinite(W.grad).all()


def test_notears_shape_scalar():
    from fractus.causal.notears import notears_penalty
    W = torch.randn(5, 5)
    h = notears_penalty(W)
    assert h.dim() == 0


def test_notears_larger_cycle_detected():
    """Cycle de taille 4 must etre detecte.

    Note : NOTEARS est sensible a l'AMPLITUDE des poids du cycle (h mesure
    l'intensite, pas juste la presence). Avec poids 1.0, h ≈ 0.17 ; with
    poids 2.0, h ≈ 49. On utilise therefore poids 1.5 et seuil > 0.1.
    """
    from fractus.causal.notears import notears_penalty
    W = torch.zeros(4, 4)
    W[0, 1] = W[1, 2] = W[2, 3] = W[3, 0] = 1.5  # poids moderes
    h = notears_penalty(W)
    assert h.item() > 0.1, f"h(cycle 4-noeuds, poids 1.5) should etre > 0.1, eu {h.item()}"
