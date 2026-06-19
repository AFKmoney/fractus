"""Tests de notears_penalty : différentiable, =0 pour DAG, >0 pour cycle."""

import torch


def test_notears_zero_for_dag():
    """h(W) ≈ 0 si W est un DAG évident (triangulaire inférieur strict)."""
    from fractus.causal.notears import notears_penalty
    W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.4, 0.0],
    ])
    h = notears_penalty(W)
    assert abs(h.item()) < 1e-3, f"h(DAG) devrait être ~0, eu {h.item()}"


def test_notears_positive_for_cycle():
    """h(W) > 0 si W contient un cycle."""
    from fractus.causal.notears import notears_penalty
    W = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    h = notears_penalty(W)
    assert h.item() > 0.5, f"h(cycle) devrait être > 0.5, eu {h.item()}"


def test_notears_zero_for_zero_matrix():
    """h(0) = 0 (matrice nulle trivialement acyclique)."""
    from fractus.causal.notears import notears_penalty
    W = torch.zeros(4, 4)
    h = notears_penalty(W)
    assert abs(h.item()) < 1e-6


def test_notears_is_differentiable():
    """h(W) doit être différentiable."""
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
    """Cycle de taille 4 doit être détecté.

    Note : NOTEARS est sensible à l'AMPLITUDE des poids du cycle (h mesure
    l'intensité, pas juste la présence). Avec poids 1.0, h ≈ 0.17 ; avec
    poids 2.0, h ≈ 49. On utilise donc poids 1.5 et seuil > 0.1.
    """
    from fractus.causal.notears import notears_penalty
    W = torch.zeros(4, 4)
    W[0, 1] = W[1, 2] = W[2, 3] = W[3, 0] = 1.5  # poids modérés
    h = notears_penalty(W)
    assert h.item() > 0.1, f"h(cycle 4-nœuds, poids 1.5) devrait être > 0.1, eu {h.item()}"
