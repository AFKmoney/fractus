"""Tests de do_intervention : vrai do-calculus Pearl, pas column-zeroing."""

import torch


def test_do_intervention_clamps_value():
    """do(X_i = v) doit fixer la colonne i à v pour toutes les lignes."""
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3)
    intervened = do_intervention(x, var_idx=1, value=5.0)
    assert torch.allclose(intervened[:, 1], torch.full((4,), 5.0))
    assert torch.allclose(intervened[:, 0], x[:, 0])


def test_do_intervention_other_cols_unchanged():
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3)
    intervened = do_intervention(x, var_idx=0, value=-2.0)
    assert torch.allclose(intervened[:, 1], x[:, 1])
    assert torch.allclose(intervened[:, 2], x[:, 2])


def test_do_intervention_preserves_shape():
    from fractus.causal.do import do_intervention
    x = torch.randn(8, 5)
    intervened = do_intervention(x, var_idx=2, value=0.0)
    assert intervened.shape == x.shape


def test_do_intervention_is_differentiable():
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3, requires_grad=True)
    intervened = do_intervention(x, var_idx=1, value=2.0)
    loss = intervened.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_do_intervention_not_zeroing():
    """CRITÈRE L4 : do(X_i = v) ne doit PAS zerorer la colonne (le faux OMNI)."""
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3)
    intervened = do_intervention(x, var_idx=1, value=7.7)
    assert not torch.allclose(intervened[:, 1], torch.zeros(4)), \
        "do(X_i=v) ne doit pas zerorer la colonne (le faux OMNI mettait à 0)"
    assert torch.allclose(intervened[:, 1], torch.full((4,), 7.7))
