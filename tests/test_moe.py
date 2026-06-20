"""Tests of PhaseRoutedMoE : gate von Mises, top-k, load-balance, backward."""

import math
import torch
import pytest


def test_moe_output_shape():
    """Sortie (B, L, d_model) + loss auxiliaire scalar."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16)
    phases = torch.rand(2, 8, 4) * 2 * math.pi
    out, lb_loss = moe(h, phases)
    assert out.shape == (2, 8, 16)
    assert lb_loss.dim() == 0


def test_moe_is_finite():
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16) * 5
    phases = torch.rand(2, 8, 4) * 2 * math.pi
    out, lb_loss = moe(h, phases)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lb_loss)


def test_moe_load_balance_nonneg():
    """Load-balance loss >= 0 (this is a somme of carres ponderee)."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16)
    phases = torch.rand(2, 8, 4) * 2 * math.pi
    _, lb_loss = moe(h, phases)
    assert lb_loss.item() >= -1e-6


def test_moe_backward_every_param():
    """CRITERE L2b : backward propage a gradient fini ET non-nul a CHAQUE parameter."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16)
    phases = torch.rand(2, 8, 4) * 2 * math.pi
    out, lb_loss = moe(h, phases)
    loss = out.pow(2).mean() + 0.1 * lb_loss
    loss.backward()

    params = list(moe.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} n'a recu no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini"
        assert p.grad.abs().sum().item() > 0, f"{name} a recu un gradient nul"


def test_moe_top_k_at_most_n_experts():
    """top_k > n_experts must lever a error."""
    from fractus.nn.moe import PhaseRoutedMoE
    with pytest.raises(ValueError):
        PhaseRoutedMoE(d_model=16, n_experts=4, top_k=8, kappa=4.0)


def test_moe_with_uniform_phases_uses_all_experts():
    """Si all the phases are identiques, the routing not must not crasher."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16)
    phases = torch.zeros(2, 8, 4)
    out, lb_loss = moe(h, phases)
    assert torch.isfinite(out).all()
