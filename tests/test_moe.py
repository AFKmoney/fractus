"""Tests of PhaseRoutedMoE: von Mises gate, top-k, load-balance, backward."""

import math
import torch
import pytest


def test_moe_output_shape():
    """Output (B, L, d_model) + scalar auxiliary loss."""
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
    """Load-balance loss >= 0 (it is a weighted sum of squares)."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16)
    phases = torch.rand(2, 8, 4) * 2 * math.pi
    _, lb_loss = moe(h, phases)
    assert lb_loss.item() >= -1e-6


def test_moe_backward_every_param():
    """L2b CRITERION: backward propagates a finite AND non-zero gradient to EVERY parameter."""
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
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


def test_moe_top_k_at_most_n_experts():
    """top_k > n_experts must raise an error."""
    from fractus.nn.moe import PhaseRoutedMoE
    with pytest.raises(ValueError):
        PhaseRoutedMoE(d_model=16, n_experts=4, top_k=8, kappa=4.0)


def test_moe_with_uniform_phases_uses_all_experts():
    """If all phases are identical, the routing must not crash."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16)
    phases = torch.zeros(2, 8, 4)
    out, lb_loss = moe(h, phases)
    assert torch.isfinite(out).all()
