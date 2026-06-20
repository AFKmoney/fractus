"""Tests of KuramotoLayer : encode/decode, RK4, phase_loss, backward."""

import math
import torch


def test_kuramoto_output_shape():
    """Sortie phases (B, L, N_osc) for entree (B, L, d_model)."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16)
    phases = layer(x)
    assert phases.shape == (2, 10, 8)


def test_kuramoto_phases_in_unit_circle():
    """Toutes the phases ∈ [0, 2π) (wrapping modulaire after RK4)."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16) * 10
    phases = layer(x)
    assert (phases >= 0).all() and (phases < 2 * math.pi).all()


def test_kuramoto_is_finite():
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16)
    phases = layer(x)
    assert torch.isfinite(phases).all()


def test_kuramoto_backward_every_param():
    """CRITERE L2b : backward propage a gradient fini ET non-nul a CHAQUE parameter."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16)
    phases = layer(x)
    loss = phases.sum()
    loss.backward()

    params = list(layer.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} n'a recu no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini"
        assert p.grad.abs().sum().item() > 0, f"{name} a recu un gradient nul"


def test_kuramoto_phase_loss_shape_and_finite():
    """phase_loss(phases) returns a scalar fini."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16)
    phases = layer(x)
    loss = layer.phase_loss(phases)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_kuramoto_decode_to_bias_shape():
    """decode_to_bias(phases, d_model) returns (B, L, d_model)."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    phases = torch.rand(2, 10, 8) * 2 * math.pi
    bias = layer.decode_to_bias(phases, d_model=16)
    assert bias.shape == (2, 10, 16)
    assert torch.isfinite(bias).all()
