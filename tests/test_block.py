"""Tests du FractalBlock : assemblage LayerNorm → attention → residuelle.

Le test critique (test_block_backward_every_param) est l'aboutissement de L2a :
prouve que le bloc integer est differentiable et que backward propage un
gradient fini ET non-nul a CHAQUE parameter. C'est ce que FNN ne savait pas faire.
"""

import torch


def test_block_shape():
    from fractus.nn.block import FractalBlock
    block = FractalBlock(d_model=32, n_heads=4, d_head=8, n_levels=2)
    x = torch.randn(2, 10, 32)
    out = block(x)
    assert out.shape == (2, 10, 32)


def test_block_is_finite():
    from fractus.nn.block import FractalBlock
    block = FractalBlock(d_model=32, n_heads=4, d_head=8, n_levels=2)
    x = torch.randn(2, 10, 32) * 3  # valeurs un peu grandes
    out = block(x)
    assert torch.isfinite(out).all()


def test_block_residual_connection():
    """Le bloc a une connexion residuelle : with un bon init, la sortie est
    proche de l'entree (pas d'explosion)."""
    from fractus.nn.block import FractalBlock
    torch.manual_seed(0)
    block = FractalBlock(d_model=32, n_heads=4, d_head=8, n_levels=2)
    block.eval()
    x = torch.randn(1, 8, 32)
    out = block(x)
    # La residuelle garantit out ≈ x + small attn(x). On verifies juste que
    # la sortie est du meme ordre de grandeur (pas d'explosion).
    assert out.std().item() < 10.0 * x.std().item()


def test_block_backward_every_param():
    """CRITERE L2a : backward() must propager un gradient fini ET non-nul a
    CHAQUE parameter du bloc. C'est exactement ce que the original architecture echouait
    (training.rs:399 utilisait du bruit aleatoire au lieu d'un gradient)."""
    from fractus.nn.block import FractalBlock
    block = FractalBlock(d_model=32, n_heads=4, d_head=8, n_levels=2)
    x = torch.randn(2, 8, 32)
    out = block(x)
    loss = out.pow(2).sum()
    loss.backward()

    params = list(block.named_parameters())
    assert len(params) > 0, "Le bloc n'a no parameter"
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} n'a recu no gradient (parameter mort)"
        assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini (NaN/Inf)"
        grad_l1 = p.grad.abs().sum().item()
        assert grad_l1 > 0, (
            f"{name} a recu un gradient nul — l'autodiff ne propage pas "
            f"jusqu'a ce parameter (grad L1 = {grad_l1})"
        )


def test_block_full_shape_and_finite():
    from fractus.nn.block import FractalBlockFull
    block = FractalBlockFull(
        d_model=32, n_heads=4, d_head=8, n_levels=2,
        n_oscillators=8, coupling_rank=4,
        n_experts=4, top_k=2, kappa=4.0,
    )
    x = torch.randn(2, 8, 32)
    out, lb_loss = block(x)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lb_loss)


def test_block_full_backward_every_param():
    """CRITERE L2b : FractalBlockFull (attn + Kuramoto + MoE) must propager un
    gradient fini ET non-nul a CHAQUE parameter. La preuve ultime que tout
    le pipeline fractal est differentiable de bout en bout."""
    from fractus.nn.block import FractalBlockFull
    block = FractalBlockFull(
        d_model=32, n_heads=4, d_head=8, n_levels=2,
        n_oscillators=8, coupling_rank=4,
        n_experts=4, top_k=2, kappa=4.0,
    )
    x = torch.randn(2, 8, 32)
    out, lb_loss = block(x)
    loss = out.pow(2).sum() + 0.1 * lb_loss
    loss.backward()

    params = list(block.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} n'a recu no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini"
        assert p.grad.abs().sum().item() > 0, f"{name} a recu un gradient nul"
