"""Tests du FractalBlock : assemblage LayerNorm → attention → résiduelle.

Le test critique (test_block_backward_every_param) est l'aboutissement de L2a :
prouve que le bloc entier est différentiable et que backward propage un
gradient fini ET non-nul à CHAQUE paramètre. C'est ce que FNN ne savait pas faire.
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
    """Le bloc a une connexion résiduelle : avec un bon init, la sortie est
    proche de l'entrée (pas d'explosion)."""
    from fractus.nn.block import FractalBlock
    torch.manual_seed(0)
    block = FractalBlock(d_model=32, n_heads=4, d_head=8, n_levels=2)
    block.eval()
    x = torch.randn(1, 8, 32)
    out = block(x)
    # La résiduelle garantit out ≈ x + small attn(x). On vérifie juste que
    # la sortie est du même ordre de grandeur (pas d'explosion).
    assert out.std().item() < 10.0 * x.std().item()


def test_block_backward_every_param():
    """CRITÈRE L2a : backward() doit propager un gradient fini ET non-nul à
    CHAQUE paramètre du bloc. C'est exactement ce que FNN v5.0 échouait
    (training.rs:399 utilisait du bruit aléatoire au lieu d'un gradient)."""
    from fractus.nn.block import FractalBlock
    block = FractalBlock(d_model=32, n_heads=4, d_head=8, n_levels=2)
    x = torch.randn(2, 8, 32)
    out = block(x)
    loss = out.pow(2).sum()
    loss.backward()

    params = list(block.named_parameters())
    assert len(params) > 0, "Le bloc n'a aucun paramètre"
    for name, p in params:
        assert p.requires_grad, f"{name} devrait requires_grad=True"
        assert p.grad is not None, f"{name} n'a reçu aucun gradient (paramètre mort)"
        assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini (NaN/Inf)"
        grad_l1 = p.grad.abs().sum().item()
        assert grad_l1 > 0, (
            f"{name} a reçu un gradient nul — l'autodiff ne propage pas "
            f"jusqu'à ce paramètre (grad L1 = {grad_l1})"
        )
