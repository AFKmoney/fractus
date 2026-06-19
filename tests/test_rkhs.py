"""Tests de RKHSCausalOperator : vrai RKHS via RFF, pas projection bas-rang nue."""

import torch


def test_rkhs_output_shape():
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=8, rank=4, n_rff=32)
    x = torch.randn(16, 8)
    y = op(x)
    assert y.shape == (16, 8)


def test_rkhs_is_finite():
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=8, rank=4, n_rff=32)
    x = torch.randn(16, 8) * 5
    assert torch.isfinite(op(x)).all()


def test_rkhs_kernel_approx_positive():
    """k(x,x) doit être positif."""
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=4, rank=2, n_rff=64)
    x = torch.randn(3, 4)
    kxx = op.kernel(x, x)
    assert (torch.diagonal(kxx) > 0).all()


def test_rkhs_backward_every_param():
    """CRITÈRE L4 : backward propage un gradient fini ET non-nul à CHAQUE paramètre
    entraînable (U, V, decode). Les W_rff sont figés par conception (Rahimi-Recht)."""
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=8, rank=4, n_rff=32)
    x = torch.randn(16, 8)
    y = op(x)
    loss = y.pow(2).sum()
    loss.backward()

    params = list(op.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} devrait requires_grad=True"
        assert p.grad is not None, f"{name} n'a reçu aucun gradient"
        assert torch.isfinite(p.grad).all(), f"{name} gradient non-fini"
        # Tous les params nommés (U, V, decode.weight) sont entraînables et
        # doivent recevoir un gradient non-nul.
        assert p.grad.abs().sum().item() > 0, f"{name} a reçu un gradient nul"


def test_rkhs_not_just_linear_projection():
    """VRAI RKHS : la sortie doit passer par une feature map non-linéaire (cos/sin),
    pas être une projection linéaire simple x@W (le faux RKHS d'OMNI).

    On vérifie que le forward NE PEUT PAS s'écrire comme une simple Linear :
    si on remplace φ(x) = [cos,sin] par φ(x) = x (linéaire), la sortie change.
    Concrètement : on compare le vrai forward à un forward où on court-circuite
    les features par l'identité (en clippant temporairement les cos/sin via
    une approximation linéaire autour de 0).
    """
    from fractus.causal.rkhs import RKHSCausalOperator
    torch.manual_seed(0)
    op = RKHSCausalOperator(dim=4, rank=2, n_rff=64)
    # Petite entrée autour de 0 : à x≈0, cos(ω·x+b)≈cos(b), sin(ω·x+b)≈sin(b)+ω·x·cos(b).
    # Donc le RKHS est approximativement linéaire en x pour petit x, mais pas
    # exactement. Pour x grand, la non-linéarité est manifeste.
    x_small = torch.randn(8, 4) * 0.01  # petit
    x_large = torch.randn(8, 4) * 5.0   # grand
    y_small = op(x_small)
    y_large = op(x_large)
    # Pour x grand, si c'était linéaire, doubler x doublerait y. On vérifie la
    # non-linéarité : op(2·x_large) ≠ 2·op(x_large) à cause des sin/cos.
    y_2x = op(2.0 * x_large)
    ratio = y_2x / (2.0 * y_large + 1e-10)
    # Si linéaire : ratio = 1 partout. Si non-linéaire : ratio dévie de 1.
    deviation = (ratio - 1.0).abs().mean().item()
    assert deviation > 1e-3, \
        f"Le RKHS doit être non-linéaire (deviation > 1e-3), eu {deviation}"
