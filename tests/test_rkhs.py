"""Tests of RKHSCausalOperator : true RKHS via RFF, not projection bas-rang nue."""

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
    """k(x,x) must etre positif."""
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=4, rank=2, n_rff=64)
    x = torch.randn(3, 4)
    kxx = op.kernel(x, x)
    assert (torch.diagonal(kxx) > 0).all()


def test_rkhs_backward_every_param():
    """CRITERE L4 : backward propage a gradient fini ET non-nul a CHAQUE parameter
    entrainable (U, V, decode). Les W_rff are figes by conception (Rahimi-Recht)."""
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=8, rank=4, n_rff=32)
    x = torch.randn(16, 8)
    y = op(x)
    loss = y.pow(2).sum()
    loss.backward()

    params = list(op.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} n'a recu no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} gradient non-fini"
        # Tous the params nommes (U, V, decode.weight) are entrainables et
        # must recevoir a gradient non-nul.
        assert p.grad.abs().sum().item() > 0, f"{name} a recu un gradient nul"


def test_rkhs_not_just_linear_projection():
    """VRAI RKHS : the sortie must passer by a feature map non-lineaire (cos/sin),
    not etre a projection lineaire simple x@W (le false RKHS d'the original).

    On verifiess that the forward NE PEUT PAS s'ecrire comme a simple Linear :
    si on remplace φ(x) = [cos,sin] by φ(x) = x (lineaire), the sortie change.
    Concretement : on compare the true forward a a forward or on court-circuite
    the features by l'identite (en clippant temporairement the cos/sin via
    a approximation lineaire autour of 0).
    """
    from fractus.causal.rkhs import RKHSCausalOperator
    torch.manual_seed(0)
    op = RKHSCausalOperator(dim=4, rank=2, n_rff=64)
    # Petite entree autour of 0 : a x≈0, cos(ω·x+b)≈cos(b), sin(ω·x+b)≈sin(b)+ω·x·cos(b).
    # Donc the RKHS est approximativement lineaire en x for small x, but pas
    # exactment. Pour x grand, the non-linearite est manifeste.
    x_small = torch.randn(8, 4) * 0.01  # petit
    x_large = torch.randn(8, 4) * 5.0   # grand
    y_small = op(x_small)
    y_large = op(x_large)
    # Pour x grand, si c'was lineaire, doubler x doublerait y. On verifiess la
    # non-linearite : op(2·x_large) = 2·op(x_large) because ofs sin/cos.
    y_2x = op(2.0 * x_large)
    ratio = y_2x / (2.0 * y_large + 1e-10)
    # Si lineaire : ratio = 1 partout. Si non-lineaire : ratio devie of 1.
    deviation = (ratio - 1.0).abs().mean().item()
    assert deviation > 1e-3, \
        f"Le RKHS must etre non-lineaire (deviation > 1e-3), eu {deviation}"
