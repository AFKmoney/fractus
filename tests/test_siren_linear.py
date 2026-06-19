"""Tests de SirenLinear : se comporte comme nn.Linear mais W = SIREN(grid)."""

import torch


def test_siren_linear_shape():
    """SirenLinear(in, out) se comporte comme nn.Linear : (B, in) → (B, out)."""
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 16)


def test_siren_linear_is_finite():
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    x = torch.randn(4, 16)
    assert torch.isfinite(layer(x)).all()


def test_siren_linear_backward_propagates():
    """CRITÈRE L3 : backward propage un gradient fini ET non-nul à CHAQUE paramètre
    de la SIREN (qui EST la matrice de poids, dans le graphe)."""
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    x = torch.randn(4, 16)
    y = layer(x)
    loss = y.pow(2).sum()
    loss.backward()

    params = list(layer.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad
        assert p.grad is not None, f"{name} n'a reçu aucun gradient"
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum().item() > 0, f"{name} a reçu un gradient nul"


def test_siren_linear_has_no_dense_weight():
    """SirenLinear ne doit PAS avoir de nn.Parameter de poids dense séparé —
    la matrice vient entièrement de la SIREN."""
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    # Tous les params doivent venir de la SIREN (préfixe 'siren.') ou du bias.
    param_names = [n for n, _ in layer.named_parameters()]
    for n in param_names:
        assert n.startswith("siren.") or n == "bias", \
            f"Param inattendu (devrait être siren.* ou bias) : {n}"
