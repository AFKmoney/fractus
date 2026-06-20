"""Tests of TorusSirenWeight : vraie SIREN sin(ω0·), not SiLU."""

import inspect
import torch


def test_siren_uses_sin_not_silu():
    """CRITERE L3 : the SIREN must utiliser torch.sin comme non-linearite,
    PAS nn.SiLU. This is exactment the falsehood d'the original (torus_siren.py:15,17).

    On verifiess via l'inspection REELLE modules code source (AST),
    not via recherche of chaine (qui matcherait the docstring which explique
    the correction)."""
    import ast
    from fractus.nn import siren as siren_mod

    # 1. torch.sin must etre appele in the forward.
    src = inspect.getsource(siren_mod)
    assert 'torch.sin(' in src, "La SIREN must utiliser torch.sin(ω₀·)"

    # 2. Parser the source and verify qu'no Attribute dont attr='SiLU'
    #    n'est utilise comme appel (nn.SiLU()).
    tree = ast.parse(src)
    silu_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'SiLU':
                silu_calls.append(node)
    assert not silu_calls, \
        f"Aucun appel SiLU() ne must apparaitre in le code (falsehood OMNI). Trouve : {len(silu_calls)}"


def test_siren_omega0_is_30_not_56():
    """ω0 = 30 (justifie by Sitzmann 2020), PAS 56 (non justifie, heritage the original)."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    assert abs(s.omega0 - 30.0) < 1e-6, f"ω₀ should etre 30.0, eu {s.omega0}"


def test_siren_output_shape():
    """La SIREN evaluee on the grille produit a matrix (out_h, out_w)."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    W = s()
    assert W.shape == (16, 16)


def test_siren_is_finite():
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    W = s()
    assert torch.isfinite(W).all()


def test_siren_backward_propagates():
    """CRITERE L3 : backward propage a gradient fini ET non-nul a CHAQUE parameter."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    W = s()
    loss = W.pow(2).sum()
    loss.backward()

    params = list(s.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} n'a recu no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini"
        assert p.grad.abs().sum().item() > 0, f"{name} a recu un gradient nul"


def test_siren_fewer_params_than_dense():
    """La SIREN must avoir MOINS of parameters that the matrix dense equivalente."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=32, out_w=32, hidden=16)
    n_siren = sum(p.numel() for p in s.parameters())
    n_dense = 32 * 32
    assert n_siren < n_dense, \
        f"SIREN ({n_siren} params) should etre < dense ({n_dense})"
