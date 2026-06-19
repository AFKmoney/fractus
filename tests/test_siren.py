"""Tests de TorusSirenWeight : vraie SIREN sin(ω₀·), pas SiLU."""

import inspect
import torch


def test_siren_uses_sin_not_silu():
    """CRITÈRE L3 : la SIREN doit utiliser torch.sin comme non-linéarité,
    PAS nn.SiLU. C'est exactement le mensonge d'OMNI (torus_siren.py:15,17).

    On vérifie via l'inspection RÉELLE des modules du code source (AST),
    pas via recherche de chaîne (qui matcherait le docstring qui explique
    la correction)."""
    import ast
    from fractus.nn import siren as siren_mod

    # 1. torch.sin doit être appelé dans le forward.
    src = inspect.getsource(siren_mod)
    assert 'torch.sin(' in src, "La SIREN doit utiliser torch.sin(ω₀·)"

    # 2. Parser le source et vérifier qu'aucun Attribute dont attr='SiLU'
    #    n'est utilisé comme appel (nn.SiLU()).
    tree = ast.parse(src)
    silu_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'SiLU':
                silu_calls.append(node)
    assert not silu_calls, \
        f"Aucun appel SiLU() ne doit apparaître dans le code (mensonge OMNI). Trouvé : {len(silu_calls)}"


def test_siren_omega0_is_30_not_56():
    """ω₀ = 30 (justifié par Sitzmann 2020), PAS 56 (non justifié, héritage OMNI)."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    assert abs(s.omega0 - 30.0) < 1e-6, f"ω₀ devrait être 30.0, eu {s.omega0}"


def test_siren_output_shape():
    """La SIREN évaluée sur la grille produit une matrice (out_h, out_w)."""
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
    """CRITÈRE L3 : backward propage un gradient fini ET non-nul à CHAQUE paramètre."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    W = s()
    loss = W.pow(2).sum()
    loss.backward()

    params = list(s.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} devrait requires_grad=True"
        assert p.grad is not None, f"{name} n'a reçu aucun gradient"
        assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini"
        assert p.grad.abs().sum().item() > 0, f"{name} a reçu un gradient nul"


def test_siren_fewer_params_than_dense():
    """La SIREN doit avoir MOINS de paramètres que la matrice dense équivalente."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=32, out_w=32, hidden=16)
    n_siren = sum(p.numel() for p in s.parameters())
    n_dense = 32 * 32
    assert n_siren < n_dense, \
        f"SIREN ({n_siren} params) devrait être < dense ({n_dense})"
