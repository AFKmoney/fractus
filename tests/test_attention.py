"""Tests de FractalLinearAttention : forme, causalité, différentiabilité.

L'attention linéaire causale de Katharopoulos (O(L·d²) au lieu de O(L²·d)).
Portée fidèlement depuis FNN v5.0 src/attention.rs.
"""

import torch
import pytest


def test_attention_shape():
    """Entrée (B, L, d_model) → sortie (B, L, d_model)."""
    from fractus.nn.attention import FractalLinearAttention
    attn = FractalLinearAttention(d_model=32, n_heads=4, d_head=8, n_levels=2)
    x = torch.randn(2, 10, 32)
    out = attn(x)
    assert out.shape == (2, 10, 32)


def test_attention_is_finite():
    from fractus.nn.attention import FractalLinearAttention
    attn = FractalLinearAttention(d_model=32, n_heads=4, d_head=8, n_levels=2)
    x = torch.randn(2, 10, 32)
    out = attn(x)
    assert torch.isfinite(out).all()


def test_attention_causality():
    """L'attention est CAUSALE : changer le token à la position j >= t ne doit
    pas affecter la sortie à la position t < j."""
    from fractus.nn.attention import FractalLinearAttention
    torch.manual_seed(0)
    attn = FractalLinearAttention(d_model=16, n_heads=2, d_head=8, n_levels=1)
    attn.eval()  # couper tout dropout éventuel
    x = torch.randn(1, 6, 16)
    out1 = attn(x)
    # Modifier la position 4 (et après) ne doit pas changer la sortie aux pos 0..3.
    x_modified = x.clone()
    x_modified[0, 4:] = torch.randn(2, 16)  # briser les positions 4 et 5
    out2 = attn(x_modified)
    # Les 4 premières positions doivent être identiques (causalité stricte).
    assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-5), \
        "L'attention n'est pas causale : un token futur a affecté une sortie passée"


def test_attention_backward_propagates():
    """CRITÈRE L2a : backward() doit propager un gradient fini ET non-nul à
    CHAQUE paramètre. C'est exactement le test que FNN échouait."""
    from fractus.nn.attention import FractalLinearAttention
    attn = FractalLinearAttention(d_model=32, n_heads=4, d_head=8, n_levels=2)
    x = torch.randn(2, 8, 32)
    out = attn(x)
    loss = out.pow(2).sum()
    loss.backward()

    params = list(attn.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} devrait requires_grad=True"
        assert p.grad is not None, f"{name} n'a reçu aucun gradient"
        assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini"
        assert p.grad.abs().sum().item() > 0, f"{name} a reçu un gradient nul"


def test_attention_multi_levels_changes_output():
    """Test ORTHOGONAL de l'effet multi-niveaux (corrige la version
    quasi-tautologique qui tirait deux modules dans le même flux RNG).

    Idée : avec n_levels=2, si on force level_logits = [+inf, -inf] (donc le
    softmax met tout le poids sur le niveau 0, aucun sur le niveau 1), la sortie
    doit être EXACTEMENT celle d'un module n_levels=1 avec le même offset de
    niveau 0. Et inversement avec [-inf, +inf] (tout sur le niveau 1).
    Cela isole l'effet des offsets multi-niveaux indépendamment de l'init aléatoire.
    """
    from fractus.nn.attention import FractalLinearAttention
    torch.manual_seed(0)
    x = torch.randn(1, 8, 16)

    # Module mono-niveau (référence).
    attn_ref = FractalLinearAttention(d_model=16, n_heads=2, d_head=8, n_levels=1)

    # Module bi-niveau avec MÊMES poids Q/K/V/out (copie explicite).
    attn2 = FractalLinearAttention(d_model=16, n_heads=2, d_head=8, n_levels=2)
    attn2.w_qkv.data = attn_ref.w_qkv.data.clone()
    attn2.b_qkv.data = attn_ref.b_qkv.data.clone()
    attn2.w_out.data = attn_ref.w_out.data.clone()
    attn2.b_out.data = attn_ref.b_out.data.clone()

    out_ref = attn_ref(x)

    # Cas 1 : tout le poids sur le niveau 0 → doit reproduire exactement attn_ref.
    with torch.no_grad():
        attn2.level_logits.data = torch.tensor([1e9, -1e9])
    out_level0_only = attn2(x)
    assert torch.allclose(out_level0_only, out_ref, atol=1e-4), \
        "Forcer level_logits=[+inf,-inf] devrait reproduire exactement n_levels=1"

    # Cas 2 : tout le poids sur le niveau 1 → doit DIFFÉRER de attn_ref
    # (car l'offset ω_1 = (φ²)^{-1} ≠ ω_0 = 1, donc la feature map diffère).
    with torch.no_grad():
        attn2.level_logits.data = torch.tensor([-1e9, 1e9])
    out_level1_only = attn2(x)
    assert not torch.allclose(out_level1_only, out_ref, atol=1e-5), \
        "Forcer level_logits=[-inf,+inf] (offset niveau 1 ≠ niveau 0) devrait " \
        "donner une sortie différente du niveau 0 — preuve que les offsets " \
        "multi-niveaux changent réellement le calcul"


def test_attention_d_model_constraint():
    """d_model doit être divisible par n_heads (sinon erreur)."""
    from fractus.nn.attention import FractalLinearAttention
    with pytest.raises(ValueError):
        FractalLinearAttention(d_model=30, n_heads=4, d_head=8, n_levels=1)
        # 4 * 8 = 32 ≠ 30
