"""Tests of FractalLinearAttention : shape, causalite, differentiabilite.

L'attention lineaire causale of Katharopoulos (O(L·d2) instead of O(L2·d)).
Ported faithfully depuis the original architecture src/attention.rs.
"""

import torch
import pytest


def test_attention_shape():
    """Entree (B, L, d_model) → sortie (B, L, d_model)."""
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
    """L'attention est CAUSALE : changer the token a the position j >= t not must
    not affecter the sortie a the position t < j."""
    from fractus.nn.attention import FractalLinearAttention
    torch.manual_seed(0)
    attn = FractalLinearAttention(d_model=16, n_heads=2, d_head=8, n_levels=1)
    attn.eval()  # couper all dropout eventuel
    x = torch.randn(1, 6, 16)
    out1 = attn(x)
    # Modifier the position 4 (et after) not must not changer the sortie aux pos 0..3.
    x_modified = x.clone()
    x_modified[0, 4:] = torch.randn(2, 16)  # briser the positions 4 and 5
    out2 = attn(x_modified)
    # Les 4 premieres positions must be identiques (causalite stricte).
    assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-5), \
        "L'attention n'est pas causale : un token futur a affecte une sortie passee"


def test_attention_backward_propagates():
    """CRITERE L2a : backward() must propager a gradient fini ET non-nul a
    CHAQUE parameter. This is exactment the test that the original echouait."""
    from fractus.nn.attention import FractalLinearAttention
    attn = FractalLinearAttention(d_model=32, n_heads=4, d_head=8, n_levels=2)
    x = torch.randn(2, 8, 32)
    out = attn(x)
    loss = out.pow(2).sum()
    loss.backward()

    params = list(attn.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} n'a recu no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini"
        assert p.grad.abs().sum().item() > 0, f"{name} a recu un gradient nul"


def test_attention_multi_levels_changes_output():
    """Test ORTHOGONAL of l'effet multi-niveaux (correctede the version
    quasi-tautological which tirait deux modules in the same flux RNG).

    Idee : with n_levels=2, si on force level_logits = [+inf, -inf] (therefore le
    softmax met all the poids on the niveau 0, no on the niveau 1), the sortie
    must be EXACTEMENT celle d'un module n_levels=1 with the same offset de
    niveau 0. Et inversement with [-inf, +inf] (all on the niveau 1).
    Cela isole l'effet offsets multi-niveaux independamment of l'init aleatoire.
    """
    from fractus.nn.attention import FractalLinearAttention
    torch.manual_seed(0)
    x = torch.randn(1, 8, 16)

    # Module mono-niveau (reference).
    attn_ref = FractalLinearAttention(d_model=16, n_heads=2, d_head=8, n_levels=1)

    # Module bi-niveau with MEMES poids Q/K/V/out (copie explicite).
    attn2 = FractalLinearAttention(d_model=16, n_heads=2, d_head=8, n_levels=2)
    attn2.w_qkv.data = attn_ref.w_qkv.data.clone()
    attn2.b_qkv.data = attn_ref.b_qkv.data.clone()
    attn2.w_out.data = attn_ref.w_out.data.clone()
    attn2.b_out.data = attn_ref.b_out.data.clone()

    out_ref = attn_ref(x)

    # Cas 1 : all the poids on the niveau 0 → must reproduire exactment attn_ref.
    with torch.no_grad():
        attn2.level_logits.data = torch.tensor([1e9, -1e9])
    out_level0_only = attn2(x)
    assert torch.allclose(out_level0_only, out_ref, atol=1e-4), \
        "Forcer level_logits=[+inf,-inf] should reproduire exactement n_levels=1"

    # Cas 2 : all the poids on the niveau 1 → must DIFFERER of attn_ref
    # (because l'offset ω_1 = (φ2)^{-1} = ω_0 = 1, therefore the feature map differe).
    with torch.no_grad():
        attn2.level_logits.data = torch.tensor([-1e9, 1e9])
    out_level1_only = attn2(x)
    assert not torch.allclose(out_level1_only, out_ref, atol=1e-5), \
        "Forcer level_logits=[-inf,+inf] (offset niveau 1 ≠ niveau 0) should " \
        "donner une sortie differente du niveau 0 — preuve que les offsets " \
        "multi-niveaux changent reellement le computation"


def test_attention_d_model_constraint():
    """d_model must be divisible by n_heads (otherwise error)."""
    from fractus.nn.attention import FractalLinearAttention
    with pytest.raises(ValueError):
        FractalLinearAttention(d_model=30, n_heads=4, d_head=8, n_levels=1)
        # 4 * 8 = 32 = 30
