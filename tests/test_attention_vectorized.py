"""Tests d'équivalence : attention vectorisée == attention bouclée.

CRITÈRE : la version vectorisée doit donner EXACTEMENT les mêmes sorties que
la version bouclée (à 1e-5 près), pour garantir qu'on n'introduit pas de bug
en optimisant.
"""

import torch


def test_vectorized_matches_looped_small():
    """Sur un petit cas (B=2, L=8, D=4), la vectorisée == bouclée."""
    from fractus.nn.attention import FractalLinearAttention
    torch.manual_seed(0)
    attn = FractalLinearAttention(d_model=8, n_heads=2, d_head=4, n_levels=1)
    attn.eval()
    x = torch.randn(2, 8, 8)
    out_looped = attn(x)
    # La version vectorisée est appelée via _linear_attention_causal_vectorized.
    # On la compare à la bouclée sur les mêmes q,k,v.
    # Pour cela on reproduit la projection + feature map.
    from fractus.nn.stats import elu_plus_one
    B, L, _ = x.shape
    q_all = torch.einsum("bld,de->ble", x, attn.w_qkv[0]) + attn.b_qkv[0]
    k_all = torch.einsum("bld,de->ble", x, attn.w_qkv[1]) + attn.b_qkv[1]
    v_all = torch.einsum("bld,de->ble", x, attn.w_qkv[2]) + attn.b_qkv[2]
    # Une tête, niveau 0.
    q = q_all.view(B, L, 2, 4).transpose(1, 2)[:, 0]
    k = k_all.view(B, L, 2, 4).transpose(1, 2)[:, 0]
    v = v_all.view(B, L, 2, 4).transpose(1, 2)[:, 0]
    q = elu_plus_one(q + attn.level_offsets[0])
    k = elu_plus_one(k + attn.level_offsets[0])
    out_looped_one = attn._linear_attention_causal_one_head(q, k, v)
    out_vec_one = attn._linear_attention_causal_vectorized(q, k, v)
    assert torch.allclose(out_looped_one, out_vec_one, atol=1e-5), \
        f"Vectorisée != bouclée : max diff {(out_looped_one - out_vec_one).abs().max()}"


def test_vectorized_matches_looped_larger():
    """Sur un cas plus grand (B=4, L=32, D=8)."""
    from fractus.nn.attention import FractalLinearAttention
    from fractus.nn.stats import elu_plus_one
    torch.manual_seed(1)
    attn = FractalLinearAttention(d_model=16, n_heads=2, d_head=8, n_levels=1)
    attn.eval()
    x = torch.randn(4, 32, 16)
    B, L, _ = x.shape
    q_all = torch.einsum("bld,de->ble", x, attn.w_qkv[0]) + attn.b_qkv[0]
    k_all = torch.einsum("bld,de->ble", x, attn.w_qkv[1]) + attn.b_qkv[1]
    v_all = torch.einsum("bld,de->ble", x, attn.w_qkv[2]) + attn.b_qkv[2]
    q = q_all.view(B, L, 2, 8).transpose(1, 2)[:, 0]
    k = k_all.view(B, L, 2, 8).transpose(1, 2)[:, 0]
    v = v_all.view(B, L, 2, 8).transpose(1, 2)[:, 0]
    q = elu_plus_one(q + attn.level_offsets[0])
    k = elu_plus_one(k + attn.level_offsets[0])
    out_looped = attn._linear_attention_causal_one_head(q, k, v)
    out_vec = attn._linear_attention_causal_vectorized(q, k, v)
    assert torch.allclose(out_looped, out_vec, atol=1e-5)


def test_vectorized_preserves_causality():
    """La version vectorisée doit préserver la causalité."""
    from fractus.nn.attention import FractalLinearAttention
    torch.manual_seed(0)
    attn = FractalLinearAttention(d_model=8, n_heads=2, d_head=4, n_levels=1)
    attn.eval()
    x = torch.randn(1, 6, 8)
    out1 = attn(x)
    x_mod = x.clone()
    x_mod[0, 4:] = torch.randn(2, 8)
    out2 = attn(x_mod)
    assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-5), \
        "Vectorisée doit rester causale"


def test_vectorized_faster_than_looped():
    """La version vectorisée doit être plus rapide que la bouclée.
    On ne fait pas un benchmark strict, juste une vérification que c'est
    significativement plus rapide (facteur > 2)."""
    import time
    from fractus.nn.attention import FractalLinearAttention
    from fractus.nn.stats import elu_plus_one
    torch.manual_seed(0)
    attn = FractalLinearAttention(d_model=16, n_heads=2, d_head=8, n_levels=1)
    attn.eval()
    B, L, D = 4, 64, 16
    x = torch.randn(B, L, D)
    q_all = torch.einsum("bld,de->ble", x, attn.w_qkv[0]) + attn.b_qkv[0]
    k_all = torch.einsum("bld,de->ble", x, attn.w_qkv[1]) + attn.b_qkv[1]
    v_all = torch.einsum("bld,de->ble", x, attn.w_qkv[2]) + attn.b_qkv[2]
    q = q_all.view(B, L, 2, 8).transpose(1, 2)[:, 0]
    k = k_all.view(B, L, 2, 8).transpose(1, 2)[:, 0]
    v = v_all.view(B, L, 2, 8).transpose(1, 2)[:, 0]
    q = elu_plus_one(q + attn.level_offsets[0])
    k = elu_plus_one(k + attn.level_offsets[0])

    # Warmup.
    attn._linear_attention_causal_one_head(q, k, v)
    attn._linear_attention_causal_vectorized(q, k, v)

    # Bouclée.
    t0 = time.time()
    for _ in range(3):
        attn._linear_attention_causal_one_head(q, k, v)
    t_looped = (time.time() - t0) / 3
    # Vectorisée.
    t0 = time.time()
    for _ in range(3):
        attn._linear_attention_causal_vectorized(q, k, v)
    t_vec = (time.time() - t0) / 3
    speedup = t_looped / max(t_vec, 1e-9)
    print(f"\nBouclée : {t_looped*1000:.1f}ms, Vectorisée : {t_vec*1000:.1f}ms, speedup : {speedup:.1f}x")
    assert speedup > 2.0, f"Vectorisée devrait être >2x plus rapide, eu {speedup:.1f}x"
