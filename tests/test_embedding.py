"""Tests de l'embedding fractal : char features, base de Fourier, FractalEmbedding.

Le critère critique (hérité de FNN qui échouait là) : la forward doit être
différentiable et backward() doit propager des gradients finis partout.
"""

import torch
import pytest


# ---------------------------------------------------------------------------
# CharClassFeatures (16 features morphologiques)
# ---------------------------------------------------------------------------

def test_char_features_shape():
    """16 features pour tout token id."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("a"))
    assert f.shape == (16,)


def test_char_features_vowel():
    """'a' est voyelle (feature 0 = 1)."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("a"))
    assert f[0].item() == 1.0  # is_vowel


def test_char_features_digit_value():
    """'5' est un chiffre de valeur 5 (feature 11)."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("5"))
    assert f[2].item() == 1.0   # is_digit
    assert f[11].item() == 5.0  # digit_value


def test_char_features_batch_consistency():
    """La même lettre donne le même vecteur de features."""
    from fractus.nn.char_features import CharClassFeatures
    f1 = CharClassFeatures.extract(ord("z"))
    f2 = CharClassFeatures.extract(ord("z"))
    assert torch.equal(f1, f2)


# ---------------------------------------------------------------------------
# MandelbrotFourierBasis (base de Fourier à décroissance (φ²)^{-k})
# ---------------------------------------------------------------------------

def test_fourier_basis_shape():
    """Pour vocab 128 et 32 fréquences : matrice (vocab, n_freq) en entrée du calcul."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=128, n_frequencies=32)
    M = basis.matrix()  # (vocab, 2*n_freq)
    assert M.shape == (128, 64)  # 2*32 colonnes (sin+cos)


def test_fourier_basis_is_finite():
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=128, n_frequencies=16)
    M = basis.matrix()
    assert torch.isfinite(M).all()


def test_fourier_frequencies_decay():
    """Les fréquences ω_k = (φ²)^{-k} doivent décroître géométriquement."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=10, n_frequencies=4)
    # ω_0 = 1.0, ω_1 = 1/φ², ω_2 = 1/φ⁴, ...
    phi_sq = ((1 + 5 ** 0.5) / 2) ** 2
    expected = [phi_sq ** (-k) for k in range(4)]
    for k, exp in enumerate(expected):
        assert abs(basis.frequencies[k].item() - exp) < 1e-5, \
            f"freq[{k}] = {basis.frequencies[k].item()}, attendu {exp}"


def test_fourier_matrix_is_deterministic():
    """Deux appels donnent la même matrice (pas d'aléa)."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    b1 = MandelbrotFourierBasis(vocab_size=64, n_frequencies=16)
    b2 = MandelbrotFourierBasis(vocab_size=64, n_frequencies=16)
    assert torch.allclose(b1.matrix(), b2.matrix())


# ---------------------------------------------------------------------------
# FractalEmbedding (assemblage + projection entraînable)
# ---------------------------------------------------------------------------

def test_fractal_embedding_shape():
    """Sortie (N, d_model) pour entrée (N,) d'ids."""
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=128, d_model=64, n_frequencies=16)
    ids = torch.tensor([0, 1, 2, 65, 97])  # mix
    out = emb(ids)
    assert out.shape == (5, 64)


def test_fractal_embedding_is_finite():
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=128, d_model=64, n_frequencies=16)
    ids = torch.arange(128)
    out = emb(ids)
    assert torch.isfinite(out).all()


def test_fractal_embedding_backward_propagates():
    """CRITIQUE : backward() doit propager des gradients finis à tous les params.

    C'est exactement le test que FNN v5.0 échouait (training.rs:399 utilisait du
    bruit aléatoire au lieu d'un gradient). Ici, la projection Linear est dans
    le graphe autodiff, donc les gradients doivent être non-nuls et finis.
    """
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=128, d_model=64, n_frequencies=16)
    ids = torch.tensor([0, 1, 2, 3, 4])
    out = emb(ids)
    loss = out.pow(2).sum()
    loss.backward()

    has_param_with_grad = False
    for name, p in emb.named_parameters():
        assert p.requires_grad, f"{name} devrait requires_grad=True"
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini"
            if p.grad.abs().sum().item() > 0:
                has_param_with_grad = True
    assert has_param_with_grad, "Aucun paramètre n'a reçu de gradient non-nul"


def test_fractal_embedding_respects_vocab_bounds():
    """Un id >= vocab_size doit lever une erreur (pas de crash silencieux)."""
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=100, d_model=32, n_frequencies=8)
    with pytest.raises(IndexError):
        emb(torch.tensor([100]))  # hors borne
