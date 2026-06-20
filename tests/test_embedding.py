"""Tests of l'embedding fractal : char features, Fourier basis, FractalEmbedding.

Le critere critique (herite of the original which echouait la) : the forward must etre
differentiable and backward() must propager gradients finis parall.
"""

import torch
import pytest


# ---------------------------------------------------------------------------
# CharClassFeatures (16 morphological features)
# ---------------------------------------------------------------------------

def test_char_features_shape():
    """16 features for all token id."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("a"))
    assert f.shape == (16,)


def test_char_features_vowel():
    """'a' est voyelle (feature 0 = 1)."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("a"))
    assert f[0].item() == 1.0  # is_vowel


def test_char_features_digit_value():
    """'5' est a chiffre of value 5 (feature 11)."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("5"))
    assert f[2].item() == 1.0   # is_digit
    assert f[11].item() == 5.0  # digit_value


def test_char_features_batch_consistency():
    """La same lettre donne the same vector of features."""
    from fractus.nn.char_features import CharClassFeatures
    f1 = CharClassFeatures.extract(ord("z"))
    f2 = CharClassFeatures.extract(ord("z"))
    assert torch.equal(f1, f2)


# ---------------------------------------------------------------------------
# MandelbrotFourierBasis (Fourier basis a decroissance (φ2)^{-k})
# ---------------------------------------------------------------------------

def test_fourier_basis_shape():
    """Pour vocab 128 and 32 frequences : matrix (vocab, 2·n_freq) (sin+cos)."""
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
    """Les frequences ω_k = (φ2)^{-k} must decroitre geometriquement."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=10, n_frequencies=4)
    # ω_0 = 1.0, ω_1 = 1/φ2, ω_2 = 1/φ4, ...
    phi_sq = ((1 + 5 ** 0.5) / 2) ** 2
    expected = [phi_sq ** (-k) for k in range(4)]
    for k, exp in enumerate(expected):
        assert abs(basis.frequencies[k].item() - exp) < 1e-5, \
            f"freq[{k}] = {basis.frequencies[k].item()}, attendu {exp}"


def test_fourier_matrix_is_deterministic():
    """Deux appels donnent the same matrix (pas d'alea)."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    b1 = MandelbrotFourierBasis(vocab_size=64, n_frequencies=16)
    b2 = MandelbrotFourierBasis(vocab_size=64, n_frequencies=16)
    assert torch.allclose(b1.matrix(), b2.matrix())


# ---------------------------------------------------------------------------
# FractalEmbedding (assemblage + projection trainable)
# ---------------------------------------------------------------------------

def test_fractal_embedding_shape():
    """Sortie (N, d_model) for input (N,) d'ids."""
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
    """CRITIQUE : backward() must propager a gradient fini ET non-nul a CHAQUE parameter.

    This is exactment the test that the original architecture echouait (training.rs:399 utilisait du
    bruit aleatoire au lieu d'un gradient). Ici, the projection Linear est in
    the graphe autodiff, therefore the gradients must be non-nuls and finis.

    On verifiesss CHAQUE parameter individuellement (et not juste « au less a »),
    parce qu'un parameter mort (gradient nul) in the MLP vortex for example
    indiquerait a autodiff casse sislowly — exactment the defaut que
    cette refonte must eliminer.
    """
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=128, d_model=64, n_frequencies=16)
    ids = torch.tensor([0, 1, 2, 3, 4])
    out = emb(ids)
    loss = out.pow(2).sum()
    loss.backward()

    params = list(emb.named_parameters())
    assert len(params) > 0, "Le modele n'a no parameter entrainable"
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} n'a recu no gradient (parameter mort)"
        assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini (NaN/Inf)"
        grad_l1 = p.grad.abs().sum().item()
        assert grad_l1 > 0, (
            f"{name} a recu un gradient nul — l'autodiff ne propage pas "
            f"jusqu'a ce parameter (grad L1 = {grad_l1})"
        )


def test_fractal_embedding_respects_vocab_bounds():
    """Un id >= vocab_size must lever a error (no crash silencieux)."""
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=100, d_model=32, n_frequencies=8)
    with pytest.raises(IndexError):
        emb(torch.tensor([100]))  # hors borne
