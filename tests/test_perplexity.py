"""Tests de honest_perplexity : vraie perplexite, pas proxy."""

import math
import torch
import torch.nn as nn


class _DummyModel(nn.Module):
    """Modele trivial : logits = embedding lineaire."""

    def __init__(self, vocab, d):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        return self.head(self.emb(ids))


def test_perplexity_returns_float_ge_one():
    """ppl >= 1.0 (1.0 = prediction parfaite)."""
    from fractus.metrics.perplexity import honest_perplexity
    model = _DummyModel(vocab=10, d=4)
    inp = torch.randint(0, 10, (2, 5))
    tgt = torch.randint(0, 10, (2, 5))
    ppl = honest_perplexity(model, inp, tgt)
    assert isinstance(ppl, float)
    assert ppl >= 1.0


def test_perplexity_uniform_init_close_to_vocab():
    """Un modele non-entraine (logits ~ uniforme) → ppl ≈ vocab."""
    from fractus.metrics.perplexity import honest_perplexity
    # Avec init embedding aleatoire petite, logits ~ uniforme → CE ≈ log(vocab).
    torch.manual_seed(0)
    model = _DummyModel(vocab=10, d=4)
    inp = torch.randint(0, 10, (4, 8))
    tgt = torch.randint(0, 10, (4, 8))
    ppl = honest_perplexity(model, inp, tgt)
    # ppl ≈ exp(log(10)) = 10. Tolerance large because l'init n'est pas parfaitement uniforme.
    assert 3.0 < ppl < 30.0, f"ppl attendu ~10, eu {ppl}"


def test_perplexity_perfect_model_close_to_one():
    """Un modele qui predit exactement → ppl ≈ 1.0."""
    from fractus.metrics.perplexity import honest_perplexity
    model = _DummyModel(vocab=5, d=8)

    # Surfit parfait : on force head for que logits[arg] = grand.
    inp = torch.tensor([[0, 1, 2]])
    tgt = torch.tensor([[0, 1, 2]])
    with torch.no_grad():
        # Bias for que le bon token ait un logit enorme.
        for i in range(3):
            model.head.weight[i, :] = 0
            model.head.weight[i, i] = 100.0
            model.head.bias[i] = 0
            model.emb.weight[i, i] = 1.0
    ppl = honest_perplexity(model, inp, tgt)
    assert ppl < 1.5, f"ppl modele parfait attendu ~1.0, eu {ppl}"


def test_perplexity_not_just_embedding_norm():
    """CRITERE L6 : honest_perplexity must faire un VRAI forward + cross_entropy,
    pas un proxy base sur la norme de l'embedding (le falsehood FNN model.rs:537)."""
    import inspect
    from fractus.metrics import perplexity as ppl_mod
    src = inspect.getsource(ppl_mod)
    assert "cross_entropy" in src, "Doit utiliser cross_entropy (pas un proxy)"
    assert "model(input_ids)" in src or "model(" in src, "Doit faire un true forward"
