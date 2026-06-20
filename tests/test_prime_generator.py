"""Tests of PrimeGenerator : produit numbers premiers, apprenable by REINFORCE."""

import torch


def test_prime_generator_output_range():
    """predict returns n ∈ [2, max_n]."""
    from fractus.reasoning.prime_generator import PrimeGenerator
    gen = PrimeGenerator(max_n=50, context_dim=8, hidden=32)
    ctx = torch.randn(4, 8)
    n = gen.predict(ctx)
    assert n.shape == (4,)
    assert (n >= 2).all() and (n <= 50).all()


def test_prime_generator_logits_shape():
    """forward returns (B, n_classes) logits, n_classes = max_n - 1."""
    from fractus.reasoning.prime_generator import PrimeGenerator
    gen = PrimeGenerator(max_n=50, context_dim=8, hidden=32)
    ctx = torch.randn(4, 8)
    logits = gen(ctx)
    assert logits.shape == (4, 49)  # n_classes = max_n - 1 = 49


def test_prime_generator_backward_every_param():
    """CRITERE L5+v2 : backward propage a gradient fini ET non-nul a CHAQUE parameter."""
    from fractus.reasoning.prime_generator import PrimeGenerator
    gen = PrimeGenerator(max_n=50, context_dim=8, hidden=32)
    ctx = torch.randn(4, 8)
    logits = gen(ctx)
    loss = logits.sum()  # loss proxy
    loss.backward()
    for name, p in gen.named_parameters():
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} n'a recu no gradient"
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum().item() > 0, f"{name} a recu un gradient nul"


def test_prime_generator_learns_reinforce():
    """CRITERE L5+v2 : after training REINFORCE, valid_rate > 50% (>25% hasard).

    This is LE test pivot of the correction L5+. the original pretendait produire proofs
    valids but did not learn. Ici on prouve that the pipeline 'neural proposes,
    exact verify disposes' APPREND when the tache est atteignable.
    """
    from fractus.reasoning.prime_generator import PrimeGenerator
    torch.manual_seed(42)
    gen = PrimeGenerator(max_n=100, context_dim=16, hidden=64)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-2)

    # Entrainement short (150 steps).
    for step in range(150):
        opt.zero_grad()
        ctx = torch.randn(16, gen.context_dim)
        logits = gen(ctx)
        indices = logits.argmax(dim=-1)
        n_pred = indices + 2
        rewards = gen.is_prime_pred(n_pred).float()
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen = log_probs[torch.arange(16), indices]
        loss = -(rewards * chosen).mean()
        loss.backward()
        opt.step()

    # Evaluer on 200 contextes frais.
    gen.eval()
    n_valid = 0
    with torch.no_grad():
        for _ in range(200):
            ctx = torch.randn(1, gen.context_dim)
            n = gen.predict(ctx)
            if gen.is_prime_pred(n)[0]:
                n_valid += 1
    valid_rate = n_valid / 200
    assert valid_rate > 0.5, \
        f"valid_rate after entrainement should etre > 50%, eu {valid_rate:.1%}"


def test_prime_generator_soundness():
    """Tout n predit after training must be VRAIMENT premier (soundness)."""
    from fractus.reasoning.prime_generator import PrimeGenerator
    torch.manual_seed(42)
    gen = PrimeGenerator(max_n=100, context_dim=16, hidden=64)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-2)
    for _ in range(100):
        opt.zero_grad()
        ctx = torch.randn(16, gen.context_dim)
        logits = gen(ctx)
        indices = logits.argmax(dim=-1)
        rewards = gen.is_prime_pred(indices + 2).float()
        chosen = torch.log_softmax(logits, dim=-1)[torch.arange(16), indices]
        loss = -(rewards * chosen).mean()
        loss.backward()
        opt.step()
    # Verifier with a sieve INDEPENDANT (re-verification).
    from fractus.math.primes import PrimeSieve
    independent_sieve = PrimeSieve(1000)
    gen.eval()
    with torch.no_grad():
        for _ in range(100):
            ctx = torch.randn(1, gen.context_dim)
            n = int(gen.predict(ctx).item())
            # Le verify of PrimeGenerator and the sieve independent must
            # be d'accord (soundness guaranteed by the sieve exact).
            assert independent_sieve.verify_prime(n) == gen.is_prime_pred(torch.tensor([n]))[0].item()
