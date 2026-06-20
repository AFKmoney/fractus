"""Demo L5+ v2 : PrimeGenerator apprend a produire numbers premiers.

REDESIGN : the tache 'convergesr toward target a 1e-3' (L5 original) was
inatteignable. Nouvelle tache : produire integers premiers, verifiesss par
le sieve exact (soundness guaranteed).

REINFORCE : reward = 1 si n predit is prime, 0 otherwise. Loss = -reward · log_prob.
Comme 25% integers in [2,100] are premiers, the signal n'est not sparse.

Critere honesty : valid_rate must depasser 25% (le hasard) after training,
idealement atteindre >70%.

Run :
    python scripts/demo_prime_reinforce.py
"""

import torch
from fractus.reasoning.prime_generator import PrimeGenerator


def evaluate_valid_rate(gen: PrimeGenerator, n_eval: int = 200) -> float:
    """Fraction of n predits which are premiers."""
    gen.eval()
    n_valid = 0
    with torch.no_grad():
        for _ in range(n_eval):
            ctx = torch.randn(1, gen.context_dim)
            n = gen.predict(ctx)
            if gen.is_prime_pred(n)[0]:
                n_valid += 1
    return n_valid / n_eval


def main():
    torch.manual_seed(42)
    gen = PrimeGenerator(max_n=100, context_dim=16, hidden=64)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-2)

    initial_rate = evaluate_valid_rate(gen)
    print(f"Taux de primalite initial (random) : {initial_rate:.1%}")
    print(f"  (densite de premiers in [2,100] ≈ 25%)")
    print()

    n_steps = 500
    eval_every = 100
    for step in range(n_steps):
        gen.train()
        opt.zero_grad()
        ctx = torch.randn(16, gen.context_dim)  # batch of 16 contextes.
        logits = gen(ctx)
        indices = logits.argmax(dim=-1)
        n_pred = indices + 2
        # Reward = 1 si premier, 0 otherwise.
        rewards = gen.is_prime_pred(n_pred).float()  # (16,)
        # REINFORCE : on veut maximiser E[reward]. Loss = -reward · log_prob(n).
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen_log_probs = log_probs[torch.arange(16), indices]
        loss = -(rewards * chosen_log_probs).mean()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == n_steps - 1:
            vr = evaluate_valid_rate(gen)
            print(f"step {step:3d}  valid_rate = {vr:.1%}  "
                  f"(batch reward moyen = {rewards.mean().item():.2f})")

    final_rate = evaluate_valid_rate(gen)
    print()
    print(f"Taux de primalite initial : {initial_rate:.1%}")
    print(f"Taux de primalite final   : {final_rate:.1%}")
    print(f"Amelioration              : {(final_rate - initial_rate)*100:+.1f} points")

    if final_rate > 0.5:
        print(f"\nOK : PrimeGenerator APPREND a produire des premiers "
              f"({final_rate:.0%} vs 25% au hasard).")
        print(f"  Tous les n acceptes sont mathematiquement premiers (soundness).")
    elif final_rate > initial_rate + 0.05:
        print(f"\n~ : amelioration modeste ({final_rate:.0%}). REINFORCE marche")
        print(f"  but converge lentement. Plus de steps aideraient.")
    else:
        print(f"\n~ : pas d'amelioration. Investiguer.")


if __name__ == "__main__":
    main()
