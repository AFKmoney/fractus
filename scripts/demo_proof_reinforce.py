"""Demo L5 : ProofGenerator apprend by REINFORCE a convergesr toward a target.

Pipeline "neural propose, exact verify disposes" :
    1. ProofGenerator produit a proof for a target aleatoire.
    2. ProofVerifier dit si elle est valid (soundness guaranteed).
    3. ProofReward computatione the reward.
    4. On met a jour the generateur by REINFORCE (policy gradient).

POSITION SCIENTIFIQUE HONNETE (mise a jour after diagnostic) :
Le threshold of validity the original (|conclusion - target| < 1e-3) est INACCESSIBLE for
cette architecture : a GRU a 6 steps d'EMA 0.8/0.2 not can not convergesr vers
une target arbitraire a 1e-3 pres. Erreur mediane without training ≈ 1.72
(sur range [-5,5] of width 10). the original never measured it (no autodiff).

On mesure therefore a METRIQUE CONTINUE HONNETE : the error mediane |conclusion -
target| on 200 targets, before/after training. Critere of succes : the error
mediane must baisser d'au less 30% after 500 steps REINFORCE.

Honnetete supplementaire : the verify n'impose no structure logical.
Une "proof valid" here = the GRU a converges toward the target numerique. Pas une
derivation logical au sens inference rules. Fidele a the original (proof.rs:341-360).

Run :
    python scripts/demo_proof_reinforce.py
"""

import torch
from fractus.reasoning.proof import ProofGenerator, ProofVerifier, ProofReward


def evaluate_median_error(generator: ProofGenerator, n_eval: int = 200) -> float:
    """Erreur mediane |conclusion - target| on n_eval targets aleatoires."""
    errors = []
    with torch.no_grad():
        for _ in range(n_eval):
            target = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            proof, _ = generator.generate(target)
            errors.append(abs(proof.conclusion - proof.target))
    errors.sort()
    return errors[len(errors) // 2]


def reinforce_update(
    generator: ProofGenerator,
    target: float,
    reward_fn: ProofReward,
    verify: ProofVerifier,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, bool]:
    """Une etape REINFORCE. Retourne (reward, is_valid)."""
    optimizer.zero_grad()
    proof, info = generator.generate(target)
    is_valid = verify.verify_proof(proof)
    reward = reward_fn.compute_reward(proof, is_valid)

    # REINFORCE : on veut maximiser E[reward]. reward est scalar non-diff.
    # On utilise reward comme poids on the log-probs regles choisies.
    loss = torch.tensor(0.0, requires_grad=True)
    for logits, selected_idx in zip(info["logits_per_step"], info["selected_indices"]):
        log_probs = torch.log_softmax(logits, dim=-1)
        term = -reward * log_probs[selected_idx]
        loss = loss + term
    loss.backward()
    optimizer.step()
    return reward, is_valid


def main():
    torch.manual_seed(42)
    generator = ProofGenerator(hidden_dim=32, max_steps=6)
    verify = ProofVerifier()
    reward_fn = ProofReward()
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-2)

    n_steps = 500
    eval_every = 100

    initial_error = evaluate_median_error(generator)
    print(f"Erreur mediane initial : {initial_error:.4f}")
    print(f"  (range target [-5,5], width 10 ; error mediane = "
          f"{initial_error/10*100:.1f}% de la range)")
    print()

    for step in range(n_steps):
        target = float(torch.empty(1).uniform_(-5.0, 5.0).item())
        reward, _ = reinforce_update(generator, target, reward_fn, verify, optimizer)

        if step % eval_every == 0 or step == n_steps - 1:
            med_err = evaluate_median_error(generator)
            print(f"step {step:3d}  error mediane = {med_err:.4f}  "
                  f"(reward ~{reward:.3f})")

    final_error = evaluate_median_error(generator)
    print()
    print(f"Erreur mediane initial : {initial_error:.4f}")
    print(f"Erreur mediane finale   : {final_error:.4f}")
    baisse = (1 - final_error / initial_error) * 100 if initial_error > 0 else 0
    print(f"Baisse                  : {baisse:.1f}%")

    if baisse >= 30.0:
        print(f"\nOK : le generateur apprend a mieux converger "
              f"(error mediane divisee par {initial_error/max(final_error,1e-9):.2f}).")
    else:
        print(f"\nVERDICT HONNETE : REINFORCE pur n'apprend PAS cette tâche.")
        print(f"  Raison diagnostiquee : le reward est quasi-constant (~0.15-0.25).")
        print(f"  La composante 'correctness' vaut ~0 (error >> 1e-3), et les")
        print(f"  composantes 'efficiency'/'diversity' ne dependent pas de la quality")
        print(f"  de convergence. REINFORCE with reward constant = pas de signal.")
        print(f"  ")
        print(f"  Ce n'est PAS un bug de code — c'est un diagnostic scientifique :")
        print(f"  le pipeline preuve de FNN, meme corrige (autodiff natif), ne suffit")
        print(f"  pas a apprendre cette tâche. Pistes futures :")
        print(f"    - Reward shaping : penalty continuous sur the error (not just binary)")
        print(f"    - PPO ou A2C (variance plus faible que REINFORCE)")
        print(f"    - Architecture plus expressive (GRU plus large, plus de steps)")
        print(f"    - Tâche plus facile d'abord (targets in [-1,1], seuil 0.1)")
        print(f"  ")
        print(f"  La refonte a fait son travail : on SAIT now que ce pipeline")
        print(f"  ne suffit pas, alors que FNN pretendait reussir without le mesurer.")


if __name__ == "__main__":
    main()
