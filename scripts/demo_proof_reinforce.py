"""Démo L5 : ProofGenerator apprend par REINFORCE à converger vers une target.

Pipeline "neural propose, exact verifier disposes" :
    1. ProofGenerator produit une preuve pour une target aléatoire.
    2. ProofVerifier dit si elle est valide (soundness garantie).
    3. ProofReward calcule la récompense.
    4. On met à jour le générateur par REINFORCE (policy gradient).

POSITION SCIENTIFIQUE HONNÊTE (mise à jour après diagnostic) :
Le seuil de validité FNN (|conclusion - target| < 1e-3) est INACCESSIBLE pour
cette architecture : un GRU à 6 steps d'EMA 0.8/0.2 ne peut pas converger vers
une target arbitraire à 1e-3 près. Erreur médiane sans entraînement ≈ 1.72
(sur plage [-5,5] de largeur 10). FNN n'a jamais mesuré ça (pas d'autodiff).

On mesure donc une MÉTRIQUE CONTINUE HONNÊTE : l'erreur médiane |conclusion -
target| sur 200 targets, avant/après entraînement. Critère de succès : l'erreur
médiane doit baisser d'au moins 30% après 500 steps REINFORCE.

Honnêteté supplémentaire : le vérificateur n'impose aucune structure logique.
Une "preuve valide" ici = le GRU a convergé vers la target numérique. Pas une
dérivation logique au sens des règles d'inférence. Fidèle à FNN (proof.rs:341-360).

Run :
    python scripts/demo_proof_reinforce.py
"""

import torch
from fractus.reasoning.proof import ProofGenerator, ProofVerifier, ProofReward


def evaluate_median_error(generator: ProofGenerator, n_eval: int = 200) -> float:
    """Erreur médiane |conclusion - target| sur n_eval targets aléatoires."""
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
    verifier: ProofVerifier,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, bool]:
    """Une étape REINFORCE. Retourne (reward, is_valid)."""
    optimizer.zero_grad()
    proof, info = generator.generate(target)
    is_valid = verifier.verify_proof(proof)
    reward = reward_fn.compute_reward(proof, is_valid)

    # REINFORCE : on veut maximiser E[reward]. reward est scalaire non-diff.
    # On utilise reward comme poids sur les log-probs des règles choisies.
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
    verifier = ProofVerifier()
    reward_fn = ProofReward()
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-2)

    n_steps = 500
    eval_every = 100

    initial_error = evaluate_median_error(generator)
    print(f"Erreur médiane initiale : {initial_error:.4f}")
    print(f"  (plage target [-5,5], largeur 10 ; erreur médiane = "
          f"{initial_error/10*100:.1f}% de la plage)")
    print()

    for step in range(n_steps):
        target = float(torch.empty(1).uniform_(-5.0, 5.0).item())
        reward, _ = reinforce_update(generator, target, reward_fn, verifier, optimizer)

        if step % eval_every == 0 or step == n_steps - 1:
            med_err = evaluate_median_error(generator)
            print(f"step {step:3d}  erreur médiane = {med_err:.4f}  "
                  f"(reward ~{reward:.3f})")

    final_error = evaluate_median_error(generator)
    print()
    print(f"Erreur médiane initiale : {initial_error:.4f}")
    print(f"Erreur médiane finale   : {final_error:.4f}")
    baisse = (1 - final_error / initial_error) * 100 if initial_error > 0 else 0
    print(f"Baisse                  : {baisse:.1f}%")

    if baisse >= 30.0:
        print(f"\nOK : le générateur apprend à mieux converger "
              f"(erreur médiane divisée par {initial_error/max(final_error,1e-9):.2f}).")
    else:
        print(f"\nVERDICT HONNÊTE : REINFORCE pur n'apprend PAS cette tâche.")
        print(f"  Raison diagnostiquée : le reward est quasi-constant (~0.15-0.25).")
        print(f"  La composante 'correctness' vaut ~0 (erreur >> 1e-3), et les")
        print(f"  composantes 'efficiency'/'diversity' ne dépendent pas de la qualité")
        print(f"  de convergence. REINFORCE avec reward constant = pas de signal.")
        print(f"  ")
        print(f"  Ce n'est PAS un bug de code — c'est un diagnostic scientifique :")
        print(f"  le pipeline preuve de FNN, même corrigé (autodiff natif), ne suffit")
        print(f"  pas à apprendre cette tâche. Pistes futures :")
        print(f"    - Reward shaping : pénalité continue sur l'erreur (pas juste binaire)")
        print(f"    - PPO ou A2C (variance plus faible que REINFORCE)")
        print(f"    - Architecture plus expressive (GRU plus large, plus de steps)")
        print(f"    - Tâche plus facile d'abord (targets dans [-1,1], seuil 0.1)")
        print(f"  ")
        print(f"  La refonte a fait son travail : on SAIT maintenant que ce pipeline")
        print(f"  ne suffit pas, alors que FNN prétendait réussir sans le mesurer.")


if __name__ == "__main__":
    main()
