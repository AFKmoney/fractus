"""Demo L5+ : ProofGenerator apprend by curriculum + reward shaping + baseline.

CORRECTION DU VERDICT L5 : REINFORCE pur did not learn parce that the tache
was trop dure d'emblee (targets ±5, error 1.7, reward ecrase a 0).

Cette demo utilise ProofTrainer with :
    - reward shaping continu (-log(1+err), gradient non-nul partout)
    - baseline subtraction (reduit the variance REINFORCE)
    - curriculum (targets ±0.1 → ±5 progressivement)

Critere honestete : the error mediane a ±5 must baisser d'au less 30% after
training, et/ou the taux of validity must augmenter significativement.

Run :
    python scripts/demo_proof_curriculum.py
"""

import torch
from fractus.reasoning.proof import ProofGenerator, ProofVerifier
from fractus.reasoning.proof_trainer import ProofTrainer


def main():
    torch.manual_seed(42)
    generator = ProofGenerator(hidden_dim=32, max_steps=6)
    verify = ProofVerifier()
    trainer = ProofTrainer(generator, verify, lr=1e-2)

    metrics = trainer.train(verbose=True)

    print()
    print("=" * 60)
    print("VERDICT L5+ (curriculum + reward shaping + baseline)")
    print("=" * 60)
    print(f"Erreur mediane (±5) : {metrics['initial_error']:.4f} -> {metrics['final_error']:.4f}")
    baisse = (1 - metrics['final_error'] / max(metrics['initial_error'], 1e-9)) * 100
    print(f"Baisse : {baisse:.1f}%")
    print(f"Taux validite (±5) : {metrics['initial_valid_rate']:.1%} -> {metrics['final_valid_rate']:.1%}")
    print(f"Paliers atteints : {metrics['levels_reached']}/5")

    if baisse >= 30.0 or metrics['final_valid_rate'] > metrics['initial_valid_rate'] + 0.05:
        print("\nOK : le generateur APPREND (curriculum + reward shaping + baseline).")
        print("  Correction du verdict L5 : REINFORCE pur n'was pas le problem,")
        print("  c'was la tâche trop dure without curriculum.")
    else:
        print("\n~ : amelioration modeste. Considerer : lr plus eleve, plus de")
        print("  paliers, ou PPO for reduire encore la variance.")


if __name__ == "__main__":
    main()
