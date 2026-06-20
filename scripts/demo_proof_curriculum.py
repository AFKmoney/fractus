"""Démo L5+ : ProofGenerator apprend par curriculum + reward shaping + baseline.

CORRECTION DU VERDICT L5 : REINFORCE pur n'apprenait pas parce que la tâche
était trop dure d'emblée (targets ±5, erreur 1.7, reward écrasé à 0).

Cette démo utilise ProofTrainer avec :
    - reward shaping continu (-log(1+err), gradient non-nul partout)
    - baseline subtraction (réduit la variance REINFORCE)
    - curriculum (targets ±0.1 → ±5 progressivement)

Critère honnête : l'erreur médiane à ±5 doit baisser d'au moins 30% après
entraînement, et/ou le taux de validité doit augmenter significativement.

Run :
    python scripts/demo_proof_curriculum.py
"""

import torch
from fractus.reasoning.proof import ProofGenerator, ProofVerifier
from fractus.reasoning.proof_trainer import ProofTrainer


def main():
    torch.manual_seed(42)
    generator = ProofGenerator(hidden_dim=32, max_steps=6)
    verifier = ProofVerifier()
    trainer = ProofTrainer(generator, verifier, lr=1e-2)

    metrics = trainer.train(verbose=True)

    print()
    print("=" * 60)
    print("VERDICT L5+ (curriculum + reward shaping + baseline)")
    print("=" * 60)
    print(f"Erreur médiane (±5) : {metrics['initial_error']:.4f} -> {metrics['final_error']:.4f}")
    baisse = (1 - metrics['final_error'] / max(metrics['initial_error'], 1e-9)) * 100
    print(f"Baisse : {baisse:.1f}%")
    print(f"Taux validité (±5) : {metrics['initial_valid_rate']:.1%} -> {metrics['final_valid_rate']:.1%}")
    print(f"Paliers atteints : {metrics['levels_reached']}/5")

    if baisse >= 30.0 or metrics['final_valid_rate'] > metrics['initial_valid_rate'] + 0.05:
        print("\nOK : le générateur APPREND (curriculum + reward shaping + baseline).")
        print("  Correction du verdict L5 : REINFORCE pur n'était pas le problème,")
        print("  c'était la tâche trop dure sans curriculum.")
    else:
        print("\n~ : amélioration modeste. Considérer : lr plus élevé, plus de")
        print("  paliers, ou PPO pour réduire encore la variance.")


if __name__ == "__main__":
    main()
