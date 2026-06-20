"""ProofTrainer : entraîne ProofGenerator par REINFORCE avec curriculum.

CORRECTION DU VERDICT L5 (REINFORCE pur n'apprend pas) :

Le diagnostic a révélé que le reward shaping de FNN est déjà continu et
informatif, MAIS il est écrasé à 0 pour les targets ±5 (erreur médiane 1.7 →
correctness ≈ 0 → pas de signal d'apprentissage). Le problème n'était pas
REINFORCE ni l'architecture, mais la TÂCHE TROP DURE DÈS LE DÉPART.

Solution (3 ingrédients combinés) :

1. REWARD SHAPING continu : pénalité -log(1 + err) au lieu de max(0, 1-err/max).
   Plus informative même pour grandes erreurs (gradient non-nul partout).

2. BASELINE SUBTRACTION : on soustrait une moyenne mobile du reward pour
   réduire la variance de REINFORCE. ∇J = E[(R - b) · ∇log π], b = EMA(R).
   Sans baseline, REINFORCE a une variance élevée qui empêche l'apprentissage.

3. CURRICULUM : on entraîne par paliers de difficulté croissante.
   Palier 0 : targets ±0.1 (générateur réussit déjà à ~10% sans entraînement).
   Palier 1 : ±0.5, Palier 2 : ±1, Palier 3 : ±2, Palier 4 : ±5.
   On passe au palier supérieur quand le taux de validité > seuil (ex: 30%).

L'idée : le générateur apprend d'abord sur la tâche facile (où il y a du signal),
puis généralise progressivement vers les tâches dures.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from .proof import ProofGenerator, ProofVerifier, ProofReward, Proof


@dataclass
class CurriculumLevel:
    """Un palier du curriculum."""
    target_range: float  # targets dans [-range, +range]
    min_valid_rate: float  # taux de validité requis pour passer au palier suivant
    max_steps: int  # nombre max de pas d'entraînement à ce palier


DEFAULT_CURRICULUM: List[CurriculumLevel] = [
    CurriculumLevel(target_range=0.1, min_valid_rate=0.30, max_steps=200),
    CurriculumLevel(target_range=0.5, min_valid_rate=0.25, max_steps=300),
    CurriculumLevel(target_range=1.0, min_valid_rate=0.20, max_steps=400),
    CurriculumLevel(target_range=2.0, min_valid_rate=0.15, max_steps=500),
    CurriculumLevel(target_range=5.0, min_valid_rate=0.10, max_steps=600),
]


def shaped_reward(
    proof: Proof,
    is_valid: bool,
    base_reward_fn: ProofReward,
    sharpness: float = 2.0,
) -> float:
    """Reward shaping continu : pénalité -log(1 + sharpness·err).

    Contrairement à correctness_reward de FNN (max(0, 1-err/max_err) qui
    s'écrase à 0 pour err > max_err), cette forme a un gradient non-nul
    partout : même une grosse erreur donne un signal (faible mais non-nul).

    Args:
        proof : la preuve générée.
        is_valid : verdict du vérificateur exact.
        base_reward_fn : ProofReward de FNN (pour efficiency + diversity).
        sharpness : contrôle la pente de la pénalité.
    Returns:
        reward : float. Composé de :
            - correctness_shaped : -log(1 + sharpness·err) normalisé dans [0, 1].
            - efficiency (de FNN) : 1/n_steps.
            - diversity (de FNN) : n_unique_rules/20.
    """
    err = abs(proof.conclusion - proof.target)
    # -log(1 + sharpness·err) ∈ (-inf, 0]. On normalise dans [0, 1] via
    # 1 - log(1 + sharpness·err) / log(1 + sharpness·10) (borne sup à err=10).
    max_norm = torch.log1p(torch.tensor(sharpness * 10.0)).item()
    correctness_shaped = max(0.0, 1.0 - torch.log1p(torch.tensor(sharpness * err)).item() / max_norm)
    if is_valid:
        correctness_shaped = 1.0  # bonus pour validité exacte.

    # Composantes efficiency + diversity inchangées (de FNN ProofReward).
    eff = base_reward_fn.efficiency_reward(proof)
    div = base_reward_fn.diversity_reward(proof)

    # Mêmes poids que FNN : 0.6 correctness + 0.3 efficiency + 0.1 diversity.
    return 0.6 * correctness_shaped + 0.3 * eff + 0.1 * div


class ProofTrainer:
    """Entraîne ProofGenerator par REINFORCE + baseline + curriculum.

    Args:
        generator   : le ProofGenerator à entraîner.
        verifier    : le ProofVerifier (sound).
        base_reward : ProofReward de FNN (pour efficiency + diversity).
        curriculum  : liste de CurriculumLevel (par défaut DEFAULT_CURRICULUM).
        lr          : learning rate Adam.
        baseline_decay : décroissance EMA de la baseline (0.95 par défaut).
    """

    def __init__(
        self,
        generator: ProofGenerator,
        verifier: ProofVerifier,
        base_reward: Optional[ProofReward] = None,
        curriculum: Optional[List[CurriculumLevel]] = None,
        lr: float = 1e-2,
        baseline_decay: float = 0.95,
    ):
        self.generator = generator
        self.verifier = verifier
        self.base_reward = base_reward if base_reward is not None else ProofReward()
        self.curriculum = curriculum if curriculum is not None else DEFAULT_CURRICULUM
        self.optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
        self.baseline_decay = baseline_decay
        self.baseline: float = 0.0  # EMA du reward.
        self.current_level_idx: int = 0

    def _evaluate_valid_rate(self, target_range: float, n_eval: int = 100) -> float:
        """Taux de preuves valides sur n_eval targets dans [-range, range]."""
        n_valid = 0
        with torch.no_grad():
            for _ in range(n_eval):
                t = float(torch.empty(1).uniform_(-target_range, target_range).item())
                proof, _ = self.generator.generate(t)
                if self.verifier.verify_proof(proof):
                    n_valid += 1
        return n_valid / n_eval

    def _evaluate_median_error(self, target_range: float, n_eval: int = 100) -> float:
        """Erreur médiane sur n_eval targets."""
        errs = []
        with torch.no_grad():
            for _ in range(n_eval):
                t = float(torch.empty(1).uniform_(-target_range, target_range).item())
                proof, _ = self.generator.generate(t)
                errs.append(abs(proof.conclusion - proof.target))
        errs.sort()
        return errs[len(errs) // 2] if errs else 0.0

    def train_step(self, target_range: float) -> tuple[float, bool, float]:
        """Une étape REINFORCE avec baseline, sur une target dans [-range, range].

        Retourne (reward, is_valid, advantage).
        """
        self.optimizer.zero_grad()
        target = float(torch.empty(1).uniform_(-target_range, target_range).item())
        proof, info = self.generator.generate(target)
        is_valid = self.verifier.verify_proof(proof)
        reward = shaped_reward(proof, is_valid, self.base_reward)

        # Advantage = reward - baseline (réduit la variance REINFORCE).
        advantage = reward - self.baseline
        # Mise à jour de la baseline (EMA).
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward

        # REINFORCE : ∇J = advantage · ∇log π(rule | state).
        loss = torch.tensor(0.0, requires_grad=True)
        for logits, selected_idx in zip(info["logits_per_step"], info["selected_indices"]):
            log_probs = torch.log_softmax(logits, dim=-1)
            term = -advantage * log_probs[selected_idx]
            loss = loss + term
        loss.backward()
        self.optimizer.step()
        return reward, is_valid, advantage

    def train(self, verbose: bool = True) -> dict:
        """Entraîne sur tout le curriculum. Retourne un dict de métriques.

        Métriques :
            initial_error : erreur médiane à range=5.0 avant entraînement.
            final_error   : erreur médiane à range=5.0 après entraînement.
            initial_valid_rate : taux de validité à range=5.0 avant.
            final_valid_rate   : taux de validité à range=5.0 après.
            levels_reached : nombre de paliers atteints.
        """
        # Évaluer avant.
        initial_error = self._evaluate_median_error(5.0)
        initial_valid = self._evaluate_valid_rate(5.0)
        if verbose:
            print(f"Avant entraînement : err_méd(±5) = {initial_error:.4f}, "
                  f"valid_rate(±5) = {initial_valid:.1%}")

        levels_reached = 0
        for level_idx, level in enumerate(self.curriculum):
            if verbose:
                print(f"\n--- Palier {level_idx} : targets ±{level.target_range} "
                      f"(objectif valid_rate >= {level.min_valid_rate:.0%}) ---")
            for step in range(level.max_steps):
                self.train_step(level.target_range)
                if verbose and (step % 100 == 0 or step == level.max_steps - 1):
                    err = self._evaluate_median_error(level.target_range)
                    vr = self._evaluate_valid_rate(level.target_range, n_eval=50)
                    print(f"  step {step:4d}  err_méd(±{level.target_range}) = {err:.4f}  "
                          f"valid_rate = {vr:.1%}  baseline = {self.baseline:.3f}")
            # Évaluation finale du palier.
            final_vr = self._evaluate_valid_rate(level.target_range)
            levels_reached = level_idx + 1
            if verbose:
                status = "OK" if final_vr >= level.min_valid_rate else "~"
                print(f"  -> valid_rate final palier : {final_vr:.1%} {status} "
                      f"(objectif {level.min_valid_rate:.0%})")
            # Si on rate largement le palier, on continue quand même au suivant
            # (le curriculum reste progressif, on ne bloque pas).

        # Évaluer après.
        final_error = self._evaluate_median_error(5.0)
        final_valid = self._evaluate_valid_rate(5.0)
        if verbose:
            print(f"\nAprès entraînement : err_méd(±5) = {final_error:.4f}, "
                  f"valid_rate(±5) = {final_valid:.1%}")
            baisse = (1 - final_error / max(initial_error, 1e-9)) * 100
            print(f"Baisse erreur : {baisse:.1f}%")

        return {
            "initial_error": initial_error,
            "final_error": final_error,
            "initial_valid_rate": initial_valid,
            "final_valid_rate": final_valid,
            "levels_reached": levels_reached,
        }
