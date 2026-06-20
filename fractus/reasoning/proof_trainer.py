"""ProofTrainer : entraine ProofGenerator par REINFORCE with curriculum.

CORRECTION DU VERDICT L5 (REINFORCE pur n'apprend pas) :

Le diagnostic a revele que le reward shaping de FNN est deja continu et
informatif, MAIS il est ecrase a 0 for les targets ±5 (error mediane 1.7 →
correctness ≈ 0 → pas de signal d'apprentissage). Le problem n'was pas
REINFORCE ni l'architecture, but la TÂCHE TROP DURE DES LE DEPART.

Solution (3 ingredients combines) :

1. REWARD SHAPING continu : penalty -log(1 + err) au lieu de max(0, 1-err/max).
   Plus informative meme for grandes errors (gradient non-nul partout).

2. BASELINE SUBTRACTION : on soustrait une moyenne mobile du reward for
   reduire la variance de REINFORCE. ∇J = E[(R - b) · ∇log π], b = EMA(R).
   Sans baseline, REINFORCE a une variance elevee qui empeche l'apprentissage.

3. CURRICULUM : on entraine par paliers de difficulte croissante.
   Palier 0 : targets ±0.1 (generateur reussit deja a ~10% without entrainement).
   Palier 1 : ±0.5, Palier 2 : ±1, Palier 3 : ±2, Palier 4 : ±5.
   On passe au palier superieur quand le taux de validite > seuil (ex: 30%).

L'idee : le generateur apprend d'abord sur la tâche facile (ou il y a du signal),
then generalise progressivement vers les tâches dures.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from .proof import ProofGenerator, ProofVerifier, ProofReward, Proof


@dataclass
class CurriculumLevel:
    """Un palier du curriculum."""
    target_range: float  # targets in [-range, +range]
    min_valid_rate: float  # taux de validite requis for passer au palier suivant
    max_steps: int  # number max de pas d'entrainement a ce palier


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
    """Reward shaping continu : penalty -log(1 + sharpness·err).

    Contrairement a correctness_reward de FNN (max(0, 1-err/max_err) qui
    s'ecrase a 0 for err > max_err), cette shape a un gradient non-nul
    partout : meme une grosse error donne un signal (faible but non-nul).

    Args:
        proof : la preuve generee.
        is_valid : verdict du verify exact.
        base_reward_fn : ProofReward de FNN (for efficiency + diversity).
        sharpness : controle la pente de la penalty.
    Returns:
        reward : float. Compose de :
            - correctness_shaped : -log(1 + sharpness·err) normalise in [0, 1].
            - efficiency (de FNN) : 1/n_steps.
            - diversity (de FNN) : n_unique_rules/20.
    """
    err = abs(proof.conclusion - proof.target)
    # -log(1 + sharpness·err) ∈ (-inf, 0]. On normalise in [0, 1] via
    # 1 - log(1 + sharpness·err) / log(1 + sharpness·10) (borne sup a err=10).
    max_norm = torch.log1p(torch.tensor(sharpness * 10.0)).item()
    correctness_shaped = max(0.0, 1.0 - torch.log1p(torch.tensor(sharpness * err)).item() / max_norm)
    if is_valid:
        correctness_shaped = 1.0  # bonus for validite exact.

    # Composantes efficiency + diversity inchangees (de FNN ProofReward).
    eff = base_reward_fn.efficiency_reward(proof)
    div = base_reward_fn.diversity_reward(proof)

    # Memes poids que FNN : 0.6 correctness + 0.3 efficiency + 0.1 diversity.
    return 0.6 * correctness_shaped + 0.3 * eff + 0.1 * div


class ProofTrainer:
    """Entraine ProofGenerator par REINFORCE + baseline + curriculum.

    Args:
        generator   : le ProofGenerator a entrainer.
        verify    : le ProofVerifier (sound).
        base_reward : ProofReward de FNN (for efficiency + diversity).
        curriculum  : liste de CurriculumLevel (par defaut DEFAULT_CURRICULUM).
        lr          : learning rate Adam.
        baseline_decay : decroissance EMA de la baseline (0.95 par defaut).
    """

    def __init__(
        self,
        generator: ProofGenerator,
        verify: ProofVerifier,
        base_reward: Optional[ProofReward] = None,
        curriculum: Optional[List[CurriculumLevel]] = None,
        lr: float = 1e-2,
        baseline_decay: float = 0.95,
    ):
        self.generator = generator
        self.verify = verify
        self.base_reward = base_reward if base_reward is not None else ProofReward()
        self.curriculum = curriculum if curriculum is not None else DEFAULT_CURRICULUM
        self.optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
        self.baseline_decay = baseline_decay
        self.baseline: float = 0.0  # EMA du reward.
        self.current_level_idx: int = 0

    def _evaluate_valid_rate(self, target_range: float, n_eval: int = 100) -> float:
        """Taux de preuves valides sur n_eval targets in [-range, range]."""
        n_valid = 0
        with torch.no_grad():
            for _ in range(n_eval):
                t = float(torch.empty(1).uniform_(-target_range, target_range).item())
                proof, _ = self.generator.generate(t)
                if self.verify.verify_proof(proof):
                    n_valid += 1
        return n_valid / n_eval

    def _evaluate_median_error(self, target_range: float, n_eval: int = 100) -> float:
        """Erreur mediane sur n_eval targets."""
        errs = []
        with torch.no_grad():
            for _ in range(n_eval):
                t = float(torch.empty(1).uniform_(-target_range, target_range).item())
                proof, _ = self.generator.generate(t)
                errs.append(abs(proof.conclusion - proof.target))
        errs.sort()
        return errs[len(errs) // 2] if errs else 0.0

    def train_step(self, target_range: float) -> tuple[float, bool, float]:
        """Une etape REINFORCE with baseline, sur une target in [-range, range].

        Retourne (reward, is_valid, advantage).
        """
        self.optimizer.zero_grad()
        target = float(torch.empty(1).uniform_(-target_range, target_range).item())
        proof, info = self.generator.generate(target)
        is_valid = self.verify.verify_proof(proof)
        reward = shaped_reward(proof, is_valid, self.base_reward)

        # Advantage = reward - baseline (reduit la variance REINFORCE).
        advantage = reward - self.baseline
        # Mise a jour de la baseline (EMA).
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
        """Entraine sur tout le curriculum. Retourne un dict de metriques.

        Metriques :
            initial_error : error mediane a range=5.0 before entrainement.
            final_error   : error mediane a range=5.0 after entrainement.
            initial_valid_rate : taux de validite a range=5.0 before.
            final_valid_rate   : taux de validite a range=5.0 after.
            levels_reached : number de paliers atteints.
        """
        # Evaluer before.
        initial_error = self._evaluate_median_error(5.0)
        initial_valid = self._evaluate_valid_rate(5.0)
        if verbose:
            print(f"Avant entrainement : err_med(±5) = {initial_error:.4f}, "
                  f"valid_rate(±5) = {initial_valid:.1%}")

        levels_reached = 0
        for level_idx, level in enumerate(self.curriculum):
            if verbose:
                print(f"\n--- Palier {level_idx} : targets ±{level.target_range} "
                      f"(objective valid_rate >= {level.min_valid_rate:.0%}) ---")
            for step in range(level.max_steps):
                self.train_step(level.target_range)
                if verbose and (step % 100 == 0 or step == level.max_steps - 1):
                    err = self._evaluate_median_error(level.target_range)
                    vr = self._evaluate_valid_rate(level.target_range, n_eval=50)
                    print(f"  step {step:4d}  err_med(±{level.target_range}) = {err:.4f}  "
                          f"valid_rate = {vr:.1%}  baseline = {self.baseline:.3f}")
            # Evaluation finale du palier.
            final_vr = self._evaluate_valid_rate(level.target_range)
            levels_reached = level_idx + 1
            if verbose:
                status = "OK" if final_vr >= level.min_valid_rate else "~"
                print(f"  -> valid_rate final palier : {final_vr:.1%} {status} "
                      f"(objective {level.min_valid_rate:.0%})")
            # Si on rate largement le palier, on continuous quand meme au suivant
            # (le curriculum reste progressif, on ne bloque pas).

        # Evaluer after.
        final_error = self._evaluate_median_error(5.0)
        final_valid = self._evaluate_valid_rate(5.0)
        if verbose:
            print(f"\nApres entrainement : err_med(±5) = {final_error:.4f}, "
                  f"valid_rate(±5) = {final_valid:.1%}")
            baisse = (1 - final_error / max(initial_error, 1e-9)) * 100
            print(f"Baisse error : {baisse:.1f}%")

        return {
            "initial_error": initial_error,
            "final_error": final_error,
            "initial_valid_rate": initial_valid,
            "final_valid_rate": final_valid,
            "levels_reached": levels_reached,
        }
