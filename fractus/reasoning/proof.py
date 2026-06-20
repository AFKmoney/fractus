"""Pipeline de preuves : neural propose, exact verify disposes.

Porte depuis the original architecture (src/proof.rs) en PyTorch pur.

Architecture :
    ProofGenerator : GRU autoregressif qui produit une sequence de ProofStep.
        Chaque step : choisit une regle (argmax sur logits) + predit une valeur
        scalar (conclusion progressive vers la cible).
    ProofVerifier : VERIFICATION EXACTE. N'utilise PAS les regles d'inference —
        teste uniquement des identites arithmetiques concretes (addition,
        primalite, divisibilite, Fermat, Wilson, GCD, modulaire). Soundness
        garantie : si le verify dit "valide", c'est mathematiquement true.
    ProofReward : R = 0.6·correctness + 0.3·efficiency + 0.1·diversity.

CORRECTION vs FNN : FNN n'avait pas d'autodiff (training.rs:399 = bruit). Ici
le ProofGenerator est entrainable par REINFORCE (policy gradient) sur la
recompense du verify. Voir demo L5.

HONNETETE : le verify ne verifies que la CONCLUSION numerique (|conclusion - target|<1e-3),
pas la structure logical de la preuve. C'est faithful a FNN (proof.rs:341-360).
Une preuve "acceptee" est therefore garantie d'atteindre la bonne valeur numerique,
but pas d'etre une derivation logiquement valide etape par etape.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from ..math.primes import PrimeSieve


# ---------------------------------------------------------------------------
# Regles d'inference (les 20 regles de FNN proof.rs:14-36)
# ---------------------------------------------------------------------------

class InferenceRule:
    """Labels des 20 regles d'inference (comme FNN proof.rs:14-36).

    NOTE : ce ne sont que des labels. Le generateur predit un index de regle,
    but le verify ne lit PAS la regle — il verifies uniquement la
    conclusion numerique. Fidele a FNN.
    """

    NAMES = [
        "AddBothSides",       # a=b → a+c=b+c
        "MultiplyBothSides",  # a=b → a*c=b*c
        "Distributive",       # a*(b+c) = a*b + a*c
        "Commutative",        # a+b = b+a
        "Associative",        # (a+b)+c = a+(b+c)
        "Transitive",         # a=b, b=c → a=c
        "Substitution",       # replace variable with value
        "Reflexive",          # a=a
        "Symmetric",          # a=b → b=a
        "FermatLittle",       # a^(p-1) ≡ 1 mod p
        "WilsonTheorem",      # (p-1)! ≡ -1 mod p
        "EuclidGCD",          # gcd(a,b) = gcd(b, a mod b)
        "Divisibility",       # a|b et a|c → a|(b+c)
        "ModularArithmetic",  # a ≡ b mod n → a+c ≡ b+c mod n
        "Contradiction",      # assume ¬P, derive contradiction → P
        "DoubleNegation",     # ¬(¬P) → P
        "ModusPonens",        # P, P→Q → Q
        "UniversalInstant",   # ∀x:P(x) → P(a)
        "ExistentialIntro",   # P(a) → ∃x:P(x)
        "ProofByInduction",   # cas de base + pas inductif
    ]

    @classmethod
    def n_rules(cls) -> int:
        return len(cls.NAMES)

    @classmethod
    def name(cls, idx: int) -> str:
        return cls.NAMES[idx]


def all_rules() -> List[str]:
    """Retourne la liste des noms des 20 regles."""
    return list(InferenceRule.NAMES)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProofStep:
    """Une etape de preuve (comme FNN proof.rs:73-79)."""
    rule_index: int
    rule_name: str
    premise_indices: List[int]
    conclusion: float
    confidence: float


@dataclass
class Proof:
    """Une preuve complete (comme FNN proof.rs:83-88)."""
    steps: List[ProofStep] = field(default_factory=list)
    target: float = 0.0
    conclusion: float = 0.0
    is_valid: bool = False


# ---------------------------------------------------------------------------
# ProofGenerator : GRU autoregressif (FNN proof.rs:91-260)
# ---------------------------------------------------------------------------

class ProofGenerator(nn.Module):
    """Generateur de preuves GRU autoregressif.

    Produit une sequence de ProofStep : a each step, choisit une regle
    (argmax sur logits) et predit une valeur scalar (conclusion progressive
    via EMA 0.8/0.2 vers la cible).

    Args:
        hidden_dim : dimension de l'etat cache du GRU (32 par defaut, FNN).
        n_rules    : number de regles (= 20, InferenceRule.NAMES).
        max_steps  : number d'etapes de preuve generees (6 par defaut, FNN).
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        n_rules: int = None,
        max_steps: int = 6,
    ):
        super().__init__()
        if n_rules is None:
            n_rules = InferenceRule.n_rules()
        self.hidden_dim = hidden_dim
        self.n_rules = n_rules
        self.max_steps = max_steps

        # Poids (comme FNN proof.rs:108-141). Init Xavier-ish uniform.
        scale_rule = math.sqrt(2.0 / (hidden_dim + n_rules))
        self.w_rule = nn.Parameter(torch.empty(hidden_dim, n_rules).uniform_(-scale_rule, scale_rule))
        scale_value = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_value = nn.Parameter(torch.empty(hidden_dim).uniform_(-scale_value, scale_value))

        # GRU : input = [hidden ; scalar_input] → hidden. Poids [hidden_dim, hidden_dim+1].
        gru_scale = math.sqrt(2.0 / (hidden_dim + hidden_dim + 1))
        self.w_update = nn.Parameter(torch.empty(hidden_dim, hidden_dim + 1).uniform_(-gru_scale, gru_scale))
        self.w_reset = nn.Parameter(torch.empty(hidden_dim, hidden_dim + 1).uniform_(-gru_scale, gru_scale))
        self.w_candidate = nn.Parameter(torch.empty(hidden_dim, hidden_dim + 1).uniform_(-gru_scale, gru_scale))

    def _gru_step(self, hidden: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        """Un pas GRU (comme FNN proof.rs:210-232). Pas de biais separe.

        hidden : (hidden_dim,). x_scalar : scalar.
        """
        # input = concat([hidden, x_scalar]) → (hidden_dim+1,).
        x = torch.cat([hidden, x_scalar.reshape(1)])
        z = torch.sigmoid(self.w_update @ x)
        r = torch.sigmoid(self.w_reset @ x)
        # candidate : combined = concat([r ⊙ hidden, x_scalar]).
        combined = torch.cat([r * hidden, x_scalar.reshape(1)])
        h_tilde = torch.tanh(self.w_candidate @ combined)
        return (1 - z) * hidden + z * h_tilde

    def generate(self, target: float) -> tuple[Proof, dict]:
        """Genere une preuve for atteindre `target`.

        Retourne (proof, info) ou info contient les tenseurs for REINFORCE :
            info['logits_per_step'] : liste de (n_rules,) logits par step.
            info['selected_indices'] : liste d'ints (regle choisie par step).
        """
        device = self.w_rule.device
        steps: List[ProofStep] = []
        current_value = float(target)

        # Init etat cache : hidden[i] = target · sin(i+1) · 0.1 (FNN proof.rs:149-151).
        idx = torch.arange(1, self.hidden_dim + 1, dtype=torch.float32, device=device)
        hidden = torch.tensor(target, dtype=torch.float32, device=device) * torch.sin(idx) * 0.1

        logits_per_step = []
        selected_indices = []
        value_preds_per_step = []

        for step_idx in range(self.max_steps):
            # input = [hidden ; current_value · 0.1].
            x_scalar = torch.tensor(current_value * 0.1, dtype=torch.float32, device=device)
            hidden = self._gru_step(hidden, x_scalar)

            # logits = hidden @ w_rule.
            logits = hidden @ self.w_rule  # (n_rules,)
            rule_index = int(logits.argmax().item())

            value_pred = (hidden @ self.w_value)  # tenseur scalar, for gradient

            premise_indices = [step_idx - 1] if step_idx > 0 else []

            # confidence = softmax(logits)[rule_index] (stable).
            max_logit = logits.max()
            exps = torch.exp(logits - max_logit)
            sum_exp = exps.sum()
            if sum_exp > 1e-10:
                confidence = float((exps[rule_index] / sum_exp).item())
            else:
                confidence = 1.0 / self.n_rules

            # EMA 0.8/0.2 vers value_pred (FNN proof.rs:189).
            value_pred_float = float(value_pred.item())
            current_value = 0.8 * current_value + 0.2 * value_pred_float

            steps.append(ProofStep(
                rule_index=rule_index,
                rule_name=InferenceRule.name(rule_index),
                premise_indices=premise_indices,
                conclusion=current_value,
                confidence=confidence,
            ))
            logits_per_step.append(logits)
            selected_indices.append(rule_index)
            value_preds_per_step.append(value_pred)

        is_valid = abs(current_value - target) < 1e-3
        proof = Proof(steps=steps, target=float(target), conclusion=current_value, is_valid=is_valid)
        info = {
            "logits_per_step": logits_per_step,
            "selected_indices": selected_indices,
            "value_preds_per_step": value_preds_per_step,
        }
        return proof, info


# ---------------------------------------------------------------------------
# ProofVerifier : VERIFICATION EXACTE (FNN proof.rs:263-360)
# ---------------------------------------------------------------------------

def _mod_pow(base: int, exp: int, modulus: int) -> int:
    """Exponentiation modulaire rapide par carres (FNN proof.rs:364-378)."""
    if modulus == 1:
        return 0
    result = 1
    base = base % modulus
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % modulus
        exp //= 2
        base = (base * base) % modulus
    return result


def _gcd(a: int, b: int) -> int:
    """Euclide iteratif (FNN proof.rs:381-388)."""
    while b:
        a, b = b, a % b
    return abs(a)


class ProofVerifier:
    """Verificateur EXACT de preuves (soundness garantie).

    N'utilise PAS les regles d'inference — teste des identites arithmetiques
    concretes. Toute preuve acceptee est mathematiquement vraie sur la
    conclusion numerique (|conclusion - target| < 1e-3).
    """

    def __init__(self, sieve_limit: int = 10000):
        self.sieve = PrimeSieve(sieve_limit)

    def verify_arithmetic(self, a: int, b: int, claimed_sum: int) -> bool:
        """Verifie a + b == claimed_sum (FNN proof.rs:275-277)."""
        return a + b == claimed_sum

    def verify_primality(self, n: int, is_prime: bool) -> bool:
        """Verifie que n est premier ssi is_prime (FNN proof.rs:280-282)."""
        return self.sieve.verify_prime(n) == is_prime

    def verify_divisibility(self, a: int, b: int) -> bool:
        """Verifie que b divise a (FNN proof.rs:285-290)."""
        if b == 0:
            return False
        return a % b == 0

    def verify_modular(self, a: int, b: int, n: int) -> bool:
        """Verifie a ≡ b mod n (FNN proof.rs:293-298)."""
        if n == 0:
            return False
        return (a - b) % n == 0

    def verify_fermat(self, a: int, p: int) -> bool:
        """Verifie Fermat petit : a^(p-1) ≡ 1 mod p for p premier et a non divisible par p."""
        if p < 2 or not self.sieve.verify_prime(p):
            return False
        if a % p == 0:
            return True  # cas trivial
        return _mod_pow(a, p - 1, p) == 1

    def verify_wilson(self, p: int) -> bool:
        """Verifie Wilson : (p-1)! ≡ p-1 mod p for p premier."""
        if p < 2 or not self.sieve.verify_prime(p):
            return False
        if p == 2:
            return True
        fact_mod = 1
        for i in range(1, p):
            fact_mod = (fact_mod * (i % p)) % p
        return fact_mod == p - 1

    def verify_gcd(self, a: int, b: int) -> bool:
        """Verifie l'identite gcd(a,b)·lcm(a,b) = a·b (FNN proof.rs:329-338)."""
        g = _gcd(a, b)
        if a == 0 or b == 0:
            return g == max(a, b)
        lcm = (a // g) * b  # ordre anti-overflow
        return g * lcm == a * b

    def verify_proof(self, proof: Proof) -> bool:
        """Verifie globalement une preuve (FNN proof.rs:341-360).

        Critere : (a) all les rule_index sont valides (< n_rules), et
        (b) |conclusion - target| < 1e-3.
        """
        if not proof.steps:
            return False
        n_rules = InferenceRule.n_rules()
        for step in proof.steps:
            if step.rule_index < 0 or step.rule_index >= n_rules:
                return False
        return abs(proof.conclusion - proof.target) < 1e-3


# ---------------------------------------------------------------------------
# ProofReward : R = 0.6·correctness + 0.3·efficiency + 0.1·diversity (FNN proof.rs:391-456)
# ---------------------------------------------------------------------------

class ProofReward:
    """Recompense composite for REINFORCE.

    R = 0.6·correctness + 0.3·efficiency + 0.1·diversity.
    """

    def __init__(
        self,
        correctness_weight: float = 0.6,
        efficiency_weight: float = 0.3,
        diversity_weight: float = 0.1,
    ):
        self.correctness_weight = correctness_weight
        self.efficiency_weight = efficiency_weight
        self.diversity_weight = diversity_weight

    def correctness_reward(self, proof: Proof, is_valid: bool) -> float:
        """1.0 si valide, sinon decroit with the error (FNN proof.rs:417-426)."""
        if is_valid:
            return 1.0
        error = abs(proof.conclusion - proof.target)
        max_error = max(abs(proof.target), 1.0)
        return max(0.0, 1.0 - min(error / max_error, 1.0))

    def efficiency_reward(self, proof: Proof) -> float:
        """1.0 si 1 step, sinon 1/n_steps (FNN proof.rs:429-438). 'Brevity'."""
        n = len(proof.steps)
        if n <= 1:
            return 1.0
        return 1.0 / n

    def diversity_reward(self, proof: Proof) -> float:
        """Fraction des 20 regles distinctes utilisees (FNN proof.rs:441-456). 'Novelty'."""
        if not proof.steps:
            return 0.0
        n_rules = InferenceRule.n_rules()
        used = [False] * n_rules
        for step in proof.steps:
            if 0 <= step.rule_index < n_rules:
                used[step.rule_index] = True
        n_unique = sum(used)
        return min(n_unique / n_rules, 1.0)

    def compute_reward(self, proof: Proof, is_valid: bool) -> float:
        """R = 0.6·correctness + 0.3·efficiency + 0.1·diversity."""
        c = self.correctness_reward(proof, is_valid)
        e = self.efficiency_reward(proof)
        d = self.diversity_reward(proof)
        return (
            self.correctness_weight * c
            + self.efficiency_weight * e
            + self.diversity_weight * d
        )
