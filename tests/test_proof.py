"""Tests du pipeline preuves : generateur, verify exact, recompense."""

import torch
import pytest


# --- InferenceRule ---

def test_inference_rules_count_20():
    """Il y a exactement 20 regles (FNN proof.rs:14-36)."""
    from fractus.reasoning.proof import InferenceRule, all_rules
    assert InferenceRule.n_rules() == 20
    assert len(all_rules()) == 20


def test_inference_rule_names_consistent():
    """Index et nom coherents."""
    from fractus.reasoning.proof import InferenceRule
    assert InferenceRule.name(0) == "AddBothSides"
    assert InferenceRule.name(19) == "ProofByInduction"


# --- ProofGenerator ---

def test_generator_produces_proof():
    """generate(target) returns une preuve with max_steps etapes."""
    from fractus.reasoning.proof import ProofGenerator
    gen = ProofGenerator(hidden_dim=32, max_steps=6)
    proof, info = gen.generate(target=5.0)
    assert len(proof.steps) == 6
    assert proof.target == 5.0
    # Chaque step a un rule_index valide.
    for step in proof.steps:
        assert 0 <= step.rule_index < 20


def test_generator_backward_every_param():
    """CRITERE L5 : backward propage un gradient fini ET non-nul a CHAQUE parameter.

    La loss must toucher A LA FOIS les logits de regles (w_rule) ET les valeurs
    predites (w_value), sinon w_value ne recoit pas de gradient. On construit
    therefore une loss qui combine les deux : REINFORCE-like.
    """
    from fractus.reasoning.proof import ProofGenerator
    gen = ProofGenerator(hidden_dim=32, max_steps=6)
    proof, info = gen.generate(target=3.0)
    # Loss = somme des logits (touche w_rule) + somme des value_preds (touche w_value).
    # Les deux sont des tenseurs in le graphe autodiff.
    loss = sum(logits.sum() for logits in info["logits_per_step"])
    loss = loss + sum(vp for vp in info["value_preds_per_step"])
    loss.backward()

    for name, p in gen.named_parameters():
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} n'a recu no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} a un gradient non-fini"
        assert p.grad.abs().sum().item() > 0, f"{name} a recu un gradient nul"


# --- ProofVerifier ---

def test_verify_arithmetic_correct():
    from fractus.reasoning.proof import ProofVerifier
    v = ProofVerifier()
    assert v.verify_arithmetic(2, 3, 5) is True
    assert v.verify_arithmetic(2, 3, 6) is False


def test_verify_primality():
    from fractus.reasoning.proof import ProofVerifier
    v = ProofVerifier()
    assert v.verify_primality(7, True) is True
    assert v.verify_primality(8, True) is False
    assert v.verify_primality(8, False) is True


def test_verify_fermat():
    """Fermat petit : 2^(7-1) ≡ 1 mod 7."""
    from fractus.reasoning.proof import ProofVerifier
    v = ProofVerifier()
    assert v.verify_fermat(2, 7) is True
    assert v.verify_fermat(3, 7) is True


def test_verify_wilson():
    """Wilson : 6! ≡ 6 mod 7."""
    from fractus.reasoning.proof import ProofVerifier
    v = ProofVerifier()
    assert v.verify_wilson(7) is True
    assert v.verify_wilson(4) is False  # 4 non premier


def test_verify_gcd():
    from fractus.reasoning.proof import ProofVerifier
    v = ProofVerifier()
    assert v.verify_gcd(12, 18) is True  # gcd=6, lcm=36, 6*36=216=12*18


def test_verify_rejects_invalid_proof():
    """Une preuve with conclusion loin de la target must etre rejetee."""
    from fractus.reasoning.proof import ProofVerifier, Proof, ProofStep
    v = ProofVerifier()
    steps = [ProofStep(rule_index=0, rule_name="AddBothSides", premise_indices=[],
                       conclusion=99.0, confidence=0.5)]
    proof = Proof(steps=steps, target=5.0, conclusion=99.0, is_valid=False)
    assert v.verify_proof(proof) is False


def test_verify_accepts_valid_proof():
    """Une preuve dont la conclusion atteint la target must etre acceptee.

    Note : le seuil 1e-3 est strict (<, pas <=), therefore |conclusion - target| must
    etre STRICTEMENT inferieur a 0.001. On utilise 0.0005 for etre sur.
    """
    from fractus.reasoning.proof import ProofVerifier, Proof, ProofStep
    v = ProofVerifier()
    steps = [ProofStep(rule_index=0, rule_name="AddBothSides", premise_indices=[],
                       conclusion=5.0005, confidence=0.9)]
    proof = Proof(steps=steps, target=5.0, conclusion=5.0005, is_valid=True)
    assert v.verify_proof(proof) is True


# --- ProofReward ---

def test_reward_valid_proof_high():
    """Une preuve valide a une recompense elevee."""
    from fractus.reasoning.proof import ProofReward, Proof, ProofStep
    r = ProofReward()
    steps = [ProofStep(rule_index=0, rule_name="AddBothSides", premise_indices=[],
                       conclusion=5.0, confidence=0.9)]
    proof = Proof(steps=steps, target=5.0, conclusion=5.0, is_valid=True)
    reward = r.compute_reward(proof, is_valid=True)
    assert reward > 0.8  # 0.6 + 0.3 + 0.1·(1/20)


def test_reward_diversity_increases_with_rules():
    """Plus de regles distinctes → recompense diversity plus haute."""
    from fractus.reasoning.proof import ProofReward, Proof, ProofStep
    r = ProofReward()
    steps1 = [ProofStep(rule_index=0, rule_name="A", premise_indices=[], conclusion=5.0, confidence=1.0)]
    steps20 = [ProofStep(rule_index=i, rule_name=str(i), premise_indices=[], conclusion=5.0, confidence=1.0)
               for i in range(20)]
    proof1 = Proof(steps=steps1, target=5.0, conclusion=5.0, is_valid=True)
    proof20 = Proof(steps=steps20, target=5.0, conclusion=5.0, is_valid=True)
    assert r.diversity_reward(proof20) > r.diversity_reward(proof1)


def test_reward_weights_sum_to_one():
    """Les poids par defaut somment a 1.0 (0.6 + 0.3 + 0.1)."""
    from fractus.reasoning.proof import ProofReward
    r = ProofReward()
    total = r.correctness_weight + r.efficiency_weight + r.diversity_weight
    assert abs(total - 1.0) < 1e-6
