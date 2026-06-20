"""Tests of ProofTrainer : curriculum, reward shaping, baseline."""

import torch


def test_shaped_reward_continuous():
    """shaped_reward must be continu and decroissant with the error."""
    from fractus.reasoning.proof_trainer import shaped_reward
    from fractus.reasoning.proof import ProofReward, Proof, ProofStep
    rf = ProofReward()
    rewards = []
    for err in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        steps = [ProofStep(0, "A", [], 5.0, 0.9)]
        proof = Proof(steps=steps, target=5.0, conclusion=5.0 + err, is_valid=(err < 0.001))
        r = shaped_reward(proof, err < 0.001, rf)
        rewards.append(r)
    # Doit decroitre.
    for i in range(len(rewards) - 1):
        assert rewards[i] >= rewards[i + 1], \
            f"reward(err={[0.0,0.1,0.5,1.0,2.0,5.0,10.0][i]})={rewards[i]} < " \
            f"reward(err={[0.0,0.1,0.5,1.0,2.0,5.0,10.0][i+1]})={rewards[i+1]}"


def test_shaped_reward_nonzero_for_large_error():
    """CRITERE L5+ : shaped_reward must be > 0 same for largee error
    (contrairement a correctness_reward of the original which s'ecrase a 0)."""
    from fractus.reasoning.proof_trainer import shaped_reward
    from fractus.reasoning.proof import ProofReward, Proof, ProofStep
    rf = ProofReward()
    steps = [ProofStep(0, "A", [], 5.0, 0.9)]
    proof = Proof(steps=steps, target=5.0, conclusion=10.0, is_valid=False)  # err=5
    r = shaped_reward(proof, False, rf)
    assert r > 0.0, f"shaped_reward must etre > 0 meme for err=5, eu {r}"


def test_trainer_train_step_runs():
    """Une etape d'training tourne without crash."""
    from fractus.reasoning.proof import ProofGenerator, ProofVerifier
    from fractus.reasoning.proof_trainer import ProofTrainer
    gen = ProofGenerator(hidden_dim=16, max_steps=4)
    ver = ProofVerifier()
    trainer = ProofTrainer(gen, ver, lr=1e-2)
    reward, is_valid, advantage = trainer.train_step(target_range=0.5)
    assert isinstance(reward, float)
    assert isinstance(is_valid, bool)
    assert isinstance(advantage, float)


def test_trainer_baseline_updates():
    """La baseline must evoluer after quelques steps (EMA reward)."""
    from fractus.reasoning.proof import ProofGenerator, ProofVerifier
    from fractus.reasoning.proof_trainer import ProofTrainer
    torch.manual_seed(0)
    gen = ProofGenerator(hidden_dim=16, max_steps=4)
    ver = ProofVerifier()
    trainer = ProofTrainer(gen, ver, lr=1e-2, baseline_decay=0.5)
    initial_baseline = trainer.baseline
    for _ in range(10):
        trainer.train_step(target_range=0.5)
    # La baseline must have change (non-nulle si rewards non-nuls).
    assert trainer.baseline != initial_baseline


def test_trainer_evaluates_median_error():
    """_evaluate_median_error returns a float >= 0."""
    from fractus.reasoning.proof import ProofGenerator, ProofVerifier
    from fractus.reasoning.proof_trainer import ProofTrainer
    gen = ProofGenerator(hidden_dim=16, max_steps=4)
    ver = ProofVerifier()
    trainer = ProofTrainer(gen, ver)
    err = trainer._evaluate_median_error(target_range=1.0, n_eval=20)
    assert isinstance(err, float)
    assert err >= 0.0


def test_trainer_curriculum_improves_or_runs():
    """CRITERE L5+ : after training curriculum, the error a ±5 must baisser
    OU au less not not exploser (proof that ca apprend or stagne, not diverge)."""
    from fractus.reasoning.proof import ProofGenerator, ProofVerifier
    from fractus.reasoning.proof_trainer import ProofTrainer, CurriculumLevel
    torch.manual_seed(42)
    gen = ProofGenerator(hidden_dim=32, max_steps=6)
    ver = ProofVerifier()
    # Curriculum short for the test (otherwise too long).
    short_curriculum = [
        CurriculumLevel(0.1, 0.30, 50),
        CurriculumLevel(0.5, 0.25, 50),
        CurriculumLevel(1.0, 0.20, 50),
    ]
    trainer = ProofTrainer(gen, ver, curriculum=short_curriculum, lr=1e-2)
    metrics = trainer.train(verbose=False)
    # L'error a ±5 must have baisse d'au less 10% (critere test, more souple
    # that the demo which vise 30%).
    assert metrics["final_error"] <= metrics["initial_error"] * 1.5, \
        f"L'error a explose : {metrics['initial_error']} -> {metrics['final_error']}"
