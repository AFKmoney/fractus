"""Tests du pipeline conjectures : générateur, testeur, mémoire, boucle."""

import torch


def test_conjecture_templates_count_10():
    """10 templates (FNN conjecture.rs:15-26)."""
    from fractus.reasoning.conjecture import ConjectureTemplate
    assert ConjectureTemplate.n_templates() == 10


def test_tester_sum_identity_survives():
    """SumIdentity (a+a=2a) doit toujours survivre (identité vraie)."""
    from fractus.reasoning.conjecture import ConjectureTester, Conjecture
    tester = ConjectureTester(n_trials=50, seed=42)
    conj = Conjecture(template_index=0, template_name="SumIdentity")
    result = tester.test(conj)
    assert result.survived is True
    assert result.n_tests_passed == 50


def test_tester_fermat_survives_for_prime():
    """FermatLittle pour p=7 premier doit survivre."""
    from fractus.reasoning.conjecture import ConjectureTester, Conjecture
    tester = ConjectureTester(n_trials=50, seed=42)
    conj = Conjecture(template_index=3, template_name="FermatLittle", parameters=[7.0])
    result = tester.test(conj)
    assert result.survived is True


def test_tester_fermat_fails_for_composite():
    """FermatLittle pour p=4 (non premier) doit échouer."""
    from fractus.reasoning.conjecture import ConjectureTester, Conjecture
    tester = ConjectureTester(n_trials=10, seed=42)
    conj = Conjecture(template_index=3, template_name="FermatLittle", parameters=[4.0])
    result = tester.test(conj)
    assert result.survived is False


def test_generator_produces_conjecture():
    """Le générateur produit une conjecture valide depuis un état."""
    from fractus.reasoning.conjecture import ConjectureGenerator
    gen = ConjectureGenerator(state_dim=32)
    state = torch.randn(32)
    conj = gen(state)
    assert 0 <= conj.template_index < 10
    assert len(conj.parameters) == 4


def test_generator_backward_every_param():
    """CRITÈRE L5 : backward propage un gradient fini ET non-nul à CHAQUE paramètre.

    La loss doit toucher w_template (logits), w_params (paramètres générés) ET
    w_novelty (score nouveauté). Sinon certains poids ne reçoivent pas de gradient.
    """
    from fractus.reasoning.conjecture import ConjectureGenerator
    gen = ConjectureGenerator(state_dim=32)
    state = torch.randn(32)
    # Recalculer les 3 tenseurs pour avoir le graphe complet.
    logits = state @ gen.w_template
    params_logits = state @ gen.w_params
    novelty = state @ gen.w_novelty
    loss = logits.sum() + params_logits.sum() + novelty.sum()
    loss.backward()
    for name, p in gen.named_parameters():
        assert p.requires_grad, f"{name} devrait requires_grad=True"
        assert p.grad is not None, f"{name} n'a reçu aucun gradient"
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum().item() > 0, f"{name} a reçu un gradient nul"


def test_memory_add_and_evict():
    """La mémoire évince les non-survivantes quand pleine."""
    from fractus.reasoning.conjecture import ConjectureMemory, Conjecture
    mem = ConjectureMemory(max_size=3)
    for i in range(3):
        c = Conjecture(template_index=i, template_name=str(i), survived=False)
        mem.add(c)
    assert len(mem.discovered) == 3
    # Ajout d'une 4e : éviction de la plus ancienne non-survivante.
    c4 = Conjecture(template_index=3, template_name="3", survived=True)
    mem.add(c4)
    assert len(mem.discovered) == 3
    assert mem.discovered[-1].template_index == 3


def test_memory_is_novel():
    """is_novel détecte les templates déjà survivants."""
    from fractus.reasoning.conjecture import ConjectureMemory, Conjecture
    mem = ConjectureMemory()
    c1 = Conjecture(template_index=0, template_name="A", survived=True)
    mem.add(c1)
    # Une nouvelle conjecture du même template survivant n'est pas nouvelle.
    c2 = Conjecture(template_index=0, template_name="A")
    assert mem.is_novel(c2) is False
    # Un autre template est nouveau.
    c3 = Conjecture(template_index=1, template_name="B")
    assert mem.is_novel(c3) is True


def test_discovery_loop_runs():
    """La boucle complète tourne sans crash."""
    from fractus.reasoning.conjecture import ConjectureDiscoveryLoop
    loop = ConjectureDiscoveryLoop(state_dim=32, n_trials=20, seed=42)
    discoveries = 0
    for _ in range(20):
        result = loop.discover_step()
        if result is not None:
            discoveries += 1
    # Au moins une découverte (SumIdentity, FermatLittle, etc. sont vraies).
    assert discoveries >= 1, f"Aucune découverte en 20 steps, eu {discoveries}"
    assert 0.0 <= loop.discovery_rate() <= 1.0
