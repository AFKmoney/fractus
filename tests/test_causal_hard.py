"""Tests du SCM non-linéaire + validation NOTEARS au-delà du cas jouet."""

import torch


def test_nonlinear_scm_shape():
    """generate_nonlinear_scm retourne W (n,n) et X (samples, n)."""
    from data.causal.generate_scm_hard import generate_nonlinear_scm
    W, X = generate_nonlinear_scm(n_vars=5, n_samples=100, seed=42)
    assert W.shape == (5, 5)
    assert X.shape == (100, 5)


def test_nonlinear_scm_is_dag():
    """W_true doit être acyclique (c'est un DAG par construction)."""
    from data.causal.generate_scm_hard import generate_nonlinear_scm
    from fractus.causal.notears import notears_penalty
    W, _ = generate_nonlinear_scm(n_vars=5, n_samples=100, seed=42)
    h = notears_penalty(W)
    assert abs(h.item()) < 1e-3, f"W_true devrait être DAG (h≈0), eu {h.item()}"


def test_nonlinear_scm_not_triangular():
    """W_true ne doit PAS être triangulaire (ordre topo caché)."""
    from data.causal.generate_scm_hard import generate_nonlinear_scm
    # Sur plusieurs seeds, au moins un W doit être non-triangulaire.
    found_nontrian = False
    for seed in range(20):
        W, _ = generate_nonlinear_scm(n_vars=5, n_samples=10, seed=seed)
        # Triangulaire supérieur strict : W[i,j] != 0 implique i < j.
        # Non-triangulaire : existe i > j avec W[i,j] != 0.
        for i in range(5):
            for j in range(i):
                if W[i, j] != 0:
                    found_nontrian = True
                    break
    assert found_nontrian, "Aucun W non-triangulaire trouvé en 20 seeds"


def test_notears_recovers_nonlinear_dag():
    """CRITÈRE L4+ : NOTEARS récupère un DAG non-linéaire à ordre inconnu (SHD <= 3).

    C'est la validation au-delà du cas jouet L4 (linéaire + triangulaire).
    """
    from data.causal.generate_scm_hard import generate_nonlinear_scm
    from fractus.causal.notears import notears_penalty
    from fractus.metrics.causal import structural_hamming_distance
    torch.manual_seed(42)
    W_true, X = generate_nonlinear_scm(n_vars=5, n_samples=2000, edge_prob=0.5, seed=11)
    n_vars = W_true.shape[0]
    W_pred = torch.zeros(n_vars, n_vars, requires_grad=True)
    torch.nn.init.normal_(W_pred, std=0.1)
    opt = torch.optim.Adam([W_pred], lr=0.05)
    for _ in range(800):
        opt.zero_grad()
        X_pred = X @ W_pred
        recon = ((X_pred - X) ** 2).mean()
        h = notears_penalty(W_pred)
        loss = recon + h.abs()
        loss.backward()
        opt.step()
    shd = structural_hamming_distance(W_true, W_pred.detach(), threshold=0.3)
    assert shd <= 3, f"NOTEARS devrait récupérer le DAG non-linéaire (SHD<=3), eu {shd}"
