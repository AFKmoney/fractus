"""Pénalité d'acyclicité NOTEARS : h(W) = tr(e^{W⊙W}) − n.

Portée fidèlement depuis FNN v5.0 (src/causal.rs:159-196) en PyTorch pur.

Math (Zheng et al. 2018, "DAGs with NO TEARS") :
    h(W) = tr(expm(W ⊙ W)) − n
    où expm est l'exponentielle matricielle et ⊙ le produit d'Hadamard.

    Propriété : h(W) = 0 ssi W est acyclique (DAG).
    h(W) > 0 si W contient un cycle.
    Différentiable → on peut l'optimiser par gradient descent.

Approximation : expm via série de Taylor à 20 termes (comme FNN).

CORRECTION vs OMNI : OMNI n'avait PAS de contrainte d'acyclicité du tout
(rkhs_causal.py n'imposait aucun DAG). Ici on a un vrai NOTEARS différentiable.
"""

import torch


def notears_penalty(W: torch.Tensor, n_terms: int = 20) -> torch.Tensor:
    """Calcule h(W) = tr(e^{W⊙W}) − n, scalaire.

    Args:
        W : matrice d'adjacence (n, n), differentiable.
        n_terms : nombre de termes de la série de Taylor (20 par défaut).
    Returns:
        h : scalaire. =0 si W est un DAG, >0 si W contient un cycle.
    """
    n = W.shape[0]
    assert W.shape == (n, n), f"W doit être carrée, eu {W.shape}"

    M = W * W  # produit d'Hadamard (W ⊙ W).

    eye = torch.eye(n, dtype=W.dtype, device=W.device)
    result = eye.clone()
    term = eye.clone()  # term_k = M^k / k!, init à M^0/0! = I
    for k in range(1, n_terms + 1):
        term = (term @ M) / k
        result = result + term
        if term.norm() < 1e-10:
            break

    trace = torch.diagonal(result).sum()
    return trace - n
