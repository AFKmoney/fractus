"""Penalite d'acyclicite NOTEARS : h(W) = tr(e^{W⊙W}) − n.

Portee faithfully depuis the original architecture (src/causal.rs:159-196) en PyTorch pur.

Math (Zheng et al. 2018, "DAGs with NO TEARS") :
    h(W) = tr(expm(W ⊙ W)) − n
    ou expm est l'exponentielle matricielle et ⊙ le produit d'Hadamard.

    Propriete : h(W) = 0 ssi W est acyclique (DAG).
    h(W) > 0 si W contient un cycle.
    Differentiable → on can l'optimiser par gradient descent.

Approximation : expm via serie de Taylor a 20 termes (comme FNN).

CORRECTION vs OMNI : OMNI n'avait PAS de contrainte d'acyclicite du tout
(rkhs_causal.py n'imposait no DAG). Ici on a un true NOTEARS differentiable.
"""

import torch


def notears_penalty(W: torch.Tensor, n_terms: int = 20) -> torch.Tensor:
    """Calcule h(W) = tr(e^{W⊙W}) − n, scalar.

    Args:
        W : matrix d'adjacence (n, n), differentiable.
        n_terms : number de termes de la serie de Taylor (20 par defaut).
    Returns:
        h : scalar. =0 si W est un DAG, >0 si W contient un cycle.
    """
    n = W.shape[0]
    assert W.shape == (n, n), f"W must etre carree, eu {W.shape}"

    M = W * W  # produit d'Hadamard (W ⊙ W).

    eye = torch.eye(n, dtype=W.dtype, device=W.device)
    result = eye.clone()
    term = eye.clone()  # term_k = M^k / k!, init a M^0/0! = I
    for k in range(1, n_terms + 1):
        term = (term @ M) / k
        result = result + term
        if term.norm() < 1e-10:
            break

    trace = torch.diagonal(result).sum()
    return trace - n
