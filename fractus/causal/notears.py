"""NOTEARS acyclicity penalty: h(W) = tr(e^{W⊙W}) − n.

Ported faithfully from the original architecture (src/causal.rs) in pure PyTorch.

Math (Zheng et al. 2018, "DAGs with NO TEARS"):
    h(W) = tr(expm(W ⊙ W)) − n
    where expm is the matrix exponential and ⊙ is the Hadamard product.

    Property: h(W) = 0 if and only if W is acyclic (a DAG).
    h(W) > 0 if W contains a cycle.
    Differentiable: we can optimize it via gradient descent.

Approximation: expm via Taylor series with 20 terms.
"""

import torch


def notears_penalty(W: torch.Tensor, n_terms: int = 20) -> torch.Tensor:
    """Compute h(W) = tr(e^{W⊙W}) − n, a scalar.

    Args:
        W: adjacency matrix (n, n), differentiable.
        n_terms: number of Taylor series terms (20 by default).
    Returns:
        h: scalar. =0 if W is a DAG, >0 if W contains a cycle.
    """
    n = W.shape[0]
    assert W.shape == (n, n), f"W must be square, got {W.shape}"

    M = W * W  # Hadamard product (W ⊙ W).

    eye = torch.eye(n, dtype=W.dtype, device=W.device)
    result = eye.clone()
    term = eye.clone()  # term_k = M^k / k!, init to M^0/0! = I
    for k in range(1, n_terms + 1):
        term = (term @ M) / k
        result = result + term
        if term.norm() < 1e-10:
            break

    trace = torch.diagonal(result).sum()
    return trace - n
