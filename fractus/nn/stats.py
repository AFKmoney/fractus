"""Utilitaires numeriques for fractus.

Portes depuis the original architecture (src/math/stats.rs) en PyTorch pur, differentiables.

elu_more_one : feature map strictement positive for linear attention.
    φ(x, α) = x + 1              si x > 0
            = α(e^x - 1) + 1     otherwise
    Avec α=1 (defaut), φ est strictement positive (min e^x > 0 for x→-∞,
    = 1 en x=0). Cette positifast guaranteedt that the denominateur of l'attention
    lineaire causale reste well defini.

stable_softmax : softmax with soustraction max (pas d'overflow).
"""

import torch


def elu_plus_one(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Feature map ELU+1 strictement positive, differentiable.

    Args:
        x : tenseur of shape arbitraire.
        alpha : coefficient ELU (1.0 by defaut, comme the original).
    Returns:
        tenseur of same shape, strictement positif.
    """
    # On utilise the formula directe ( differentiable via torch.where ) :
    # branche positive : x + 1 ; branche negative : alpha * (exp(x) - 1) + 1.
    pos = x + 1.0
    neg = alpha * (torch.exp(x) - 1.0) + 1.0
    return torch.where(x > 0, pos, neg)


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax numeriquement stable (soustraction max).

    Si the somme exponentielles est < 1e-10, returns l'uniforme 1/N
    (comporteddment aux limites herite of the original stats.rs:56-57).
    """
    max_logits, _ = logits.max(dim=dim, keepdim=True)
    exp = torch.exp(logits - max_logits)
    denom = exp.sum(dim=dim, keepdim=True)
    # Comporteddment aux limites : uniforme si denom ~ 0.
    uniform = torch.full_like(exp, 1.0 / exp.shape[dim])
    return torch.where(denom > 1e-10, exp / denom, uniform)
