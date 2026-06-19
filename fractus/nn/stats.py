"""Utilitaires numériques pour fractus.

Portés depuis FNN v5.0 (src/math/stats.rs) en PyTorch pur, différentiables.

elu_plus_one : feature map strictement positive pour linear attention.
    φ(x, α) = x + 1              si x > 0
            = α(e^x - 1) + 1     sinon
    Avec α=1 (défaut), φ est strictement positive (min e^x > 0 pour x→-∞,
    = 1 en x=0). Cette positivité garantit que le dénominateur de l'attention
    linéaire causale reste bien défini.

stable_softmax : softmax avec soustraction du max (pas d'overflow).
"""

import torch


def elu_plus_one(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Feature map ELU+1 strictement positive, différentiable.

    Args:
        x : tenseur de forme arbitraire.
        alpha : coefficient ELU (1.0 par défaut, comme FNN).
    Returns:
        tenseur de même forme, strictement positif.
    """
    # On utilise la formule directe ( différentiable via torch.where ) :
    # branche positive : x + 1 ; branche négative : alpha * (exp(x) - 1) + 1.
    pos = x + 1.0
    neg = alpha * (torch.exp(x) - 1.0) + 1.0
    return torch.where(x > 0, pos, neg)


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax numériquement stable (soustraction du max).

    Si la somme des exponentielles est < 1e-10, retourne l'uniforme 1/N
    (comportement aux limites hérité de FNN stats.rs:56-57).
    """
    max_logits, _ = logits.max(dim=dim, keepdim=True)
    exp = torch.exp(logits - max_logits)
    denom = exp.sum(dim=dim, keepdim=True)
    # Comportement aux limites : uniforme si denom ~ 0.
    uniform = torch.full_like(exp, 1.0 / exp.shape[dim])
    return torch.where(denom > 1e-10, exp / denom, uniform)
