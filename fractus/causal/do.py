"""do_intervention : vrai do-calculus de Pearl.

CORRECTION DU FAUX DO-CALCULUS D'OMNI :
- OMNI (rkhs_causal.py:21-25) faisait 'intervened[:, do_mask] = 0.0' — juste
  mettre la colonne à 0. Ce n'est PAS do-calculus.
- Ici : do(X_i = v) fixe X_i à v pour tous les échantillons (intervention
  Pearl), ce qui permet de comparer P(Y | do(X=v)) vs P(Y | X=v).

Différentiable (pour estimer l'effet causal par gradient quand le modèle
est différentiable).
"""

import torch


def do_intervention(
    x: torch.Tensor, var_idx: int, value: float
) -> torch.Tensor:
    """Applique do(X_{var_idx} = value) à un batch de données.

    Args:
        x       : tenseur (N, d) de variables observées.
        var_idx : indice de la variable à intervenir.
        value   : valeur à imposer (peut être non-nulle — c'est l'intervention).
    Returns:
        x_intervened : (N, d) avec la colonne var_idx mise à `value`.
    """
    x_intervened = x.clone()
    x_intervened[:, var_idx] = value
    return x_intervened
