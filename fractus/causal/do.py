"""do_intervention : primitive d'intervention atomique (opérateur do de Pearl).

CORRECTION DU FAUX DO-CALCULUS D'OMNI :
- OMNI (rkhs_causal.py:21-25) faisait 'intervened[:, do_mask] = 0.0' — juste
  mettre la colonne à 0. Ce n'est même pas une intervention atomique correcte.
- Ici : do(X_i = v) fixe X_i à v pour tous les échantillons.

HONNÊTETÉ SUR LE SCOPE :
Ceci est la PRIMITIVE d'intervention atomique (la brique de base de Pearl).
Le "do-calculus" complet de Pearl (règles d'identification backdoor/front-door,
P(Y | do(X)) à partir de P observationnel) nécessite en plus un graphe causal
connu et des règles d'identification — non implémenté ici. Pour estimer
P(Y | do(X=v)) en pratique, on applique do_intervention puis un passage forward
dans un modèle entraîné (voir démo L4).
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

