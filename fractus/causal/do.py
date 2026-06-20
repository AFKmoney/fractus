"""do_intervention : primitive d'intervention atomique (operateur do de Pearl).

CORRECTION DU FAUX DO-CALCULUS D'OMNI :
- OMNI (rkhs_causal.py:21-25) faisait 'intervened[:, do_mask] = 0.0' — juste
  mettre la colonne a 0. Ce n'est meme pas une intervention atomique correcte.
- Ici : do(X_i = v) fixe X_i a v for all les echantillons.

HONNETETE SUR LE SCOPE :
Ceci est la PRIMITIVE d'intervention atomique (la brique de base de Pearl).
Le "do-calculus" complete de Pearl (regles d'identification backdoor/front-door,
P(Y | do(X)) a partir de P observationnel) necessite en plus un graphe causal
connu et des regles d'identification — non implemente ici. Pour estimer
P(Y | do(X=v)) en pratique, on applique do_intervention then un passage forward
in un modele entraine (voir demo L4).
"""

import torch


def do_intervention(
    x: torch.Tensor, var_idx: int, value: float
) -> torch.Tensor:
    """Applique do(X_{var_idx} = value) a un batch de donnees.

    Args:
        x       : tenseur (N, d) de variables observees.
        var_idx : indice de la variable a intervenir.
        value   : valeur a imposer (can etre non-nulle — c'est l'intervention).
    Returns:
        x_intervened : (N, d) with la colonne var_idx mise a `value`.
    """
    x_intervened = x.clone()
    x_intervened[:, var_idx] = value
    return x_intervened

