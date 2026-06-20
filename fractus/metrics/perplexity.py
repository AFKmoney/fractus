"""honest_perplexity : vraie perplexite = exp(loss de validation).

CORRECTION DE LA PERPLEXITE FICTIVE D'OMNI/FNN :
- FNN (model.rs:537-546) retournait un PROXY base sur la norme de l'embedding,
  pas une vraie perplexite. Le commentaire admettait :
  "Real perplexity requires a full forward pass".
- Ici : vraie perplexite = exp(cross-entropy moyenne sur un dataset de validation).
"""

import math
import torch
import torch.nn as nn


def honest_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> float:
    """Calcule la vraie perplexite = exp(CE loss moyenne).

    Args:
        model      : un modele qui prend input_ids et returns des logits.
        input_ids  : (B, L) tenseur d'ids d'entree.
        target_ids : (B, L) tenseur d'ids cibles (typiquement input decale de 1).
    Returns:
        ppl : float >= 1.0. ppl = 1.0 = prediction parfaite.
    """
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)  # (B, L, vocab) ou autre selon le modele
        if isinstance(logits, tuple):
            logits = logits[0]  # certains modeles return (logits, aux_loss)
        B, L, V = logits.shape
        ce = nn.functional.cross_entropy(
            logits.reshape(-1, V),
            target_ids.reshape(-1),
        )
    return math.exp(float(ce.item()))
