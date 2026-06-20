"""honest_perplexity : vraie perplexite = exp(loss of validation).

CORRECTION DE LA PERPLEXITE FICTIVE D'the original/the original :
- the original (model.rs:537-546) retournait a PROXY base on the norme of l'embedding,
  not a vraie perplexite. Le commentaire admettait :
  "Real perplexity requires a full forward pass".
- Ici : vraie perplexite = exp(cross-entropy moyenne on a dataset of validation).
"""

import math
import torch
import torch.nn as nn


def honest_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> float:
    """Calcule the vraie perplexite = exp(CE loss moyenne).

    Args:
        model      : a modele which prend input_ids and returns logits.
        input_ids  : (B, L) tenseur d'ids d'entree.
        target_ids : (B, L) tenseur d'ids targets (typiquement input decale of 1).
    Returns:
        ppl : float >= 1.0. ppl = 1.0 = prediction parfaite.
    """
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)  # (B, L, vocab) or other selon the modele
        if isinstance(logits, tuple):
            logits = logits[0]  # certains modeles return (logits, aux_loss)
        B, L, V = logits.shape
        ce = nn.functional.cross_entropy(
            logits.reshape(-1, V),
            target_ids.reshape(-1),
        )
    return math.exp(float(ce.item()))
