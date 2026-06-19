"""honest_perplexity : vraie perplexité = exp(loss de validation).

CORRECTION DE LA PERPLEXITÉ FICTIVE D'OMNI/FNN :
- FNN (model.rs:537-546) retournait un PROXY basé sur la norme de l'embedding,
  pas une vraie perplexité. Le commentaire admettait :
  "Real perplexity requires a full forward pass".
- Ici : vraie perplexité = exp(cross-entropy moyenne sur un dataset de validation).
"""

import math
import torch
import torch.nn as nn


def honest_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> float:
    """Calcule la vraie perplexité = exp(CE loss moyenne).

    Args:
        model      : un modèle qui prend input_ids et retourne des logits.
        input_ids  : (B, L) tenseur d'ids d'entrée.
        target_ids : (B, L) tenseur d'ids cibles (typiquement input décalé de 1).
    Returns:
        ppl : float >= 1.0. ppl = 1.0 = prédiction parfaite.
    """
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)  # (B, L, vocab) ou autre selon le modèle
        if isinstance(logits, tuple):
            logits = logits[0]  # certains modèles retournent (logits, aux_loss)
        B, L, V = logits.shape
        ce = nn.functional.cross_entropy(
            logits.reshape(-1, V),
            target_ids.reshape(-1),
        )
    return math.exp(float(ce.item()))
