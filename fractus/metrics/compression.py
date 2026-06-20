"""Mesure HONNETE du ratio de compression d'un modele.

CORRECTION DU MENSONGE D'the original design :
- OMNI hardcodait "compression_ratio": 20.4 in training_loop.py:52.
- Ici, le ratio est MESURE : on compte les parameters reellement utilises et
  on les compare a la taille qu'auraient les matrices si elles etaient denses.

Definition du ratio :
    ratio = (somme des tailles denses equivalentes des SirenLinear) /
            (somme des params SIREN + params denses restants)

Pour une SirenLinear(in, out, hidden=h) :
    - taille dense equivalente = in·out (la matrix qu'elle remplace)
    - params SIREN = 2·h + h·h + h·1 + biases ≈ h² + 3h
    Le ratio de CETTE couche = in·out / params_SIREN.

Pour un modele mixte (SirenLinear + nn.Linear), le ratio global est :
    (Σ tailles denses equivalentes) / (Σ params totaux).

On ne pretend PAS 20.4×. On mesure. La demo L3 montrera le true chiffre.
"""

import torch
import torch.nn as nn

from ..nn.siren_linear import SirenLinear


def _count_params(module: nn.Module) -> int:
    """Nombre total de parameters entrainables d'un module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def measure_compression_ratio(model: nn.Module) -> float:
    """Mesure REELLEMENT le ratio de compression d'un modele.

    Args:
        model : un nn.Module pouvant contenir des SirenLinear et/ou des nn.Linear.
    Returns:
        ratio > 0. Ratio = 1.0 si le modele est 100% dense.
        Ratio > 1 si le modele contient des SirenLinear (compression effective).
        Ratio < 1 est possible but rare (SIREN plus grosse que la matrix).

    LIMITE CONNUE : ne gere que SirenLinear, nn.Linear, nn.LayerNorm, nn.Embedding.
    Tout autre module (Conv, BatchNorm, RNN, etc.) est ignore silencieusement —
    le ratio serait alors under-estime. Pour usage general, etendre la liste ou
    ajouter un warn. Suffisant for la demo L3 (MLP) et le transformer fractus.
    """
    total_dense_equivalent = 0
    total_actual_params = 0

    for module in model.modules():
        if isinstance(module, SirenLinear):
            # Taille dense equivalente : in·out + out (matrix + bias).
            dense_eq = module.in_features * module.out_features
            if module.bias is not None:
                dense_eq += module.out_features
            actual = _count_params(module)
            total_dense_equivalent += dense_eq
            total_actual_params += actual
        elif isinstance(module, nn.Linear) and not isinstance(module, SirenLinear):
            # nn.Linear : dense_eq == actual (pas de compression).
            dense_eq = module.in_features * module.out_features
            if module.bias is not None:
                dense_eq += module.out_features
            actual = _count_params(module)
            total_dense_equivalent += dense_eq
            total_actual_params += actual
        elif isinstance(module, (nn.LayerNorm, nn.Embedding)):
            # Autres modules : pas de compression (comptes a leur taille real).
            actual = _count_params(module)
            total_dense_equivalent += actual
            total_actual_params += actual

    if total_actual_params == 0:
        return 1.0
    return total_dense_equivalent / total_actual_params
