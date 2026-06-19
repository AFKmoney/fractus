"""Mesure HONNÊTE du ratio de compression d'un modèle.

CORRECTION DU MENSONGE D'OMNI-FRACTAL :
- OMNI hardcodait "compression_ratio": 20.4 dans training_loop.py:52.
- Ici, le ratio est MESURÉ : on compte les paramètres réellement utilisés et
  on les compare à la taille qu'auraient les matrices si elles étaient denses.

Définition du ratio :
    ratio = (somme des tailles denses équivalentes des SirenLinear) /
            (somme des params SIREN + params denses restants)

Pour une SirenLinear(in, out, hidden=h) :
    - taille dense équivalente = in·out (la matrice qu'elle remplace)
    - params SIREN = 2·h + h·h + h·1 + biases ≈ h² + 3h
    Le ratio de CETTE couche = in·out / params_SIREN.

Pour un modèle mixte (SirenLinear + nn.Linear), le ratio global est :
    (Σ tailles denses équivalentes) / (Σ params totaux).

On ne prétend PAS 20.4×. On mesure. La démo L3 montrera le vrai chiffre.
"""

import torch
import torch.nn as nn

from ..nn.siren_linear import SirenLinear


def _count_params(module: nn.Module) -> int:
    """Nombre total de paramètres entraînables d'un module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def measure_compression_ratio(model: nn.Module) -> float:
    """Mesure RÉELLEMENT le ratio de compression d'un modèle.

    Args:
        model : un nn.Module pouvant contenir des SirenLinear et/ou des nn.Linear.
    Returns:
        ratio > 0. Ratio = 1.0 si le modèle est 100% dense.
        Ratio > 1 si le modèle contient des SirenLinear (compression effective).
        Ratio < 1 est possible mais rare (SIREN plus grosse que la matrice).

    LIMITE CONNUE : ne gère que SirenLinear, nn.Linear, nn.LayerNorm, nn.Embedding.
    Tout autre module (Conv, BatchNorm, RNN, etc.) est ignoré silencieusement —
    le ratio serait alors sous-estimé. Pour usage général, étendre la liste ou
    ajouter un warn. Suffisant pour la démo L3 (MLP) et le transformer fractus.
    """
    total_dense_equivalent = 0
    total_actual_params = 0

    for module in model.modules():
        if isinstance(module, SirenLinear):
            # Taille dense équivalente : in·out + out (matrice + bias).
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
            # Autres modules : pas de compression (comptés à leur taille réelle).
            actual = _count_params(module)
            total_dense_equivalent += actual
            total_actual_params += actual

    if total_actual_params == 0:
        return 1.0
    return total_dense_equivalent / total_actual_params
