"""Mesure HONNETE ratio of compression d'un modele.

CORRECTION DU MENSONGE D'the original design :
- the original hardcodait "compression_ratio": 20.4 in training_loop.py:52.
- Ici, the ratio est MESURE : on compte the parameters reallement utilises et
  on the compare a the taille qu'auraient the matrixs si elles etaient denses.

Definition ratio :
    ratio = (somme tailles denses equivalentes SirenLinear) /
            (somme params SIREN + params denses restants)

Pour a SirenLinear(in, out, hidden=h) :
    - taille dense equivalente = in·out (la matrix qu'elle remplace)
    - params SIREN = 2·h + h·h + h·1 + biases ≈ h2 + 3h
    Le ratio of CETTE couche = in·out / params_SIREN.

Pour a modele mixte (SirenLinear + nn.Linear), the ratio global est :
    (Σ tailles denses equivalentes) / (Σ params totaux).

On not pretend PAS 20.4×. On mesure. La demo L3 montrera the true chiffre.
"""

import torch
import torch.nn as nn

from ..nn.siren_linear import SirenLinear


def _count_params(module: nn.Module) -> int:
    """Nombre total of parameters entrainables d'un module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def measure_compression_ratio(model: nn.Module) -> float:
    """Mesure REELLEMENT the ratio of compression d'un modele.

    Args:
        model : a nn.Module pouvant contenir SirenLinear et/ou nn.Linear.
    Returns:
        ratio > 0. Ratio = 1.0 si the modele est 100% dense.
        Ratio > 1 si the modele contient SirenLinear (compression effective).
        Ratio < 1 est possible but rare (SIREN more grosse that the matrix).

    LIMITE CONNUE : not gere that SirenLinear, nn.Linear, nn.LayerNorm, nn.Embedding.
    Tout other module (Conv, BatchNorm, RNN, etc.) est ignore silencieusement —
    the ratio serait alors under-estime. Pour usage general, etendre the liste ou
    ajouter a warn. Suffisant for the demo L3 (MLP) and the transformer fractus.
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
            # nn.Linear : dense_eq == actual (pas of compression).
            dense_eq = module.in_features * module.out_features
            if module.bias is not None:
                dense_eq += module.out_features
            actual = _count_params(module)
            total_dense_equivalent += dense_eq
            total_actual_params += actual
        elif isinstance(module, (nn.LayerNorm, nn.Embedding)):
            # Autres modules : not of compression (comptes a their taille real).
            actual = _count_params(module)
            total_dense_equivalent += actual
            total_actual_params += actual

    if total_actual_params == 0:
        return 1.0
    return total_dense_equivalent / total_actual_params
