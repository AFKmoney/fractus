"""Tests de measure_compression_ratio : mesure RÉELLE, pas de hardcode."""

import inspect
import torch


def test_compression_no_hardcoded_204():
    """CRITÈRE L3 : le CODE de mesure ne doit PAS hardcoder un ratio de retour.
    On tolère la mention de '20.4×' dans les docstrings (qui expliquent le
    mensonge corrigé d'OMNI), mais on interdit tout 'return 20.4' ou ratio
    numérique figé dans la logique."""
    import re
    from fractus.metrics import compression
    src = inspect.getsource(compression)
    # On retire commentaires/docstrings avant de chercher.
    code_lines = []
    in_docstring = False
    for line in src.split('\n'):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_docstring = not in_docstring or stripped.count('"""') == 2
            continue
        if in_docstring:
            continue
        if stripped.startswith('#'):
            continue
        code_lines.append(line)
    code_only = '\n'.join(code_lines)
    # Aucun littéral 20.4 dans le code exécutable.
    assert not re.search(r'\b20\.4\b', code_only), \
        "Le littéral 20.4 est interdit dans le code exécutable (mensonge OMNI)"


def test_compression_pure_dense_returns_one():
    """Un modèle 100% dense (pas de SirenLinear) → ratio ~1.0.

    Note : le ratio pour un nn.Linear pur n'est pas EXACTEMENT 1.0 car on
    compte le bias dans les params réels (in·out + out) et dans l'équivalent
    dense (in·out + out aussi) — donc ~1.0 à l'epsilon près.
    """
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Linear(16, 16)
    ratio = measure_compression_ratio(m)
    assert abs(ratio - 1.0) < 1e-6, f"Ratio dense pur attendu 1.0, eu {ratio}"


def test_compression_with_siren_gt_one():
    """Un modèle avec SirenLinear → ratio > 1 (moins de params que l'équivalent dense)."""
    from fractus.nn.siren_linear import SirenLinear
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Sequential(
        SirenLinear(32, 32, hidden=16),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )
    ratio = measure_compression_ratio(m)
    assert ratio > 1.0, f"Ratio attendu > 1, eu {ratio}"


def test_compression_returns_finite():
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Linear(8, 8)
    r = measure_compression_ratio(m)
    assert isinstance(r, float)
    assert r > 0
