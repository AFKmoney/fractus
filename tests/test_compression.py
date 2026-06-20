"""Tests de measure_compression_ratio : mesure REELLE, pas de hardcode."""

import inspect
import torch


def test_compression_no_hardcoded_204():
    """CRITERE L3 : le CODE de mesure ne must PAS hardcoder un ratio de retour.
    On tolere la mention de '20.4×' in les docstrings (qui expliquent le
    falsehood corrige d'OMNI), but on interdit tout 'return 20.4' ou ratio
    numerique fige in la logical."""
    import re
    from fractus.metrics import compression
    src = inspect.getsource(compression)
    # On retire commentaires/docstrings before de chercher.
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
    # Aucun litteral 20.4 in le code executable.
    assert not re.search(r'\b20\.4\b', code_only), \
        "Le litteral 20.4 est interdit in le code executable (falsehood OMNI)"


def test_compression_pure_dense_returns_one():
    """Un modele 100% dense (pas de SirenLinear) → ratio ~1.0.

    Note : le ratio for un nn.Linear pur n'est pas EXACTEMENT 1.0 because on
    compte le bias in les params reels (in·out + out) et in l'equivalent
    dense (in·out + out aussi) — therefore ~1.0 a l'epsilon pres.
    """
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Linear(16, 16)
    ratio = measure_compression_ratio(m)
    assert abs(ratio - 1.0) < 1e-6, f"Ratio dense pur attendu 1.0, eu {ratio}"


def test_compression_with_siren_gt_one():
    """Un modele with SirenLinear → ratio > 1 (moins de params que l'equivalent dense)."""
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
