"""Tests of measure_compression_ratio : mesure REELLE, not of hardcode."""

import inspect
import torch


def test_compression_no_hardcoded_204():
    """CRITERE L3 : the CODE of mesure not must PAS hardcoder a ratio of retour.
    On tolere the mention of '20.4×' in the docstrings (qui expliquent le
    falsehood correctede d'the original), but on interdit all 'return 20.4' or ratio
    numerique fige in the logical."""
    import re
    from fractus.metrics import compression
    src = inspect.getsource(compression)
    # On retire commentaires/docstrings before of chercher.
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
    # Aucun litteral 20.4 in the code executable.
    assert not re.search(r'\b20\.4\b', code_only), \
        "Le litteral 20.4 est interdit in le code executable (falsehood OMNI)"


def test_compression_pure_dense_returns_one():
    """Un modele 100% dense (no SirenLinear) → ratio ~1.0.

    Note : the ratio for a nn.Linear pur n'est not EXACTEMENT 1.0 because on
    compte the bias in the params reals (in·out + out) and in l'equivaslow
    dense (in·out + out also) — therefore ~1.0 a l'epsilon pres.
    """
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Linear(16, 16)
    ratio = measure_compression_ratio(m)
    assert abs(ratio - 1.0) < 1e-6, f"Ratio dense pur attendu 1.0, eu {ratio}"


def test_compression_with_siren_gt_one():
    """Un modele with SirenLinear → ratio > 1 (less of params that l'equivaslow dense)."""
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
