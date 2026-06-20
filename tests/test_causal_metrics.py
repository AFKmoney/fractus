"""Tests de structural_hamming_distance : mesure honnete, pas de clamp a 0.98."""

import inspect
import torch


def test_shd_perfect_match_zero():
    from fractus.metrics.causal import structural_hamming_distance
    W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.4, 0.0],
    ])
    shd = structural_hamming_distance(W, W, threshold=0.1)
    assert shd == 0


def test_shd_counts_missing_edges():
    from fractus.metrics.causal import structural_hamming_distance
    true_W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.4, 0.0],
    ])
    pred_W = torch.zeros(3, 3)
    shd = structural_hamming_distance(true_W, pred_W, threshold=0.1)
    assert shd == 3


def test_shd_counts_extra_edges():
    from fractus.metrics.causal import structural_hamming_distance
    true_W = torch.zeros(3, 3)
    pred_W = torch.tensor([
        [0.0, 0.5, 0.3],
        [0.0, 0.0, 0.4],
        [0.0, 0.0, 0.0],
    ])
    shd = structural_hamming_distance(true_W, pred_W, threshold=0.1)
    assert shd == 3


def test_shd_threshold_filters_small_values():
    from fractus.metrics.causal import structural_hamming_distance
    true_W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    pred_W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    shd = structural_hamming_distance(true_W, pred_W, threshold=0.1)
    assert shd == 1


def test_shd_no_clamp_to_098():
    """CRITERE L4 : le code EXECUTABLE ne must pas contenir de clamp a 0.98
    (le falsehood d'OMNI benchmarks.py:43-46 : min(causal_acc, 0.98)).

    On tolere '0.98' in les docstrings (qui expliquent le falsehood corrige),
    but on l'interdit in les expressions Python hors-commentaires."""
    import ast
    from fractus.metrics import causal as causal_mod

    src = inspect.getsource(causal_mod)
    tree = ast.parse(src)
    # Chercher tout litteral Constant de valeur 0.98 qui n'est pas in un
    # docstring (ast.Expr → Constant str).
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            docstring_nodes.add(id(node))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 0.98:
            # Verifier que ce n'est pas in un docstring.
            parent_in_doc = any(id(node) in docstring_nodes for _ in [0])
            if not parent_in_doc:
                # Le litteral 0.98 apparait in une expression executable.
                # On verifies qu'il n'est pas in un docstring en remontant.
                # (Simplification : on interdit tout 0.98 hors docstring.)
                # ast ne donne pas le parent direct ; on accepte si le node est
                # un Argument/default ou in une function docstring.
                pass
    # Methode plus simple : extraire le code hors docstring par lignes.
    code_lines = []
    in_docstring = False
    for line in src.split('\n'):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                continue  # docstring sur une ligne
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if stripped.startswith('#'):
            continue
        code_lines.append(line)
    code_only = '\n'.join(code_lines)
    assert "0.98" not in code_only, \
        "Le litteral 0.98 est interdit in le code executable (falsehood OMNI)"


def test_causal_accuracy_no_clamp():
    """causal_accuracy ne must pas etre clampee."""
    from fractus.metrics.causal import causal_accuracy
    true_W = torch.eye(3)
    pred_W = torch.eye(3) * 2.0
    acc = causal_accuracy(true_W, pred_W, threshold=0.5)
    assert abs(acc - 1.0) < 1e-6
