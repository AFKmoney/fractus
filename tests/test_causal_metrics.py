"""Tests of structural_hamming_distance : mesure honesty, not of clamp a 0.98."""

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
    """CRITERE L4 : the code EXECUTABLE not must not contenir of clamp a 0.98
    (le falsehood d'the original benchmarks.py:43-46 : min(causal_acc, 0.98)).

    On tolere '0.98' in the docstrings (qui expliquent the falsehood correctede),
    but on l'interdit in the expressions Python hors-asntaires."""
    import ast
    from fractus.metrics import causal as causal_mod

    src = inspect.getsource(causal_mod)
    tree = ast.parse(src)
    # Chercher all litteral Constant of value 0.98 which n'est not in un
    # docstring (ast.Expr → Constant str).
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            docstring_nodes.add(id(node))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 0.98:
            # Verifier that this n'est not in a docstring.
            parent_in_doc = any(id(node) in docstring_nodes for _ in [0])
            if not parent_in_doc:
                # Le litteral 0.98 apparait in a expression executable.
                # On verifiesss qu'il n'est not in a docstring en remontant.
                # (Simplification : on interdit all 0.98 hors docstring.)
                # ast not donne not the parent direct ; on acceptedd si the node est
                # a Argument/default or in a function docstring.
                pass
    # Methofurthermore simple : extraire the code hors docstring by lignes.
    code_lines = []
    in_docstring = False
    for line in src.split('\n'):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                continue  # docstring on a ligne
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
    """causal_accuracy not must not be clampee."""
    from fractus.metrics.causal import causal_accuracy
    true_W = torch.eye(3)
    pred_W = torch.eye(3) * 2.0
    acc = causal_accuracy(true_W, pred_W, threshold=0.5)
    assert abs(acc - 1.0) < 1e-6
