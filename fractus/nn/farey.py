"""Suite of Farey and selection of phases for the MoE a routing of phase.

Porte depuis the original architecture (src/math/farey.rs).

La Farey sequence F_n est l'ensemble trie fractions irreductibles p/q in
[0, 1] with q <= n. Elle est generatede iterativement by the property of mediante.

Pour the MoE : on prend F_{2E} (ordre double number d'experts) and on selectionne
uniformement E angles parmi the fractions, convertis en angles 2π·p/q ∈ [0, 2π).
Cela donne a distribution of phases dense, non-collapsante and deterministic —
l'interet for the routing von Mises.
"""

import math
from typing import List, Tuple


def farey_sequence(n: int) -> List[Tuple[int, int]]:
    """Genere the Farey sequence F_n comme liste of (p, q) triee croissante.

    Algorithme by mediante (comme farey.rs:18-49).
    F_n contient exactment 1 + Σ_{q=1}^{n} φ(q) termes (φ = indicatrice d'Euler).
    """
    if n < 1:
        raise ValueError("n must etre >= 1")
    fractions: List[Tuple[int, int]] = []
    a, b = 0, 1
    c, d = 1, n
    fractions.append((a, b))
    while c <= n:
        k = (n + b) // d
        next_c = k * c - a
        next_d = k * d - b
        a, b = c, d
        c, d = next_c, next_d
        fractions.append((a, b))
    return fractions


def expert_phases(n_experts: int) -> List[float]:
    """Selectionne n_experts angles ∈ [0, 2π) depuis F_{2·n_experts}.

    Comme farey.rs:53-64 : on construit F_{2E} (ordre double), then on selectionne
    uniformement E angles parmi the n_frac = len(F_{2E}) fractions disponibles.
    """
    if n_experts < 1:
        raise ValueError("n_experts must etre >= 1")
    fractions = farey_sequence(2 * n_experts)
    n_frac = len(fractions)
    angles_all = [2.0 * math.pi * p / q for (p, q) in fractions]
    phases: List[float] = []
    for i in range(n_experts):
        idx = min(int(i * n_frac / n_experts), n_frac - 1)
        phases.append(angles_all[idx])
    return phases
