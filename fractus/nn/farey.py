"""Suite de Farey et sélection de phases pour le MoE à routing de phase.

Porté depuis FNN v5.0 (src/math/farey.rs).

La suite de Farey F_n est l'ensemble trié des fractions irréductibles p/q dans
[0, 1] avec q <= n. Elle est générée itérativement par la propriété de médiante.

Pour le MoE : on prend F_{2E} (ordre double du nombre d'experts) et on sélectionne
uniformément E angles parmi les fractions, convertis en angles 2π·p/q ∈ [0, 2π).
Cela donne une distribution de phases dense, non-collapsante et déterministe —
l'intérêt pour le routing von Mises.
"""

import math
from typing import List, Tuple


def farey_sequence(n: int) -> List[Tuple[int, int]]:
    """Génère la suite de Farey F_n comme liste de (p, q) triée croissante.

    Algorithme par médiante (comme farey.rs:18-49).
    F_n contient exactement 1 + Σ_{q=1}^{n} φ(q) termes (φ = indicatrice d'Euler).
    """
    if n < 1:
        raise ValueError("n doit être >= 1")
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
    """Sélectionne n_experts angles ∈ [0, 2π) depuis F_{2·n_experts}.

    Comme farey.rs:53-64 : on construit F_{2E} (ordre double), puis on sélectionne
    uniformément E angles parmi les n_frac = len(F_{2E}) fractions disponibles.
    """
    if n_experts < 1:
        raise ValueError("n_experts doit être >= 1")
    fractions = farey_sequence(2 * n_experts)
    n_frac = len(fractions)
    angles_all = [2.0 * math.pi * p / q for (p, q) in fractions]
    phases: List[float] = []
    for i in range(n_experts):
        idx = min(int(i * n_frac / n_experts), n_frac - 1)
        phases.append(angles_all[idx])
    return phases
