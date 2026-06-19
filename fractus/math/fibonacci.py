"""Suite de Fibonacci précalculée + formule de Binet pour grands n.

Porté depuis FNN v5.0 (src/math/fibonacci.rs).
"""

from typing import List


class FibonacciSequence:
    """Suite de Fibonacci précalculée.

    Args:
        n : nombre de termes à précalculer.
    """

    def __init__(self, n: int):
        if n < 0:
            raise ValueError("n doit être >= 0")
        self.n = n
        self.values: List[int] = []
        if n >= 1:
            self.values.append(0)
        if n >= 2:
            self.values.append(1)
        for i in range(2, n):
            self.values.append(self.values[i - 1] + self.values[i - 2])

    def get(self, i: int) -> int:
        """Retourne F(i). Pour i < n : table. Pour i >= n : formule de Binet."""
        if i < 0:
            raise ValueError("i doit être >= 0")
        if i < len(self.values):
            return self.values[i]
        # Binet : F(n) = (φ^n - ψ^n) / √5, ψ = -1/φ.
        sqrt5 = 5 ** 0.5
        phi = (1 + sqrt5) / 2
        psi = (1 - sqrt5) / 2
        return round((phi ** i - psi ** i) / sqrt5)
