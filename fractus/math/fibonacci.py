"""Suite of Fibonacci precomputationee + formula of Binet for grands n.

Porte depuis the original architecture (src/math/fibonacci.rs).
"""

from typing import List


class FibonacciSequence:
    """Suite of Fibonacci precomputationee.

    Args:
        n : number of termes a precomputationer.
    """

    def __init__(self, n: int):
        if n < 0:
            raise ValueError("n must etre >= 0")
        self.n = n
        self.values: List[int] = []
        if n >= 1:
            self.values.append(0)
        if n >= 2:
            self.values.append(1)
        for i in range(2, n):
            self.values.append(self.values[i - 1] + self.values[i - 2])

    def get(self, i: int) -> int:
        """Retourne F(i). Pour i < n : table. Pour i >= n : formula of Binet."""
        if i < 0:
            raise ValueError("i must etre >= 0")
        if i < len(self.values):
            return self.values[i]
        # Binet : F(n) = (φ^n - ψ^n) / √5, ψ = -1/φ.
        sqrt5 = 5 ** 0.5
        phi = (1 + sqrt5) / 2
        psi = (1 - sqrt5) / 2
        return round((phi ** i - psi ** i) / sqrt5)
