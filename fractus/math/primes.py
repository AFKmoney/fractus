"""Crible d'Ératosthène pour vérification de primalité.

Porté depuis FNN v5.0 (src/math/primes.rs). Le crible est précalculé une fois
jusqu'à une limite, puis verify_prime(n) est O(1) pour n <= limite.
"""

from typing import List


class PrimeSieve:
    """Crible d'Ératosthène précalculé.

    Args:
        limit : borne supérieure (inclusive) du crible.
    """

    def __init__(self, limit: int):
        if limit < 0:
            raise ValueError("limit doit être >= 0")
        self.limit = limit
        # is_prime[i] = True si i est premier.
        self.is_prime: List[bool] = [True] * (limit + 1)
        if limit >= 0:
            self.is_prime[0] = False
        if limit >= 1:
            self.is_prime[1] = False
        # Crible classique.
        i = 2
        while i * i <= limit:
            if self.is_prime[i]:
                for multiple in range(i * i, limit + 1, i):
                    self.is_prime[multiple] = False
            i += 1

    def verify_prime(self, n: int) -> bool:
        """Retourne True si n est premier (n <= limit), sinon vérifie par essai de division."""
        if n < 2:
            return False
        if n <= self.limit:
            return self.is_prime[n]
        # n > limit : essai de division jusqu'à sqrt(n).
        d = 2
        while d * d <= n:
            if n % d == 0:
                return False
            d += 1
        return True
