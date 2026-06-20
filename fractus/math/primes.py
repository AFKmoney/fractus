"""Sieve of Eratosthenes for primality verification.

Ported from the original architecture (src/math/primes.rs). Precomputed once
up to a limit, then verify_prime(n) is O(1) for n <= limit.
"""

from typing import List


class PrimeSieve:
    """Crible d'Eratosthene precomputatione.

    Args:
        limit : borne superieure (inclusive) crible.
    """

    def __init__(self, limit: int):
        if limit < 0:
            raise ValueError("limit must etre >= 0")
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
        """Retourne True si n est premier (n <= limit), sinon verifiess by essai of division."""
        if n < 2:
            return False
        if n <= self.limit:
            return self.is_prime[n]
        # n > limit : essai of division jusqu'a sqrt(n).
        d = 2
        while d * d <= n:
            if n % d == 0:
                return False
            d += 1
        return True
