import math
from typing import List


def decompose_factors(n: int, n_factors: int) -> List[int]:
    if n_factors == 1:
        return [n]

    h = math.ceil(n ** (1 / n_factors))
    for i in range(h, 0, -1):
        if n % i == 0:
            w = n // i
            return decompose_factors(w, n_factors - 1) + [i]

    raise ValueError("Factorization failed.")
