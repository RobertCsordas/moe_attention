from dataclasses import dataclass
from typing import Union
import math
import numpy as np
import statistics


@dataclass
class Stat:
    mean: Union[np.ndarray, float]
    std: Union[np.ndarray, float]
    n: int

    def __repr__(self) -> str:
        return f"Stat(mean: {self.mean}, std: {self.std}, n: {self.n})"

    def __mul__(self, other):
        return Stat(self.mean * other, self.std * other, self.n)

    def __truediv__(self, other):
        return self * (1.0 / other)


class StatTracker:
    def __init__(self):
        self.sum = 0
        self.sqsum = 0
        self.n = 0

    def add(self, v: float):
        if isinstance(v, np.ndarray):
            v = v.astype(np.float32)
        self.sum = self.sum + v
        self.sqsum = self.sqsum + v**2
        self.n += 1

    def get(self) -> Stat:
        assert self.n > 0
        mean = self.sum / self.n
        var = (self.sqsum / self.n - mean ** 2) * self.n/(self.n-1) if self.n>1 else 0

        return Stat(mean, np.sqrt(np.maximum(var,0)), self.n)

    def __repr__(self) -> str:
        return self.get().__repr__()

    def __add__(self, other):
        res = StatTracker()
        res.sum = other.sum + self.sum
        res.sqsum = other.sqsum + self.sqsum
        res.n = other.n + self.n
        return res


class MedianTracker:
    def __init__(self):
        self.elems = []

    def add(self, v: float):
        if isinstance(v, np.ndarray):
            v = v.astype(np.float32)
        self.elems.append(v)

    def get(self) -> float:
        assert len(self.elems) > 0
        return statistics.median(self.elems)

    def __repr__(self) -> str:
        return f"Median({self.get()})"

    def __add__(self, other):
        res = MedianTracker()
        res.elems = self.elems + other.elems
        return res
