from typing import List
import bisect


class StepLrSched:
    def __init__(self, lr: float, steps: List[int], gamma: float):
        self.steps = [0] + list(sorted(steps))
        self.lrs = [lr * (gamma ** i) for i in range(len(self.steps))]

    def get(self, step: int) -> float:
        return self.lrs[bisect.bisect(self.steps, step) - 1]
