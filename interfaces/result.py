import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


class Result:
    outputs: torch.Tensor
    loss: torch.Tensor

    batch_dim = 0

    def plot(self) -> Dict[str, Any]:
        return {}

    @property
    def batch_size(self) -> int:
        return self.outputs.shape[self.batch_dim]

    @staticmethod
    def merge(l: List, batch_weights: Optional[List[float]] = None):
        if len(l) == 1:
            return l[0]
        batch_weights = batch_weights if batch_weights is not None else [1] * len(l)
        loss = sum([r.loss * w for r, w in zip(l, batch_weights)]) / sum(batch_weights)
        out = torch.cat([r.outputs for r in l], l[0].batch_dim)
        return l[0].__class__(out, loss)


@dataclass
class RecurrentResult(Result):
    outputs: torch.Tensor
    loss: torch.Tensor

    batch_dim = 1


@dataclass
class FeedforwardResult(Result):
    outputs: torch.Tensor
    loss: torch.Tensor

    batch_dim = 0
