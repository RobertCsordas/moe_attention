import torch
import math


def entropy(t: torch.Tensor) -> torch.Tensor:
    return -(t.clamp(torch.finfo(t.dtype).eps).log() * t).sum(-1)


def reative_entropy(t: torch.Tensor) -> torch.Tensor:
    return entropy(t) / math.log(t.shape[-1])


def perplexity(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(entropy(t))


def relative_perplexity(t: torch.Tensor) -> torch.Tensor:
    return perplexity(t) / t.shape[-1]


def entropy_l(l: torch.Tensor) -> torch.Tensor:
    return - (l * l.exp()).sum(-1)


def relative_perplexity_l(l: torch.Tensor) -> torch.Tensor:
    return torch.exp(entropy_l(l) - math.log(l.shape[-1]))
