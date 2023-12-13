import torch


def cossim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: (batch, seq1, dim)
    # b: (batch, seq2, dim)
    # return: (batch, seq1, seq2)

    # For some reason PyTorch internal implementation uses infinite amount of memory (2023.04.23)

    dots = (a @ b.transpose(-1, -2))
    dots = dots / a.norm(p=2, dim=-1)[..., None]
    dots = dots / b.norm(p=2, dim=-1).unsqueeze(-2)
    return dots
