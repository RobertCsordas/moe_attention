import torch


def gumbel_noise_like(x: torch.Tensor) -> torch.Tensor:
    eps = 3e-4 if x.dtype == torch.float16 else 1e-10
    uniform = torch.empty_like(x).uniform_(eps, 1 - eps)

    return - (- uniform.log()).log()


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    noise = gumbel_noise_like(logits)
    res = torch.softmax((logits + noise) / tau, dim=dim)

    if hard:
        res = (torch.zeros_like(res).scatter_(dim, res.argmax(dim, keepdim=True), 1.0) - res).detach() + res

    return res
