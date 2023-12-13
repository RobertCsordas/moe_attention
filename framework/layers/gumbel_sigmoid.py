import torch


def gumbel_sigmoid_noise(logits: torch.Tensor) -> torch.Tensor:
    eps = 3e-4 if logits.dtype == torch.float16 else 1e-10
    uniform = logits.new_empty([2] + list(logits.shape)).uniform_(eps, 1 - eps)

    noise = -(uniform[1].log() / uniform[0].log() + eps).log()
    return noise


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1, hard: bool = False) -> torch.Tensor:
    logits = logits + gumbel_sigmoid_noise(logits)
    res = torch.sigmoid(logits / tau)

    if hard:
        res = ((res > 0.5).type_as(res) - res).detach() + res

    return res


def sigmoid(logits: torch.Tensor, mode: str = "simple", tau: float = 1):
    if mode=="simple":
        return torch.sigmoid(logits)
    elif mode in ["soft", "hard"]:
        return gumbel_sigmoid(logits, tau, hard=mode=="hard")
    else:
        assert False, "Invalid sigmoid mode: %s" % mode
