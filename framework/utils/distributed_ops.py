import torch
import torch.distributed
import math


def logsumexp(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    x = x.float()
    if not torch.distributed.is_initialized():
        return x.logsumexp(dim=dim, keepdim=keepdim)

    # Calculate numerically stable distributed logsumexp
    xmax = x.max(dim=dim, keepdim=True).values
    torch.distributed.all_reduce(xmax, op=torch.distributed.ReduceOp.MAX)

    xe = (x - xmax).exp().sum(dim=dim, keepdim=True)
    torch.distributed.all_reduce(xe, op=torch.distributed.ReduceOp.SUM)

    res = (xmax + xe.log())
    if not keepdim:
        res = res.squeeze(dim)

    return res


def log_mean(x: torch.Tensor, dim: int = 0, sync_distributed: bool = True) -> torch.Tensor:
    assert x.shape[dim] > 0
    x = x.float()
    if torch.distributed.is_initialized() and sync_distributed:
        xlse = logsumexp(x, dim=dim)

        # Normalize
        n = torch.tensor(x.shape[dim]).to(x.device)
        torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
        return xlse - n.log()
    else:
        return x.logsumexp(dim) - math.log(x.shape[dim])
