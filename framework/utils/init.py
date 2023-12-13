import torch


def trunc_normal_(param: torch.Tensor, std: float):
    torch.nn.init.trunc_normal_(param)
    with torch.no_grad():
        param.mul_(std / param.std())
