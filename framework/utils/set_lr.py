import torch


def set_lr(optim: torch.optim.Optimizer, lr: float):
    for param_group in optim.param_groups:
        param_group['lr'] = lr


def get_lr(optim: torch.optim.Optimizer) -> float:
    lr = None
    for param_group in optim.param_groups:
        if lr is None:
            lr = param_group['lr']
        else:
            assert lr == param_group['lr']

    return lr
