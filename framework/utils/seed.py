import torch
import torch.utils.data
import torch.cuda
import torch.backends.cudnn
import random
import numpy as np
from typing import Optional


def fix(offset: int = 0, fix_cudnn: bool = True):
    random.seed(0x12345678 + offset)
    torch.manual_seed(0x0DABA52 + offset)
    torch.cuda.manual_seed(0x0DABA52 + 1 + offset)
    np.random.seed(0xC1CAFA52 + offset)

    if fix_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_randstate(seed: Optional[int] = None) -> np.random.RandomState:
    if seed is None:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = worker_info.seed
        else:
            seed = random.randint(0, 0x7FFFFFFF)

    return np.random.RandomState(seed)
