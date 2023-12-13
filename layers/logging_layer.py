import torch
import torch.nn
from typing import Dict, Any, Callable
from framework.utils import U
import numpy as np
import os


class LoggingLayer:
    def __init__(self) -> None:
        super().__init__()
        self._logs = {}
        self._log_counts = {}
        self._custom_reductions = {}

    def custom_reduction(self, name: str, reduction):
        self._custom_reductions[name] = reduction

    def log(self, name: str, value: Any, drop_old: bool = False):
        value = U.apply_to_tensors(value, lambda x: x.detach())

        drop_old = drop_old or (not isinstance(value, (torch.Tensor, np.ndarray, float, int)))

        if name in self._custom_reductions:
            if name not in self._logs:
                self._logs[name] = []

            self._logs[name].append(value)
        else:
            if name not in self._logs or drop_old:
                self._logs[name] = value
                self._log_counts[name] = 1
            else:
                self._logs[name] = self._logs[name] + value
                self._log_counts[name] = self._log_counts[name] + 1

    def get_logs(self) -> Dict[str, Any]:
        res = {}
        for k, v in self._logs.items():
            if k in self._custom_reductions:
                res[k] = self._custom_reductions[k](v)
            elif isinstance(v, (torch.Tensor, np.ndarray, int, float)):
                res[k] = v / self._log_counts[k]
            else:
                res[k] = v

        self._logs = {}
        self._log_counts = {}
        return res

    def dump_logs(self, save_dir: str):
        pass


def get_logs(module: torch.nn.Module) -> Dict[str, Any]:
    res = {}
    for n, m in module.named_modules():
        if isinstance(m, LoggingLayer):
            logs = m.get_logs()
            res.update({f"{n}/{k}": v for k, v in logs.items()})
    return res


def dump_logs(module: torch.nn.Module, save_dir: str):
    for n, m in module.named_modules():
        if isinstance(m, LoggingLayer):
            m.dump_logs(os.path.join(save_dir, n))
