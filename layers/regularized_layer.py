import torch
import torch.nn
from typing import Dict, Any, Callable, Tuple, Optional, Set


class RegularizedLayer:
    def __init__(self) -> None:
        super().__init__()
        self.reg_accumulated = {}
        self.regularization_present = False

    @property
    def reg_enabled(self) -> bool:
        return self.training and self.regularization_present

    def add_reg(self, l: Callable[[], torch.Tensor], name="reg"):
        if self.reg_enabled:
            self.reg_accumulated[name] = self.reg_accumulated.get(name, 0) + l()

    def get_reg_loss(self) -> torch.Tensor:
        rl = self.reg_accumulated
        self.reg_accumulated = {}
        return rl


class LayerRegularizer:
    def __init__(self, module: torch.nn.Module, stop_after: Optional[int] = None, scales: Dict[str, float] = {},
                 lin_decay: Set[str] = set(), options: Dict[str, Any] = {}):

        self.modules = []
        self.scales = scales
        self.stop_after = stop_after
        self.lin_decay = set(lin_decay)

        if self.lin_decay and stop_after is None:
            raise ValueError("Please specify stop_after when using lin_decay.")

        for n, m in module.named_modules():
            if isinstance(m, RegularizedLayer):
                self.modules.append((n, m))
                m.regularization_present = True

    def get(self, iter: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        res = {}
        for _, m in self.modules:
            for k, v in m.get_reg_loss().items():
                res[k] = res.get(k, 0) + v

        to_log = {k: v.detach() for k, v in res.items()}

        for k, v in res.items():
            res[k] = v * self.scales.get(k, 1)

        for k in self.lin_decay:
            res[k] *= 1 - iter / self.stop_after
        return sum(res.values()), to_log
