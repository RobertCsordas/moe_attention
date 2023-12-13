import torch
import torch.nn
from typing import Dict, Any


class LayerWithVisualization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visualization_enabled = False

    def prepare(self):
        # Should be called before the training step
        pass

    def plot(self, options: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()


class LayerVisualizer:
    def __init__(self, module: torch.nn.Module, options: Dict[str, Any] = {}):
        self.modules = []
        self.options = options
        self.curr_options = None
        for n, m in module.named_modules():
            if isinstance(m, LayerWithVisualization):
                self.modules.append((n, m))

    def plot(self) -> Dict[str, Any]:
        res = {}
        for n, m in self.modules:
            res.update({f"{n}/{k}": v for k, v in m.plot(self.curr_options).items()})
            m.visualization_enabled = False

        self.curr_options = None
        return res

    def prepare(self, options: Dict[str, Any] = {}):
        self.curr_options = self.options.copy()
        self.curr_options.update(options)

        for _, m in self.modules:
            m.prepare()
            m.visualization_enabled = True
