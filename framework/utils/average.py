import torch
from typing import Union, Any, Dict


class Average:
    SAVE = ["sum", "cnt"]

    def __init__(self):
        self.reset()

    def add(self, data: Union[int, float, torch.Tensor]):
        if torch.is_tensor(data):
            data = data.detach()

        self.sum += data
        self.cnt += 1

    def reset(self):
        self.sum = 0
        self.cnt = 0

    def get(self, reset=True) -> Union[float, torch.Tensor]:
        res = self.sum / self.cnt
        if reset:
            self.reset()

        return res

    def state_dict(self) -> Dict[str, Any]:
        return {k: self.__dict__[k] for k in self.SAVE}

    def load_state_dict(self, state: Dict[str, Any]):
        self.__dict__.update(state or {})


class MovingAverage(Average):
    SAVE = ["sum", "cnt", "history"]

    def __init__(self, window_size: int):
        self.window_size = window_size
        super().__init__()

    def reset(self):
        self.history = []
        super().reset()

    def add(self, data: Union[int, float, torch.Tensor]):
        super().add(data)
        if self.cnt > self.window_size:
            self.sum -= self.history.pop(0)
            self.cnt -= 1

        assert self.cnt <= self.window_size


class DictAverage:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avgs = {}

    def add(self, data: Dict[str, Union[int, float, torch.Tensor]]):
        for k, v in data.items():
            if k not in self.avgs:
                self.avgs[k] = Average()

            self.avgs[k].add(v)

    def get(self, reset=True) -> Dict[str, Union[float, torch.Tensor]]:
        return {k: v.get(reset) for k, v in self.avgs.items()}

    def state_dict(self) -> Dict[str, Any]:
        return {k: v.state_dict() for k, v in self.avgs.items()}

    def load_state_dict(self, state: Dict[str, Any]):
        self.avgs = {k: Average() for k in state.keys()}
        for k, v in state.items():
            self.avgs[k].load_state_dict(v)
