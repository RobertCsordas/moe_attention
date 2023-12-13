import torch.utils.data
from typing import Any
from ..helpers.distributed import get_work_slice


class DatasetSplitter(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, n_partitions: int, current: int):
        self.dataset = dataset
        self.my_offset, self.my_len = get_work_slice(len(self.dataset), n_partitions, current)

    def __len__(self) -> int:
        return self.my_len

    def __getitem__(self, item: int) -> Any:
        return self.dataset[self.my_offset + item]
