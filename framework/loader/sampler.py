import torch
import torch.utils.data
from .. import utils
from ..helpers.distributed import get_work_slice
import numpy as np
import math
from typing import Dict, Any, List, Optional, Tuple
from framework.utils import parallel_map

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset, replacement=False, seed=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self.seed = utils.seed.get_randstate(seed)

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            while True:
                yield self.seed.randint(0, n, dtype=np.int64)
        else:
            i_list = None
            pos = n
            while True:
                if pos >= n:
                    i_list = self.seed.permutation(n).tolist()
                    pos = 0

                sample = i_list[pos]
                pos += 1
                yield sample

    def __len__(self):
        return 0x7FFFFFFF


class FixedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset):
        super().__init__(data_source)
        self.data_source = data_source
        self.order = utils.seed.get_randstate(0xB0C1FA53).permutation(len(self.data_source)).tolist()

    def __iter__(self):
        for i in self.order:
            yield i

    def __len__(self):
        return len(self.data_source)


class SubsetSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset, n_max: int):
        super().__init__(data_source)
        self.data_source = data_source
        self._len = min(len(self.data_source), n_max)
        self.order = utils.seed.get_randstate(0xB0C1FA53).choice(len(self.data_source), self._len, replace=False)

    def __iter__(self):
        for i in self.order:
            yield i

    def __len__(self):
        return self._len


class MultibatchSequentialSampler:
    def __init__(self, data_source: torch.utils.data.Dataset, batch_size: int, world_size: int = 1,
                 rank: Optional[int] = None):

        self.ds_len = len(data_source)
        self.len = self.ds_len // batch_size
        self.pos = None

        self.batch_offset, self.batch_size = get_work_slice(batch_size, world_size, rank)

    def __iter__(self):
        if self.pos is None:
            self.pos = 0

        while self.pos < self.len:
            p = self.pos
            self.pos += 1
            yield [b * self.len + p for b in range(self.batch_offset, self.batch_offset + self.batch_size)]

        self.pos = None

    def __len__(self):
        return self.len

    def state_dict(self) -> Dict[str, Any]:
        return {"pos": self.pos}

    def load_state_dict(self, state: Dict[str, Any]):
        self.pos = state["pos"]


class BucketedSampler(torch.utils.data.Sampler):
    def get_data_slice(self, s: int) -> Tuple[int, int]:
        return get_work_slice(s, self.world_size, self.rank)

    def get_variance(self, k):
        l = [self.data_source[i][k] for i in range(len(self.data_source))]
        return (sum([a**2 for a in l]) - sum(l)**2/len(l))/(len(l)-1)

    def __init__(self, data_source: torch.utils.data.Dataset, batch_size: int,
                 length_key_names: List[str] = ["in_len", "out_len"],
                 infinite: bool = False, seed: Optional[int] = None, drop_last: bool = False, long_first: bool = False,
                 random_order: bool = True, world_size: int = 1, rank: Optional[int] = None):
        super().__init__(data_source)

        self.data_source = data_source
        variances = [self.get_variance(n) for n in length_key_names]
        length_key_names = [length_key_names[i] for i in sorted(range(len(variances)), key=lambda i: variances[i], reverse=True)]

        self.lens = [tuple(data_source[i][k] for k in length_key_names) for i in range(len(data_source))]
        self.batch_size = batch_size
        self.seed = np.random.RandomState(seed)
        self.infinite = infinite
        self.drop_last = drop_last
        self.reverse = long_first
        self.random_order = random_order
        assert (not long_first) or (not self.random_order)
        self.world_size = world_size or 1
        self.rank = rank
        assert (self.world_size == 1) or (seed is not None)

        self.last_batch_size = None

        assert (not drop_last) or (batch_size <= len(data_source))

        if self.world_size > 1:
            self.batch_offset, self.real_batch_size = self.get_data_slice(self.batch_size)

            # The size of the last batch might differ from the others
            last_batch_size = len(data_source) % batch_size
            if (not self.drop_last) and (last_batch_size != 0):
                if last_batch_size < self.world_size:
                    # If the last batch is smaller than the world size, some workers would not have anything to do
                    # Handling this is very painful, just drop the batch. The world size should not be so big anyways.
                    print("WARNING: last_batch_size < world_size. Setting drop_last anyways.")
                    self.drop_last = True
                else:
                    self.last_batch_offset, self.last_batch_size = self.get_data_slice(last_batch_size)

    def makebins(self) -> List[List[int]]:
        # First shuffle all
        order = self.seed.permutation(len(self.lens)).tolist()
        if self.drop_last and (len(order) % self.batch_size) != 0:
            order = order[:-(len(order) % self.batch_size)]
        # Sort preverses the order of the same-length elements, thus the previous shuffle makes random elements to be
        # binned together
        order = list(sorted(order, key=lambda i: self.lens[i], reverse=self.reverse))
        return [order[i: i + self.batch_size] for i in range(0, len(order), self.batch_size)]

    def __iter__(self):
        while True:
            batches = self.makebins()

            if self.random_order:
                if not self.infinite:
                    # Make sure that the last, incomplete batch is always the last
                    i = self.seed.permutation(len(batches) - 1).tolist() + [len(batches) - 1]
                else:
                    i = self.seed.permutation(len(batches))
            else:
                i = range(len(batches))

            for o in i:
                indices = batches[o]
                if self.world_size > 1:
                    if (o == len(batches) - 1) and (self.last_batch_size is not None):
                        yield indices[self.last_batch_offset: self.last_batch_offset + self.last_batch_size]
                    else:
                        yield indices[self.batch_offset: self.batch_offset + self.real_batch_size]
                else:
                    yield indices

            if not self.infinite:
                break

    def __len__(self):
        return math.ceil(len(self.lens) / self.batch_size)
