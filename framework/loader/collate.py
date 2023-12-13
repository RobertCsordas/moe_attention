import numpy as np
import torch
from operator import mul
from functools import reduce
from typing import List
import warnings


class VarLengthCollate:
    def __init__(self, ignore_symbol=0, batch_dim: int = 1):
        self.ignore_symbol = ignore_symbol
        self.batch_dim = batch_dim

    @staticmethod
    def _measure_array_max_dim(batch: List[torch.Tensor]):
        s=list(batch[0].size())
        different=[False] * len(s)
        for i in range(1, len(batch)):
            ns = batch[i].size()
            different = [different[j] or s[j]!=ns[j] for j in range(len(s))]
            s=[max(s[j], ns[j]) for j in range(len(s))]
        return s, different

    def _merge_var_len_array(self, batch: List[torch.Tensor]):
        max_size, different = self._measure_array_max_dim(batch)
        s=max_size[:self.batch_dim] + [len(batch)] + max_size[self.batch_dim:]
        storage = batch[0].storage()._new_shared(reduce(mul, s, 1))
        out = batch[0].new(storage).view(s).fill_(self.ignore_symbol if self.ignore_symbol is not None else 0)
        for i, d in enumerate(batch):
            bdim = self.batch_dim if len(out.shape)>self.batch_dim else 0
            this_o = out.narrow(bdim, i,  1).squeeze(bdim)
            for j, diff in enumerate(different):
                if different[j]:
                    this_o = this_o.narrow(j, 0, d.size(j))

            this_o.copy_(d)
        return out

    def __call__(self, batch):
        if isinstance(batch[0], dict):
            return {k: self([b[k] for b in batch]) for k in batch[0].keys()}
        elif isinstance(batch[0], np.ndarray):
            with warnings.catch_warnings():
                # If the source data is mmapped from a file, from_numpy will throw a warning that it is readonly.
                # However it does not matter, since all batches will be merged anyway, which copies the data.
                warnings.filterwarnings("ignore", category=UserWarning)
                return self([torch.from_numpy(a) for a in batch])
        elif torch.is_tensor(batch[0]):
            return self._merge_var_len_array(batch)
        elif isinstance(batch[0], (int, float, np.int16, np.int8, np.int32)):
            return torch.Tensor(batch)
        else:
            assert False, "Unknown type: %s" % type(batch[0])


class MetaCollate:
    def __init__(self, meta_name="meta", collate=VarLengthCollate()):
        self.meta_name = meta_name
        self.collate = collate

    def __call__(self, batch):
        if isinstance(batch[0], dict):
            meta = [b[self.meta_name] for b in batch]
            batch = [{k: v for k, v in b.items() if k != self.meta_name} for b in batch]
        else:
            meta = None

        res = self.collate(batch)
        if meta is not None:
            res[self.meta_name] = meta

        return res
