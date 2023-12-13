from typing import Callable, List, Any
import torch


def apply_recursive(d, fn: Callable, filter: Callable = None):
    if isinstance(d, list):
        return [apply_recursive(da, fn, filter) for da in d]
    elif isinstance(d, tuple):
        return tuple(apply_recursive(list(d), fn, filter))
    elif isinstance(d, dict):
        return {k: apply_recursive(v, fn, filter) for k, v in d.items()}
    else:
        if filter is None or filter(d):
            return fn(d)
        else:
            return d


def apply_to_tensors(d, fn: Callable):
    return apply_recursive(d, fn, torch.is_tensor)


def reduce_resursive(ds: List[Any], fn: Callable):
    assert len(ds) > 0

    for d in ds:
        assert type(d) == type(ds[0]), "All list elements must be the same"

    if isinstance(ds[0], (list, tuple)):
        r = [reduce_resursive([d[i] for d in ds], fn) for i in range(len(ds[0]))]
        if isinstance(ds[0], tuple):
            r = tuple(r)
        return r
    elif isinstance(ds[0], dict):
        return {k: reduce_resursive([d[k] for d in ds], fn) for k in ds[0].keys()}
    else:
        return fn(ds)
