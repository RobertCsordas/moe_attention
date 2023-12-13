from collections import defaultdict
from typing import Dict, Any, Union


class DotDefaultDict(defaultdict):
    def __getattr__(self, item):
        if item not in self:
            raise AttributeError
        return self.get(item)

    __setattr__ = defaultdict.__setitem__
    __delattr__ = defaultdict.__delitem__


class DotDict(dict):
    def __getattr__(self, item):
        if item not in self:
            raise AttributeError
        return self.get(item)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def create_recursive_dot_dict(data: Dict[str, Any], cls=DotDict) -> Union[DotDict, DotDefaultDict]:
    """
    Takes a dict of string keys and arbitrary values, and creates a tree of DotDicts.

    The keys might contain . in which case child DotDicts are created.

    :param data: Input dict with string keys potentially containing .s.
    :param cls: Either DotDict or DotDefaultDict
    :return: tree DotDict or DotDefaultDict where the keys are split by .
    """
    res = cls()
    for k, v in data.items():
        k = k.split(".")
        target = res
        for i in range(0, len(k)-1):
            t2 = target.get(k[i])
            if t2 is None:
                t2 = cls()
                target[k[i]] = t2

            assert isinstance(t2, cls), f"Trying to overwrite key {'.'.join(k[:i+1])}"
            target = t2

        assert isinstance(target, cls), f"Trying to overwrite key {'.'.join(k)}"
        target[k[-1]] = v
    return res
