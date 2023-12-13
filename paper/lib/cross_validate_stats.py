from lib.stat_tracker import StatTracker, Stat
from typing import List, Dict, Any, Union


class CrossValidatedStats:
    def __init__(self, gen_key: str, valid_key: Union[str, List[str]], extra_keys: List[str] = []) -> None:
        self.gen_key = gen_key
        self.valid_keys = [valid_key] if isinstance(valid_key, str) else valid_key
        self.extra_keys = extra_keys

    def cross_validate_run(self, r, test):
        hist = r.history(keys=[self.gen_key] + self.valid_keys + self.extra_keys, pandas=False)
        # best = max(range(len(hist)), key=lambda d: hist[d][self.valid_key])
        hist = [h for h in hist if test(h)]
        vals = [sum([h[v] for v in self.valid_keys])/len(self.valid_keys) for h in hist]
        best = max(vals)
        for h in reversed(range(len(hist))):
            if vals[h] == best:
                return hist[h][self.gen_key]
        assert False, "This could not happen"

    def __call__(self, group_of_runs: Dict[str, List[Any]], tracker=StatTracker, test = lambda x: True) -> Dict[str, Dict[str, Stat]]:
        res = {}
        for k, rlist in group_of_runs.items():
            t = tracker()
            for r in rlist:
                t.add(self.cross_validate_run(r, test))
            res[k] = t.get()

        return res
