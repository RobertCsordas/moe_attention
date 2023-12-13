from typing import List, Dict, Callable
from .stat_tracker import StatTracker
from typing import Any


def construct_name(config_names: List[str], get_name: Callable[[str], str]) -> str:
    return "/".join([f"{c}_{get_name(c)}" for c in config_names])


def group(runs, config_names: List[str], get_config=None) -> Dict[str, Any]:
    if get_config is None:
        get_config = lambda run, name: run.config[name]

    res = {}
    for r in runs:
        cval = construct_name(config_names, lambda name: get_config(r, name))
        if cval not in res:
            res[cval] = []

        res[cval].append(r)

    return res

def calc_stat(group_of_runs: Dict[str, List[Any]], filter, tracker=StatTracker) -> Dict[str, Dict[str, StatTracker]]:
    all_stats = {}

    for k, rn in group_of_runs.items():
        if k not in all_stats:
            all_stats[k] = {}

        stats = all_stats[k]

        for r in rn:
            for k, v in r.summary.items():
                if not filter(k):
                    continue

                if k not in stats:
                    stats[k] = tracker()

                stats[k].add(v)

    return all_stats
