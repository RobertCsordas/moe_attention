from torch import multiprocessing
from typing import Iterable, Callable, Any, List, Optional
import time
import math


class ParallelMapPool:
    def __init__(self, callback: Callable[[Any], Any], max_parallel: Optional[int] = None, debug: bool = False):
        self.n_proc = max_parallel or multiprocessing.cpu_count()
        self.callback = callback
        self.debug = debug

    def __enter__(self):
        if self.debug:
            return self

        self.in_queues = [multiprocessing.Queue() for _ in range(self.n_proc)]
        self.out_queues = [multiprocessing.Queue() for _ in range(self.n_proc)]

        def run_process(in_q: multiprocessing.Queue, out_q: multiprocessing.Queue):
            while True:
                args = in_q.get()
                if args is None:
                    break
                res = [self.callback(a) for a in args]
                out_q.put(res)

        self.processes = [multiprocessing.Process(target=run_process, args=(iq, oq), daemon=True)
                          for iq, oq in zip(self.in_queues, self.out_queues)]
        for p in self.processes:
            p.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.debug:
            return

        for q in self.in_queues:
            q.put(None)

        for p in self.processes:
            p.join()

        self.in_queues = None
        self.out_queues = None

    def map(self, args: List) -> List:
        if len(args) == 0:
            return []

        if self.debug:
            return [self.callback(a) for a in args]

        a_per_proc = math.ceil(len(args) / self.n_proc)
        chunks = [args[a : a + a_per_proc] for a in range(0, len(args), a_per_proc)]

        for q, c in zip(self.in_queues, chunks):
            q.put(c)

        res = [q.get() for q in self.out_queues[:len(chunks)]]
        return sum(res, [])


def parallel_map(tasks: Iterable, callback = Callable[[Any], None], max_parallel: int = 32) -> List:
    tasks = list(tasks)
    if len(tasks) == 0:
        return []

    limit = min(multiprocessing.cpu_count(), max_parallel)
    with ParallelMapPool(callback, max_parallel=limit) as p:
        return p.map(tasks)
