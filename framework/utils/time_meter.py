import time


class ElapsedTimeMeter:
    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def _curr_timer(self) -> float:
        if self.start_time is None:
            return 0

        return time.time() - self.start_time

    def stop(self):
        self.sum += self._curr_timer()
        self.start_time = None

    def get(self, reset=False) -> float:
        res = self.sum + self._curr_timer()
        if reset:
            self.reset()
        return res

    def reset(self):
        self.start_time = None
        self.sum = 0

    def __enter__(self):
        assert self.start_time is None
        self.start()

    def __exit__(self, *args):
        self.stop()
