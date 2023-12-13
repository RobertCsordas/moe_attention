# from torch.multiprocessing import Event, Queue, Process
from queue import Queue
from multiprocessing import Event
from threading import Thread
import atexit
from queue import Empty as EmptyQueue, Full as FullQueue


class StoppingParallelLoop:
    def __init__(self, target=None, args=(), kwargs={}):
        self.stop_event = Event()
        atexit.register(self.finish)
        self.target = target

        self.thread = Thread(target=self.infloop, args=(self.stop_event, *args), kwargs=kwargs, daemon=True)
        self.thread.start()

    def infloop(self, stop_event: Event, *args, **kwargs):
        while not stop_event.is_set():
            self.target(*args, **kwargs)

    def finish(self):
        if self.stop_event.is_set():
            return

        self.stop_event.set()
        self.thread.join()


class StoppingParallelProducer(StoppingParallelLoop):
    class Stopped(Exception):
        pass

    def __init__(self, target=None, args=(), kwargs={}, q_size: int = 1):
        self.queue = Queue(q_size)
        super().__init__(target=target, args=args, kwargs=kwargs)

    def infloop(self, stop_event: Event, *args, **kwargs):
        while not stop_event.is_set():
            res = self.target(*args, **kwargs)
            while not stop_event.is_set():
                try:
                    self.queue.put(res, timeout=0.1)
                    break
                except FullQueue:
                    pass

    def get(self):
        while not self.stop_event.is_set():
            try:
                return self.queue.get(timeout=0.1)
            except EmptyQueue:
                pass

        raise self.Stopped()
