import os
import torch.distributed
from typing import Tuple
import datetime
from typing import Optional

hostlist = None


def has_extra_work(len: int, world_size: Optional[int], rank: Optional[int]) -> bool:
    if (world_size or 1) == 1:
        return False

    rem = len % world_size
    return rank < rem


def is_work_uneven(len: int, world_size: Optional[int]) -> bool:
    return world_size is not None and len % world_size != 0


def get_work_slice(len: int, world_size: Optional[int], rank: Optional[int]) -> Tuple[int, int]:
    if (world_size or 1) == 1:
        return 0, len

    assert rank is not None, "If world_size > 1, rank must be specified"
    rem = len % world_size

    real_batch_size = len // world_size
    batch_offset = real_batch_size * rank + min(rank, rem)
    real_batch_size += int(rank < rem)

    return batch_offset, real_batch_size


class DistributedEnv:
    is_distributed: bool
    rank: int
    world_size: int

    def __init__(self) -> None:
        self.is_distributed = False

        self.detect_env()

        if not self.is_distributed:
            self.world_size = 1
            self.rank = 0

    def detect_env(self):
        pass

    def init_env(self):
        pass

    def is_master(self):
        return (not self.is_distributed) or (self.rank == 0)

    def has_extra_work(self, work_size: int) -> bool:
        return self.is_distributed and has_extra_work(work_size, self.world_size, self.rank)

    def is_work_uneven(self, work_size: int) -> bool:
        return self.is_distributed and is_work_uneven(work_size, self.world_size)

    def __repr__(self) -> str:
        if not self.is_distributed:
            return "No distributed environment detected"
        else:
            return f"Distributed environment: {self.__class__.__name__}. World size: {self.world_size}, rank: {self.rank}"


class SLURMEnv(DistributedEnv):
    def detect_env(self):
        global hostlist

        self.rank = os.getenv("SLURM_PROCID")
        self.world_size = os.getenv("SLURM_NPROCS")
        self.hostnames = os.getenv('SLURM_JOB_NODELIST')
        self.gpu_ids = os.getenv('SLURM_STEP_GPUS')

        self.is_distributed = self.rank is not None and self.world_size is not None and self.hostnames is not None and\
                              self.gpu_ids is not None

        if self.is_distributed:
            if hostlist is None:
                import hostlist

            self.rank = int(self.rank)
            self.world_size = int(self.world_size)
            self.hostnames = hostlist.expand_hostlist(self.hostnames)
            self.gpu_ids = self.gpu_ids.split(",")

            self.port = 12345 + int(min(self.gpu_ids))

            self.is_distributed = self.world_size > 1


    def init_env(self):
        if not self.is_distributed:
            return

        print(f"Initializing distributed environment. World size: {self.world_size}, master: {self.hostnames[0]}, "
              f"my rank {self.rank}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpu_ids)
        torch.distributed.init_process_group('nccl', rank=self.rank, world_size=self.world_size,
                                             init_method=f"tcp://{self.hostnames[0]}:{self.port}",
                                             timeout=datetime.timedelta(0, 6000))

    def __repr__(self) -> str:
        r = super().__repr__()
        if self.is_distributed:
            r = r + f" hostnames: {self.hostnames}, gpu_ids: {self.gpu_ids}"
        return r


class LocalEnv(DistributedEnv):
    def detect_env(self):
        self.world_size = os.getenv('WORLD_SIZE')
        self.rank = os.getenv('RANK')
        self.local_rank = os.getenv('LOCAL_RANK')
        self.master_addr = os.getenv('MASTER_ADDR')
        self.master_port = os.getenv('MASTER_PORT')

        self.is_distributed = self.rank is not None and self.world_size is not None and \
                              self.master_addr is not None and self.master_port is not None \
                              and self.local_rank is not None

        if self.is_distributed:
            self.rank = int(self.rank)
            self.local_rank = int(self.local_rank)
            self.world_size = int(self.world_size)

    def init_env(self):
        if not self.is_distributed:
            return

        torch.cuda.set_device(self.local_rank)

        print(f"Initializing local multigpu environment. World size: {self.world_size}, my rank {self.rank}")
        torch.distributed.init_process_group('nccl', rank=self.rank, world_size=self.world_size,
                                             init_method=f"tcp://{self.master_addr}:{self.master_port}")
        print("Done initializing local multigpu environment")

    def __repr__(self) -> str:
        r = super().__repr__()
        if self.is_distributed:
            r = r + f" local_rank: {self.local_rank}, master_addr: {self.master_addr}, master_port: {self.master_port}"
        return r


def get_dist_env() -> DistributedEnv:
    envs = [SLURMEnv, LocalEnv]

    for env in envs:
        e = env()
        if e.is_distributed:
            return e

    return DistributedEnv()
