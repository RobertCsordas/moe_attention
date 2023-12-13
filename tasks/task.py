from numpy.lib.function_base import digitize
import framework
from interfaces import Result, ModelInterface
import torch
import torch.utils.data
from tqdm import tqdm
from typing import Dict, Any, Iterable, Tuple, Optional
import os
import optimizer
from dataclasses import dataclass
import torch.distributed
import time
from layers import logging_layer


@dataclass
class LastBestMarker:
    iter: int
    loss: float
    accuracy: float


class Task:
    valid_loaders: framework.data_structures.DotDict
    model_interface: ModelInterface
    batch_dim: int
    TRAIN_NUM_WORKERS = 1
    VALID_NUM_WORKERS = 1

    def __init__(self, helper: framework.helpers.TrainingHelper):
        self.helper = helper
        self.helper.state.best_losses = {}
        self.helper.state.best_accuracies = {}
        self.valid_sets = framework.data_structures.DotDict()
        self.loss_average = framework.utils.Average()
        self.forward_time_meter = framework.utils.ElapsedTimeMeter()
        self.load_time_meter = framework.utils.ElapsedTimeMeter()
        self.plot_time_meter = framework.utils.ElapsedTimeMeter()
        self.total_n_token_in_period = 0
        self.last_token_measure_time = time.time()

    def create_lr_scheduler(self):
        if self.helper.args.lr_sched.type == "step":
            self.lr_scheduler = optimizer.StepLrSched(self.helper.args.lr, self.helper.args.lr_sched.steps,
                                                      self.helper.args.lr_sched.gamma)
        elif self.helper.args.lr_sched.type == "cos":
            if self.helper.args.stop_after is None:
                raise ValueError("Cosine annealing requires stop_after to be set")
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.helper.args.stop_after)
        else:
            assert False

    def use_length_bucketed(self, vset: torch.utils.data.Dataset) -> bool:
        return "in_len" in vset[0]

    def create_valid_loader(self, vset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        # Do bucketed testing even when the bucketed training is not enabled
        if self.use_length_bucketed(vset):
            batch_size = self.test_batch_size
            batch_sampler = framework.loader.sampler.BucketedSampler(vset, batch_size, infinite=False, long_first=True,
                                                                     random_order=False,
                                                                     world_size=self.helper.dist_env.world_size,
                                                                     rank=self.helper.dist_env.rank,
                                                                     seed=0)
            batch_size = 1
        else:
            batch_size = self.helper.get_batch_size(self.test_batch_size)
            if self.helper.dist_env.is_distributed:
                vset = framework.loader.DatasetSplitter(vset, self.helper.dist_env.world_size, self.helper.dist_env.rank)
            batch_sampler = None

        return torch.utils.data.DataLoader(vset, batch_size=batch_size, batch_sampler=batch_sampler,
                                   collate_fn=framework.loader.collate.VarLengthCollate(batch_dim=self.batch_dim),
                                   num_workers=self.VALID_NUM_WORKERS, persistent_workers=self.VALID_NUM_WORKERS > 0)


    def create_loaders(self):
        self.train_loader = self.create_train_loader(self.train_set, mask = False)
        self.valid_loaders = framework.data_structures.DotDict()
        self.valid_loaders.update({k: self.create_valid_loader(v) for k, v in self.valid_sets.items()})

    def replace_valid_set(self, name: str, vset: torch.utils.data.Dataset):
        self.valid_sets[name] = vset
        self.valid_loaders[name] = self.create_valid_loader(vset)

    def create_train_loader_bs(self, loader: torch.utils.data.Dataset, batch_size: int, seed: Optional[int] = None) \
                            -> torch.utils.data.DataLoader:

        if self.helper.args.length_bucketed_sampling and self.use_length_bucketed(loader):
            batch_sampler = framework.loader.sampler.BucketedSampler(loader, batch_size, infinite=True, drop_last=True,
                                                                     random_order=True,
                                                                     world_size=self.helper.dist_env.world_size,
                                                                     rank=self.helper.dist_env.rank,
                                                                     seed=0)
            sampler = None
            batch_size = 1
        else:
            batch_size = self.helper.get_batch_size(batch_size)
            if self.helper.dist_env.is_distributed:
                loader = framework.loader.DatasetSplitter(loader, self.helper.dist_env.world_size,
                                                          self.helper.dist_env.rank)

            batch_sampler = None
            sampler = framework.loader.sampler.InfiniteSampler(loader, seed = seed)

        return torch.utils.data.DataLoader(loader, batch_size=batch_size,
                                           sampler=sampler, batch_sampler=batch_sampler,
                                           collate_fn=framework.loader.collate.VarLengthCollate(
                                               batch_dim=self.batch_dim),
                                           num_workers=self.TRAIN_NUM_WORKERS, pin_memory=True,
                                           persistent_workers=self.TRAIN_NUM_WORKERS > 0)

    def create_validate_on_train(self, set: torch.utils.data.Dataset):
        self.valid_sets.train = set

        if self.helper.dist_env.is_distributed:
            set = framework.loader.DatasetSplitter(set, self.helper.dist_env.world_size,
                                                      self.helper.dist_env.rank)

        self.valid_loaders.train = torch.utils.data.DataLoader(set, batch_size=self.helper.get_batch_size(),
                                   collate_fn=framework.loader.collate.VarLengthCollate(batch_dim=self.batch_dim),
                                   sampler=framework.loader.sampler.SubsetSampler(set, (len(self.valid_sets.iid)
                                        if "iid" in self.valid_sets else 1000) // self.helper.dist_env.world_size),
                                   num_workers=self.VALID_NUM_WORKERS, persistent_workers=self.VALID_NUM_WORKERS > 0)

    def clip_gradients(self):
        if self.helper.args.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.helper.args.grad_clip)

    def set_optimizer_lr(self, lr: float):
        framework.utils.set_lr(self.optimizer, lr)

    def set_linear_warmup(self, curr_step: int, n_steps: int, final: float) -> float:
        if curr_step >= n_steps:
            lr = final
        else:
            lr = final / n_steps * (curr_step+1)

        self.set_optimizer_lr(lr)
        return lr

    def set_lr(self):
        has_builtin_warmup = self.helper.args.lr_sched.type in {"noam"}
        offset = 0 if has_builtin_warmup else self.helper.args.lr_warmup

        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
            self.lr_scheduler.step(max(0, self.helper.state.iter - offset))
        else:
            self.set_optimizer_lr(self.lr_scheduler.get(max(0, self.helper.state.iter - offset)))

        if self.helper.args.lr_warmup > self.helper.state.iter and not has_builtin_warmup:
            self.set_linear_warmup(self.helper.state.iter, self.helper.args.lr_warmup,
                                   framework.utils.get_lr(self.optimizer))

        if self.helper.state.iter % 100 == 0:
            self.helper.log({"lr": framework.utils.get_lr(self.optimizer)})

    def prepare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.helper.to_device(data)

    def run_model(self, data: Dict[str, torch.Tensor], ubatch: int = 0) -> Tuple[Result, Dict[str, Any]]:
        return self.model_interface(data, self.helper.state.iter, ubatch)

    def validate_on(self, set: torch.utils.data.Dataset, loader: torch.utils.data.DataLoader) -> Tuple[Any, float]:
        self.model.eval()

        bucketed = self.use_length_bucketed(set)
        with torch.no_grad():
            loss_sum = 0
            count = 0

            test = set.start_test()
            l = len(loader)
            lmax = l

            if self.helper.dist_env.is_distributed:
                lmax = torch.tensor(lmax, dtype=torch.int32).cuda()
                torch.distributed.all_reduce(lmax, torch.distributed.ReduceOp.MAX)
                lmax = lmax.item()

            for d in tqdm(loader):
                d = self.prepare_data(d)
                res, _ = self.run_model(d)
                digits = self.model_interface.decode_outputs(res)
                this_loss = res.loss.sum().item() * res.batch_size

                if self.helper.dist_env.is_distributed:
                    # gather is not supported, so send to everyone
                    alist = [None] * self.helper.dist_env.world_size
                    torch.distributed.all_gather_object(alist, (digits, d, this_loss, res.batch_size))

                    for digits, d, this_loss, bs in alist:
                        if digits is None:
                            continue
                        loss_sum += this_loss
                        count += bs
                        test.step(digits, d)
                else:
                    loss_sum += this_loss
                    count += res.batch_size
                    test.step(digits, d)

            for _ in range(lmax - l):
                # if the work is not even for all workers, send dummy messages around to not get blocked
                alist = [None] * self.helper.dist_env.world_size
                torch.distributed.all_gather_object(alist, (None, None, None, None))

        print(f"Validation done on worker {self.helper.dist_env.rank}.")
        self.model.train()
        return test, loss_sum / count

    def validate_on_name(self, name: str) -> Tuple[Any, float]:
        return self.validate_on(self.valid_sets[name], self.valid_loaders[name])

    def fix_loaded_best_losses(self):
        # Loading destroys the class.
        to_fix = [self.helper.state.best_losses, self.helper.state.best_accuracies]
        for f in to_fix:
            for k, v in f.items():
                if isinstance(v, dict):
                    new_v = LastBestMarker(0, 0, 0)
                    new_v.__dict__.update(v)
                    f[k] = new_v

    def update_best_accuracies(self, name: str, accuracy: float, loss: float):
        self.fix_loaded_best_losses()

        if name not in self.helper.state.best_losses or loss < self.helper.state.best_losses[name].loss:
                self.helper.state.best_losses[name] = LastBestMarker(self.helper.state.iter, loss, accuracy)

        if name not in self.helper.state.best_accuracies or accuracy > \
                self.helper.state.best_accuracies[name].accuracy:
            self.helper.state.best_accuracies[name] = LastBestMarker(self.helper.state.iter, loss, accuracy)

        return {
            f"{name}/time_since_best_loss": self.helper.state.iter - self.helper.state.best_losses[name].iter,
            f"{name}/time_since_best_accuracy": self.helper.state.iter - self.helper.state.best_accuracies[name].iter
        }

    def validate_on_names(self, name_it: Iterable[str]) -> Dict[str, Any]:
        charts = {}
        sum_accuracy = 0
        sum_all_losses = 0

        logging_layer.get_logs(self.model)

        for name in name_it:
            test, loss = self.validate_on_name(name)

            if self.helper.args.dump_logs:
                logging_layer.dump_logs(self.model, self.helper.get_storage_path("log_dumps") + f"/{self.helper.state.iter}/valid/{name}/")
            logs = logging_layer.get_logs(self.model)

            print(f"Validation accuracy on {name}: {test.accuracy}")
            charts[f"{name}/loss"] = loss
            sum_all_losses += loss
            charts.update({f"{name}/{k}": v for k, v in test.plot().items()})
            if self.helper.args.val_log_details:
                charts.update({f"{name}/{k}": v for k, v in logs.items()})
            sum_accuracy += test.accuracy

            charts.update(self.update_best_accuracies(name, test.accuracy, loss))

        charts["mean_accuracy"] = sum_accuracy / max(len(self.valid_sets), 1)
        charts["mean_loss"] = sum_all_losses / max(len(self.valid_sets), 1)
        return charts

    def validate(self) -> Dict[str, Any]:
        return self.validate_on_names(self.valid_sets.keys())

    def plot(self, res: Result) -> Dict[str, Any]:
        plots = {}

        self.loss_average.add(res.loss)

        if self.helper.state.iter % 200 == 0:
            plots.update(res.plot())

        if self.helper.state.iter % 20 == 0:
            if self.total_n_token_in_period:
                now = time.time()
                plots["timing/total_ms_per_token"] = (now - self.last_token_measure_time)*1000/ \
                                                                     (20*self.total_n_token_in_period)
                plots["timing/ms_per_token"] = self.forward_time_meter.get(False)*1000/ \
                                                                     (20*self.total_n_token_in_period)
                self.total_n_token_in_period = 0
                self.last_token_measure_time = now

            plots["train/loss"] = self.loss_average.get()
            plots["timing/ms_per_iter"] = self.forward_time_meter.get(True)*1000/20
            plots["timing/ms_per_load"] = self.load_time_meter.get(True)*1000/20
            plots["timing/ms_per_plot"] = self.plot_time_meter.get(True)*1000/20

        if self.helper.state.iter % self.helper.args.test_interval == 0:
            plots.update({f"validation/{k}": v for k, v in self.validate().items()})

        return plots

    def train_step_reconfig(self):
        pass

    def create_model_interface(self):
        raise NotImplementedError()

    def create_datasets(self):
        raise NotImplementedError()

    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def save_weights(self):
        pass

    def load_weights(self, file_path: str):
        pass

    def post_train(self):
        pass

    def finish(self):
        pass

    @property
    def test_batch_size(self):
        return self.helper.args.test_batch_size or self.helper.args.batch_size
