import torch
import torch.nn
import torch.optim
import framework
import torch.utils.data
import torch.cuda.amp
from typing import Optional, Dict, Any, Tuple, List
from interfaces import Result
from ..task import Task
from .. import args
from layers import LayerRegularizer
import torch.distributed
from layers.logging_layer import get_logs, dump_logs
from layers.once_per_iter_layer import call_post_iter, call_pre_iter, call_before_loss
from framework.utils import U


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-reg_scales", default="", parser=parser.float_params_parser)
    parser.add_argument("-reg_lin_decay", default="", parser=parser.str_list_parser)


class SimpleTask(Task):
    MAX_LENGHT_PER_BATCH = None
    train_set: torch.utils.data.Dataset
    train_loader: torch.utils.data.DataLoader
    model: torch.nn.Module

    def create_datasets(self):
        raise NotImplementedError()

    def create_model_interface(self):
        raise NotImplementedError()

    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    def create_state(self):
        pass

    @property
    def amp_enabled(self):
        return torch.cuda.is_available() and self.helper.args.amp

    @property
    def time_dim(self) -> int:
        return 1 - self.batch_dim

    def __init__(self, helper: framework.helpers.TrainingHelper):
        super().__init__(helper)

        self.avg_num_chunks = framework.utils.Average()
        self.reg_loss_average = framework.utils.DictAverage()
        self.max_grad = 0
        self.time_sum = 0

        self.create_datasets()
        self.create_loaders()
        self.model = self.create_model()
        self.model = self.model.to(self.helper.device)

        self.create_model_interface()
        self.create_optimizer()
        self.create_lr_scheduler()

        self.regularizer = LayerRegularizer(
            self.model, self.helper.args.stop_after, self.helper.args.reg_scales,  self.helper.args.reg_lin_decay)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.helper.saver["scaler"] = self.scaler

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of model parameters: {n_params}")

        self.helper.saver["model"] = self.model
        self.create_state()
        self.helper.restore()

        self.fetcher = None
        self.helper.log({"n_params": n_params})

        if self.helper.args.nan_detect:
            # based on https://discuss.pytorch.org/t/finding-source-of-nan-in-forward-pass/51153/3
            def nan_hook(self, inp, output):
                if not isinstance(output, tuple):
                    outputs = [output]
                else:
                    outputs = output

                for i, out in enumerate(outputs):
                    def detect(out):
                        nan_mask = ~torch.isfinite(out)
                        if nan_mask.any():
                            print("In", self.__class__.__name__)
                            raise RuntimeError(f"Found non-finite in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

                    U.apply_recursive(out, detect, torch.is_tensor)

            for submodule in self.model.modules():
                submodule.register_forward_hook(nan_hook)

    def fetch_thread(self):
        data = self.prepare_data(self.get_train_batch())
        n_chunks = self.get_n_chunks(data)
        d_chunks = self.chunk_batch_dim(data, n_chunks)

        return data, d_chunks

    def create_train_loader(self, loader: torch.utils.data.Dataset, seed: Optional[int] = None,
                            batch_size: Optional[int] = None) -> torch.utils.data.DataLoader:

        return super().create_train_loader_bs(loader, batch_size or self.helper.args.batch_size, seed)

    def set_train_set(self, ds: torch.utils.data.Dataset, seed: Optional[int] = None):
        self.train_set = ds
        self.train_loader = self.create_train_loader(self.train_set, seed)
        self.data_iter = iter(self.train_loader)

    def create_loaders(self):
        self.train_loader = self.create_train_loader(self.train_set)
        self.valid_loaders = framework.data_structures.DotDict()
        self.valid_loaders.update({k: self.create_valid_loader(v) for k, v in self.valid_sets.items()})

    def get_optimizer_param_list(self):
        return self.model.parameters()

    def create_optimizer(self):
        if self.helper.args.optimizer in ["adam", "adamw"]:
            opt = torch.optim.Adam if self.helper.args.optimizer == "adam" else torch.optim.AdamW
            self.set_optimizer(opt(self.get_optimizer_param_list(), self.helper.args.lr,
                                                weight_decay=self.helper.args.wd, betas=self.helper.args.adam.betas,
                                                eps=self.helper.args.adam.eps))
        elif self.helper.args.optimizer == "adagrad":
            self.set_optimizer(torch.optim.Adagrad(self.get_optimizer_param_list(), self.helper.args.lr,
                                                    weight_decay=self.helper.args.wd))
        elif self.helper.args.optimizer == "sgd":
            self.set_optimizer(torch.optim.SGD(self.get_optimizer_param_list(), self.helper.args.lr,
                                               weight_decay=self.helper.args.wd, momentum=0.9))
        else:
            assert False, f"Unsupported optimizer: {self.helper.args.optimizer}"

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.helper.saver.register("optimizer", self.optimizer, replace=True)

    def get_train_batch(self) -> Dict[str, Any]:
        return next(self.data_iter)

    def chunk_batch_dim(self, data: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
        if n == 1:
            return [data]

        res = [{} for _ in range(n)]
        for k, v in data.items():
            assert torch.is_tensor(v), "Only tensors are supported by autosplitting"

            bd = self.batch_dim if self.batch_dim < v.ndimension() else 0
            assert v.shape[bd] % n == 0, f"Batch (dim {bd} of input {k} of shape {v.shape} is not divisible by {n})"

            for i, c in enumerate(v.chunk(n, dim=bd)):
                res[i][k] = c

        # Avoid unnecessary computation.
        if "in" in data and "in_len" in data:
            for r in res:
                r["in"] = r["in"].narrow(1 - self.batch_dim, 0, int(r["in_len"].max().item()))

        if "out" in data and "out_len" in data and data["out"].ndim > 1:
            for r in res:
                r["out"] = r["out"].narrow(1 - self.batch_dim, 0, int(r["out_len"].max().item()))

        return res

    def is_seq2seq_task(self, data: Dict[str, Any]) -> bool:
        return "in_len" in data and "out_len" in data

    def get_seq_length(self, data: Dict[str, Any]) -> int:
        # This assumes separate encoder and decoder
        return max(data["in"].shape[self.time_dim], data["out"].shape[self.time_dim] if data["out"].ndim > 1 else 0)

    def get_n_chunks(self, data: Dict[str, Any]) -> int:
        if self.helper.args.n_microbatch:
            return self.helper.args.n_microbatch

        max_length_per_batch = self.helper.args.max_length_per_batch or self.MAX_LENGHT_PER_BATCH
        if self.is_seq2seq_task(data) and max_length_per_batch:
            # The formula below assumes quadratic memory consumption
            return int(2**int(self.get_seq_length(data) / max_length_per_batch))
        return 1

    def post_backward(self) -> Dict[str, Any]:
        return {}

    def get_batch_size(self, data: Dict[str, Any]) -> int:
        for v in data.values():
            if torch.is_tensor(v) and v.ndim > self.batch_dim:
                return v.shape[self.batch_dim]

        raise ValueError("Unable to automatically determine the local batch size.")

    def train_step(self) -> Tuple[Result, Dict[str, Any]]:
        plots = {}

        if self.helper.args.speedtest=="iter":
            torch.cuda.synchronize()

        with self.forward_time_meter:
            self.set_lr()
            self.optimizer.zero_grad(set_to_none=True)

            data, d_chunks = self.fetcher.get()

            res_list = []
            weights = []

            self.avg_num_chunks.add(len(d_chunks))

            total_batch_size = self.get_batch_size(data)

            profiler = None
            # if self.helper.state.iter == 3:
            #     profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True, profile_memory=True)
            #     profiler.__enter__()


            call_pre_iter(self.model)
            for ubatch, d in enumerate(d_chunks):
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    res, custom_plots = self.run_model(d, ubatch)
                    call_before_loss(self.model)
                    res_list.append(res)
                    if ubatch == 0:
                        plots.update(custom_plots)

                # weights for microbatch accumulation
                weights.append(self.get_batch_size(d) / total_batch_size)
                reg_loss, reg_log = self.regularizer.get(self.helper.state.iter)
                self.reg_loss_average.add(reg_log)
                total_loss = (res_list[-1].loss + reg_loss * self.helper.args.reg) * self.helper.get_loss_scaling()

                if not torch.isfinite(total_loss):
                    for n, p in self.model.named_parameters():
                        if not torch.isfinite(p).all():
                            print(f"Found non-finite weight {n}")

                    for n, p in self.model.named_buffers():
                        if not torch.isfinite(p).all():
                            print(f"Found non-finite buffer {n}")

                    assert False, f"Loss not finite ({total_loss})"

                self.scaler.scale(total_loss * weights[-1]).backward()
                pbwout = self.post_backward()
                if ubatch == 0:
                    plots.update(pbwout)

            if self.helper.dist_env.is_distributed:
                aops = []
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    aops.append(torch.distributed.all_reduce(p.grad.contiguous(), async_op=True))

                for a in aops:
                    a.wait()


            call_post_iter(self.model)

            self.scaler.unscale_(self.optimizer)

            if self.helper.args.grad_clip:
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.helper.args.grad_clip)
                self.max_grad = max(self.max_grad, gn)


            if self.helper.args.log_grad_norms:
                for n, p in self.model.named_parameters():
                    plots[f"grad_norms/{n}"] = p.detach().norm().item()


            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.helper.state.iter += 1
            res = res_list[0].__class__.merge(res_list, weights)

            if self.helper.args.speedtest in {"iter"}:
                torch.cuda.synchronize()

            if profiler is not None:
                profiler.__exit__(None, None, None)
                profiler.export_chrome_trace("trace_all.json")
                assert False


            # if self.helper.state.iter % 20 == 0:

        if "in_len" in data:
            n_total_tokens = (data["in_len"] + data["out_len"]).sum()
            if self.helper.dist_env.is_distributed:
                torch.distributed.all_reduce(n_total_tokens)

            self.total_n_token_in_period += n_total_tokens

        return res, plots

    def plot(self, res: Result) -> Dict[str, Any]:
        res = super().plot(res)

        if self.helper.args.dump_logs and self.helper.dist_env.is_master():
            dump_logs(self.model, self.helper.get_storage_path("log_dumps") + f"/{self.helper.state.iter}")

        if self.helper.state.iter % 20 == 1:
            res.update(get_logs(self.model))

            res["average_num_chunks"] = self.avg_num_chunks.get()
            for k, v in self.reg_loss_average.get().items():
                res[f"train/reg_loss/{k}"] = v

            if self.helper.args.grad_clip:
                res["max_grad"] = self.max_grad
                self.max_grad = 0

        return res

    def train(self):
        self.loss_average.reset()

        self.data_iter = iter(self.train_loader)
        self.fetcher = framework.helpers.StoppingParallelProducer(self.fetch_thread)

        try:
            while (self.helper.args.stop_after or 10e10) > self.helper.state.iter:
                self.load_time_meter.stop()

                res, plots = self.train_step()
                plots.update(self.plot(res))

                with self.plot_time_meter:
                    self.helper.log(plots)

                self.load_time_meter.start()

                self.helper.tick()
        except self.fetcher.Stopped:
            pass
