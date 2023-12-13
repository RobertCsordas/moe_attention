import framework
import torch
import torch.nn
import torch.utils.data
from models import TransformerLanguageModel
from ... import task, args
import dataset
from .transformer_lm_mixin import TransformerLMMixin
from ..simple_task import SimpleTask
from typing import Tuple, Any, Dict, List, Union
from interfaces import LanguageModelInterface
import random


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-lm.state_drop_probability", default=0.0)
    parser.add_argument("-lm.lstm_weight_drop", default=0.0)
    parser.add_argument("-lm.unroll", default=100)
    parser.add_argument("-lm.unroll_eval", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-lm.example_context", default=100)
    parser.add_argument("-lm.example_window", default=40)


@task()
class Enwik8Transformer(TransformerLMMixin, SimpleTask):
    VALID_NUM_WORKERS = 1
    TRAIN_NUM_WORKERS = 2

    def create_state(self):
        self.helper.state.epoch = 0

    def create_model_interface(self):
        self.model_interface = LanguageModelInterface(
            self.model, drop_state_prob=self.helper.args.lm.state_drop_probability, dist_env=self.helper.dist_env,
            n_ubatches=self.helper.args.n_microbatch)
        self.helper.saver["interface"] = self.model_interface

    def validate_on(self, set: torch.utils.data.Dataset, loader: torch.utils.data.DataLoader) -> Tuple[Any, float]:
        state = self.model_interface.state
        self.model_interface.reset_state()
        res = super().validate_on(set, loader)
        self.model_interface.state = state
        return res

    def log_epoch(self):
        self.helper.log({"epoch": self.helper.state.epoch})

    def start_next_epoch(self):
        self.model_interface.reset_state()
        self.helper.state.epoch += 1
        self.log_epoch()

    def get_train_batch(self) -> Dict[str, Any]:
        try:
            return next(self.data_iter)
        except StopIteration:
            self.start_next_epoch()
            self.data_iter = iter(self.train_loader)
            return next(self.data_iter)

    def create_sampler(self, loader: torch.utils.data.Dataset, batch_size: int) -> \
                       framework.loader.sampler.MultibatchSequentialSampler:

        return framework.loader.sampler.MultibatchSequentialSampler(loader, batch_size,
                            world_size=self.helper.dist_env.world_size, rank=self.helper.dist_env.rank)

    def create_valid_loader(self, vset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(vset,
                                   batch_sampler=self.create_sampler(vset, self.test_batch_size),
                                   collate_fn=framework.loader.collate.VarLengthCollate(batch_dim=self.batch_dim),
                                   num_workers=self.VALID_NUM_WORKERS)

    def create_train_loader(self, loader: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        sampler = self.create_sampler(loader, self.helper.args.batch_size)
        self.helper.saver.register("sampler", sampler, replace=True)

        return torch.utils.data.DataLoader(loader, batch_sampler=sampler, num_workers=self.TRAIN_NUM_WORKERS,
                                           pin_memory=True, collate_fn=framework.loader.collate.VarLengthCollate(
                                           batch_dim=self.batch_dim))

    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.Enwik8("train", self.helper.args.lm.unroll)
        self.valid_sets.val = dataset.Enwik8("valid", self.helper.args.lm.unroll_eval or self.helper.args.lm.unroll)
        self.valid_sets.test = dataset.Enwik8("test", self.helper.args.lm.unroll_eval or self.helper.args.lm.unroll)

    def train(self):
        self.log_epoch()
        super().train()
