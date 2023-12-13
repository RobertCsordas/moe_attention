from typing import Any, Dict
import torch
import torch.nn
from .wikitext103_sp_transformer import Wikitext103SPTransformer
from .enwik8_transformer import Enwik8Transformer
from ... import task, args
import dataset
import framework
from framework.helpers import TrainingHelper
from layers.transformer import PrelnRelativeTransformerEncoderLayer


class TransformerReluCountAnalyzeMixin:
    helper: framework.helpers.TrainingHelper

    def __init__(self, helper: TrainingHelper):
        super().__init__(helper)
        if not isinstance(self.model.layers[0], PrelnRelativeTransformerEncoderLayer):
            raise ValueError("Expected PrelnRelativeTransformerEncoderLayer")

        self.counts = {}
        self.sq_counts = {}
        self.norm = {}
        self.mod_id_to_layer = {}

        def fw_pre_hook(m, inputs):
            i = self.mod_id_to_layer[id(m)]
            inp = inputs[0]
            print("hook", i, inp.shape)
            cl = (inp > 0).float().sum(-1)
            self.counts[i] = self.counts.get(i, 0) + cl.sum()
            self.sq_counts[i] = self.sq_counts.get(i, 0) + (cl**2).sum()

            self.norm[i] = self.norm.get(i, 0) + cl.numel()

        for i, l in enumerate(self.model.layers):
            self.mod_id_to_layer[id(l.linear2)] = i

            l.linear2.register_forward_pre_hook(fw_pre_hook)

    def validate(self) -> Dict[str, Any]:
        self.counts = {}
        self.norm = {}

        res = super().validate()

        means = {k: (c/self.norm[k]).item() for k, c in self.counts.items()}
        stds = {k: ((self.sq_counts[k]/self.norm[k] - means[k]**2)**0.5).item() for k in self.counts.keys()}
        torch.save({
            "means": means,
            "stds": stds,
        }, "counts.pth")


@task()
class Wikitext103SPTransformerAnalyze(TransformerReluCountAnalyzeMixin, Wikitext103SPTransformer):
    pass


@task()
class Enwik8TransformerAnalyze(TransformerReluCountAnalyzeMixin, Enwik8Transformer):
    pass

