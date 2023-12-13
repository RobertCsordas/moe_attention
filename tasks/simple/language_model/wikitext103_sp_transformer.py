import torch
from .enwik8_transformer import Enwik8Transformer
from ... import task, args
import dataset
import framework

@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-sentencepiece.n_pieces", default=8000)


@task()
class Wikitext103SPTransformer(Enwik8Transformer):
    helper: framework.helpers.TrainingHelper

    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.Wikitext103SentencePiece("train", self.helper.args.lm.unroll, n_pieces=self.helper.args.sentencepiece.n_pieces)
        self.valid_sets.val = dataset.Wikitext103SentencePiece("valid", self.helper.args.lm.unroll_eval or self.helper.args.lm.unroll, n_pieces=self.helper.args.sentencepiece.n_pieces)
        self.valid_sets.test = dataset.Wikitext103SentencePiece("test", self.helper.args.lm.unroll_eval or self.helper.args.lm.unroll, n_pieces=self.helper.args.sentencepiece.n_pieces)
