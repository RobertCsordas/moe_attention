from .enwik8_transformer import Enwik8Transformer
from ... import task, args
import dataset
import framework

@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-sentencepiece.n_pieces", default=8000)
    parser.add_argument("-lmds.valid_ratio", default=1.0)


@task()
class C4Transformer(Enwik8Transformer):
    helper: framework.helpers.TrainingHelper

    def create_datasets(self):
        self.batch_dim = 1

        if self.helper.args.stop_after is not None:
            train_token_limit = self.helper.args.lm.unroll * self.helper.args.batch_size * (self.helper.args.stop_after + 100)
        else:
            train_token_limit = None

        # Magic number for backward compatibility
        test_token_limit = int((206453965 // self.helper.args.lm.unroll) * self.helper.args.lmds.valid_ratio) * self.helper.args.lm.unroll

        self.train_set = dataset.C4(self.helper.args.lm.unroll, split="train", n_tokens=self.helper.args.sentencepiece.n_pieces, token_limit=train_token_limit)
        self.valid_sets.val = dataset.C4(self.helper.args.lm.unroll, split="validation", n_tokens=self.helper.args.sentencepiece.n_pieces, token_limit=test_token_limit)
