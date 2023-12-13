
import os
from typing import List, Union, Dict, Any, Union, Iterator


class SentencepieceVocabulary:
    def __init__(self, path: str, train_data: Union[str, Iterator], vocab_size: int):
        global spm
        import sentencepiece as spm

        model_file = path + ".model"

        if not os.path.exists(model_file):
            if isinstance(train_data, str):
                spm.SentencePieceTrainer.train(input=train_data, model_prefix=path, vocab_size=vocab_size, split_digits=True, model_type="bpe")
            else:
                spm.SentencePieceTrainer.train(sentence_iterator=train_data, model_prefix=path, vocab_size=vocab_size, split_digits=True, model_type="bpe")

        self.path = path
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(model_file)
        pass

    def __len__(self) -> int:
        return self.tokenizer.get_piece_size()

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]):
        pass

    def indices_to_sentence(self, indices: List[int]) -> List[str]:
        return [self.tokenizer.IdToPiece(i) for i in indices]

    def sentence_to_indices(self, sentence: str) -> List[int]:
        return self.tokenizer.encode_as_ids(sentence)

    def __call__(self, seq: Union[List[Union[str, int]], str]) -> List[Union[int, str]]:
        if seq is None or (isinstance(seq, list) and not seq):
            return seq

        if isinstance(seq, str) or isinstance(seq[0], str):
            return self.sentence_to_indices(seq)
        else:
            return self.indices_to_sentence(seq)

    def to_string(self, seq: List[int]) -> str:
        return self.tokenizer.decode_ids(seq)
