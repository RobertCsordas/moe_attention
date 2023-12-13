import re
from typing import List, Union, Optional, Dict, Any, Set
import numpy as np


class Vocabulary:
    def __len__(self) -> int:
        raise NotImplementedError()

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def load_state_dict(self, state: Dict[str, Any]):
        raise NotImplementedError()

    def to_string(self, seq: List[int]) -> str:
        raise NotImplementedError()


class WordVocabulary(Vocabulary):
    def __init__(self, list_of_sentences: Optional[List[Union[str, List[str]]]] = None, allow_any_word: bool = False,
                 split_punctuation: bool = True):
        self.words: Dict[str, int] = {}
        self.inv_words: Dict[int, str] = {}
        self.to_save = ["words", "inv_words", "_unk_index", "allow_any_word", "split_punctuation"]
        self.allow_any_word = allow_any_word
        self.initialized = False
        self.split_punctuation = split_punctuation

        if list_of_sentences is not None:
            words = set()
            for s in list_of_sentences:
                words |= set(self.split_sentence(s))

            self._add_set(words)
            self.finalize()

    def finalize(self):
        self._unk_index = self.words.get("<UNK>", self.words.get("<unk>"))
        if self.allow_any_word:
            assert self._unk_index is not None

        self.initialized = True

    def _add_word(self, w: str):
        next_id = len(self.words)
        self.words[w] = next_id
        self.inv_words[next_id] = w

    def _add_set(self, words: Set[str]):
        for w in sorted(words):
            self._add_word(w)

    def _process_word(self, w: str) -> int:
        res = self.words.get(w)
        if res is None:
            assert self.allow_any_word, f"WARNING: unknown word: '{w}'"
            res = self._unk_index
        return res

    def _process_index(self, i: int) -> str:
        res = self.inv_words.get(i, None)
        if res is None:
            return f"<!INV: {i}!>"
        return res

    def __getitem__(self, item: Union[int, str]) -> Union[str, int]:
        if isinstance(item, int):
            return self._process_index(item)
        else:
            return self._process_word(item)

    def split_sentence(self, sentence: Union[str, List[str]]) -> List[str]:
        if isinstance(sentence, list):
            # Already tokenized.
            return sentence

        if self.split_punctuation:
            return re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
        else:
            return [x for x in sentence.split(" ") if len(x) > 0]

    def sentence_to_indices(self, sentence: Union[str, List[str]]) -> List[int]:
        assert self.initialized
        words = self.split_sentence(sentence)
        return [self._process_word(w) for w in words]

    def indices_to_sentence(self, indices: List[int]) -> List[str]:
        assert self.initialized
        return [self._process_index(i) for i in indices]

    def __call__(self, seq: Union[List[Union[str, int]], str]) -> List[Union[int, str]]:
        if seq is None or (isinstance(seq, list) and not seq):
            return seq

        if isinstance(seq, str) or isinstance(seq[0], str):
            return self.sentence_to_indices(seq)
        else:
            return self.indices_to_sentence(seq)

    def __len__(self) -> int:
        return len(self.words)

    def state_dict(self) -> Dict[str, Any]:
        return {
            k: self.__dict__[k] for k in self.to_save
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.initialized = True
        self.__dict__.update(state)

    def __add__(self, other):
        res = WordVocabulary(allow_any_word=self.allow_any_word and other.allow_any_word,
                             split_punctuation=self.split_punctuation)
        res._add_set(set(self.words.keys()) | set(other.words.keys()))
        res.finalize()
        return res

    def mapfrom(self, other) -> Dict[int, int]:
        return {other.words[w]: i for w, i in self.words.items() if w in other.words}

    def mapfrom_numpy(self, other) -> np.ndarray:
        m = self.mapfrom(other)
        maxkey = max(m.keys())

        res = np.zeros(maxkey + 1, dtype=np.int32)
        for k, v in m.items():
            res[k] = v

        return res

    def to_string(self, seq: List[int]) -> str:
        return " ".join(self.indices_to_sentence(seq))

    def __str__(self) -> str:
        return "WordVucabulary(["+(", ".join([self.inv_words[i] for i in range(len(self.words))]))+"])"


class CharVocabulary(Vocabulary):
    def __init__(self, chars: Optional[Set[str]]):
        self.initialized = False
        if chars is not None:
            self.from_set(chars)

    def from_set(self, chars: Set[str]):
        chars = list(sorted(chars))
        self.to_index = {c: i for i, c in enumerate(chars)}
        self.from_index = {i: c for i, c in enumerate(chars)}
        self.initialized = True

    def __len__(self):
        return len(self.to_index)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "chars": set(self.to_index.keys())
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.from_set(state["chars"])

    def str_to_ind(self, data: str) -> List[int]:
        return [self.to_index[c] for c in data]

    def ind_to_str(self, data: List[int]) -> str:
        return "".join([self.from_index[i] for i in data])

    def _is_string(self, i):
        return isinstance(i, str)

    def __call__(self, seq: Union[List[int], str]) -> Union[List[int], str]:
        assert self.initialized
        if seq is None or (isinstance(seq, list) and not seq):
            return seq

        if self._is_string(seq):
            return self.str_to_ind(seq)
        else:
            return self.ind_to_str(seq)

    def mapfrom(self, other) -> Dict[int, int]:
        return {other.to_index[c]: i for c, i in self.to_index.items() if c in other.to_index}

    def mapfrom_numpy(self, other) -> np.ndarray:
        m = self.mapfrom(other)
        maxkey = max(m.keys())

        res = np.zeros(maxkey + 1, dtype=np.int32)
        for k, v in m.items():
            res[k] = v

        return res

    def to_string(self, seq: List[int]) -> str:
        return self.ind_to_str(seq)

    def __add__(self, other):
        return self.__class__(set(self.to_index.values()) | set(other.to_index.values()))


class ByteVocabulary(CharVocabulary):
    def ind_to_str(self, data: List[int]) -> bytearray:
        return bytearray([self.from_index[i] for i in data])

    def _is_string(self, i):
        return isinstance(i, (bytearray, bytes))

    def to_string(self, seq: List[int]) -> str:
        return self.ind_to_str(seq).decode(errors='ignore')
