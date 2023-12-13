from copy import deepcopy
import torch
import numpy as np
from typing import Optional, Union, Tuple, Dict, List, Any, Set
import os
from framework.utils import download
import framework
from dataclasses import dataclass
import bisect
import torch.nn.functional as F
import copy
import hashlib
from torch.nn.modules.adaptive import _ASMoutput
from ..sequence_dataset import SequenceDataset
from .tokenizers.sentencepiece import SentencepieceVocabulary


class CharLevelLanguageModelTestState:
    def __init__(self, batch_dim: int = 1):
        self.loss_sum = 0
        self.n_ok = 0
        self.n_total = 0
        self.batch_dim = batch_dim
        self.time_dim = 1 - self.batch_dim

    def step(self, net_out: Union[torch.Tensor, _ASMoutput], data: Dict[str, torch.Tensor]):
        with torch.no_grad():
            target = data["data"].narrow(self.time_dim, 1, data["data"].shape[self.time_dim] - 1).contiguous()
            if isinstance(net_out, _ASMoutput):
                self.loss_sum += net_out.loss.item() * net_out.output.numel()
                out = net_out.output
            else:
                self.loss_sum += F.cross_entropy(net_out.flatten(0, -2), target.flatten().long(),
                                    reduction='sum').cpu().item()
                out = net_out.argmax(-1)

            assert out.shape == target.shape
            self.n_total += target.numel()
            self.n_ok += (out == target).float().sum().cpu().item()

    @property
    def accuracy(self) -> float:
        return self.n_ok / self.n_total

    def plot(self) -> Dict[str, Any]:
        loss = self.loss_sum / self.n_total
        bpc = np.log2(np.exp(loss))
        return {
            "loss": loss,
            "accuracy": self.accuracy,
            "bpc": bpc
        }


class WordLevelLanguageModelTestState(CharLevelLanguageModelTestState):
    def plot(self) -> Dict[str, Any]:
        loss = self.loss_sum / self.n_total
        bpc = np.exp(loss)
        return {
            "loss": loss,
            "accuracy": self.accuracy,
            "perplexity": bpc
        }

@dataclass
class LMFile:
    split: str
    url: str
    filename: Optional[str]
    size: Optional[Union[int, float]]

    def __init__(self, split: str, url: str, size: Optional[Union[int, float]] = None):
        sparts = url.split("//")
        offset = 1 if sparts[0].lower() in {"http:", "https:"} else 0

        if len(sparts) - offset == 1:
            self.url = url
            self.filename = None
        elif len(sparts) - offset == 2:
            self.url = "//".join(sparts[:-1])
            self.filename = sparts[-1]
        else:
            assert False, f"Invalid URL: {url}"

        self.split = split
        self.size = size

    def get_local_filename(self) -> str:
        return self.filename or self.url.split("/")[-1]

    def get_size(self, full_size: int):
        if self.size is None:
            return None
        elif isinstance(self.size, float):
            return int(full_size * self.size)
        elif isinstance(self.size, int):
            return self.size
        else:
            assert False, "Invalid size for split specification."

@dataclass
class DataSlice:
    offset: int
    len: int
    data: np.ndarray


class CharLanguageDataset(SequenceDataset):
    VERISON = 1
    DOWNLOAD_VERSION = 1

    vocabulary = None
    data = None

    def get_files_to_download(self) -> Set[str]:
        return {s.url for s in self.splits}

    def get_all_files(self) -> Set[str]:
        return {s.get_local_filename() for s in self.splits}

    def get_id(self) -> str:
        return hashlib.md5("".join(sorted(self.get_all_files())).encode()).hexdigest()

    def get_version(self, download: bool) -> Optional[int]:
        verfile = os.path.join(self.download_dir if download else self.cache_dir, "version")
        if os.path.isfile(verfile):
            with open(verfile, "r") as f:
                return int(f.read())
        return None

    def write_version(self, download: bool):
        verfile = os.path.join(self.download_dir if download else self.cache_dir, "version")
        with open(verfile, "w+") as f:
            f.write(str(self.DOWNLOAD_VERSION if download else self.VERISON))

    def update_data_type(self):
        # Avoid unnecessary copying
        if len(self.vocabulary) >= 2**31 - 1:
            self.data_dtype = np.int64
        elif len(self.vocabulary) >= 2**15 - 1:
            self.data_dtype = np.int32
        elif len(self.vocabulary) >= 2**8:
            self.data_dtype = np.int16
        else:
            self.data_dtype = np.uint8

    def get_tokens(self, line: str):
        return line

    def create_vocab(self, tokens: Set[str]):
        return framework.data_structures.CharVocabulary(tokens)

    def open_file(self, fname: str):
        return open(fname, "r")

    def build_vocabulary(self, files: List[str]):
        voc = set()
        for f in files:
            split_fn = os.path.join(self.download_dir, f)

            with self.open_file(split_fn) as f:
                for l in f:
                    voc.update(self.get_tokens(l))

        return self.create_vocab(voc)

    def download(self):
        if self.get_version(True) != self.DOWNLOAD_VERSION:
            print(f"{self.__class__.__name__}: Downloading...")
            for url in self.get_files_to_download():
                print(f"    {url}")
                download(url, self.download_dir + "/")

            self.write_version(True)

    def load(self):
        if self.vocabulary is not None or self.data is not None:
            return

        voc_cache = os.path.join(self.cache_dir, "vocabulary.pth")

        if self.get_version(False) != self.VERISON:
            print(f"{self.__class__.__name__}: Constructing vocabulary...")
            # First pass: construct vocabulary
            self.__class__.vocabulary = self.build_vocabulary(self.get_all_files())
            self.update_data_type()

            # Second pass: translate files
            for fname in self.get_all_files():
                print(f"{self.__class__.__name__}: Tokeinizing file '{fname}'...")
                split_fn = os.path.join(self.download_dir, fname)

                with self.open_file(split_fn) as f:
                    path = os.path.join(self.cache_dir, "raw", fname)
                    os.makedirs(os.path.dirname(path), exist_ok=True)

                    with open(path, "wb") as out_f:
                        for l in f:
                            l = self.vocabulary(self.get_tokens(l))
                            np.asarray(l, dtype=self.data_dtype).tofile(out_f)

            torch.save(self.__class__.vocabulary, voc_cache)
            self.write_version(False)
        else:
            self.__class__.vocabulary = torch.load(voc_cache)
            self.update_data_type()

        print(f"{self.__class__.__name__}: Vocabulary size: {len(self.vocabulary)}")

        self.__class__.files = {}
        for f in self.get_all_files():
            self.files[f] = np.memmap(os.path.join(self.cache_dir, "raw", f), dtype=self.data_dtype, mode='r')

        self.__class__.in_vocabulary = self.__class__.out_vocabulary = self.vocabulary

    def initialize_split(self, split: str):
        file_offsets = {}
        self.slices = []
        for a in self.splits:
            fname = a.get_local_filename()
            start_pos = file_offsets.get(fname, 0)
            full_len = self.files[fname].shape[0]
            len = a.get_size(full_len)

            file_offsets[fname] = start_pos + len
            assert file_offsets[fname] <= full_len

            if a.split == split:
                self.slices.append(DataSlice(start_pos, len, self.files[fname]))

        self.offsets = np.cumsum([s.len for s in self.slices]).tolist()

    def set_filesizes(self):
        # Needed to be able to allow "None" size in any order. Replace Nones with the remaning number of elements.
        none_found = set()
        size_used = {}
        for s in self.splits:
            name = s.get_local_filename()
            if s.size is None:
                assert name not in none_found, "It can be only one split without size specification referring to the same file."
                none_found.add(name)
            else:
                size_used[name] = size_used.get(name, 0) + s.get_size(self.files[name].shape[0])

        for s in self.splits:
            if s.size is None:
                name = s.get_local_filename()
                s.size = self.files[name].shape[0] - size_used.get(name, 0)

    def __init__(self, splits: List[LMFile], split: str, unroll_len: int, n_extra: int = 1,
                 cache_dir: str = "./cache/"):
        self.splits = copy.deepcopy(splits)
        self.download_dir = os.path.join(cache_dir, self.__class__.__name__, "downloaded")
        self.cache_dir = os.path.join(cache_dir, self.__class__.__name__, self.get_id())
        self.n_extra = n_extra
        self.unroll_len = unroll_len
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)

        with framework.utils.LockFile(os.path.join(self.download_dir, "lock")):
            self.download()

        with framework.utils.LockFile(os.path.join(self.cache_dir, "lock")):
            self.load()

        self.set_filesizes()
        self.initialize_split(split)
        print(f"{self.__class__.__name__}: Split: {split}, Vocabulary size: {len(self.vocabulary)}, Length: {len(self)}")

    def __len__(self) -> int:
        return self.linear_len() // self.unroll_len

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return {
            "data": self.get_linear(item * self.unroll_len, self.unroll_len + self.n_extra)
        }

    def get_linear(self, offset: int, length: int) -> np.ndarray:
        # If the data is not glued together from multiple slices, just return
        if len(self.slices) == 1:
            return self.slices[0].data[self.slices[0].offset + offset: self.slices[0].offset + offset + length]

        # If it is from multiple pieces, find the initial piece/offset
        s_index = bisect.bisect(self.offsets, offset)
        offset = offset - (self.offsets[s_index - 1] if s_index > 0 else 0)

        # Concatenate slices until we read enoguh
        selected = []
        while length > 0:
            sl = self.slices[s_index]
            selected.append(sl.data[sl.offset + offset : sl.offset + min(offset + length, sl.len)])
            length -= selected[-1].shape[0]
            offset = 0
            s_index += 1

        return np.concatenate(selected, 0)

    def linear_len(self) -> int:
        return self.offsets[-1]

    def start_test(self) -> CharLevelLanguageModelTestState:
        return CharLevelLanguageModelTestState()


class ByteLanguageDataset(CharLanguageDataset):
    def open_file(self, fname: str):
        return open(fname, "rb")

    def create_vocab(self, tokens: Set[str]):
        return framework.data_structures.ByteVocabulary(tokens)


class WordLanguageDataset(CharLanguageDataset):
    def __init__(self, splits: List[LMFile], split: str, unroll_len: int, n_extra: int = 1,
                 split_punctuation: bool = False, cache_dir: str = "./cache/"):
        self.split_punctuation = split_punctuation
        super().__init__(splits, split, unroll_len, n_extra, cache_dir)

    def get_tokens(self, line: str):
        return [w for w in line.split(" ") if len(w)>0]

    def open_file(self, fname: str):
        return open(fname, "r")

    def create_vocab(self, tokens: Set[str]):
        return framework.data_structures.WordVocabulary(tokens, split_punctuation=self.split_punctuation)

    def start_test(self) -> WordLevelLanguageModelTestState:
        return WordLevelLanguageModelTestState()


class SentencepieceLanguageDataset(CharLanguageDataset):
    def __init__(self, splits: List[LMFile], split: str, unroll_len: int, n_extra: int = 1,
                 split_punctuation: bool = False, cache_dir: str = "./cache/", n_pieces: int = 32000):
        self.split_punctuation = split_punctuation
        self.n_pieces = n_pieces

        global spm
        import sentencepiece as spm

        super().__init__(splits, split, unroll_len, n_extra, cache_dir)

    def open_file(self, fname: str):
        return open(fname, "r")

    # def get_raw_dir(self):
        # return os.path.join(self.cache_dir, "raw")

    def get_id(self) -> str:
        return f"{super().get_id()}-{self.n_pieces}"

    def build_vocabulary(self, _):
        model = os.path.join(self.cache_dir, "tokenizer.spm")
        train_file = None
        for s in self.splits:
            if s.split == "train":
                train_file = s.get_local_filename()
                break

        if train_file is None:
            raise ValueError(f"Train split not found.")

        train_file = os.path.join(self.download_dir, train_file)
        return SentencepieceVocabulary(model, train_file, self.n_pieces)

    def create_vocab(self, tokens: Set[str]):
        return self.build_vocabulary(None)

    def start_test(self) -> WordLevelLanguageModelTestState:
        return WordLevelLanguageModelTestState()