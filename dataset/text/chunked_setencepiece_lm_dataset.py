# Based on https://huggingface.co/datasets/c4/blob/main/c4.py

from .tokenizers.sentencepiece import SentencepieceVocabulary
from framework.utils.download import UrlStream
from framework.utils import LockFile
import gzip
import json
from typing import List, Optional, Dict, Any
import numpy as np
import os
import bisect
import time
import torch.multiprocessing as mp
from .lm_dataset import WordLevelLanguageModelTestState
import math


class GenToIt:
    def __init__(self, gen, *args, **kwargs):
        self.gen = gen
        self.args  = args
        self.kwargs = kwargs
        self.gen_inst = None
        self.__iter__()
        self.initialized = False

    def __iter__(self):
        assert (self.gen_inst is None) or (self.initialized == False)
        self.initialized = True
        self.gen_inst = self.gen(*self.args, **self.kwargs)
        return self

    def __next__(self):
        try:
            n = next(self.gen_inst)
            return n
        except StopIteration:
            self.gen_inst = None
            raise


class ChunkedSentencepieceLMDataset:
    TOKENIZER_N_FILES = 10

    def _get_variant_id(self) -> str:
        return f"{self.__class__.__name__}-{self.n_tokens}"

    def gzip_line_iterator(self, url: str):
        stream = UrlStream(url)
        print(f"Opening shard {url}, size {stream.size()}")
        for l in gzip.GzipFile(fileobj=stream):
            txt = json.loads(l.decode("utf-8"))["text"]
            if txt:
                yield txt + "<STORY_SEP>"

    def get_url(self, index: int, split: Optional[str] = None) -> str:
        raise NotImplementedError()

    def get_n_shards(self, split: Optional[str] = None) -> int:
        raise NotImplementedError()

    def get_tokenizer_train_sentences(self):
        n_files = min(self.TOKENIZER_N_FILES, self.get_n_shards("train"))
        for i in range(n_files):
            url = self.get_url(i, "train")
            for txt in self.gzip_line_iterator(url):
                yield txt

    def _chunk_fname(self, index: int) -> str:
        return os.path.join(self._chunk_dir, f"chunk_{index}.bin")

    def tokenize_chunk(self, chunk_index):
        fname = self._chunk_fname(chunk_index)
        if not os.path.exists(fname):
            print(f"Tokenizing chunk {chunk_index}...")

            url = self.get_url(chunk_index)
            with open(fname+".tmp", "wb") as out_f:
                for l in self.gzip_line_iterator(url):
                    np.asarray(self.vocabulary(l), dtype=self.data_dtype).tofile(out_f)

            os.rename(fname+".tmp", fname)
            print(f"Tokenizing chunk {chunk_index} done.")

    def do_mmap(self, index: int):
        if self.chunk_mmap[index] is None:
            self.chunk_mmap[index] = np.memmap(self._chunk_fname(index), dtype=self.data_dtype, mode='r')

    def update_data_type(self):
        # Avoid unnecessary copying
        if self.n_tokens >= 2**31 - 1:
            self.data_dtype = np.int64
        elif self.n_tokens >= 2**15 - 1:
            self.data_dtype = np.int32
        elif self.n_tokens >= 2**8:
            self.data_dtype = np.int16
        else:
            self.data_dtype = np.uint8

    def get_chunk_sizes(self) -> List[int]:
        chunk_sizes = []
        for i in range(self._n_chunks):
            fn = self._chunk_fname(i)
            if os.path.exists(fn):
                chunk_sizes.append(os.path.getsize(fn) // self.data_dtype(0).itemsize)
            else:
                break
        return chunk_sizes

    def get_ready_tokens(self) -> int:
        return sum(self.get_chunk_sizes())

    def __init__(self, unroll_len: int, n_extra: int = 1, split: str = 'train',
                 cache_dir: str = "./cache/", n_tokens: int = 8000, token_limit: Optional[int] = None) -> None:
        self.split = split
        self.n_tokens = n_tokens
        self.unroll_len = unroll_len
        self.n_extra = n_extra
        self.update_data_type()

        self._cache_dir = os.path.join(cache_dir, self.__class__.__name__, self._get_variant_id())
        self._chunk_dir = os.path.join(self._cache_dir, "tokenized_chunks", split)
        self._n_chunks = self.get_n_shards()
        self.chunk_sizes = [0] * self._n_chunks
        self.chunk_offsets = [0] * self._n_chunks
        self.chunk_mmap = [None] * self._n_chunks
        self.last_available_chunk = -1
        self.last_accessed_chunk = -1
        self.token_limit = int(math.ceil(token_limit)) if token_limit is not None else None

        os.makedirs(self._chunk_dir, exist_ok=True)

        self._sp_model_name = os.path.join(self._cache_dir, "tokenizer.model")

        with LockFile(self._cache_dir + "/lock"):
            self.vocabulary = SentencepieceVocabulary(self._sp_model_name, GenToIt(self.get_tokenizer_train_sentences), n_tokens)
            print(f"{self.__class__.__name__}: Loaded tokenizer.")

            missing = [i for i in range(self._n_chunks) if not os.path.exists(self._chunk_fname(i))]
            print(f"{self.__class__.__name__}: {len(missing)} chunks missing")
            if missing:
                if token_limit is not None:
                    pool = mp.Pool(min(mp.cpu_count(), len(missing)))

                    while True:
                        tokens_ready = self.get_ready_tokens()
                        if tokens_ready >= token_limit:
                            print("Token limit reached. No need to tokenize more.")
                            break

                        print(f"{self.__class__.__name__}: {tokens_ready/token_limit*100:.2f}% ready.")

                        chunks_ready = len(self.get_chunk_sizes())

                        if chunks_ready == 0:
                            print("Tokenizing first chunk to estimate the number of required chunks...")
                            pool.map(self.tokenize_chunk, [0])
                            continue
                        elif chunks_ready >= self._n_chunks:
                            print("All chunks ready. No need to tokenize more.")
                            break

                        n_estimated = int(math.ceil(chunks_ready * (token_limit / tokens_ready)))

                        print(f"{self.__class__.__name__}: Tokenizing {n_estimated} estimated chunks...")
                        pool.map(self.tokenize_chunk, [a for a in range(chunks_ready, n_estimated) if a in missing])

                    print(f"Limiting to {token_limit} tokens")
                    missing = missing[:token_limit // self.unroll_len]

                    del pool
                else:
                    mp.Pool(min(mp.cpu_count(), len(missing))).map(self.tokenize_chunk, missing)

        self.chunk_sizes = self.get_chunk_sizes()
        self.chunk_offsets = self.chunk_offsets[:len(self.chunk_sizes)]
        self.chunk_mmap = self.chunk_mmap[:len(self.chunk_sizes)]
        lim_found = False
        for i in range(1, len(self.chunk_sizes)):
            self.chunk_offsets[i] = self.chunk_offsets[i - 1] + self.chunk_sizes[i]
            if self.token_limit is not None and not lim_found and self.chunk_offsets[i] >= self.token_limit:
                print(f"{self.__class__.__name__}: Limiting to first {i} chunks because limited to {self.token_limit} tokens")
                lim_found = True

    def __len__(self):
        l = self.linear_len()
        if self.token_limit is not None:
            l = min(l, self.token_limit)

        return l // self.unroll_len

    def linear_len(self):
        return self.chunk_sizes[-1] + self.chunk_offsets[-1]

    def get_linear(self, offset: int, clen: int):
        chunk_index = bisect.bisect(self.chunk_offsets, offset) - 1
        chunk_offset = offset - self.chunk_offsets[chunk_index]

        self.do_mmap(chunk_index)

        if chunk_offset + clen > self.chunk_sizes[chunk_index]:
            # Wrapping over chunk boundary
            next_chunk = (chunk_index + 1) % len(self.chunk_sizes)
            self.do_mmap(next_chunk)

            d1 = self.chunk_mmap[chunk_index][chunk_offset:]
            d2 = self.chunk_mmap[next_chunk][:clen-len(d1)]

            r = np.concatenate([d1, d2])
        else:
            r = self.chunk_mmap[chunk_index][chunk_offset:chunk_offset+clen]

        assert r.shape[0] == clen
        return r

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return {
            "data": self.get_linear(item * self.unroll_len, self.unroll_len + self.n_extra)
        }

    def start_test(self) -> WordLevelLanguageModelTestState:
        return WordLevelLanguageModelTestState()

