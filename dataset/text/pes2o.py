# Based on https://huggingface.co/datasets/c4/blob/main/c4.py


from typing import List, Optional, Dict, Any

from .chunked_setencepiece_lm_dataset import ChunkedSentencepieceLMDataset


_N_SHARDS_PER_SPLIT = {
    "v1": {"train": 20, "validation": 2},
    "v2": {"train": 20, "validation": 2},
}

_DATA_URL = "https://huggingface.co/datasets/allenai/peS2o/resolve/main/data/{name}/{split}-{index:05d}-of-{n_shards:05d}.json.gz"


class PES2O(ChunkedSentencepieceLMDataset):
    TOKENIZER_N_FILES = 1

    def _get_variant_id(self) -> str:
        return f"{self.__class__.__name__}-{self.variant}-{self.n_tokens}"

    def get_url(self, index: int, split: Optional[str] = None) -> str:
        split = split or self.split
        return _DATA_URL.format(
            name=self.variant, split=split, index=index, n_shards=_N_SHARDS_PER_SPLIT[self.variant][split])

    def get_n_shards(self, split: Optional[str] = None) -> int:
        split = split or self.split
        return _N_SHARDS_PER_SPLIT[self.variant][split]

    def __init__(self, unroll_len: int, n_extra: int = 1, split: str = 'train', variant: str = 'v2',
                 cache_dir: str = "./cache/", n_tokens: int = 8000, token_limit: Optional[int] = None) -> None:
        self.variant = variant
        super().__init__(unroll_len, n_extra, split, cache_dir, n_tokens, token_limit=token_limit)
