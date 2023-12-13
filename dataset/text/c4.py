# Based on https://huggingface.co/datasets/c4/blob/main/c4.py


from typing import List, Optional, Dict, Any

from .chunked_setencepiece_lm_dataset import ChunkedSentencepieceLMDataset

_VARIANTS = ["en", "realnewslike", "en.noblocklist", "en.noclean"]

_N_SHARDS_PER_SPLIT = {
    "en": {"train": 1024, "validation": 8},
    "realnewslike": {"train": 512, "validation": 1},
    "en.noblocklist": {"train": 1024, "validation": 8},
    "en.noclean": {"train": 7168, "validation": 64},
}

_DATA_URL = "https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/{name}/c4-{split}.{index:05d}-of-{n_shards:05d}.json.gz"


class C4(ChunkedSentencepieceLMDataset):
    def _get_variant_id(self) -> str:
        return f"{self.__class__.__name__}-{self.variant}-{self.n_tokens}"

    def get_url(self, index: int, split: Optional[str] = None) -> str:
        split = split or self.split
        return _DATA_URL.format(
            name=self.variant, split=split, index=index, n_shards=_N_SHARDS_PER_SPLIT[self.variant][split])

    def get_n_shards(self, split: Optional[str] = None) -> int:
        split = split or self.split
        return _N_SHARDS_PER_SPLIT[self.variant][split]

    def __init__(self, unroll_len: int, n_extra: int = 1, split: str = 'train', variant: str = 'en',
                 cache_dir: str = "./cache/", n_tokens: int = 8000, token_limit: Optional[int] = None) -> None:
        self.variant = variant
        super().__init__(unroll_len, n_extra, split, cache_dir, n_tokens, token_limit)
