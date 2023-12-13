from .lm_dataset import ByteLanguageDataset, LMFile


class Enwik8(ByteLanguageDataset):
    def __init__(self, split: str, unroll_len: int, n_extra: int = 1, cache_dir: str = "./cache/"):
        super().__init__([
            LMFile("train", "https://cs.fit.edu/~mmahoney/compression/enwik8.zip//enwik8"),
            LMFile("valid", "https://cs.fit.edu/~mmahoney/compression/enwik8.zip//enwik8", 5000000),
            LMFile("test", "https://cs.fit.edu/~mmahoney/compression/enwik8.zip//enwik8", 5000000)
        ], split, unroll_len, n_extra, cache_dir)
