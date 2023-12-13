from .lm_dataset import SentencepieceLanguageDataset, LMFile


class Wikitext2SentencePiece(SentencepieceLanguageDataset):
    def __init__(self, split: str, unroll_len: int, n_extra: int = 1, cache_dir: str = "./cache/"):
        super().__init__([LMFile(
            st, "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip//"
            f"wikitext-2/wiki.{st}.tokens", None) for st in ["train", "test", "valid"]],
            split, unroll_len, n_extra, False, cache_dir)


class Wikitext103SentencePiece(SentencepieceLanguageDataset):
    def __init__(self, split: str, unroll_len: int, n_extra: int = 1, cache_dir: str = "./cache/",
                 n_pieces: int = 32000):
        super().__init__([LMFile(
            st, "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip//"
            f"wikitext-103/wiki.{st}.tokens", None) for st in ["train", "test", "valid"]],
            split, unroll_len, n_extra, False, cache_dir, n_pieces=n_pieces)

