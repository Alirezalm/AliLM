type Vocabulary = dict[int, bytes]
type Merges = list[tuple[bytes, bytes]]
from collections.abc import Iterable, Iterator
import pickle

from tokenizer.bpe_trainer import merge_pair, pretokenize_for_encode


class Tokenizer:
    """A tokenizer that uses byte pair encoding (BPE) to convert text into tokens and vice versa."""

    def __init__(
        self, vocab: Vocabulary, merges: Merges, special_tokens: list[str] | None = None
    ):

        if len(vocab) < 256:
            raise ValueError("Vocabulary must contain at least 256 tokens.")

        vocab_size = len(vocab)

        self.id_to_byte = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}

        self.merges = merges

        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(
        cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None
    ):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encodes the given text into a list of token IDs using the vocab and merges."""
        tokens = []

        pretokens = pretokenize_for_encode(text)

        for pretoken in pretokens:
            for merge in self.merges:
                while merge in zip(pretoken, pretoken[1:]):
                    pretoken = merge_pair(merge, pretoken)

            for token in pretoken:
                tokens.append(self.byte_to_id[token])

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encodes the given text into an iterable of token IDs using the vocab and merges."""
        for text in iterable:
            yield self.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decodes the given list of token IDs back into a string using the vocab."""
        text = "".join(self.id_to_byte[token].decode("utf-8") for token in tokens)
        return text

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={len(self.id_to_byte)})"
