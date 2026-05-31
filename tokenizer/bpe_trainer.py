from collections import Counter
import regex as re
import pickle

type Vocabulary = dict[int, bytes]
type Merges = list[tuple[bytes, bytes]]
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def init_vocabulary(special_tokens: list[str]) -> Vocabulary:
    """Initializes the vocabulary with byte-level tokens and special tokens."""
    return {i: bytes([i]) for i in range(256)} | {
        i + 256: token.encode("utf-8") for i, token in enumerate(special_tokens)
    }


def string_to_tuple(s: str) -> tuple[bytes, ...]:
    """Converts a string to a tuple of bytes for BPE merges."""
    return tuple(bytes([b]) for b in s.encode("utf-8"))


def pretokenize(text: str, special_tokens: list[str]) -> dict[tuple[bytes, bytes], int]:
    # TODO: Parallelize

    text = re.split("|".join(map(re.escape, special_tokens)), text)

    splits = [re.findall(PAT, d) for d in text]

    pretokenized = [string_to_tuple(s) for split in splits for s in split]

    return dict(Counter(pretokenized))


def get_pair_frequencies(
    pretokens: dict[tuple[bytes, bytes], int],
) -> dict[tuple[bytes, bytes], int]:
    """Calculates the frequency of adjacent byte pairs in the pretokenized data."""

    pair_freqs: dict[tuple[bytes, bytes], int] = {}

    for token_tuple, freq in pretokens.items():
        for i, j in zip(token_tuple, token_tuple[1:]):
            pair = (i, j)
            if pair in pair_freqs:
                pair_freqs[pair] += freq
            else:
                pair_freqs[pair] = freq

    return pair_freqs


def get_the_most_frequent_pair(
    pair_frequencies: dict[tuple[bytes, bytes], int],
) -> tuple[bytes, bytes]:
    """Returns the most frequent byte pair from the list of pair frequencies."""
    pair = max(pair_frequencies.items(), key=lambda item: (item[1], item[0]))[0]
    return pair


def merge_pair(
    pair: tuple[bytes, bytes], pretoken: tuple[bytes, ...]
) -> tuple[bytes, ...]:
    """Merges the specified byte pair in the pretokenized data."""

    i = 0
    merged_token = []
    while i < len(pretoken) - 1:
        token_pair = (pretoken[i], pretoken[i + 1])
        if token_pair == pair:
            merged_token.append(
                token_pair[0] + token_pair[1]
            )  # Merge the pair into a single token
            i += 2
        else:
            merged_token.append(pretoken[i])
            i += 1

    # Append the last token if it's not part of a pair
    if i < len(pretoken):
        merged_token.append(pretoken[i])

    return tuple(merged_token)


def get_updated_pretokens(
    pair: tuple[bytes, bytes], pretokens: dict[tuple[bytes, bytes], int]
) -> dict[tuple[bytes, bytes], int]:

    return {merge_pair(pair, pair_freq): freq for pair_freq, freq in pretokens.items()}


def read_file(file_path: str) -> str:
    """Reads the content of a file and returns it as a string."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[Vocabulary, Merges]:
    """Trains a Byte Pair Encoding (BPE) tokenizer on the given input data."""

    vocab = init_vocabulary(special_tokens)
    print("Reading input file...")
    text = read_file(input_path)
    print("Input file read.")

    print("Pretokenizing input data...")
    pretokenized = pretokenize(text, special_tokens)
    print("Input data pretokenized.")

    merges: Merges = []

    for i in range(vocab_size - len(vocab)):

        # Get the frequency of adjacent byte pairs in the pretokenized data
        pair_frequencies = get_pair_frequencies(pretokenized)

        if not pair_frequencies:
            break

        # Get the most frequent byte pair
        the_most_frequent_pair = get_the_most_frequent_pair(pair_frequencies)

        # Merge the most frequent byte pair in the pretokenized data
        pretokenized = get_updated_pretokens(the_most_frequent_pair, pretokenized)

        merges.append(the_most_frequent_pair)

        # information logging
        if (i + 1) % 100 == 0:
            print(
                f"Merge {i + 1}/{vocab_size - len(vocab)}: {the_most_frequent_pair} with frequency {pair_frequencies[the_most_frequent_pair]}"
            )

    vocab.update(
        {i + len(vocab): merge[0] + merge[1] for i, merge in enumerate(merges)}
    )
    return vocab, merges


def pretokenize_for_encode(
    text: str, special_tokens: list[str] = ["<|endoftext|>"]
) -> list[tuple[bytes, ...]]:

    text = re.split("|".join(map(re.escape, special_tokens)), text)

    splits = [re.findall(PAT, d) for d in text]

    return [string_to_tuple(s) for split in splits for s in split]


def encode(text: str, vocab: Vocabulary, merges: Merges) -> list[int]:
    tokens = []
    vocab_inverse = {v: k for k, v in vocab.items()}

    pretokens = pretokenize_for_encode(text)

    for pretoken in pretokens:
        for merge in merges:
            while merge in zip(pretoken, pretoken[1:]):
                pretoken = merge_pair(merge, pretoken)

        for token in pretoken:
            tokens.append(vocab_inverse[token])

    return tokens


def decode(tokens: list[int], vocab: Vocabulary) -> str:

    text = "".join(vocab[token].decode("utf-8") for token in tokens)
    return text


def persist_vocab_and_merges(
    vocab: Vocabulary, merges: Merges, vocab_path: str, merges_path: str
):
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
