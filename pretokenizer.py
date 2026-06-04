"""CLI entry point for converting a text corpus into token shard files.

This script parses command-line arguments and forwards them to
``utils.tokenize_dataset`` so raw text datasets can be pretokenized before
training.
"""

import argparse
from utils import tokenize_dataset
import multiprocessing as mp


def cli():
    """Parse CLI arguments and run dataset tokenization."""

    args = argparse.ArgumentParser(description="Tokenize a text dataset into shards.")
    args.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input text file to be tokenized.",
    )
    args.add_argument(
        "--output_prefix",
        type=str,
        default="tokenized_data/shard_",
        help="Prefix for the output shard files (e.g., 'tokenized_data/train_shard_').",
    )
    args.add_argument(
        "--max_tokens_per_shard",
        type=int,
        default=50_000_000,
        help="Maximum number of tokens to include in each output shard file.",
    )
    args = args.parse_args()

    print(f"Tokenizing dataset from {args.input_path}...")
    tokenize_dataset(
        input_path=args.input_path,
        output_prefix=args.output_prefix,
        max_tokens_per_shard=args.max_tokens_per_shard,
    )
    print("Tokenization complete.")


if __name__ == "__main__":
    cli()
