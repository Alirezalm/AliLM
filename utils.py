from pathlib import Path
import time

import torch
from transformers import AutoTokenizer
from jaxtyping import Int


def tokenize_dataset(
    input_path: str,
    output_prefix: str,
    max_tokens_per_shard: int = 50_000_000,
):
    """
    Tokenize a text dataset and save token IDs into shard files.

    Args:
        input_path: Path to input text file.
        output_prefix: Prefix for output shard files.
        max_tokens_per_shard: Maximum tokens per output shard.
    """

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    eot_id = tokenizer.eos_token_id

    shard_tokens = []
    shard_idx = 0

    total_tokens = 0
    total_lines = 0

    start_time = time.perf_counter()

    with open(input_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):

            # Remove trailing newline characters
            line = line.rstrip("\n")

            if not line:  # Skip empty lines
                continue

            ids = tokenizer.encode(
                line,
                add_special_tokens=False,
            )

            # Append end-of-text token after each document/line
            ids.append(eot_id)

            shard_tokens.extend(ids)

            total_tokens += len(ids)
            total_lines += 1

            if len(shard_tokens) >= max_tokens_per_shard:
                save_shard(
                    shard_tokens,
                    output_prefix,
                    shard_idx,
                )

                print(f"Saved shard {shard_idx} " f"({len(shard_tokens):,} tokens)")

                shard_tokens = []
                shard_idx += 1

            if (line_idx + 1) % 100_000 == 0:
                elapsed = time.perf_counter() - start_time
                throughput = total_tokens / max(elapsed, 1e-9)

                print(
                    f"Processed {line_idx + 1:,} lines | "
                    f"{total_tokens:,} tokens | "
                    f"{throughput:,.0f} tokens/sec"
                )
    # Save any remaining tokens in the last shard
    if shard_tokens:
        save_shard(
            shard_tokens,
            output_prefix,
            shard_idx,
        )

        print(f"Saved shard {shard_idx} " f"({len(shard_tokens):,} tokens)")

    elapsed = time.perf_counter() - start_time

    print("\nFinished")
    print(f"Lines processed : {total_lines:,}")
    print(f"Tokens produced : {total_tokens:,}")
    print(f"Elapsed time    : {elapsed:.2f} sec")
    print(
        f"Average throughput: " f"{total_tokens / max(elapsed, 1e-9):,.0f} tokens/sec"
    )


def save_shard(tokens, output_prefix, shard_idx):
    tensor = torch.tensor(tokens, dtype=torch.int32)

    output_path = f"{output_prefix}_shard_{shard_idx:05d}.pt"

    torch.save(tensor, output_path)


def data_loading(
    x: torch.Tensor, batch_size: int, context_length: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:

    n = x.shape[0]

    start_idx = torch.randint(0, n - context_length, (batch_size,))

    X = torch.stack([x[idx : idx + context_length] for idx in start_idx])
    Y = torch.stack([x[idx + 1 : idx + 1 + context_length] for idx in start_idx])

    return X.to(device), Y.to(device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    src_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    checkpoint = torch.load(src_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return epoch
