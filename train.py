import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
import time

from model import AliLM
from utils import *
import argparse


def train(
    pretokenized_training_data_path: str,
    num_training_tokens: int,
    *,
    max_training_time: float = 60 * 60,  # 1 hour
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    print_interval: int = 200,
    checkpoint_path: str = "checkpoints/model_checkpoint.pt",
    checkpoint_interval: int = 1000,
):

    # Check if the pretokenized data path exists and contains files

    if not os.path.exists(pretokenized_training_data_path):
        raise FileNotFoundError(
            f"Pretokenized training data path '{pretokenized_training_data_path}' does not exist."
        )
    if not os.path.isdir(pretokenized_training_data_path):
        raise NotADirectoryError(
            f"Pretokenized training data path '{pretokenized_training_data_path}' is not a directory."
        )

    shard_files = [
        f for f in os.listdir(pretokenized_training_data_path) if f.endswith(".pt")
    ]

    if not shard_files:
        raise FileNotFoundError(
            f"No pretokenized shard files found in '{pretokenized_training_data_path}'."
        )

    print_section("Data")
    print("Loading token sequences...")

    token_sequence = torch.concat(
        [
            torch.load(os.path.join(pretokenized_training_data_path, shard))
            for shard in shard_files
        ]
    )
    print_stat("Shards loaded:", f"{len(shard_files)}")
    print_stat("Tokens loaded:", f"{len(token_sequence):,}")

    with open("config.json", "r") as f:
        config = json.load(f)

    print_section("Config")
    for key, value in config.items():
        print_stat(key, value)

    print("Initializing model...")
    model = (
        AliLM(
            vocab_size=config["vocab_size"],
            context_length=config["context_length"],
            d_model=config["embedding_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            d_ff=config["d_ff"],
        )
        .to(device)
        .to(dtype)
    )
    print("Model initialized.")
    print_section("Model")
    print_stat("Parameters:", f"{param_count(model)/1_000_000:.2f}M")
    print_stat("Model size (MB):", f"{model_size(model):.2f}")
    print_stat("Device:", device)
    print_stat("Dtype:", str(dtype).replace("torch.", ""))

    max_training_tokens = num_training_tokens  # maximum number of tokens to train on

    total_steps = max_training_tokens // (
        config["Batch_size"] * config["context_length"]
    )  # maximum number of training steps

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print_section("Training")
    print_stat("Total steps:", f"{total_steps:,}")
    print_stat("Batch size:", f"{config['Batch_size']:,}")
    print_stat("Context length:", f"{config['context_length']:,}")
    print_stat("Learning rate:", f"{config['learning_rate']:.1e}")
    print_stat("Max training tokens:", f"{max_training_tokens / 1_000_000:0.3f}M")
    print_stat("Max training time:", f"{max_training_time / 60:.0f} minutes")
    print_stat("Maximum number of steps:", f"{total_steps:,}")

    total_loss = 0.0
    num_batches = 0
    loss_values = []
    total_number_of_tokens = 0
    model.train()
    start = time.perf_counter()
    for epoch in range(total_steps):
        X, Y = data_loading(
            token_sequence,
            batch_size=config["Batch_size"],
            context_length=config["context_length"],
            device=device,
        )

        total_number_of_tokens += X.numel()

        optimizer.zero_grad()

        logits = model(X)

        loss_value = loss_fn(
            logits.view(-1, config["vocab_size"]), Y.view(-1).to(torch.long)
        )

        loss_values.append(loss_value.item())

        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        scheduler.step()

        total_loss += loss_value.item()
        num_batches += 1
        average_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(average_loss)).item()
        end = time.perf_counter()
        elapsed = end - start
        token_per_second = total_number_of_tokens / (end - start)
        if (epoch + 1) % print_interval == 0:
            print(
                f"[{epoch + 1:>5}/{total_steps}] "
                f"loss={average_loss:.4f} | "
                f"ppl={perplexity:.4f} | "
                f"tokens={total_number_of_tokens/1000000:.3f}M | "
                f"tok/s={token_per_second/1000:.2f}k | "
                f"elapsed={elapsed / 60 :.3f}m"
            )
        if elapsed > max_training_time:
            print("Reached maximum training time. Stopping training.")
            break
        if total_number_of_tokens >= max_training_tokens:
            print("Reached maximum training tokens. Stopping training.")
            break
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

    save_model(model, f"models/{config['model_name']}.pt")
    print_section("Checkpoint")
    print_stat("Saved model:", f"models/{config['model_name']}.pt")

    return model, config, device


def validate(
    pretokenized_validation_data_path: str,
    model: torch.nn.Module,
    config: dict,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else torch.device("cpu")
    ),
    num_batches: int = 100,
):
    print_section("Validation")
    print("Loading validation data...")
    shard_files = [
        f for f in os.listdir(pretokenized_validation_data_path) if f.endswith(".pt")
    ]
    if not shard_files:
        raise FileNotFoundError(
            f"No pretokenized shard files found in '{pretokenized_validation_data_path}'."
        )
    token_sequence = torch.concat(
        [
            torch.load(os.path.join(pretokenized_validation_data_path, shard))
            for shard in shard_files
        ]
    )
    print_stat("Tokens loaded:", f"{len(token_sequence):,}")

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    validation_start = time.perf_counter()
    was_training = model.training

    model.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            X, Y = data_loading(
                token_sequence,
                batch_size=config["Batch_size"],
                context_length=config["context_length"],
                device=device,
            )
            logits = model(X)
            loss_value = loss_fn(
                logits.view(-1, config["vocab_size"]), Y.view(-1).to(torch.long)
            )
            total_loss += loss_value.item()

    average_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(average_loss)).item()
    elapsed = time.perf_counter() - validation_start
    tokens_evaluated = num_batches * config["Batch_size"] * config["context_length"]

    print_stat("Validation batches:", f"{num_batches:,}")
    print_stat("Validation loss:", f"{average_loss:.4f}")
    print_stat("Validation perplexity:", f"{perplexity:.4f}")
    print_stat("Tokens evaluated:", f"{tokens_evaluated:,}")
    print_stat("Elapsed:", f"{elapsed:.2f}s")

    if was_training:
        model.train()

    return average_loss, perplexity


def main():

    args = argparse.ArgumentParser(description="Train the AliLM model.")
    args.add_argument(
        "--train_data_path",
        type=str,
        default="./tokenized_data",
        help="Path to the directory containing pretokenized training data shards.",
    )
    args.add_argument(
        "--validation_data_path",
        type=str,
        default="./tokenized_data",
        help="Path to the directory containing pretokenized validation data shards.",
    )
    args.add_argument(
        "--num_training_tokens",
        type=int,
        default=500_000_000,
        help="Total number of training tokens to process (e.g., 500000000 for 500M).",
    )
    args.add_argument(
        "--max_training_time",
        type=float,
        default=60 * 60,
        help="Maximum training time in seconds (e.g., 3600 for 1 hour).",
    )

    args = args.parse_args()

    model, config, device = train(
        pretokenized_training_data_path=args.train_data_path,
        num_training_tokens=args.num_training_tokens,
        max_training_time=args.max_training_time,
    )
    validate(
        pretokenized_validation_data_path=args.validation_data_path,
        model=model,
        config=config,
        device=device,
        num_batches=100,
    )


if __name__ == "__main__":
    main()
