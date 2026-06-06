"""Training entry point for AliLM.

Loads pretokenized shard files, builds the model from ``config.json``,
runs the training loop with periodic logging and checkpointing, then saves
the final model weights to ``models/<model_name>.pt``.

Usage::

    uv run train.py --train_data_path ./tokenized_data/tiny_story/training
    uv run train.py --train_data_path ./tokenized_data/tiny_story/training \\
                    --num_training_tokens 500000000 --max_training_time 3600
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler

from model import AliLM
from utils import (
    data_loading,
    load_config,
    load_shards,
    model_size,
    param_count,
    print_section,
    print_stat,
    save_checkpoint,
    load_checkpoint,
    save_model,
)


def build_model(
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> AliLM:
    """Instantiate AliLM from the global config and move it to device/dtype.

    Args:
        config: Configuration dictionary loaded from ``config.json``.
        device: Target device for model parameters.
        dtype: Target floating-point dtype for model parameters.

    Returns:
        Freshly initialised AliLM model on the requested device and dtype.
    """
    return (
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


def train(
    pretokenized_training_data_path: str,
    num_training_tokens: int,
    *,
    max_training_time: float = 60 * 60,
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    dtype: torch.dtype = (
        torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ),
    print_interval: int = 200,
    checkpoint_path: str = "checkpoints/model_checkpoint.pt",
    checkpoint_interval: int = 1000,
) -> tuple[AliLM, dict, torch.device]:
    """Load data, build the model, and run the full training loop.

    Args:
        pretokenized_training_data_path: Directory containing ``.pt`` shard files.
        num_training_tokens: Total token budget; training stops when reached.
        max_training_time: Wall-clock time budget in seconds.
        device: Device to train on.
        dtype: Floating-point precision for model weights and activations.
        print_interval: Log a status line every this many optimiser steps.
        checkpoint_path: Base path used when writing periodic checkpoint files.
        checkpoint_interval: Save a checkpoint every this many steps.

    Returns:
        Tuple of ``(model, config, device)`` after training completes.
    """
    # ── Load and concatenate training shards ──────────────────────────────────
    print_section("Data")
    print("  Loading training shards...")
    token_sequence = load_shards(pretokenized_training_data_path)
    print_stat("  Shards directory:", pretokenized_training_data_path)
    print_stat("  Tokens loaded:", f"{len(token_sequence):,}")

    # ── Read global config from config.json ───────────────────────────────────
    config = load_config()
    print_section("Config")
    for key, value in config.items():
        print_stat(f"  {key}:", value)

    # ── Build the model ────────────────────────────────────────────────────────
    print_section("Model")
    print("  Initialising model...")
    model = build_model(config, device, dtype)
    print_stat("  Parameters:", f"{param_count(model) / 1_000_000:.2f}M")
    print_stat("  Size:", f"{model_size(model):.2f} MB")
    print_stat("  Device:", device)
    print_stat("  Dtype:", str(dtype).replace("torch.", ""))

    # ── Optimizer, scheduler, and loss ────────────────────────────────────────
    # Total steps is determined by the token budget divided by tokens per step
    total_steps = num_training_tokens // (
        config["Batch_size"] * config["context_length"]
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    # Cosine annealing smoothly decays the LR from its initial value to ~0
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print_section("Training")
    print_stat("  Steps budget:", f"{total_steps:,}")
    print_stat("  Batch size:", f"{config['Batch_size']:,}")
    print_stat("  Context length:", f"{config['context_length']:,}")
    print_stat("  Initial LR:", f"{config['learning_rate']:.1e}")
    print_stat("  Token budget:", f"{num_training_tokens / 1_000_000:.1f}M")
    print_stat("  Time budget:", f"{max_training_time / 60:.0f} min")
    print()

    # ── Resume From Checkpoint ──────────────────────────────────────────────────
    if os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
        print_stat("  Resuming from checkpoint:", checkpoint_path)
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

        for _ in range(start_epoch):
            scheduler.step()
    else:
        print_stat("  No checkpoint found, starting fresh training.")
        start_epoch = 0

    # ── Training loop ──────────────────────────────────────────────────────────
    total_loss = 0.0
    num_batches = 0
    total_tokens_processed = 0
    model.train()
    start = time.perf_counter()

    for step in range(total_steps):
        # Sample a random batch of (input, target) windows from the token stream
        X, Y = data_loading(
            token_sequence,
            batch_size=config["Batch_size"],
            context_length=config["context_length"],
            device=device,
        )

        # Accumulate total tokens seen this run (batch_size × context_length)
        total_tokens_processed += X.numel()

        # ── Forward pass ──────────────────────────────────────────────────────
        optimizer.zero_grad()
        logits = model(X)  # (B, T, vocab_size)

        # Reshape to (B*T, vocab_size) / (B*T,) as required by CrossEntropyLoss
        step_loss = loss_fn(
            logits.view(-1, config["vocab_size"]),
            Y.view(-1).to(torch.long),
        )

        # ── Backward pass and parameter update ────────────────────────────────
        step_loss.backward()
        # Clip gradients to max_norm=1.0 to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += step_loss.item()
        num_batches += 1
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        elapsed = time.perf_counter() - start
        tok_per_sec = total_tokens_processed / max(elapsed, 1e-9)
        current_lr = scheduler.get_last_lr()[0]

        if (step + 1) % print_interval == 0:
            print(
                f"  step [{step + 1:>6}/{total_steps}]"
                f"  loss={step_loss.item():.4f}"
                f"  avg={avg_loss:.4f}"
                f"  ppl={perplexity:.2f}"
                f"  lr={current_lr:.2e}"
                f"  tok/s={tok_per_sec / 1000:.1f}k"
                f"  tokens={total_tokens_processed / 1e6:.2f}M"
                f"  elapsed={elapsed / 60:.1f}m"
            )

        # ── Periodic checkpoint ────────────────────────────────────────────────
        if (step + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step + 1, checkpoint_path)
            print(f"  Checkpoint saved at step {step + 1}.")

        # ── Early-exit if time or token budget is exhausted ───────────────────
        if elapsed > max_training_time:
            print(
                f"\n  Stopping: reached time limit "
                f"({elapsed / 60:.1f} min / {max_training_time / 60:.0f} min)."
            )
            break
        if total_tokens_processed >= num_training_tokens:
            print(
                f"\n  Stopping: processed {total_tokens_processed / 1e6:.2f}M "
                f"/ {num_training_tokens / 1e6:.2f}M tokens."
            )
            break

    # ── Save final model weights ───────────────────────────────────────────────
    output_path = f"models/{config['model_name']}.pt"
    save_model(model, output_path)

    print_section("Training Complete")
    print_stat("  Steps completed:", f"{num_batches:,}")
    print_stat("  Final avg loss:", f"{avg_loss:.4f}")
    print_stat("  Final perplexity:", f"{perplexity:.2f}")
    print_stat("  Total tokens:", f"{total_tokens_processed / 1e6:.2f}M")
    print_stat("  Elapsed:", f"{elapsed / 60:.2f} min")
    print_stat("  Saved model:", output_path)

    return model, config, device


def main() -> None:
    """Parse CLI arguments and launch training."""
    parser = argparse.ArgumentParser(description="Train the AliLM language model.")
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Directory containing pretokenized training shard files (.pt).",
    )
    parser.add_argument(
        "--num_training_tokens",
        type=int,
        default=500_000_000_000,
        help="Total token budget for training (default: 500B).",
    )
    parser.add_argument(
        "--max_training_time",
        type=float,
        default=5 * 60 * 60,
        help="Maximum wall-clock training time in seconds (default: 5 hours).",
    )
    parser.add_argument(
        "--print_interval",
        type=int,
        default=200,
        help="Log a status line every this many steps (default: 200).",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="Save a checkpoint every this many steps (default: 1000).",
    )
    args = parser.parse_args()

    train(
        pretokenized_training_data_path=args.train_data_path,
        num_training_tokens=args.num_training_tokens,
        max_training_time=args.max_training_time,
        print_interval=args.print_interval,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
