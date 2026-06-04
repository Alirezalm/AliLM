"""Standalone validation script for AliLM.

Loads a trained model checkpoint and evaluates it on pretokenized validation
shards, reporting cross-entropy loss and perplexity.

Usage::

    uv run validate.py --data_path ./tokenized_data/tiny_story/validation
    uv run validate.py --data_path ./tokenized_data/owt/validation --num_batches 200
"""

import argparse
import time

import torch
import torch.nn as nn

from model import AliLM
from utils import (
    data_loading,
    load_config,
    load_shards,
    model_size,
    param_count,
    print_section,
    print_stat,
)


def load_model(
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> AliLM:
    """Build AliLM from config and load the saved weights into it.

    Args:
        config: Configuration dictionary loaded from ``config.json``.
        device: Target device for inference.
        dtype: Target floating-point dtype.

    Returns:
        AliLM model with weights restored, moved to *device* and *dtype*.
    """
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
    # Restore weights written by train.py's save_model call
    weights_path = config["model_weights_path"]
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


def validate(
    pretokenized_validation_data_path: str,
    *,
    config_path: str = "config.json",
    num_batches: int = 100,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[float, float]:
    """Evaluate a trained AliLM checkpoint on pretokenized validation shards.

    Args:
        pretokenized_validation_data_path: Directory containing ``.pt`` shard files.
        config_path: Path to the global JSON config file.
        num_batches: Number of random batches to evaluate.
        device: Device to run inference on.

    Returns:
        Tuple of ``(avg_loss, perplexity)``.
    """
    config = load_config(config_path)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # ── Load validation token shards ──────────────────────────────────────────
    print_section("Validation Data")
    print("  Loading validation shards...")
    token_sequence = load_shards(pretokenized_validation_data_path)
    print_stat("  Shards directory:", pretokenized_validation_data_path)
    print_stat("  Tokens loaded:", f"{len(token_sequence):,}")

    # ── Build model and restore weights ───────────────────────────────────────
    print_section("Model")
    weights_path = config["model_weights_path"]
    print_stat("  Loading weights:", weights_path)
    model = load_model(config, device, dtype)
    print_stat("  Parameters:", f"{param_count(model) / 1_000_000:.2f}M")
    print_stat("  Size:", f"{model_size(model):.2f} MB")
    print_stat("  Device:", device)
    print_stat("  Dtype:", str(dtype).replace("torch.", ""))

    # ── Evaluation loop ────────────────────────────────────────────────────────
    print_section("Evaluation")
    print_stat("  Batches:", f"{num_batches:,}")
    print_stat("  Batch size:", f"{config['Batch_size']:,}")
    print_stat("  Context length:", f"{config['context_length']:,}")
    print()

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    start = time.perf_counter()

    model.eval()
    with torch.no_grad():
        for i in range(num_batches):
            # Sample a random batch of (input, target) windows from the token stream
            X, Y = data_loading(
                token_sequence,
                batch_size=config["Batch_size"],
                context_length=config["context_length"],
                device=device,
            )
            logits = model(X)  # (B, T, vocab_size)
            # Reshape for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = loss_fn(
                logits.view(-1, config["vocab_size"]),
                Y.view(-1).to(torch.long),
            )
            total_loss += loss.item()

            # Print running stats every 10 batches so progress is visible
            if (i + 1) % 10 == 0:
                running_loss = total_loss / (i + 1)
                running_ppl = torch.exp(torch.tensor(running_loss)).item()
                print(
                    f"  batch [{i + 1:>4}/{num_batches}]"
                    f"  loss={running_loss:.4f}"
                    f"  ppl={running_ppl:.2f}"
                )

    elapsed = time.perf_counter() - start
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    tokens_evaluated = num_batches * config["Batch_size"] * config["context_length"]
    tok_per_sec = tokens_evaluated / max(elapsed, 1e-9)

    print_section("Results")
    print_stat("  Batches evaluated:", f"{num_batches:,}")
    print_stat("  Tokens evaluated:", f"{tokens_evaluated:,}")
    print_stat("  Validation loss:", f"{avg_loss:.4f}")
    print_stat("  Perplexity:", f"{perplexity:.2f}")
    print_stat("  Tok/s:", f"{tok_per_sec / 1000:.1f}k")
    print_stat("  Elapsed:", f"{elapsed:.2f}s")

    return avg_loss, perplexity


def main() -> None:
    """Parse CLI arguments and run validation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained AliLM checkpoint on validation data."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Directory containing pretokenized validation shard files (.pt).",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.json",
        help="Path to the global config JSON file (default: config.json).",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=100,
        help="Number of random batches to evaluate (default: 100).",
    )
    args = parser.parse_args()

    validate(
        pretokenized_validation_data_path=args.data_path,
        config_path=args.config_path,
        num_batches=args.num_batches,
    )


if __name__ == "__main__":
    main()
