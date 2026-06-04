import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
import time

from model import AliLM
from utils import data_loading


# count non-embedding parameters
def param_count(model, include_embedding=True):
    if include_embedding:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(
            p.numel()
            for name, p in model.named_parameters()
            if p.requires_grad and "embed" not in name
        )


def model_size(model):
    return sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    ) / (1024**2)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def print_section(title):
    print(f"\n{'=' * 16} {title} {'=' * 16}")


def print_stat(label, value):
    print(f"{label:<24} {value}")


def train():
    print_section("Data")
    print("Loading token sequences...")
    token_sequence = torch.concat(
        [torch.load(f"./data/ow_train_shard_0000{i}.pt") for i in range(10)]
    )
    print_stat("Tokens loaded:", f"{len(token_sequence):,}")

    vocabulary_size = 50257
    print_stat("Vocabulary size:", f"{vocabulary_size:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    config = {
        "vocab_size": vocabulary_size,
        "context_length": 256,
        "embedding_dim": 512,
        "num_heads": 16,
        "num_layers": 16,
        "d_ff": 1344,
        "Batch_size": 32,
        "learning_rate": 1e-3,
    }

    print_section("Model")
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
    print_stat("Parameters:", f"{param_count(model):,}")
    print_stat("Model size (MB):", f"{model_size(model):.2f}")
    print_stat("Device:", device)
    print_stat("Dtype:", str(dtype).replace("torch.", ""))

    max_training_tokens = len(token_sequence)  # maximum number of tokens to train on
    total_steps = max_training_tokens // (
        config["Batch_size"] * config["context_length"]
    )  # maximum number of training steps

    max_training_time = 120 * 60  #
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
        if (epoch + 1) % 200 == 0:
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

    save_model(model, "model_weights.pt")
    print_section("Checkpoint")
    print_stat("Saved model:", "model_weights.pt")

    return model, config, device


def validate(model, config, device, num_batches: int = 100):
    print_section("Validation")
    print("Loading validation data...")
    token_sequence = torch.load("./data/ow_valid_shard_00000.pt")
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
    model, config, device = train()
    validate(model, config, device)


if __name__ == "__main__":
    main()
