import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler

from model import AliLM
from utils import data_loading


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size(model):
    return sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    ) / (1024**2)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def main():
    token_sequence = torch.concat(
        [torch.load(f"./data/ts_train_shard_0000{i}.pt") for i in range(3)]
    )
    print(f"Number of tokens: {len(token_sequence):,}")

    vocabulary_size = 50257
    print(f"Vocabulary size: {vocabulary_size:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    config = {
        "vocab_size": vocabulary_size,
        "context_length": 256,
        "embedding_dim": 500,
        "num_heads": 10,
        "num_layers": 20,
        "d_ff": 5,
        "Batch_size": 16,
        "learning_rate": 1e-4,
    }

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

    print(f"Number of trainable parameters: {param_count(model):,}")
    print(f"Model size (MB): {model_size(model):.2f}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")

    total_steps = 5000
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    total_loss = 0.0
    num_batches = 0
    loss_values = []

    model.train()
    for epoch in range(total_steps):
        X, Y = data_loading(
            token_sequence,
            batch_size=config["Batch_size"],
            context_length=config["context_length"],
            device=device,
        )

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

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1} / {total_steps}, Average Loss: {average_loss:.4f}, Perplexity: {perplexity:.4f}"
            )

    save_model(model, "model_weights.pt")
    print("Model saved to model_weights.pt")


if __name__ == "__main__":
    main()
