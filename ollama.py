import torch
from transformers import AutoTokenizer

from model import AliLM
from generate import generate


def run_engine(model, tokenizer, device):

    while True:
        prompt = input("Enter a prompt (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        generate(
            model, tokenizer, prompt, num_tokens=512, device=device, temperature=0.8
        )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    config = {
        "vocab_size": 50257,
        "context_length": 4096,
        "embedding_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "d_ff": 4 * 768,
        "Batch_size": 32,
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
    print(f"Loading model on {device}...")
    model.load_state_dict(torch.load("model_weights.pt", map_location=device))
    print(f"Model loaded on {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    run_engine(model, tokenizer, device)


if __name__ == "__main__":
    main()
