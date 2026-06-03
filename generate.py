import torch
from transformers import AutoTokenizer

from model import AliLM


def generate(model, tokenizer, prompt: str, num_tokens: int, device: str) -> str:
    model.eval()
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = prompt

    for i in range(num_tokens):
        with torch.no_grad():
            logits = model(inputs)
            probabilities = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
            predicted_token_id = torch.multinomial(probabilities, num_samples=1)

        inputs = torch.cat([inputs, predicted_token_id], dim=1)
        token_text = tokenizer.decode(predicted_token_id[0], skip_special_tokens=True)
        generated += token_text

        print(token_text, end="", flush=True)
        if i % 20 == 0:
            print("\n", end="", flush=True)

    print("\n")
    return generated


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    config = {
        "vocab_size": 50257,
        "context_length": 256,
        "embedding_dim": 500,
        "num_heads": 10,
        "num_layers": 30,
        "d_ff": 5,
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

    model.load_state_dict(torch.load("model_weights.pt", map_location=device))
    print(f"Model loaded on {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = " "
    generate(model, tokenizer, prompt, num_tokens=50, device=device)


if __name__ == "__main__":
    main()
