"""Interactive inference engine for AliLM.

Loads the trained model and tokenizer using settings from ``config.json``,
then enters an interactive prompt loop where the user types a prompt and
receives streamed generated text.

Usage::

    uv run ollama.py
"""

import argparse

import torch
from transformers import AutoTokenizer

from model import AliLM
from generate import generate
from utils import load_config, print_section, print_stat


def run_engine(
    model: AliLM,
    tokenizer: AutoTokenizer,
    config: dict,
    device: str,
    verbose: bool = False,
) -> None:
    """Run an interactive generation loop until the user types ``exit``.

    Args:
        model: Loaded AliLM model ready for inference.
        tokenizer: GPT-2 tokenizer used to encode prompts and decode tokens.
        config: Global configuration dictionary from ``config.json``.
        device: Device string used for tensor allocation.
        verbose: When ``True``, print section headers and Ollama-style timing
            metrics after every generation.
    """
    # Read generation hyperparameters from config, with sensible defaults
    temperature = config.get("inference_temperature", 0.8)
    num_tokens = config.get("inference_num_tokens", 256)

    print_section("Interactive Engine")
    print_stat("  Temperature:", f"{temperature:.2f}")
    print_stat("  Max new tokens:", f"{num_tokens:,}")
    print("  Type 'exit' to quit.\n")

    while True:
        prompt = input("Prompt> ").strip()
        if not prompt:
            continue
        if prompt.lower() == "exit":
            print("Exiting ...")
            break
        generate(
            model,
            tokenizer,
            prompt,
            num_tokens=num_tokens,
            device=device,
            temperature=temperature,
            verbose=verbose,
        )


def main() -> None:
    """Parse CLI arguments, load model from ``config.json``, and start the engine."""
    parser = argparse.ArgumentParser(description="AliLM interactive inference engine.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print section headers, prompt echo, and detailed Ollama-style "
            "timing metrics (load duration, prompt eval rate, eval rate, etc.) "
            "after every generation."
        ),
    )
    args = parser.parse_args()

    config = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use half-precision on GPU for faster inference; full precision on CPU
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print_section("AliLM Inference Engine")
    print_stat("  Model:", config["model_name"])
    print_stat("  Device:", device)
    print_stat("  Dtype:", str(dtype).replace("torch.", ""))

    # ── Build model architecture ───────────────────────────────────────────────
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

    # Load the weights saved by train.py
    weights_path = config["model_weights_path"]
    print_stat("  Loading weights:", weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print_stat("  Status:", "Ready")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    run_engine(model, tokenizer, config, device, verbose=args.verbose)


if __name__ == "__main__":
    main()
