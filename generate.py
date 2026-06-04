import torch
import shutil
import textwrap
from transformers import AutoTokenizer

from model import AliLM
import time


def print_section(title):
    print(f"\n{'=' * 16} {title} {'=' * 16}")


def print_stat(label, value):
    print(f"{label:<24} {value}")


def format_text_block(text: str) -> str:
    terminal_width = shutil.get_terminal_size(fallback=(100, 20)).columns
    content_width = max(terminal_width - 4, 40)
    paragraphs = text.splitlines() or [text]
    wrapped_paragraphs = []

    for paragraph in paragraphs:
        if not paragraph.strip():
            wrapped_paragraphs.append("")
            continue
        wrapped_paragraphs.append(
            textwrap.fill(
                paragraph,
                width=content_width,
                replace_whitespace=False,
                drop_whitespace=False,
            )
        )

    return "\n".join(wrapped_paragraphs)


def stream_text(text: str, current_line_length: int) -> int:
    terminal_width = shutil.get_terminal_size(fallback=(100, 20)).columns
    content_width = max(terminal_width - 4, 40)

    for char in text:
        if char == "\n":
            print()
            current_line_length = 0
            continue

        print(char, end="", flush=True)
        current_line_length += 1

        if current_line_length >= content_width and char == " ":
            print()
            current_line_length = 0

    return current_line_length


def generate(
    model,
    tokenizer,
    prompt: str,
    num_tokens: int,
    device: str,
    temperature: float = 1.0,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    model.eval()
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = prompt
    start = time.perf_counter()
    total_tokens_generated = 0

    print_section("Generation")
    print_stat("Device:", device)
    print_stat("Prompt tokens:", f"{inputs.shape[1]:,}")
    print_stat("Target tokens:", f"{num_tokens:,}")
    print_stat("Temperature:", f"{temperature:.2f}")

    separator = "-" * min(shutil.get_terminal_size(fallback=(100, 20)).columns, 100)
    print_section("Prompt")
    print(separator)
    print(format_text_block(prompt))
    print(separator)

    print_section("Generated Text")
    print(separator)
    current_line_length = 0

    for _ in range(num_tokens):
        with torch.no_grad():
            logits = model(inputs)
            next_token_logits = logits[:, -1, :] / temperature
            probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)
            predicted_token_id = torch.multinomial(probabilities, num_samples=1)

        inputs = torch.cat([inputs, predicted_token_id], dim=1)
        token_text = tokenizer.decode(predicted_token_id[0], skip_special_tokens=True)
        generated += token_text
        total_tokens_generated += 1
        current_line_length = stream_text(token_text, current_line_length)

    end = time.perf_counter()
    elapsed = end - start
    tok_per_sec = total_tokens_generated / elapsed if elapsed > 0 else 0.0

    if current_line_length:
        print()
    print(separator)

    print_section("Generation Stats")
    print_stat("Generated tokens:", f"{total_tokens_generated:,}")
    print_stat("Elapsed:", f"{elapsed:.2f}s")
    print_stat("Tokens/sec:", f"{tok_per_sec:,.2f}")

    return generated
