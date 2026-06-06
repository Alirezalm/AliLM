"""Text generation utilities for AliLM.

Provides autoregressive token-by-token generation with temperature sampling,
live character-level streaming to the terminal, and generation statistics.

This module is consumed by ``ollama.py`` for interactive inference.
"""

import shutil
import textwrap
import time

import torch
from transformers import AutoTokenizer

from model import AliLM
from utils import print_section, print_stat


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Uses microseconds below 1 ms, milliseconds below 1 s, and seconds
    otherwise — matching the format used by the real Ollama CLI.
    """
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f}\u00b5s"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f}ms"
    return f"{seconds:.3f}s"


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
    model: AliLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_tokens: int,
    device: str,
    temperature: float = 1.0,
    verbose: bool = False,
) -> str:
    """Generate text autoregressively from a prompt using temperature sampling.

    Args:
        model: Trained AliLM model.
        tokenizer: GPT-2 tokenizer used to encode the prompt and decode tokens.
        prompt: Input text to condition generation on.
        num_tokens: Number of new tokens to generate.
        device: Device string for tensor allocation.
        temperature: Softmax temperature — lower values are sharper/greedier,
            higher values are more random. Must be strictly positive.
        verbose: When ``True``, print section headers, a prompt echo, and
            detailed Ollama-style timing metrics after generation.

    Returns:
        Full generated string including the original prompt.

    Raises:
        ValueError: If ``temperature`` is not strictly positive.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    model.eval()

    overall_start = time.perf_counter()

    # Encode the prompt; time this separately as "load duration"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.shape[1]
    load_duration = time.perf_counter() - overall_start

    generated = prompt
    total_tokens_generated = 0
    sep = "─" * min(shutil.get_terminal_size(fallback=(100, 20)).columns, 100)

    if verbose:
        print_section("Generation")
        print_stat("  Device:", device)
        print_stat("  Prompt tokens:", f"{prompt_len:,}")
        print_stat("  Target new tokens:", f"{num_tokens:,}")
        print_stat("  Temperature:", f"{temperature:.2f}")
        print_section("Prompt")
        print(sep)
        print(format_text_block(prompt))
        print(sep)
        print_section("Generated Text")
        print(sep)

    current_line_length = 0

    # ── Prompt eval: single forward pass over the full prompt ─────────────────
    # This mirrors how Ollama separates "prompt eval" (batch prefill) from
    # autoregressive "eval" (one token at a time).
    prompt_eval_start = time.perf_counter()
    with torch.no_grad():
        logits = model(inputs)  # (1, T, vocab_size)
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        first_token_id = torch.multinomial(probs, num_samples=1)
    prompt_eval_duration = time.perf_counter() - prompt_eval_start

    # Stream the first generated token
    inputs = torch.cat([inputs, first_token_id], dim=1)
    token_text = tokenizer.decode(first_token_id[0], skip_special_tokens=True)
    generated += token_text
    total_tokens_generated += 1
    current_line_length = stream_text(token_text, current_line_length)

    # ── Generation loop for the remaining tokens ───────────────────────────────
    eval_start = time.perf_counter()
    for _ in range(num_tokens - prompt_len - 1):
        with torch.no_grad():
            logits = model(inputs)  # (1, growing_T, vocab_size)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            # Sample one token index from the distribution
            predicted_token_id = torch.multinomial(probs, num_samples=1)

        # Append the new token and stream it to the terminal
        inputs = torch.cat([inputs, predicted_token_id], dim=1)
        token_text = tokenizer.decode(predicted_token_id[0], skip_special_tokens=True)
        generated += token_text
        total_tokens_generated += 1
        current_line_length = stream_text(token_text, current_line_length)

        if predicted_token_id.item() == tokenizer.eos_token_id:
            break

    eval_duration = time.perf_counter() - eval_start
    total_duration = time.perf_counter() - overall_start

    # Rates: exclude the first token from eval_rate since it was timed separately
    prompt_eval_rate = (
        prompt_len / prompt_eval_duration if prompt_eval_duration > 0 else 0.0
    )
    eval_tokens = max(total_tokens_generated - 1, 0)  # tokens produced in the eval loop
    eval_rate = eval_tokens / eval_duration if eval_duration > 0 else 0.0

    if current_line_length:
        print()  # newline after last streamed character

    if verbose:
        print(sep)
        print_section("Generation Stats")
        print_stat("  Prompt tokens:", f"{prompt_len:,}")
        print_stat("  Generated tokens:", f"{total_tokens_generated:,}")
        print_stat("  Total context length:", f"{inputs.shape[1]:,}")
        print_stat("  Tok/s:", f"{total_tokens_generated / total_duration:.1f}")
        print_stat("  Elapsed:", f"{total_duration:.2f}s")
        # Ollama-style timing breakdown
        print()
        col = 24
        print(f"  {'total duration:':<{col}} {_fmt_duration(total_duration)}")
        print(f"  {'load duration:':<{col}} {_fmt_duration(load_duration)}")
        print(f"  {'prompt eval count:':<{col}} {prompt_len} token(s)")
        print(
            f"  {'prompt eval duration:':<{col}} {_fmt_duration(prompt_eval_duration)}"
        )
        print(f"  {'prompt eval rate:':<{col}} {prompt_eval_rate:.2f} tokens/s")
        print(f"  {'eval count:':<{col}} {total_tokens_generated} token(s)")
        print(f"  {'eval duration:':<{col}} {_fmt_duration(eval_duration)}")
        print(f"  {'eval rate:':<{col}} {eval_rate:.2f} tokens/s")

    return generated
