# AliLM — A Decoder-Only Transformer Language Model

AliLM is a from-scratch decoder-only transformer language model built entirely in PyTorch. It implements the modern transformer recipe — RoPE positional encodings, RMSNorm, SwiGLU feed-forward networks, and causal multi-head self-attention — without relying on any higher-level model libraries. The full pipeline covers dataset download, BPE-based pretokenization, training, validation, and interactive inference.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Environment Setup](#environment-setup)
5. [Datasets](#datasets)
   - [Download](#download)
   - [Tokenize](#tokenize)
6. [Training](#training)
7. [Validation](#validation)
8. [Interactive Inference](#interactive-inference)
9. [Checkpoints](#checkpoints)
10. [Dependencies](#dependencies)

---

## Project Structure

```
AliLM/
├── config.json            # Global hyperparameter config shared by all scripts
├── model.py               # Full transformer model definition
├── train.py               # Training entry point
├── validate.py            # Standalone validation script
├── generate.py            # Autoregressive generation with streaming output
├── ollama.py              # Interactive inference engine (REPL)
├── pretokenizer.py        # CLI for converting raw text to shard files
├── utils.py               # Shared helpers (data loading, checkpoints, printing)
├── download.sh            # Downloads raw dataset files
├── pyproject.toml         # uv/pip project definition and dependencies
├── data/                  # Raw text files (created by download.sh)
├── tokenized_data/        # Pretokenized shard files (.pt)
│   ├── tiny_story/
│   │   ├── training/
│   │   └── validation/
│   └── owt/
│       ├── training/
│       └── validation/
└── models/                # Saved model weights (created by train.py)
```

---

## Architecture

AliLM is a **decoder-only transformer** (GPT-style) with the following design choices:

### Token Embedding
Tokens are mapped to dense vectors via a learned `Embedding` table of shape `(vocab_size, d_model)`. Weights are initialised from a truncated normal distribution.

### Rotary Positional Encoding (RoPE)
Position information is injected into queries and keys via **Rotary Positional Embedding** rather than absolute or learned positional embeddings. For a query/key vector of dimension `d_k`, each consecutive pair of dimensions `(x_{2i}, x_{2i+1})` is rotated by angle:

$$\theta_{pos,i} = \frac{pos}{\Theta^{2i / d_k}}, \quad \Theta = 10000$$

The rotation is applied elementwise using precomputed `cos` and `sin` buffers of shape `[max_seq_len, d_k/2]`. RoPE enables the model to generalise to sequence lengths not seen during training and encodes relative position naturally within the attention dot product.

### Causal Multi-Head Self-Attention
Each transformer block contains a `CausalMultiHeadSelfAttention` layer with:
- Separate `W_q`, `W_k`, `W_v` projections (no bias) of size `d_model → num_heads × d_k`
- RoPE applied to queries and keys after head splitting
- A lower-triangular causal mask so each token can only attend to past tokens
- Scaled dot-product attention: $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$
- Output projection `W_o` of size `num_heads × d_k → d_model`

### SwiGLU Feed-Forward Network
Each block's feed-forward sublayer uses **SwiGLU** gating:

$$\text{FFN}(x) = W_2\!\left(W_3(x) \cdot \text{SiLU}(W_1(x))\right)$$

where the intermediate dimension is `⌊8 × d_model / 3⌋`. This gives an efficient gated activation without a separate gating parameter.

### RMSNorm
Layer normalisation uses **Root Mean Square Norm** (no mean subtraction), computed in `float32` for numerical stability then cast back to the working dtype:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot g$$

Pre-norm is applied before both the attention and feed-forward sublayers (residual connections wrap each).

### Output Head
After the final RMSNorm, a weight-tied (or separate) linear projection maps `d_model → vocab_size` to produce per-token logits over the vocabulary.

### Default Hyperparameters

| Parameter        | Value   | Notes                          |
|-----------------|---------|--------------------------------|
| `vocab_size`    | 50 257  | GPT-2 BPE vocabulary           |
| `context_length`| 256     | Tokens per sequence window     |
| `embedding_dim` | 512     | `d_model`                      |
| `num_heads`     | 16      | Attention heads (d_k = 32)     |
| `num_layers`    | 16      | Transformer blocks             |
| `d_ff`          | 1 344   | Feed-forward intermediate dim  |
| `Batch_size`    | 32      | Training batch size            |
| `learning_rate` | 1e-3    | Initial LR (cosine annealed)   |

All values live in [`config.json`](config.json) and are shared across every script.

---

## Configuration

All scripts read a single [`config.json`](config.json) at startup. Edit this file to change any hyperparameter without touching source code.

```json
{
    "model_name": "AliLM",
    "vocab_size": 50257,
    "context_length": 256,
    "embedding_dim": 512,
    "num_heads": 16,
    "num_layers": 16,
    "d_ff": 1344,
    "Batch_size": 32,
    "learning_rate": 0.001,
    "model_weights_path": "models/AliLM.pt",
    "inference_num_tokens": 256,
    "inference_temperature": 0.8
}
```

| Key                    | Used by                          | Description                                      |
|-----------------------|----------------------------------|--------------------------------------------------|
| `model_name`          | `train`, `validate`, `ollama`    | Determines the saved weights filename            |
| `vocab_size`          | all                              | Must match the tokenizer vocabulary size         |
| `context_length`      | all                              | Maximum sequence length; RoPE tables are sized to this |
| `embedding_dim`       | all                              | `d_model` — width of all hidden representations |
| `num_heads`           | all                              | Number of attention heads                        |
| `num_layers`          | all                              | Number of stacked transformer blocks             |
| `d_ff`                | all                              | Feed-forward intermediate dimension              |
| `Batch_size`          | `train`, `validate`              | Batch size for training and evaluation           |
| `learning_rate`       | `train`                          | Initial learning rate passed to AdamW            |
| `model_weights_path`  | `validate`, `ollama`             | Path to load saved weights from                  |
| `inference_num_tokens`| `ollama`                         | Tokens to generate per prompt                    |
| `inference_temperature` | `ollama`                       | Sampling temperature (0 < t ≤ 1 for focused, >1 for creative) |

---

## Environment Setup

AliLM uses [**uv**](https://github.com/astral-sh/uv) for fast, reproducible environment management.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone <repo-url>
cd AliLM
```

### 3. Create the virtual environment and install dependencies

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock`, creates a `.venv/`, and installs all dependencies including PyTorch (CUDA 12.6 build by default as specified in `pyproject.toml`).

> **CPU-only machines:** Edit `pyproject.toml` and remove the `[[tool.uv.index]]` section for `pytorch-cu126` and the `[tool.uv.sources]` block, then run `uv sync` again to pull the CPU wheels.

### 4. Verify

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Running scripts

All scripts are run through `uv run` so the correct environment is always used:

```bash
uv run train.py --train_data_path ./tokenized_data/tiny_story/training
```

---

## Datasets

Two datasets are supported out of the box:

| Dataset         | Size (raw) | Description                                              |
|----------------|-----------|----------------------------------------------------------|
| **TinyStories** | ~2.1 GB   | Short children's stories generated by GPT-4; good for quick experiments |
| **OpenWebText** | ~9 GB     | Web text sample from Stanford CS336; closer to GPT-2 pretraining data   |

### Download

Run the provided shell script. Files that already exist are skipped automatically:

```bash
bash download.sh
```

This downloads the following files into `data/`:

```
data/
├── TinyStoriesV2-GPT4-train.txt
├── TinyStoriesV2-GPT4-valid.txt
├── owt_train.txt
└── owt_valid.txt
```

### Tokenize

Before training, convert the raw text files into binary shard files using the GPT-2 BPE tokenizer. Each non-empty line is treated as one document; an end-of-text token is appended after each one. When the accumulated token count reaches `max_tokens_per_shard`, the current buffer is flushed to a numbered `.pt` file.

**TinyStories — training split:**

```bash
uv run pretokenizer.py \
    --input_path ./data/TinyStoriesV2-GPT4-train.txt \
    --output_prefix ./tokenized_data/tiny_story/training/ts_train \
    --max_tokens_per_shard 50000000
```

**TinyStories — validation split:**

```bash
uv run pretokenizer.py \
    --input_path ./data/TinyStoriesV2-GPT4-valid.txt \
    --output_prefix ./tokenized_data/tiny_story/validation/ts_valid \
    --max_tokens_per_shard 50000000
```

**OpenWebText — training split:**

```bash
uv run pretokenizer.py \
    --input_path ./data/owt_train.txt \
    --output_prefix ./tokenized_data/owt/training/ow_train \
    --max_tokens_per_shard 50000000
```

**OpenWebText — validation split:**

```bash
uv run pretokenizer.py \
    --input_path ./data/owt_valid.txt \
    --output_prefix ./tokenized_data/owt/validation/ow_valid \
    --max_tokens_per_shard 50000000
```

**`pretokenizer.py` options:**

| Flag                    | Default                      | Description                             |
|------------------------|------------------------------|-----------------------------------------|
| `--input_path`         | *(required)*                 | Path to the raw `.txt` file             |
| `--output_prefix`      | `tokenized_data/shard_`      | Directory + filename prefix for shards  |
| `--max_tokens_per_shard` | `50_000_000`               | Token cap per shard file                |

The output directory is created automatically if it does not exist.

---

## Training

Training reads all `*.pt` shard files from `--train_data_path` (sorted, so order is deterministic), builds the model from `config.json`, and runs the main loop with AdamW + cosine LR annealing.

### Basic usage

```bash
uv run train.py --train_data_path ./tokenized_data/tiny_story/training
```

### Full options

```bash
uv run train.py \
    --train_data_path      ./tokenized_data/tiny_story/training \
    --num_training_tokens  500000000 \
    --max_training_time    3600 \
    --print_interval       200 \
    --checkpoint_interval  1000
```

| Flag                     | Default       | Description                                                   |
|-------------------------|---------------|---------------------------------------------------------------|
| `--train_data_path`     | *(required)*  | Directory containing `.pt` training shards                    |
| `--num_training_tokens` | `500_000_000` | Total token budget; training stops when reached               |
| `--max_training_time`   | `3600`        | Wall-clock budget in seconds; training stops when reached     |
| `--print_interval`      | `200`         | Print a status line every N steps                             |
| `--checkpoint_interval` | `1000`        | Save a checkpoint every N steps                               |

Training stops at whichever limit is hit first (token budget or time budget).

### Training output

```
──────────── Data ────────────
  Shards directory:            ./tokenized_data/tiny_story/training
  Tokens loaded:               476,879,343

──────────── Model ────────────
  Parameters:                  44.55M
  Size:                        169.95 MB
  Device:                      cuda
  Dtype:                       bfloat16

──────────── Training ────────────
  Steps budget:                58,691
  Batch size:                  32
  Context length:              256
  Initial LR:                  1.0e-03
  Token budget:                500.0M
  Time budget:                 60 min

  step [   200/58691]  loss=3.2451  avg=4.1023  ppl=60.41  lr=9.97e-04  tok/s=42.3k  tokens=1.64M  elapsed=0.6m
  step [   400/58691]  loss=3.1102  avg=3.8234  ppl=45.73  lr=9.93e-04  tok/s=43.1k  tokens=3.28M  elapsed=1.3m
  ...
```

### Saved artefacts

| Path                                          | Description                                      |
|----------------------------------------------|--------------------------------------------------|
| `models/<model_name>.pt`                     | Final model weights (state dict only)            |
| `checkpoints/model_checkpoint_epoch_XXXXX.pt`| Periodic training checkpoints (model + optimiser)|

The `models/` and `checkpoints/` directories are created automatically.

---

## Validation

Validation is a separate script that loads a saved checkpoint, evaluates it on pretokenized validation shards, and reports cross-entropy loss and perplexity.

### Basic usage

```bash
uv run validate.py --data_path ./tokenized_data/tiny_story/validation
```

### Full options

```bash
uv run validate.py \
    --data_path   ./tokenized_data/tiny_story/validation \
    --config_path config.json \
    --num_batches 200
```

| Flag            | Default        | Description                                     |
|----------------|----------------|-------------------------------------------------|
| `--data_path`  | *(required)*   | Directory containing `.pt` validation shards    |
| `--config_path`| `config.json`  | Path to the global config file                  |
| `--num_batches`| `100`          | Number of random batches to evaluate            |

The model weights are loaded from `config["model_weights_path"]` (`models/AliLM.pt` by default).

### Validation output

```
──────────── Evaluation ────────────
  Batches:                     100
  Batch size:                  32
  Context length:              256

  batch [  10/100]  loss=2.8341  ppl=17.04
  batch [  20/100]  loss=2.8102  ppl=16.63
  ...

──────────── Results ────────────
  Batches evaluated:           100
  Tokens evaluated:            819,200
  Validation loss:             2.8120
  Perplexity:                  16.66
  Tok/s:                       38.4k
  Elapsed:                     21.32s
```

---

## Interactive Inference

`ollama.py` loads the trained model and enters an interactive prompt loop where you type a prompt and receive streamed generated text.

```bash
uv run ollama.py
```

Pass `--verbose` to also print detailed Ollama-style timing metrics after each generation:

```bash
uv run ollama.py --verbose
```

| Flag        | Default | Description                                                              |
|------------|---------|--------------------------------------------------------------------------|
| `--verbose` | off     | Print section headers, prompt echo, and a timing breakdown after each generation |

Generation parameters (`inference_num_tokens`, `inference_temperature`) are read from `config.json`. Edit those values to change generation behaviour without touching any Python code.

### Default mode (no flag)

Text streams directly to the terminal with no decoration:

```
──────────── AliLM Inference Engine ────────────
  Model:                       AliLM
  Device:                      cuda
  Dtype:                       float16
  Loading weights:             models/AliLM.pt
  Status:                      Ready

──────────── Interactive Engine ────────────
  Temperature:                 0.80
  Max new tokens:              256
  Type 'exit' to quit.

Prompt> Once upon a time in a small village
Once upon a time in a small village there lived a little girl named Lily...

Prompt> exit
Exiting.
```

### Verbose mode (`--verbose`)

Adds section headers, a prompt echo, generation stats, and a full Ollama-style timing breakdown:

```
Prompt> Once upon a time in a small village
──────────── Generation ────────────
  Device:                      cuda
  Prompt tokens:               9
  Target new tokens:           256
  Temperature:                 0.80
──────────── Prompt ────────────
────────────────────────────────────────────────────────────────
Once upon a time in a small village
────────────────────────────────────────────────────────────────
──────────── Generated Text ────────────
────────────────────────────────────────────────────────────────
Once upon a time in a small village there lived a little girl
named Lily. She loved to play in the garden with her toys...
────────────────────────────────────────────────────────────────
──────────── Generation Stats ────────────
  Prompt tokens:               9
  Generated tokens:            256
  Total context length:        265
  Tok/s:                       47.3
  Elapsed:                     5.53s

  total duration:          5.534s
  load duration:           2.1ms
  prompt eval count:       9 token(s)
  prompt eval duration:    142.3ms
  prompt eval rate:        63.24 tokens/s
  eval count:              256 token(s)
  eval duration:           5.389s
  eval rate:               47.32 tokens/s
```

The timing breakdown mirrors the real Ollama CLI output:

| Metric               | Description                                                            |
|---------------------|------------------------------------------------------------------------|
| `total duration`    | Wall-clock time from start of `generate()` to the last token          |
| `load duration`     | Time to encode the prompt and move it to device                        |
| `prompt eval count` | Number of tokens in the input prompt                                   |
| `prompt eval duration` | Time for the first (batch prefill) forward pass over the full prompt |
| `prompt eval rate`  | Prompt tokens ÷ prompt eval duration                                   |
| `eval count`        | Total new tokens generated                                             |
| `eval duration`     | Time for all autoregressive generation steps after the first token     |
| `eval rate`         | Generated tokens ÷ eval duration                                       |

### Temperature guide

| Temperature | Behaviour                                          |
|------------|---------------------------------------------------|
| `0.1–0.4`  | Very focused and repetitive; near-greedy decoding  |
| `0.6–0.9`  | Balanced creativity and coherence (recommended)    |
| `1.0`      | Full distribution; more varied but less coherent   |
| `>1.0`     | High randomness; often incoherent                  |

---

## Checkpoints

Checkpoints written during training include both the model state dict and the optimiser state dict, allowing training to be resumed exactly. They are saved to `checkpoints/model_checkpoint_epoch_XXXXX.pt`.

To resume training from a checkpoint, use `load_checkpoint` from `utils.py`:

```python
from utils import load_checkpoint
epoch = load_checkpoint("checkpoints/model_checkpoint_epoch_01000.pt", model, optimizer)
```

The final weights saved by `train.py` (at `models/<model_name>.pt`) contain only the model state dict (no optimiser state) and are used by `validate.py` and `ollama.py`.

---

## Dependencies

| Package        | Purpose                                        |
|---------------|------------------------------------------------|
| `torch`       | Core tensor operations and neural network primitives |
| `transformers`| GPT-2 BPE tokenizer for pretokenization and inference |
| `einops`      | Readable tensor rearrange/einsum operations   |
| `jaxtyping`   | Shape-annotated type hints for tensors        |
| `numpy`       | Numerical utilities                            |
| `matplotlib`  | Plotting (used in dev notebook)               |

Install all dependencies with:

```bash
uv sync
```

For development extras (Jupyter notebooks, profiling):

```bash
uv sync --group dev
```
