import math
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int


class Linear(nn.Module):
    """A fully connected linear layer without bias.

    This module maps the last dimension of the input tensor from `d_in` to
    `d_out` using a learned weight matrix with shape `(d_out, d_in)`. The
    weights are initialized from a truncated normal distribution.

    Args:
        d_in: Size of the input feature dimension.
        d_out: Size of the output feature dimension.
        device: Device on which the parameters are allocated.
        dtype: Data type used for the parameters.

    Shape:
        Input: `(..., d_in)`
        Output: `(..., d_out)`
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the linear layer parameters."""
        super().__init__()
        std = 2 / (d_in + d_out)
        mean = 0.0

        self.weight = nn.Parameter(torch.empty(d_out, d_in, dtype=dtype, device=device))

        nn.init.trunc_normal_(
            self.weight,
            mean=mean,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_out"]:
        """Apply the linear transformation to the input tensor.

        Args:
            x: Input tensor whose last dimension matches `d_in`.

        Returns:
            Tensor with the same leading dimensions as `x` and last dimension
            `d_out`.
        """

        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """A lookup table that maps token indices to dense vectors.

    This module stores a learnable embedding matrix with shape
    `(num_embeddings, embedding_dim)` and returns the rows indexed by the
    input tensor.

    Args:
        num_embeddings: Number of discrete embeddings, typically the vocabulary
            size.
        embedding_dim: Size of each embedding vector.
        device: Device on which the parameters are allocated.
        dtype: Data type used for the parameters.

    Shape:
        Input: `(batch, seq)`
        Output: `(batch, seq, embedding_dim)`
    """

    def __init__(
        self,
        num_embeddings: int,  # vocab size
        embedding_dim: int,  # d_model
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the embedding matrix."""
        super().__init__()

        self.embed = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        nn.init.trunc_normal_(self.embed, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(
        self, x: Int[torch.LongTensor, "batch seq"]
    ) -> Float[torch.Tensor, "batch seq d_model"]:
        """Return embeddings for the provided token indices.

        Args:
            x: Integer tensor of token indices.

        Returns:
            Tensor containing the embedding vector for each input index.
        """
        return self.embed[x, :]


class RMSNorm(nn.Module):
    """Root mean square normalization with a learned scale parameter.

    This module normalizes the last dimension of the input tensor by its root
    mean square and then applies a learned elementwise gain. The normalization
    is computed in `float32` for numerical stability and cast back to the
    input dtype before returning.

    Args:
        d_model: Size of the feature dimension to normalize.
        eps: Small constant added to the denominator for numerical stability.
        device: Device on which the parameters are allocated.
        dtype: Data type used for the parameters.

    Shape:
        Input: `(batch, seq, d_model)`
        Output: `(batch, seq, d_model)`
    """

    def __init__(
        self,
        d_model: int,
        *,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the RMSNorm parameters and stability constant."""
        super().__init__()

        self.g = nn.Parameter(torch.ones(1, d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq d_model"],
    ) -> Float[torch.Tensor, "batch seq d_model"]:
        """Normalize the input tensor across its last dimension.

        Args:
            x: Input tensor to normalize.

        Returns:
            Tensor with the same shape as `x`, normalized by RMS and scaled by
            the learned gain parameter.
        """

        in_dtype = x.dtype
        x = x.to(torch.float32)

        inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        result = inv_rms * self.g * x

        return result.to(in_dtype)


class SiLU(nn.Module):
    """Apply the Sigmoid Linear Unit activation function elementwise.

    This module computes `x * sigmoid(x)` for each element of the input tensor.
    It preserves the input shape.

    Shape:
        Input: `(batch, seq, d_model)`
        Output: `(batch, seq, d_model)`
    """

    def __init__(self):
        """Initialize the SiLU activation module."""
        super().__init__()

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq d_model"],
    ) -> Float[torch.Tensor, "batch seq d_model"]:
        """Apply the SiLU transformation to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Tensor with the same shape as `x` after elementwise activation.
        """

        return x * nn.functional.sigmoid(x)


class SwiGLUFFN(nn.Module):
    """A SwiGLU-based feed-forward network.

    This module applies two parallel input projections, gates one branch with a
    SiLU activation applied to the other branch, and projects the result back
    to the model dimension. The intermediate feed-forward dimension is set to
    `round(8 * d_model / 3)`.

    Args:
        d_model: Size of the input and output feature dimension.
        device: Device on which the parameters are allocated.
        dtype: Data type used for the parameters.

    Shape:
        Input: `(batch, seq, d_model)`
        Output: `(batch, seq, d_model)`
    """

    def __init__(
        self,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the SwiGLU feed-forward projections."""
        super().__init__()

        d_ff = round(8 * d_model / 3)

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

        self.silu = SiLU()

    def forward(self, x):
        """Apply the SwiGLU feed-forward transformation.

        Args:
            x: Input tensor with last dimension `d_model`.

        Returns:
            Tensor with the same shape as `x` after gated feed-forward
            processing.
        """
        return self.w2(self.w3(x) * self.silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    For a query/key vector of dimension d_k, we split it into d_k/2 pairs.
    Each pair (at index k, 1-indexed) is rotated by angle:

        θ_{i,k} = i / Θ^((2k-2)/d_k)

    where i is the token position. Rotation of a 2-vector [a, b] by angle α:

        [a·cos(α) - b·sin(α),
         a·sin(α) + b·cos(α)]

    The full d_k-dimensional rotation is block-diagonal, one 2×2 block per pair.
    We implement this efficiently with elementwise ops (no explicit matrix build).

    Buffers (non-persistent, not learned):
        cos_buf : shape [max_seq_len, d_k // 2]
        sin_buf : shape [max_seq_len, d_k // 2]
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device = None,
    ):
        """
        Args:
            theta:       Θ constant (e.g. 10_000.0)
            d_k:         Dimension of query / key vectors. Must be even.
            max_seq_len: Largest sequence length we will ever see.
            device:      Where to store the precomputed buffers.
        """
        super().__init__()

        assert d_k % 2 == 0, "d_k must be even for RoPE (we work in pairs)"

        # ── 1. Compute per-pair frequencies: freq_k = 1 / Θ^((2k-2)/d_k)
        #       k is 1-indexed in the formula, so (2k-2)/d_k gives
        #       0/d, 2/d, 4/d, … for k = 1, 2, 3, …
        #       Using 0-indexed pair index p = k-1: exponent = 2p / d_k
        pair_indices = torch.arange(d_k // 2, dtype=torch.float32, device=device)
        freqs = 1.0 / (theta ** (2.0 * pair_indices / d_k))  # [d_k/2]

        # ── 2. Outer product: angle[i, k] = position_i · freq_k
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, freqs)  # [max_seq_len, d_k/2]

        # ── 3. Precompute and cache cos / sin  (persistent=False → not saved in
        #       state_dict, but lives on the right device and moves with .to())
        self.register_buffer("cos_buf", torch.cos(angles), persistent=False)
        self.register_buffer("sin_buf", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply RoPE to x.

        Args:
            x:               Float tensor of shape (..., seq_len, d_k).
                             May have any number of leading batch dimensions.
            token_positions: Long tensor of shape (..., seq_len).
                             Gives the absolute position of each token, used
                             to index into the precomputed cos/sin buffers.

        Returns:
            Tensor of the same shape as x, with each d_k-dim vector rotated
            according to its token position.
        """
        # ── 1. Look up cos and sin for the requested positions.
        #       cos_buf / sin_buf are [max_seq_len, d_k/2].
        #       Indexing with token_positions of shape (..., seq_len) gives
        #       (..., seq_len, d_k/2).
        cos = self.cos_buf[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_buf[token_positions]  # (..., seq_len, d_k/2)

        # ── 2. Split x into the first and second element of every pair.
        #       x[..., 0::2] picks indices 0, 2, 4, … → shape (..., seq_len, d_k/2)
        #       x[..., 1::2] picks indices 1, 3, 5, … → shape (..., seq_len, d_k/2)
        x1 = x[..., 0::2]  # first  element of each 2-D pair
        x2 = x[..., 1::2]  # second element of each 2-D pair

        # ── 3. Apply the 2-D rotation formula to each pair:
        #       [a, b] · R(α) = [a·cos - b·sin,  a·sin + b·cos]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # ── 4. Interleave the rotated pairs back into a single tensor.
        #       stack on the last dim gives (..., seq_len, d_k/2, 2),
        #       then flatten the last two dims → (..., seq_len, d_k).
        x_out = torch.stack([rotated_x1, rotated_x2], dim=-1)
        x_out = x_out.flatten(start_dim=-2)  # (..., seq_len, d_k)

        return x_out


def softmax(
    x: Float[torch.Tensor, "... features"], dim: int = -1
) -> Float[torch.Tensor, "... features"]:
    """Compute a numerically stabilized softmax along a tensor dimension.

    The input is shifted by its maximum value along `dim` before
    exponentiation to improve numerical stability.

    Args:
        x: Input tensor containing unnormalized values.
        dim: Dimension along which to normalize.

    Returns:
        Tensor of the same shape as `x` whose values sum to 1 along `dim`.
    """

    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(
        exponentiated_rescaled_input, dim=dim, keepdim=True
    )


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... query d_k"],  # query
    K: Float[torch.Tensor, "... key d_k"],  # key
    V: Float[torch.Tensor, "... key d_v"],  # value
    M: Bool[torch.Tensor, "... query key"] | None = None,  # mask
) -> Float[torch.Tensor, "... query d_v"]:
    """Compute scaled dot-product attention.

    This function forms attention scores from the query and key tensors,
    scales them by the square root of the key dimension, optionally applies a
    mask, normalizes the scores with softmax, and uses the resulting weights
    to combine the value tensor.

    Args:
        Q: Query tensor.
        K: Key tensor with the same trailing key dimension as `Q`.
        V: Value tensor aligned with the sequence dimension of `Q` and `K`.
        M: Optional boolean attention mask over query-key positions.

    Returns:
        Tensor of attention outputs with the same leading dimensions and
        sequence length as `Q`, and value dimension `d_v`.
    """

    d_k = K.shape[-1]

    # Form raw attention logits and scale them to keep gradients well-behaved.
    attention_scores = einsum(
        Q, K, "... query d_k, ... key d_k -> ... query key"
    ) / math.sqrt(d_k)

    if M is not None:
        # Masked positions receive zero probability after softmax.
        attention_scores = torch.where(M, attention_scores, -float("inf"))

    attention_weights = softmax(attention_scores, dim=-1)

    return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")


class CausalMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking and rotary positions.

    This module projects the input sequence into query, key, and value
    tensors, splits them into attention heads, applies rotary positional
    encoding to queries and keys, performs causal scaled dot-product
    attention, and projects the concatenated head outputs back to `d_model`.

    Args:
        d_model: Size of the model feature dimension.
        num_heads: Number of attention heads.
        positional_encoder: Rotary positional embedding module applied to
            queries and keys.
        device: Device on which the parameters are allocated.
        dtype: Data type used for the parameters.

    Shape:
        Input:
            `x`: `(..., seq, d_model)`
            `token_positions`: `(..., seq)`
        Output: `(..., seq, d_model)`
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryPositionalEmbedding,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the attention projections and head geometry."""
        super().__init__()

        self.pos_enc = positional_encoder

        assert d_model % num_heads == 0

        self.num_heads = num_heads

        self.d_k = d_model // self.num_heads

        self.W_k = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.W_q = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.W_v = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.W_o = Linear(num_heads * self.d_k, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq d_model"],
        token_positions: Int[torch.Tensor, " ... seq"],
    ) -> Float[torch.Tensor, "... seq d_model"]:
        """Apply causal multi-head self-attention to the input sequence.

        Args:
            x: Input tensor containing token representations.
            token_positions: Absolute token positions used by the rotary
                positional encoder.

        Returns:
            Tensor with the same leading dimensions and feature size as `x`.
        """

        # Project the model states into key, query, and value spaces.
        K, Q, V = self.W_k(x), self.W_q(x), self.W_v(x)

        # Split the projected features into independent attention heads.
        K = rearrange(K, "... seq (head d) -> ... head seq d", head=self.num_heads)
        Q = rearrange(Q, "... seq (head d) -> ... head seq d", head=self.num_heads)
        V = rearrange(V, "... seq (head d) -> ... head seq d", head=self.num_heads)

        # Inject positional information into queries and keys before attention.
        Q = self.pos_enc(Q, token_positions)  # add head dim for indexing
        K = self.pos_enc(K, token_positions)

        sequence_length = x.shape[-2]
        # # A lower-triangular mask prevents each token from attending forward.
        M = (
            torch.tril(
                torch.ones(
                    sequence_length, sequence_length, dtype=torch.bool, device=x.device
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # q_len = Q.shape[-2]  # query positions
        # k_len = K.shape[-2]  # key positions (may differ with KV cache)
        # M = torch.ones(q_len, k_len, dtype=torch.bool, device=x.device)
        # M = torch.tril(M, diagonal=k_len - q_len)  # offset for past context
        attention_output = scaled_dot_product_attention(Q, K, V, M)

        # Merge the head dimension back into the model dimension.
        attention_output = rearrange(
            attention_output, "... head seq d -> ... seq (head d)"
        )
        return self.W_o(attention_output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RotaryPositionalEmbedding,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.pos_enc = positional_encoder

        self.layer_norm_1 = RMSNorm(d_model)
        self.layer_norm_2 = RMSNorm(d_model)
        self.mhsa = CausalMultiHeadSelfAttention(d_model, num_heads, self.pos_enc)
        self.ffn = SwiGLUFFN(d_model)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq d_model"],
        token_positions: Float[torch.Tensor, "... seq"],
    ):
        y = x + self.mhsa(self.layer_norm_1(x), token_positions)
        return y + self.ffn(self.layer_norm_2(y))


class AliLM(nn.Module):
    """Decoder-only language model built from stacked transformer blocks.

    The model maps token IDs to logits over the vocabulary for each sequence
    position. It uses token embeddings, rotary-positioned causal self-attention
    blocks, a final RMSNorm, and a linear output projection.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        context_length: Maximum sequence length used to precompute RoPE tables.
        d_model: Width of token representations.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads per block.
        d_ff: Feed-forward width placeholder for API compatibility.

    Shape:
        Input: `(batch, seq)` integer token IDs.
        Output: `(batch, seq, vocab_size)` next-token logits.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
    ):
        """Initialize model components and shared rotary positional encoder."""
        super().__init__()

        d_k = d_model // num_heads

        self.emb = Embedding(vocab_size, d_model)
        pos_enc = RotaryPositionalEmbedding(10000, d_k, context_length)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, pos_enc)
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = RMSNorm(d_model)
        self.output_embedding = Linear(d_in=d_model, d_out=vocab_size)

    def forward(self, x: Float[torch.Tensor, "batch seq"]):
        """Compute token logits for a batch of input token IDs.

        Args:
            x: Integer tensor of token IDs with shape `(batch, seq)`.

        Returns:
            Logits tensor with shape `(batch, seq, vocab_size)`.
        """

        embeddings = self.emb(x)
        sequence_length = x.shape[-1]
        token_positions = torch.arange(sequence_length, device=x.device)
        transformer_output = embeddings

        for block in self.transformer_blocks:
            transformer_output = block(transformer_output, token_positions)

        normalization = self.layer_norm(transformer_output)
        logits = self.output_embedding(normalization)
        return logits
