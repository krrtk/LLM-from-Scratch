"""
app/core/model.py
-----------------
From-scratch GPT architecture in PyTorch, faithfully extracted from the notebook.

Architecture overview
---------------------
Input token IDs
  → Token Embedding  +  Positional Embedding   (learnable, absolute)
  → N × TransformerBlock
        ├─ LayerNorm  → MultiHeadAttention (causal, scaled dot-product) → Dropout → residual
        └─ LayerNorm  → FeedForward (expand 4×, GELU, contract) → Dropout → residual
  → LayerNorm  → Linear head  →  logits  [B, T, vocab_size]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GPT_CONFIG_124M: dict = {
    "vocab_size": 50_257,   # tiktoken gpt2 vocab
    "context_length": 256,  # shortened from 1024 for CPU-friendly training
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """
    Custom LayerNorm — from the notebook.

    Formula:
        norm_x = (x - μ) / sqrt(σ² + ε)
        out    = scale * norm_x + shift

    Unlike PyTorch's built-in, scale and shift are learned nn.Parameter objects
    so we have full visibility into them.
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (tanh approximation).

    Formula:  0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            )
        )


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Architecture: Linear(d, 4d) → GELU → Linear(4d, d)
    The 4× expansion follows the original GPT/Transformer paper.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d = cfg["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(d, 4 * d),
            GELU(),
            nn.Linear(4 * d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """
    Efficient Multi-Head Causal Self-Attention — from the notebook.

    Key design decisions
    --------------------
    * Single projection matrices W_q, W_k, W_v (avoids per-head loops).
    * Heads are produced by reshaping + transposing, not stacking.
    * Causal mask is registered as a buffer (not a parameter) so it moves
      to GPU automatically.
    * Scaled dot-product: scores = (Q·Kᵀ) / √head_dim

    Shape flow
    ----------
    x : (B, T, d_in)
    Q = K = V : (B, T, d_out)  after projection
    → reshape to (B, T, n_heads, head_dim)
    → transpose to (B, n_heads, T, head_dim)
    scores : (B, n_heads, T, T)
    weights: softmax(masked scores / √head_dim)
    ctx    : (B, n_heads, T, head_dim) → (B, T, d_out)  after combine
    out    : (B, T, d_out)  after out_proj
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out    = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads

        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout  = nn.Dropout(dropout)

        # Upper-triangular causal mask (excluding diagonal)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        context_vec : (B, T, d_out)
        attn_weights: (B, n_heads, T, T)  — returned for the X-Ray dashboard
        """
        b, num_tokens, d_in = x.shape

        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        # Split into heads: (B, T, d_out) → (B, T, n_heads, head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # (B, T, n_heads, head_dim) → (B, n_heads, T, head_dim)
        keys    = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values  = values.transpose(1, 2)

        # Scaled dot-product attention scores
        attn_scores = queries @ keys.transpose(2, 3)  # (B, n_heads, T, T)

        # Apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, float("-inf"))

        # Softmax-normalised attention weights
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values, then combine heads
        context_vec = (attn_weights @ values).transpose(1, 2)  # (B, T, n_heads, head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec, attn_weights


class TransformerBlock(nn.Module):
    """
    Single GPT Transformer Block.

    Architecture
    ------------
    x → LN → MHA → Dropout → x (residual)
      → LN → FFN → Dropout → x (residual)
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff         = FeedForward(cfg)
        self.norm1      = LayerNorm(cfg["emb_dim"])
        self.norm2      = LayerNorm(cfg["emb_dim"])
        self.drop_short = nn.Dropout(cfg["drop_rate"])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x            : (B, T, emb_dim)
        attn_weights : (B, n_heads, T, T)
        """
        shortcut = x
        x, attn_weights = self.att(self.norm1(x))
        x = self.drop_short(x) + shortcut

        shortcut = x
        x = self.drop_short(self.ff(self.norm2(x))) + shortcut

        return x, attn_weights


class GPTModel(nn.Module):
    """
    Complete GPT-2 style language model.

    Parameters
    ----------
    cfg : dict  — use GPT_CONFIG_124M or a custom config dict.

    Forward returns
    ---------------
    logits       : (B, T, vocab_size)
    all_attn     : list of (B, n_heads, T, T)  — one per transformer layer
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.tok_emb  = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb  = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(
        self, in_idx: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_embeds + pos_embeds)

        all_attn: list[torch.Tensor] = []
        for block in self.trf_blocks:
            x, attn = block(x)
            all_attn.append(attn)

        x      = self.final_norm(x)
        logits = self.out_head(x)
        return logits, all_attn
