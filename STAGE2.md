# 🧠 Attention Mechanism — From Scratch
### Implementing Self-Attention, Causal Attention & Multi-Head Attention using PyTorch

---

## Overview

This project implements the **attention mechanism** of a Large Language Model step by step — starting from a simple dot-product attention with no trainable weights, all the way to a full **Multi-Head Attention** module.

> The sentence *"Your journey starts with one step"* is used throughout as the embedding example — 6 tokens, each represented as a 3D vector.

---

## Roadmap

```
Word Embeddings (3D vectors)
        │
        ▼
[1] Simple Self-Attention        → dot product, softmax, context vector (no weights)
        │
        ▼
[2] Scaled Dot-Product Attention → trainable W_Q, W_K, W_V matrices
        │
        ▼
[3] Causal (Masked) Attention    → mask future tokens + dropout
        │
        ▼
[4] Multi-Head Attention         → parallel heads via tensor splitting
```

---

##  Concepts Covered

### 1. Simple Self-Attention *(No Trainable Weights)*

Each token computes how much it should attend to every other token via dot products.

| Step | Operation | Formula |
|------|-----------|---------|
| Attention Score | Dot product of token vectors | `score = xᵢ · xⱼ` |
| Normalize | Softmax over scores | `w = softmax(scores)` |
| Context Vector | Weighted sum of all tokens | `z = Σ wᵢ · xᵢ` |

**Efficient computation** — instead of nested for-loops, use matrix multiplication:
```python
attn_scores  = inputs @ inputs.T        # all scores at once
attn_weights = torch.softmax(attn_scores, dim=-1)
context_vecs = attn_weights @ inputs
```

---

### 2. Scaled Dot-Product Attention *(With Trainable Weights)*

Introduces three learnable projection matrices so the model can *learn* what to query, match, and output.

| Matrix | Role | Shape |
|--------|------|-------|
| **W_Q** (Query) | "What am I looking for?" | `d_in × d_out` |
| **W_K** (Key) | "What do I contain?" | `d_in × d_out` |
| **W_V** (Value) | "What do I contribute?" | `d_in × d_out` |

Scores are **scaled by √d_k** to prevent large dot products from pushing softmax into vanishing-gradient regions:

```python
attn_scores  = Q @ K.T / d_k**0.5
attn_weights = torch.softmax(attn_scores, dim=-1)
context_vec  = attn_weights @ V
```

Two implementations are provided:

| Class | Weight Init | Notes |
|-------|-------------|-------|
| `SelfAttention_v1` | `nn.Parameter(torch.rand(...))` | Simple, manual |
| `SelfAttention_v2` | `nn.Linear(...)` | Better init, preferred |

> `nn.Linear` uses a more optimized weight initialization scheme, leading to more stable training.

---

### 3. Causal (Masked) Attention

Prevents each token from attending to **future** tokens — essential for autoregressive text generation where the model predicts the next word.

**How the mask works:**
```
Upper triangle filled with -∞, then softmax → future positions become 0

    [ 0.3,  -∞,   -∞,   -∞ ]       [ 1.0,  0.0,  0.0,  0.0 ]
    [ 0.2, 0.4,   -∞,   -∞ ]  ───► [ 0.4,  0.6,  0.0,  0.0 ]
    [ 0.1, 0.3,  0.5,   -∞ ]       [ 0.2,  0.3,  0.5,  0.0 ]
    [ 0.2, 0.1,  0.3,  0.6 ]       [ 0.2,  0.1,  0.3,  0.4 ]
```

```python
mask         = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
masked       = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / d_k**0.5, dim=-1)
```

**Dropout** is applied on top of the attention weights during training — randomly zeroing out connections to prevent over-reliance on specific tokens and improve generalization.

**`CausalAttention` class** bundles everything:
- `nn.Linear` projections for Q, K, V
- Upper-triangular mask via `register_buffer` (moves with the model to GPU automatically)
- `nn.Dropout` layer

---

### 4. Multi-Head Attention

Runs multiple attention mechanisms **in parallel**, each learning to focus on different relationships (syntax, semantics, coreference, etc.).

```
Input
  │
  ├──► [Head 1] ──┐
  ├──► [Head 2] ──┼──► Concatenate ──► Output Projection ──► Context Vectors
  └──► [Head N] ──┘
```

Two implementations:

| Class | Approach | Efficiency |
|-------|----------|------------|
| `MultiHeadAttentionWrapper` | Stack `n` separate `CausalAttention` modules | 🐢 Sequential matrix multiplications |
| `MultiHeadAttention` | Single W_Q/K/V, split via `.view()` + `.transpose()` | 🚀 One matrix multiply, parallel heads |

The efficient `MultiHeadAttention` avoids repeating expensive matrix multiplications per head by reshaping tensors instead:
```python
# (batch, tokens, d_out) → (batch, num_heads, tokens, head_dim)
queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
```

---

## Requirements

```bash
pip install torch matplotlib
```

---

## Quick Usage

```python
import torch

d_in, d_out    = 3, 2
context_length = 6
batch          = torch.rand(2, context_length, d_in)  # 2 sequences, 6 tokens, 3-dim

mha = MultiHeadAttention(
    d_in=d_in, d_out=d_out,
    context_length=context_length,
    dropout=0.0, num_heads=2
)

context_vecs = mha(batch)
print(context_vecs.shape)  # → torch.Size([2, 6, 2])
```

---

## Scale Reference

| Model | Heads | Embedding Dim | Parameters |
|-------|-------|---------------|------------|
| This project | 2 | 2–3 (toy) | — |
| GPT-2 Small | 12 | 768 | ~117M |
| GPT-2 XL | 25 | 1,600 | ~1.5B |

---

## Still In Progress

- [ ] Feed-forward layers & layer normalization
- [ ] Full GPT block (Attention + FFN + residuals)
- [ ] Training loop

---
