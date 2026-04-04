# LLM from Scratch — Stage 3: GPT Architecture

## Overview

This notebook implements the full **GPT-2 (124M parameter) architecture from scratch** using PyTorch. It is the third stage in a series building a Large Language Model step by step. By the end of this notebook, a complete, untrained GPT-2 model is assembled and capable of generating text.

---

## Prerequisites

- Completion of **Stage 1** (tokenization, embeddings) and **Stage 2** (multi-head attention)
- Python 3.x
- PyTorch
- `tiktoken` (OpenAI's tokenizer)
- `matplotlib`

Install dependencies:
```bash
pip install torch tiktoken matplotlib
```

---

## Model Configuration

The notebook uses the same hyperparameters as the original GPT-2 small model:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 12,            # Number of attention heads
    "n_layers": 12,           # Number of transformer layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias
}
```

---

## Topics Covered

### 1. Layer Normalisation
- Why it's needed: resolves vanishing/exploding gradients and internal covariate shift
- Manual normalization (mean = 0, variance = 1)
- Custom `LayerNorm` class with learnable `scale` and `shift` parameters

### 2. GELU Activation Function
- Implementation of the **Gaussian Error Linear Unit (GELU)**
- Comparison with ReLU via visualization

### 3. Feed-Forward Network
- `FeedForward` module using GELU
- Expands embedding dimension **4×** internally, then compresses back — allowing richer weight exploration

### 4. Shortcut (Residual) Connections
- Demonstrates gradient flow with and without shortcuts using a deep neural network example
- Shows how residual connections prevent vanishing gradients

### 5. Transformer Block
- Combines: **MultiHeadAttention** (from Stage 2) + **LayerNorm** + **FeedForward** + **Dropout** + **Shortcut Connections**
- Verifies that input and output shapes are preserved (shape-preserving property of transformers)

### 6. Full GPT Model (`GPTModel`)
- Token embeddings + positional embeddings
- Dropout on embeddings
- Stack of 12 `TransformerBlock` layers
- Final `LayerNorm` and linear output head
- **Weight tying** between token embedding and output layer (saves ~38M parameters)
- Total parameters: **~163M** (or **~124M** with weight tying)
- Model size: **~621 MB** (float32)

### 7. Text Generation
- `generate_text_simple()` function: greedy next-token prediction
- Demonstrates inference on a prompt: `"Hello, I am"`
- Output is decoded back to human-readable text using `tiktoken`

---

## Key Classes

| Class | Description |
|---|---|
| `LayerNorm` | Custom layer normalization with learnable parameters |
| `GELU` | GELU activation function |
| `FeedForward` | Two-layer MLP with GELU, used inside transformer blocks |
| `ExampleDeepNeuralNetwork` | Demo network to illustrate shortcut connections |
| `MultiHeadAttention` | Copied from Stage 2 — causal multi-head self-attention |
| `TransformerBlock` | Full transformer block (attention + FFN + norms + shortcuts) |
| `GPTModel` | Complete GPT-2 architecture |

---

## Usage

Run cells sequentially. After executing all cells, you will have:
1. A fully assembled GPT-2 architecture
2. A working (but **untrained**) model that can generate random text

Example output (untrained):
```
Hello, I am Featureiman Byeswickattribute argue
```

> The output is incoherent because the model has not been trained yet. Training is covered in **Stage 4**.

---

## Next Steps

> *"We have just created the basic untrained architecture for GPT-2 124M parameters. Now we just have to train it to get better results."*

**Stage 4** will cover:
- Loading pre-trained GPT-2 weights, or
- Training the model from scratch on a text corpus

---

## Series Structure

| Stage | Topic |
|---|---|
| Stage 1 | Tokenization, embeddings, data loading |
| Stage 2 | Self-attention, multi-head attention |
| **Stage 3** | **Full GPT architecture (this notebook)** |
| Stage 4 | Training & weight loading |
