# From Raw Text to Embeddings — LLM Preprocessing Pipeline

A walkthrough of the full preprocessing pipeline used to prepare text data for training a GPT-style Large Language Model. Based on the notebook **Position_Embeddings.ipynb**, which covers everything from reading a `.txt` file to producing the final input tensors ready for a transformer.

---

## The Big Picture

The notebook answers one question:

> **How does raw text become numbers that a neural network can learn from?**

The pipeline has 7 stages:

```
Raw Text → Tokenize → Build Vocabulary → Assign IDs → Create Input/Target Pairs → Token Embeddings → Positional Embeddings
```

---

## Stage 1 — Tokenization

**Goal:** Split raw text into individual *tokens* (words and punctuation) that can each be assigned a number.

The notebook uses Edith Wharton's short story *The Verdict* (~20,479 characters) as the training corpus.

Three iterations of a `re.split()` regex tokenizer are built, each more capable than the last:

| Version | Splits on | Result |
|---|---|---|
| v1 | Whitespace only | `['Hello,', 'world.']` — commas stuck to words |
| v2 | Whitespace + `[,.]` | Words and punctuation separated, but empty strings remain |
| v3 (final) | `[,.:;?_!"()']`, `--`, and whitespace | Clean tokens, whitespace stripped |

**Final output:**
```python
preprocessed = re.split(r'([,.:;?_!"()\'']|--|\\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# → ['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', ...]
# Total: 4,690 tokens
```

---

## Stage 2 — Building a Vocabulary & Assigning IDs

**Goal:** Give every unique token a unique integer ID.

```python
all_words = sorted(set(preprocessed))
vocab = {token: integer for integer, token in enumerate(all_words)}
# → {'!': 0, '"': 1, "'": 2, ..., 'younger': 1127, ...}
# Total: 1,130 unique tokens
```

### SimpleTokenizerV1

A tokenizer class with two methods:

- `encode(text)` — converts text → list of integer IDs
- `decode(ids)` — converts integer IDs → text (with a regex to clean up spaces before punctuation)

**Problem discovered:** Encoding `"Hello, do you like tea?"` raises `KeyError: 'Hello'` because "Hello" never appeared in the training story. This motivates the next upgrade.

---

## Stage 2b — Special Tokens

**Goal:** Handle unknown words and document boundaries gracefully.

Two special tokens are added to the vocabulary:

| Token | Purpose |
|---|---|
| `<\|unk\|>` | Replaces any word not found in the vocabulary |
| `<\|endoftext\|>` | Inserted between unrelated documents during training |

```python
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
# Vocabulary grows from 1,130 → 1,132 tokens
```

### SimpleTokenizerV2

The `encode()` method now substitutes `<|unk|>` for unknown words instead of crashing:

```python
preprocessed = [
    item if item in self.str_to_int else "<|unk|>"
    for item in preprocessed
]
```

**Example:**
```
Input:  "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
Output: "<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>."
# "Hello" and "palace" were not in the training story
```

---

## Stage 3 — Byte Pair Encoding (BPE) with `tiktoken`

**Goal:** Use a production-grade tokenizer that handles any unknown word without `<|unk|>`.

BPE breaks unknown words into known *subword units*:

```
"someunknownPlace" → ["some", "unknown", "Place"] → each gets an ID
```

No unknown token ever needed — every word can be decomposed into characters if necessary.

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
strings  = tokenizer.decode(integers)
```

**Key facts about the GPT-2 BPE tokenizer:**
- Vocabulary size: **50,257 tokens**
- `<|endoftext|>` = token ID **50,256** (the last one)
- The full story encodes to **5,145 tokens** (vs 4,690 with the simple tokenizer)

---

## Stage 4 — Creating Input-Target Pairs

**Goal:** Prepare `(input, target)` pairs for next-word prediction training.

The model is trained to predict the *next* token given all previous tokens. This is done with a **sliding window**:

```
Token stream:  [and, established, himself, in, a, small, studio, ...]

Input  x:      [and, established, himself, in]
Target y:            [established, himself, in, a]   ← shifted by 1
```

Each position creates one prediction task:

```
[and]                        → established
[and, established]           → himself
[and, established, himself]  → in
[and, established, himself, in] → a
```

```python
context_size = 4
x = enc_sample[:context_size]       # input tokens
y = enc_sample[1:context_size + 1]  # target tokens (shifted by 1)
```

---

## Stage 5 — The PyTorch DataLoader

**Goal:** Efficiently batch the data for training using PyTorch's built-in utilities.

```python
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids, self.target_ids = [], []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
```

### Key Parameters

| Parameter | Description |
|---|---|
| `max_length` | Number of tokens per input sequence (the context window size) |
| `stride` | How many positions to slide the window each step. `stride=1` → heavy overlap (can overfit). `stride=max_length` → no overlap |
| `batch_size` | Number of sequences processed simultaneously |
| `drop_last=True` | Drops the last incomplete batch to avoid loss spikes during training |
| `shuffle=True` | Randomizes batch order during training |

**Example output** with `batch_size=8, max_length=4, stride=4`:
```
Inputs shape:  torch.Size([8, 4])
Targets shape: torch.Size([8, 4])
```

---

## Stage 6 — Token Embeddings

**Goal:** Convert integer token IDs into dense floating-point vectors that carry semantic meaning.

An embedding layer is essentially a **lookup table** — token ID `i` returns row `i` of the weight matrix. Those weights start random and are refined during training.

```python
vocab_size = 50257   # BPE vocabulary
output_dim = 256     # embedding dimensions (GPT-3 uses 12,288)

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)
# inputs shape:          [8, 4]
# token_embeddings shape: [8, 4, 256]
```

Each of the 4 tokens in each of the 8 sequences now has a 256-dimensional vector representation.

---

## Stage 7 — Positional Embeddings

**Goal:** Tell the model *where* each token appears in the sequence, since transformers have no built-in sense of order.

The self-attention mechanism processes all tokens simultaneously and treats the same word identically regardless of its position. "dog bites man" and "man bites dog" would look the same without position information.

### The Fix: a Second Embedding Layer

```python
context_length = max_length   # = 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# Feed positions [0, 1, 2, 3] as if they were token IDs
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
# pos_embeddings shape: [4, 256]
```

### Adding to Token Embeddings

Position vectors are simply **added** to token vectors. PyTorch broadcasts the `[4, 256]` positional tensor across all 8 batches automatically:

```python
input_embeddings = token_embeddings + pos_embeddings
# shape: [8, 4, 256]
```

This is the **final output of the preprocessing pipeline** — ready to be fed into the transformer's attention layers.

### Learned vs Sinusoidal Positional Embeddings

The notebook uses **learned embeddings** (GPT-2 style) — positions are treated like tokens, start random, and the model learns what values are useful. The original *"Attention Is All You Need"* paper instead uses a fixed **sinusoidal formula**:

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

| Approach | Training needed? | Handles longer sequences? | Used by |
|---|---|---|---|
| Learned | Yes | No (fixed to training length) | GPT-2, this notebook |
| Sinusoidal | No (formula) | Yes (extrapolates) | Original Transformer |
| RoPE | Partial | Yes | LLaMA, modern LLMs |

---

## Summary

| Stage | Input | Output | Key detail |
|---|---|---|---|
| Load text | `.txt` file | Raw string | 20,479 characters |
| Tokenize | Raw string | Token list | 4,690 tokens |
| Build vocabulary | Token list | `{token: id}` dict | 1,130 unique tokens |
| Special tokens | Vocabulary | Extended vocabulary | +2 tokens (`<\|unk\|>`, `<\|endoftext\|>`) |
| BPE (tiktoken) | Raw string | Token ID list | 50,257-token vocabulary, no `<\|unk\|>` needed |
| Sliding window | Token IDs | `(x, y)` pairs | Target = input shifted by 1 |
| DataLoader | `(x, y)` pairs | Batched tensors | `[batch, seq_len]` |
| Token embedding | `[8, 4]` IDs | `[8, 4, 256]` vectors | Lookup table, trained |
| Positional embedding | Positions `[0..3]` | `[4, 256]` vectors | Added to token embeddings |
| **Final input** | — | `[8, 4, 256]` tensor | Ready for transformer layers |

---

## Dependencies

```bash
pip install tiktoken torch
```

```python
import re
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
```
