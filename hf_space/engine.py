"""
engine.py
---------
Generation, training loop, loss utilities — from the notebook.

Provides
--------
generate()           — top-k + temperature sampling (returns text + x-ray data)
train_model_simple() — training loop with eval checkpoints
calc_loss_batch()    — single-batch cross-entropy
calc_loss_loader()   — average loss over a DataLoader
fine_tune_on_text()  — quick mini fine-tune for the PDF personality feature
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPTModel


# ---------------------------------------------------------------------------
# Helpers: token ↔ text conversion
# ---------------------------------------------------------------------------

def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    """text → (1, T) tensor of token IDs."""
    ids = tokenizer.encode(text)
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    """(1, T) or (T,) tensor → decoded string."""
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    model: GPTModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    device: torch.device | str = "cpu",
) -> dict:
    """
    Generate text and return X-Ray inspection data.

    Returns
    -------
    dict with keys:
      text            : str                  — full decoded sequence (prompt + generated)
      token_ids       : list[int]            — full sequence token IDs
      token_breakdown : list[str]            — every generated token as a string (for UI pills)
      top5_probs      : list[dict]           — top-5 next-token probabilities at last step
      attn_weights    : list[list[list]]     — per-layer mean attention, shape [T, T]
      total_tokens    : int                  — total token count
    """
    model.eval()
    model.to(device)

    idx = text_to_token_ids(prompt, tokenizer).to(device)
    context_size: int = model.pos_emb.weight.shape[0]

    # Track the token IDs of the prompt so we can isolate generated tokens
    prompt_len = idx.shape[1]

    # We capture attention + logits for the final forward pass
    last_attn:  list[torch.Tensor] = []
    last_logits = None

    with torch.no_grad():
        for _step in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]

            logits, all_attn = model(idx_cond)

            # Focus on the last token position
            logits_last = logits[:, -1, :]  # (1, vocab_size)

            # Temperature scaling
            if temperature != 1.0:
                logits_last = logits_last / temperature

            # Top-k filtering
            if top_k > 0:
                top_vals, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                threshold    = top_vals[:, -1].unsqueeze(-1)
                logits_last  = logits_last.masked_fill(logits_last < threshold, float("-inf"))

            probs = F.softmax(logits_last, dim=-1)  # (1, vocab_size)

            # Sample (or argmax when temperature is very low)
            if temperature < 1e-4:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)

            # Save inspection data from the last generation step
            last_attn   = all_attn
            last_logits = probs   # softmax probs from last step

    # ---- Post-process --------------------------------------------------------
    token_ids: list[int] = idx.squeeze(0).tolist()

    # Decode every individual token (for UI pills)
    def _tok(i: int) -> str:
        return tokenizer.id_to_token(i) if hasattr(tokenizer, "id_to_token") \
               else tokenizer.decode([i])

    # token_breakdown = only the GENERATED tokens (not the prompt)
    token_breakdown: list[str] = [_tok(t) for t in token_ids[prompt_len:]]

    # Full decoded text
    generated_text = token_ids_to_text(idx, tokenizer)

    # Top-5 next-token probabilities from the last step
    top5_result  = torch.topk(last_logits.squeeze(0), k=5)
    top5_indices = top5_result.indices.tolist()
    top5_values  = top5_result.values.tolist()

    top5_probs = [
        {"token": _tok(i), "probability": round(float(p), 6)}
        for i, p in zip(top5_indices, top5_values)
    ]

    # Average attention weights across heads for each layer → [T, T] lists
    # last_attn: list of (1, n_heads, T, T) — one per transformer layer
    attn_data: list[list] = []
    T = idx.shape[1]
    for layer_attn in last_attn:            # (1, n_heads, T, T)
        mean_attn = layer_attn[0].mean(dim=0)   # (T, T)
        mean_attn = mean_attn[:T, :T]
        attn_data.append(mean_attn.cpu().tolist())

    return {
        "text":            generated_text,
        "token_ids":       token_ids,
        "token_breakdown": token_breakdown,
        "top5_probs":      top5_probs,
        "attn_weights":    attn_data,        # one [T×T] matrix per transformer layer
        "total_tokens":    len(token_ids),
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device | str,
) -> torch.Tensor:
    """Cross-entropy loss for a single batch."""
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits, _    = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device | str,
    num_batches: Optional[int] = None,
) -> float:
    """Average cross-entropy loss over `num_batches` batches of a DataLoader."""
    if len(data_loader) == 0:
        return float("nan")
    num_batches = min(num_batches or len(data_loader), len(data_loader))
    total = 0.0
    for i, (inputs, targets) in enumerate(data_loader):
        if i >= num_batches:
            break
        total += calc_loss_batch(inputs, targets, model, device).item()
    return total / num_batches


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model_simple(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    progress_callback=None,
) -> tuple[list[float], list[float], list[int]]:
    """
    Simple training loop from the notebook.

    Parameters
    ----------
    progress_callback : callable(epoch, step, train_loss, val_loss) | None
        Optional hook called after each evaluation step.
    """
    train_losses:      list[float] = []
    val_losses:        list[float] = []
    tokens_seen:       list[int]   = []
    total_tokens_seen: int         = 0
    global_step:       int         = -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            total_tokens_seen += input_batch.numel()
            global_step       += 1

            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    tr_loss = calc_loss_loader(train_loader, model, device, eval_iter)
                    va_loss = calc_loss_loader(val_loader,   model, device, eval_iter)
                model.train()

                train_losses.append(tr_loss)
                val_losses.append(va_loss)
                tokens_seen.append(total_tokens_seen)

                if progress_callback:
                    progress_callback(epoch + 1, global_step, tr_loss, va_loss)

    return train_losses, val_losses, tokens_seen


# ---------------------------------------------------------------------------
# PDF Personality mini fine-tune
# ---------------------------------------------------------------------------

def fine_tune_on_text(
    model: GPTModel,
    tokenizer,
    text: str,
    num_epochs: int = 3,
    lr: float = 4e-4,
    batch_size: int = 2,
    max_length: int = 64,
    stride: int = 32,
    device: torch.device | str = "cpu",
    progress_callback=None,
) -> GPTModel:
    """
    Perform a quick fine-tuning session on `text` to give the model
    a 'personality' derived from the supplied PDF / TXT content.

    Uses the same GPTDatasetV1 sliding-window approach from the notebook.
    Returns the mutated model (fine-tuned in place).
    """
    from data import create_dataloader_v1   # flat import — no app.core dependency

    model.to(device)
    model.train()

    # Build dataloaders with 90/10 train/val split
    split      = int(0.9 * len(text))
    train_text = text[:split]
    val_text   = text[split:]

    if len(train_text) < max_length + 1:
        # Fall back if text is very short
        train_text = text
        val_text   = text

    train_loader = create_dataloader_v1(
        train_text, tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader_v1(
        val_text, tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
        drop_last=False,
    )

    if len(train_loader) == 0:
        return model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=max(1, len(train_loader)),
        eval_iter=1,
        start_context="",
        tokenizer=tokenizer,
        progress_callback=progress_callback,
    )

    model.eval()
    return model
