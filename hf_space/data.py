"""
data.py
-------
GPTDatasetV1 — sliding-window dataset + DataLoader factory.
Extracted from the LLM-from-scratch notebook.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """
    Sliding-window character dataset for GPT pre-training / fine-tuning.

    Parameters
    ----------
    text       : raw text corpus
    tokenizer  : any object with an .encode(str) -> list[int] method
    max_length : context window length (T)
    stride     : how many tokens to advance per window (< max_length for overlap)
    """

    def __init__(
        self,
        text: str,
        tokenizer,
        max_length: int,
        stride: int,
    ) -> None:
        self.input_ids:  list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        token_ids   = tokenizer.encode(text)
        token_ids_t = torch.tensor(token_ids, dtype=torch.long)

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(token_ids_t[i : i + max_length])
            self.target_ids.append(token_ids_t[i + 1 : i + max_length + 1])

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    text: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from a raw text string."""
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
