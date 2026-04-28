"""
tokenizer.py
------------
TiktokenWrapper — thin wrapper around tiktoken's GPT-2 BPE tokeniser.

Provides encode(), decode(), tokenize(), id_to_token(), and vocab_size
so that main.py and engine.py can use a consistent interface.
"""

from __future__ import annotations

import tiktoken


class TiktokenWrapper:
    """tiktoken BPE tokeniser for GPT-2 (50 257-token vocabulary)."""

    def __init__(self, encoding: str = "gpt2") -> None:
        self._enc = tiktoken.get_encoding(encoding)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        """Text → list of token IDs."""
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, token_ids: list[int]) -> str:
        """List of token IDs → text."""
        return self._enc.decode(token_ids)

    def tokenize(self, text: str) -> list[str]:
        """Text → list of individual token strings (for X-Ray display)."""
        ids = self.encode(text)
        return [self._enc.decode([i]) for i in ids]

    def id_to_token(self, token_id: int) -> str:
        """Single token ID → token string."""
        return self._enc.decode([token_id])
