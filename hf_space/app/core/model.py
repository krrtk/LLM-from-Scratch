"""
app/core/model.py — shim that re-exports from the flat model.py in final/
"""
import sys
import os

# Ensure the final/ root is on sys.path so the flat files are importable
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Import everything from the flat model.py
from model import (  # noqa: F401, E402
    GPT_CONFIG_124M,
    GPTModel,
    LayerNorm,
    GELU,
    FeedForward,
    MultiHeadAttention,
    TransformerBlock,
)
