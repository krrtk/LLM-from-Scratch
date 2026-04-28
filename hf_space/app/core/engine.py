"""
app/core/engine.py — shim that re-exports from the flat engine.py in final/
"""
import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from engine import (  # noqa: F401, E402
    generate,
    fine_tune_on_text,
    train_model_simple,
    calc_loss_batch,
    calc_loss_loader,
    text_to_token_ids,
    token_ids_to_text,
)
