"""The ONE place that knows the HOIData field names.

If data/graph_builder.py names things differently, change it here only.
Assumed fields: `y` [P, A] multi-hot targets, `y_mask` [P, A] (1 = valid),
`pair_index` [2, P]. Falls back to all-valid mask if `y_mask` is absent.
"""
from __future__ import annotations

import torch

TARGET_NAMES = ("y", "y_action", "target", "labels")


def get_targets(batch):
    y = next((getattr(batch, n) for n in TARGET_NAMES if getattr(batch, n, None) is not None), None)
    if y is None:
        raise AttributeError(f"no target field found; tried {TARGET_NAMES}. Edit training/batch_adapter.py")
    mask = getattr(batch, "y_mask", None)
    if mask is None:
        mask = torch.ones_like(y)
    return y, mask
