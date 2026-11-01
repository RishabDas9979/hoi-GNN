"""Reproducibility helper (Phase 5.4)."""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # allows numpy-only tooling to import this
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
