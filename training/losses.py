"""Masked multi-label losses (Phase 5.1).

All losses take (logits [P, A], targets [P, A], mask [P, A]) and return a scalar.
`mask` is 1 for valid labels; invalid labels (e.g. actions with no object role)
contribute nothing. The mean divides by the number of valid entries, clamped to
>= 1, so graphs with zero pairs give a zero loss instead of NaN.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


class MaskedBCE(nn.Module):
    def __init__(self, pos_weight: Optional[Sequence[float]] = None):
        super().__init__()
        pw = None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32)
        self.register_buffer("pos_weight", pw)

    def forward(self, logits, targets, mask):
        if logits.numel() == 0:
            return logits.sum() * 0.0  # keeps the graph so .backward() still works
        loss = F.binary_cross_entropy_with_logits(
            logits, targets.to(logits.dtype), pos_weight=self.pos_weight, reduction="none"
        )
        return _masked_mean(loss, mask)


class MaskedFocal(nn.Module):
    """Focal loss (Lin et al., 2017). alpha < 0 disables the class-balance term."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, targets, mask):
        if logits.numel() == 0:
            return logits.sum() * 0.0
        t = targets.to(logits.dtype)
        ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * t + (1 - p) * (1 - t)
        loss = ce * (1 - p_t) ** self.gamma
        if self.alpha >= 0:
            loss = loss * (self.alpha * t + (1 - self.alpha) * (1 - t))
        return _masked_mean(loss, mask)


def pos_weight_from_counts(pos_counts, valid_counts, max_weight: float = 20.0):
    """neg/pos per action, clipped to [1, max_weight]. Actions with no positives get 1."""
    import numpy as np

    pos = np.asarray(pos_counts, dtype=float)
    neg = np.asarray(valid_counts, dtype=float) - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return np.clip(w, 1.0, max_weight).tolist()


def build_loss(loss_cfg: dict, pos_counts=None, valid_counts=None) -> nn.Module:
    name = loss_cfg.get("name", "bce")
    if name == "bce":
        return MaskedBCE()
    if name == "weighted_bce":
        if pos_counts is None or valid_counts is None:
            raise ValueError("weighted_bce needs train pos_counts and valid_counts")
        return MaskedBCE(pos_weight_from_counts(pos_counts, valid_counts,
                                                loss_cfg.get("max_pos_weight", 20.0)))
    if name == "focal":
        return MaskedFocal(loss_cfg.get("gamma", 2.0), loss_cfg.get("alpha", 0.25))
    raise ValueError(f"unknown loss '{name}' (use bce | focal | weighted_bce)")
