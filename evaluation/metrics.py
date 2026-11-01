"""Multi-label evaluation metrics for HOI action prediction (Phase 5.5).

Pure numpy/sklearn: no torch import, so it can be tested on any machine.

Conventions (check against data/vcoco_loader.py and adjust if different):
  y_true  [N, A] in {0, 1}   multi-hot targets, one row per human-object pair
  y_score [N, A] float       probabilities (apply sigmoid BEFORE calling)
  y_mask  [N, A] in {0, 1}   1 = this (pair, action) label is valid and counts
                             0 = ignore (e.g. action has no object role)
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score


def _prep(y_true, y_score, y_mask):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    if y_mask is None:
        y_mask = np.ones_like(y_true)
    y_mask = np.asarray(y_mask).astype(bool)
    if y_mask.shape != y_true.shape:
        raise ValueError(f"mask shape {y_mask.shape} != {y_true.shape}")
    return y_true, y_score, y_mask


def per_action_ap(y_true, y_score, y_mask=None) -> np.ndarray:
    """Average precision per action, shape [A].

    Actions with no valid positive example get NaN (AP is undefined),
    so they are excluded from mAP instead of silently counted as 0.
    """
    y_true, y_score, y_mask = _prep(y_true, y_score, y_mask)
    n_actions = y_true.shape[1]
    ap = np.full(n_actions, np.nan)
    for a in range(n_actions):
        m = y_mask[:, a]
        t = y_true[m, a]
        if t.size == 0 or t.sum() == 0:
            continue
        ap[a] = average_precision_score(t, y_score[m, a])
    return ap


def mean_ap(ap: np.ndarray) -> float:
    """mAP over actions that have a defined AP. NaN if none do."""
    valid = ~np.isnan(ap)
    return float(ap[valid].mean()) if valid.any() else float("nan")


def threshold_metrics(y_true, y_score, y_mask=None, threshold: float = 0.5) -> Dict[str, float]:
    """Micro/macro precision, recall, F1 at a fixed threshold, over valid labels only."""
    y_true, y_score, y_mask = _prep(y_true, y_score, y_mask)
    pred = (y_score >= threshold) & y_mask
    true = (y_true.astype(bool)) & y_mask

    tp = (pred & true).sum(axis=0).astype(float)
    fp = (pred & ~true).sum(axis=0).astype(float)
    fn = (~pred & true & y_mask).sum(axis=0).astype(float)

    def _safe(num, den):
        return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)

    # micro: pool counts over all actions
    p_micro = float(_safe(np.array([tp.sum()]), np.array([tp.sum() + fp.sum()]))[0])
    r_micro = float(_safe(np.array([tp.sum()]), np.array([tp.sum() + fn.sum()]))[0])
    f_micro = 0.0 if p_micro + r_micro == 0 else 2 * p_micro * r_micro / (p_micro + r_micro)

    # macro: average per-action scores over actions that have positives
    p_a, r_a = _safe(tp, tp + fp), _safe(tp, tp + fn)
    f_a = _safe(2 * p_a * r_a, p_a + r_a)
    has_pos = (tp + fn) > 0
    macro = lambda v: float(v[has_pos].mean()) if has_pos.any() else float("nan")

    return {
        "precision_micro": p_micro, "recall_micro": r_micro, "f1_micro": f_micro,
        "precision_macro": macro(p_a), "recall_macro": macro(r_a), "f1_macro": macro(f_a),
    }


def rare_common_map(ap: np.ndarray, train_pos_counts: Sequence[int], rare_below: int = 10) -> Dict[str, float]:
    """mAP split by how often each action appears in the TRAIN set.

    Rare = fewer than `rare_below` training positives. The cutoff is a choice,
    not an official V-COCO definition: report it alongside the numbers.
    """
    counts = np.asarray(train_pos_counts)
    if counts.shape != ap.shape:
        raise ValueError("train_pos_counts must have one entry per action")
    rare = counts < rare_below
    return {
        "mAP_rare": mean_ap(ap[rare]),
        "mAP_common": mean_ap(ap[~rare]),
        "n_rare": int(rare.sum()),
        "n_common": int((~rare).sum()),
    }


def evaluate_all(y_true, y_score, y_mask=None, train_pos_counts=None,
                 rare_below: int = 10, threshold: float = 0.5) -> Dict[str, object]:
    """One call that returns everything the results table needs."""
    ap = per_action_ap(y_true, y_score, y_mask)
    out: Dict[str, object] = {"mAP": mean_ap(ap), "per_action_ap": ap.tolist()}
    out.update(threshold_metrics(y_true, y_score, y_mask, threshold))
    if train_pos_counts is not None:
        out.update(rare_common_map(ap, train_pos_counts, rare_below))
    return out


def measure_latency(fn: Callable[[], object], n_warmup: int = 3, n_runs: int = 20,
                    sync: Optional[Callable[[], None]] = None) -> Dict[str, float]:
    """Wall-clock latency of fn() in ms. Pass sync=torch.cuda.synchronize on GPU."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {"latency_ms_mean": float(arr.mean()), "latency_ms_std": float(arr.std()),
            "latency_ms_p95": float(np.percentile(arr, 95))}
