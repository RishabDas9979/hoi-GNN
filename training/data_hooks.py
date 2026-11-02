"""Builds train/val/test loaders for train.py.

`--data fake`   : in-memory fake graphs (smoke tests only).
`--data vcoco`  : the real pipeline using VCOCODataset.
`--data synthetic`: Phase 3 disk-backed synthetic dataset using SyntheticVCOCODataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from training.batch_adapter import get_targets


@dataclass
class LoaderBundle:
    train_loader: Any
    val_loader: Any
    test_loader: Any
    num_actions: int
    node_dim: int
    edge_dim: int
    train_pos_counts: List[int]
    train_valid_counts: List[int]


def _real_datasets(source: str, boxes: str, model_cfg: dict):
    """Return (train_ds, val_ds, test_ds) for real or synthetic data."""
    use_gt_boxes = (boxes == "gt")
    if source == "vcoco":
        from data.dataset import VCOCODataset

        cache_dir = "cache/features_gt" if use_gt_boxes else "cache/features"
        return tuple(
            VCOCODataset(split=s, use_gt_boxes=use_gt_boxes, cache_dir=cache_dir)
            for s in ("train", "val", "test")
        )
    elif source == "synthetic":
        from data.synthetic_dataset import SyntheticVCOCODataset

        feature_dim = model_cfg.get("model", {}).get("input_dim", 56)
        train_ds = SyntheticVCOCODataset(num_samples=200, seed=42, feature_dim=feature_dim)
        val_ds = SyntheticVCOCODataset(num_samples=50, seed=43, feature_dim=feature_dim)
        test_ds = SyntheticVCOCODataset(num_samples=50, seed=44, feature_dim=feature_dim)
        return train_ds, val_ds, test_ds
    else:
        raise ValueError(f"Unknown data source: {source}. Expected 'vcoco', 'synthetic', or 'fake'.")


def _label_stats(ds, num_actions: int):
    pos = np.zeros(num_actions)
    valid = np.zeros(num_actions)
    for g in ds:
        y, m = get_targets(g)
        pos += (y * m).sum(0).numpy()
        valid += m.sum(0).numpy()
    return pos.astype(int).tolist(), valid.astype(int).tolist()


def build_loaders(source: str, boxes: str, tcfg: dict, seed: int, model_cfg: dict) -> LoaderBundle:
    if source == "fake":
        from training.fake_data import make_fake_split

        train, val, test = (make_fake_split(n, s) for n, s in ((200, seed), (50, seed + 1), (50, seed + 2)))
        collate_fn = None
    else:
        from data.dataset import hoi_collate

        train, val, test = _real_datasets(source, boxes, model_cfg)
        collate_fn = hoi_collate

    first = train[0]
    num_actions = int(get_targets(first)[0].size(-1))
    node_dim = int(first.x.size(-1))
    edge_attr = getattr(first, "edge_attr", None)
    edge_dim = int(edge_attr.size(-1)) if edge_attr is not None else 0
    pos, valid = _label_stats(train, num_actions)

    g = torch.Generator().manual_seed(seed)  # makes shuffling reproducible
    bs, nw = tcfg["batch_size"], tcfg.get("num_workers", 0)
    return LoaderBundle(
        DataLoader(train, batch_size=bs, shuffle=True, generator=g, num_workers=nw, collate_fn=collate_fn),
        DataLoader(val, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_fn),
        DataLoader(test, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_fn),
        num_actions, node_dim, edge_dim, pos, valid,
    )
