"""Builds train/val/test loaders for train.py.

`--data fake`   : in-memory fake graphs (smoke tests only).
`--data vcoco`  : the real pipeline. ADAPT `_real_datasets` below to the actual
                  constructor in data/dataset.py (I could not read that file).
`--data synthetic`: Phase 3 disk-backed synthetic set; same adapter point.
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
    """ADAPT HERE. Must return (train_ds, val_ds, test_ds), each indexable and
    yielding HOIData objects with x, edge_index, edge_attr, pair_index, y, y_mask."""
    try:
        from data.dataset import HOIDataset  # class name is a guess
    except ImportError as e:
        raise RuntimeError("Edit training/data_hooks.py::_real_datasets to match data/dataset.py") from e
    return tuple(HOIDataset(split=s, boxes=boxes) for s in ("train", "val", "test"))


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
    else:
        train, val, test = _real_datasets(source, boxes, model_cfg)

    first = train[0]
    num_actions = int(get_targets(first)[0].size(-1))
    node_dim = int(first.x.size(-1))
    edge_attr = getattr(first, "edge_attr", None)
    edge_dim = int(edge_attr.size(-1)) if edge_attr is not None else 0
    pos, valid = _label_stats(train, num_actions)

    g = torch.Generator().manual_seed(seed)  # makes shuffling reproducible
    bs, nw = tcfg["batch_size"], tcfg.get("num_workers", 0)
    return LoaderBundle(
        DataLoader(train, batch_size=bs, shuffle=True, generator=g, num_workers=nw),
        DataLoader(val, batch_size=bs, shuffle=False, num_workers=nw),
        DataLoader(test, batch_size=bs, shuffle=False, num_workers=nw),
        num_actions, node_dim, edge_dim, pos, valid,
    )
