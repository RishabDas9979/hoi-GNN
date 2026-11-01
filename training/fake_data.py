"""In-memory fake HOI graphs with a LEARNABLE signal, for smoke tests only.

Same field names as the real pipeline (x, edge_index, edge_attr, pair_index, y,
y_mask). Never report results from this data. Phase 3 (teammate) provides the
disk-backed synthetic dataset; this one needs no files, so Phase 5 can be
developed and tested independently.

Label rule: action a < n_classes is positive when the object has class a;
remaining actions are positive when human and object are close.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch_geometric.data import Data


class FakeHOIData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "pair_index":  # same batching behaviour the real HOIData must have
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def make_fake_graph(rng: np.random.Generator, num_actions: int = 6, node_dim: int = 10,
                    edge_dim: int = 9, n_classes: int = 4) -> FakeHOIData:
    assert node_dim >= 3 + n_classes and num_actions >= n_classes
    n_h, n_o = int(rng.integers(1, 3)), int(rng.integers(0, 4))  # n_o == 0 -> zero-pair graph
    n = n_h + n_o
    x = rng.normal(0, 0.05, size=(n, node_dim)).astype(np.float32)
    x[:, :2] = rng.random((n, 2))                 # box centre
    cls = np.concatenate([np.zeros(n_h, int), rng.integers(1, n_classes, size=n_o)])
    x[np.arange(n), 2 + cls] = 1.0                # one-hot class (0 = person)
    x[:n_h, 2 + n_classes] = 1.0                  # is_human flag

    pairs = [(h, n_h + o) for h in range(n_h) for o in range(n_o)]
    P = len(pairs)
    pair_index = np.array(pairs, dtype=np.int64).T if P else np.zeros((2, 0), dtype=np.int64)
    y = np.zeros((P, num_actions), dtype=np.float32)
    for k, (h, o) in enumerate(pairs):
        dist = float(np.linalg.norm(x[h, :2] - x[o, :2]))
        for a in range(num_actions):
            y[k, a] = float(cls[o] == a) if a < n_classes else float(dist < 0.15 + 0.15 * (a - n_classes))
    mask = np.ones_like(y)
    if P:
        mask[rng.random(P) < 0.2, -1] = 0.0       # last action has "no object" on some pairs

    edges = pair_index.tolist() if P else [[], []]
    ei = np.array([edges[0] + edges[1], edges[1] + edges[0]], dtype=np.int64).reshape(2, -1)
    E = ei.shape[1]
    ea = rng.normal(0, 0.1, size=(E, edge_dim)).astype(np.float32)
    if E:
        d = x[ei[1], :2] - x[ei[0], :2]
        ea[:, :2] = d
        ea[:, 2] = np.linalg.norm(d, axis=1)

    return FakeHOIData(
        x=torch.from_numpy(x), edge_index=torch.from_numpy(ei), edge_attr=torch.from_numpy(ea),
        pair_index=torch.from_numpy(pair_index), y=torch.from_numpy(y), y_mask=torch.from_numpy(mask),
    )


def make_fake_split(n_graphs: int, seed: int, **kw) -> List[FakeHOIData]:
    rng = np.random.default_rng(seed)
    return [make_fake_graph(rng, **kw) for _ in range(n_graphs)]
