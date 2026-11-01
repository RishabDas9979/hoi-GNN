"""Minimal stand-in model that follows the agreed interface: model(data) -> [P, A].

Used for tests and for developing Phase 5 before the Phase 4 models are merged.
It is NOT a result-bearing model: never put it in a results table.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self, node_dim: int, num_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * node_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_actions)
        )

    def forward(self, data) -> torch.Tensor:
        src, dst = data.pair_index
        return self.net(torch.cat([data.x[src], data.x[dst]], dim=-1))
