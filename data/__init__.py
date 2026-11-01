"""Data loading, graph construction, and dataset utilities for HOI-GNN."""

from data.vcoco_loader import (
    VCOCOLoader,
    VCOCOImageRecord,
    VCOCOEntity,
    VCOCO_ACTIONS,
    NUM_ACTIONS,
    ACTION_TO_IDX,
    IDX_TO_ACTION,
    AGENT_ONLY_ACTIONS,
    compute_box_iou,
)
from data.graph_builder import (
    GraphBuilder,
    HOIData,
    compute_spatial_features,
)
from data.dataset import (
    VCOCODataset,
    hoi_collate,
)

__all__ = [
    "VCOCOLoader",
    "VCOCOImageRecord",
    "VCOCOEntity",
    "VCOCO_ACTIONS",
    "NUM_ACTIONS",
    "ACTION_TO_IDX",
    "IDX_TO_ACTION",
    "AGENT_ONLY_ACTIONS",
    "compute_box_iou",
    "GraphBuilder",
    "HOIData",
    "compute_spatial_features",
    "VCOCODataset",
    "hoi_collate",
]
