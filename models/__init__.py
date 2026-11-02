"""HOI-GNN Models, Encoders, Heads, and Baselines."""
from typing import Any, Dict, Optional
import torch.nn as nn

from models.edge_aware_gatv2 import EdgeAwareGATv2HOIModel
from models.gat import GATHOIModel
from models.gcn import GCNHOIModel
from models.interaction_head import InteractionHead
from models.mlp_baseline import MLPBaseline
from models.node_encoder import NodeEncoder
from models.prior_baseline import ObjectClassPriorBaseline

MODEL_REGISTRY = {
    "prior": ObjectClassPriorBaseline,
    "mlp": MLPBaseline,
    "gcn": GCNHOIModel,
    "gat": GATHOIModel,
    "edge_aware_gatv2": EdgeAwareGATv2HOIModel,
}


def build_model(name: str, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> nn.Module:
    """Instantiate a model by its registry name.

    Names supported: 'prior', 'mlp', 'gcn', 'gat', 'edge_aware_gatv2'.
    Extracts runtime parameters (num_actions, node_dim, edge_dim) if present in cfg['runtime'].
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{name}'. Expected one of {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[key]
    cfg = dict(cfg or {})
    runtime = cfg.get("runtime", {})

    runtime_kwargs = {}
    if key == "prior":
        if "num_actions" in runtime:
            runtime_kwargs["num_actions"] = runtime["num_actions"]
        if "num_classes" in runtime:
            runtime_kwargs["num_classes"] = runtime["num_classes"]
    else:
        if "node_dim" in runtime:
            runtime_kwargs["input_dim"] = runtime["node_dim"]
        if "num_actions" in runtime:
            runtime_kwargs["num_actions"] = runtime["num_actions"]
        if "edge_dim" in runtime and key == "edge_aware_gatv2":
            runtime_kwargs["edge_dim"] = runtime["edge_dim"]

    runtime_kwargs.update(kwargs)

    if hasattr(cls, "from_config"):
        return cls.from_config(cfg, **runtime_kwargs)
    return cls(**runtime_kwargs)


__all__ = [
    "NodeEncoder",
    "InteractionHead",
    "ObjectClassPriorBaseline",
    "MLPBaseline",
    "GCNHOIModel",
    "GATHOIModel",
    "EdgeAwareGATv2HOIModel",
    "MODEL_REGISTRY",
    "build_model",
]
