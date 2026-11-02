"""HOI-GNN Models, Encoders, Heads, and Baselines."""
import inspect
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
    Ensures config overrides (such as feature_subset, num_layers, edge_dim, use_edge_attr,
    hidden_dim, heads) in cfg.get("model", {}) and kwargs are passed into model instantiation
    without being dropped or causing TypeErrors.
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{name}'. Expected one of {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[key]
    cfg_dict = dict(cfg or {})
    model_section = dict(cfg_dict.get("model", {} if "model" in cfg_dict else cfg_dict))
    runtime = dict(cfg_dict.get("runtime", {}))

    # Base parameters initialized from model config section
    model_params = dict(model_section)

    # Runtime fallbacks (inferred from data loaders)
    if key == "prior":
        if "num_actions" not in model_params and "num_actions" in runtime:
            model_params["num_actions"] = runtime["num_actions"]
        if "num_classes" not in model_params and "num_classes" in runtime:
            model_params["num_classes"] = runtime["num_classes"]
    else:
        if "input_dim" not in model_params and "node_dim" in runtime:
            model_params["input_dim"] = runtime["node_dim"]
        if "num_actions" not in model_params and "num_actions" in runtime:
            model_params["num_actions"] = runtime["num_actions"]
        if key == "edge_aware_gatv2" and "edge_dim" not in model_params and "edge_dim" in runtime:
            model_params["edge_dim"] = runtime["edge_dim"]

    # Explicit kwargs passed to build_model take highest precedence
    model_params.update(kwargs)

    # Normalize use_edge_attr boolean if given as string
    if "use_edge_attr" in model_params:
        raw_val = model_params["use_edge_attr"]
        if isinstance(raw_val, str):
            model_params["use_edge_attr"] = raw_val.lower() in ("true", "1", "yes")
        else:
            model_params["use_edge_attr"] = bool(raw_val)

        # If use_edge_attr is False, also harmonize use_pair_attr
        if not model_params["use_edge_attr"]:
            model_params["use_pair_attr"] = False

    if "edge_dim" in model_params and (model_params["edge_dim"] is None or model_params["edge_dim"] == 0):
        if "use_edge_attr" not in kwargs and "use_edge_attr" not in model_section:
            model_params["use_edge_attr"] = False
            model_params["use_pair_attr"] = False

    # Update cfg_dict['model'] so that from_config also sees the fully resolved dictionary
    cfg_dict["model"] = model_params

    # Filter kwargs to parameters accepted by the model's constructor
    sig = inspect.signature(cls.__init__)
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    valid_params = set(sig.parameters.keys())

    if accepts_var_kwargs:
        filtered_kwargs = model_params
    else:
        filtered_kwargs = {k: v for k, v in model_params.items() if k in valid_params}

    if hasattr(cls, "from_config"):
        return cls.from_config(cfg_dict, **filtered_kwargs)
    return cls(**filtered_kwargs)


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
