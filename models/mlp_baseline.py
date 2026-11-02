"""MLP Baseline for Human-Object Interaction detection.

Applies node-level projections via NodeEncoder without relational message passing
across edges, and classifies candidate pairs via InteractionHead.
"""
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from models.interaction_head import InteractionHead
from models.node_encoder import NodeEncoder


class MLPBaseline(nn.Module):
    """Non-relational baseline model for HOI classification.

    Follows the standard interface `model(data) -> logits [P, num_actions]`:
    1. Node features (visual, semantic, spatial) are encoded via NodeEncoder
       independently per node (no graph convolution or message passing).
    2. Candidate human-object pairs are classified by InteractionHead using the
       encoded node features and relative spatial pair features.

    Attributes:
        node_encoder (NodeEncoder): Encodes heterogeneous node attributes.
        interaction_head (InteractionHead): MLP pair classifier.
    """

    def __init__(
        self,
        input_dim: int = 896,
        hidden_dim: int = 256,
        num_actions: int = 26,
        class_embed_dim: int = 64,
        spatial_embed_dim: int = 64,
        dropout: float = 0.3,
        feature_subset: str = "all",
        use_pair_attr: bool = True,
        pair_attr_dim: int = 9,
        num_classes: int = 81,
        use_edge_attr: Optional[bool] = None,
        edge_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        heads: Optional[int] = None,
        act_type: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if use_edge_attr is not None:
            if isinstance(use_edge_attr, str):
                use_pair_attr = use_edge_attr.lower() in ("true", "1", "yes")
            else:
                use_pair_attr = bool(use_edge_attr)

        if edge_dim is not None:
            if edge_dim == 0:
                use_pair_attr = False
            pair_attr_dim = edge_dim

        self.node_encoder = NodeEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            class_embed_dim=class_embed_dim,
            spatial_embed_dim=spatial_embed_dim,
            dropout=dropout,
            feature_subset=feature_subset,
            num_classes=num_classes,
        )
        self.interaction_head = InteractionHead(
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            use_pair_attr=use_pair_attr,
            pair_attr_dim=pair_attr_dim,
            dropout=dropout,
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], **kwargs: Any) -> "MLPBaseline":
        """Instantiate MLPBaseline from a configuration dictionary.

        Args:
            cfg: Configuration dictionary with a 'model' section or flat keys.
            **kwargs: Explicit keyword arguments to override config values.

        Returns:
            MLPBaseline: Configured model instance.
        """
        model_cfg = cfg.get("model", cfg)
        raw_use_edge = model_cfg.get("use_edge_attr", True)
        if isinstance(raw_use_edge, str):
            use_edge = raw_use_edge.lower() in ("true", "1", "yes")
        else:
            use_edge = bool(raw_use_edge)

        params = {
            "input_dim": model_cfg.get("input_dim", 896),
            "hidden_dim": model_cfg.get("hidden_dim", 256),
            "num_actions": model_cfg.get("num_actions", 26),
            "class_embed_dim": model_cfg.get("class_embed_dim", 64),
            "spatial_embed_dim": model_cfg.get("spatial_embed_dim", 64),
            "dropout": model_cfg.get("dropout", 0.3),
            "feature_subset": model_cfg.get("feature_subset", "all"),
            "use_pair_attr": use_edge,
            "pair_attr_dim": model_cfg.get("edge_dim", 9),
        }
        params.update(kwargs)
        if "use_edge_attr" in kwargs and "use_pair_attr" not in kwargs:
            params["use_pair_attr"] = kwargs["use_edge_attr"]
        if "edge_dim" in kwargs and "pair_attr_dim" not in kwargs:
            params["pair_attr_dim"] = kwargs["edge_dim"]
        return cls(**params)

    def forward(self, data: Any) -> torch.Tensor:
        """Forward pass for interaction classification.

        Args:
            data: PyG Data or HOIData instance containing node features, candidate
                  pair_index [P, 2], and optional pair_attr [P, 9].

        Returns:
            torch.Tensor: [P, num_actions] unnormalized action logits.
        """
        pair_index = getattr(data, "pair_index", None)
        if pair_index is None or pair_index.numel() == 0 or pair_index.shape[0] == 0:
            dev = next(self.parameters()).device
            if hasattr(data, "x") and data.x is not None and data.x.numel() > 0:
                dev = data.x.device
            elif hasattr(data, "x_visual") and data.x_visual is not None and data.x_visual.numel() > 0:
                dev = data.x_visual.device
            return torch.zeros(
                (0, self.interaction_head.num_actions),
                dtype=torch.float32,
                device=dev,
            )

        # 1. Encode node representations (no message passing across edges)
        node_embeddings = self.node_encoder(data)

        # 2. Score candidate pairs
        pair_attr = getattr(data, "pair_attr", None)
        logits = self.interaction_head(node_embeddings, pair_index, pair_attr)
        return logits
