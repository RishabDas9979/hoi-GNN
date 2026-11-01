"""Unified Node Encoder for Human-Object Interaction graphs.

Projects heterogeneous node features (visual appearance, semantic category,
and spatial bounding boxes) into a unified latent space, supporting feature
ablation subsets ('all', 'appearance', 'spatial', 'semantic').
"""
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn


class NodeEncoder(nn.Module):
    """Encodes entity node features into a unified latent space.

    Handles four feature ablation configurations:
    - 'all': visual appearance + semantic class embedding + normalized box coordinates.
    - 'appearance': visual ROI features only.
    - 'spatial': normalized bounding boxes only.
    - 'semantic': learned COCO class embeddings only.

    Attributes:
        input_dim (int): Dimensionality of raw visual features (e.g. 896 or 56).
        hidden_dim (int): Output embedding dimension for each node.
        class_embed_dim (int): Dimension of learned class embeddings.
        spatial_embed_dim (int): Dimension of projected spatial box coordinates.
        dropout (float): Dropout probability.
        feature_subset (str): Active feature modality ('all', 'appearance', 'spatial', 'semantic').
        num_classes (int): Total number of category classes (80 COCO classes + 1 unknown at idx 80).
    """

    def __init__(
        self,
        input_dim: int = 896,
        hidden_dim: int = 256,
        class_embed_dim: int = 64,
        spatial_embed_dim: int = 64,
        dropout: float = 0.3,
        feature_subset: str = "all",
        num_classes: int = 81,
    ):
        super().__init__()
        valid_subsets = {"all", "appearance", "spatial", "semantic"}
        if feature_subset not in valid_subsets:
            raise ValueError(
                f"Invalid feature_subset: {feature_subset!r}. Must be one of {sorted(valid_subsets)}"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_embed_dim = class_embed_dim
        self.spatial_embed_dim = spatial_embed_dim
        self.dropout_rate = dropout
        self.feature_subset = feature_subset
        self.num_classes = num_classes

        # Modality projection modules
        self.visual_proj = nn.Linear(input_dim, hidden_dim)
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.spatial_proj = nn.Linear(4, spatial_embed_dim)

        # Dimension calculation for fusion projection
        if self.feature_subset == "all":
            fused_in_dim = hidden_dim + class_embed_dim + spatial_embed_dim
        elif self.feature_subset == "appearance":
            fused_in_dim = hidden_dim
        elif self.feature_subset == "spatial":
            fused_in_dim = spatial_embed_dim
        elif self.feature_subset == "semantic":
            fused_in_dim = class_embed_dim

        self.fusion = nn.Linear(fused_in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], **kwargs: Any) -> "NodeEncoder":
        """Instantiate NodeEncoder from a configuration dictionary.

        Args:
            cfg: Dictionary containing model configuration options.
            **kwargs: Overrides for configuration parameters.

        Returns:
            NodeEncoder: Configured instance.
        """
        model_cfg = cfg.get("model", cfg)
        params = {
            "input_dim": model_cfg.get("input_dim", 896),
            "hidden_dim": model_cfg.get("hidden_dim", 256),
            "class_embed_dim": model_cfg.get("class_embed_dim", 64),
            "spatial_embed_dim": model_cfg.get("spatial_embed_dim", 64),
            "dropout": model_cfg.get("dropout", 0.3),
            "feature_subset": model_cfg.get("feature_subset", "all"),
        }
        params.update(kwargs)
        return cls(**params)

    def forward(
        self,
        data: Optional[Any] = None,
        x_visual: Optional[torch.Tensor] = None,
        class_ids: Optional[torch.Tensor] = None,
        boxes_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass to compute node embeddings.

        Args:
            data: Optional PyG Data, HOIData, or Batch containing node attributes.
            x_visual: Optional direct [V, input_dim] visual features.
            class_ids: Optional direct [V] COCO class IDs (0..79, -1 for unknown).
            boxes_norm: Optional direct [V, 4] normalized xyxy boxes in [0, 1].

        Returns:
            torch.Tensor: [V, hidden_dim] node representations.
        """
        # If PyG Data or HOIData is provided, extract missing fields
        if data is not None:
            if x_visual is None:
                x_visual = getattr(data, "x_visual", None)
                if x_visual is None:
                    x_visual = getattr(data, "x", None)
            if class_ids is None:
                class_ids = getattr(data, "class_ids", None)
            if boxes_norm is None:
                boxes_norm = getattr(data, "boxes_norm", None)

        # Determine number of nodes V
        v = 0
        if data is not None and hasattr(data, "num_nodes") and data.num_nodes is not None:
            v = int(data.num_nodes)
        elif x_visual is not None:
            v = int(x_visual.shape[0])
        elif class_ids is not None:
            v = int(class_ids.shape[0])
        elif boxes_norm is not None:
            v = int(boxes_norm.shape[0])

        # Safe handling of empty node graphs (V = 0)
        if v == 0:
            dev = self.fusion.weight.device
            if x_visual is not None and x_visual.numel() > 0:
                dev = x_visual.device
            elif class_ids is not None and class_ids.numel() > 0:
                dev = class_ids.device
            elif boxes_norm is not None and boxes_norm.numel() > 0:
                dev = boxes_norm.device
            return torch.zeros((0, self.hidden_dim), dtype=torch.float32, device=dev)

        features_to_fuse = []

        # 1. Visual appearance projection
        if self.feature_subset in ("all", "appearance"):
            if x_visual is None:
                raise ValueError("x_visual (or x) is required for feature_subset in ['all', 'appearance']")
            feat_vis = self.visual_proj(x_visual.to(torch.float32))
            features_to_fuse.append(feat_vis)

        # 2. Semantic category embedding
        if self.feature_subset in ("all", "semantic"):
            if class_ids is None:
                raise ValueError("class_ids is required for feature_subset in ['all', 'semantic']")
            # Map negative (-1) or out-of-range COCO ids to index 80 (unknown/background)
            clamped_ids = torch.where(
                (class_ids < 0) | (class_ids >= 80),
                torch.full_like(class_ids, 80),
                class_ids,
            )
            feat_sem = self.class_embed(clamped_ids)
            features_to_fuse.append(feat_sem)

        # 3. Spatial bounding box projection
        if self.feature_subset in ("all", "spatial"):
            if boxes_norm is None:
                raise ValueError("boxes_norm is required for feature_subset in ['all', 'spatial']")
            feat_spat = self.spatial_proj(boxes_norm.to(torch.float32))
            features_to_fuse.append(feat_spat)

        # Concatenate selected modalities
        if len(features_to_fuse) == 1:
            combined = features_to_fuse[0]
        else:
            combined = torch.cat(features_to_fuse, dim=-1)

        # Project, normalize, and regularize
        out = self.fusion(combined)
        out = self.norm(out)
        out = self.dropout(out)
        return out
