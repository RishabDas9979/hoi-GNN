"""Graph Attention Network (GAT) for Human-Object Interaction Detection.

Stacks multi-head GATConv layers with LayerNorm, ELU activation, Dropout, and
residual skip connections, and classifies candidate pairs via InteractionHead.
"""
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from models.interaction_head import InteractionHead
from models.node_encoder import NodeEncoder


class GATHOIModel(nn.Module):
    """Relational HOI model based on Graph Attention Networks (GAT).

    Follows the standard interface `model(data) -> logits [P, num_actions]`:
    1. Node features (visual, semantic, spatial) are encoded via NodeEncoder.
    2. Context is propagated across entity nodes via a stack of multi-head GATConv
       layers with LayerNorm, ELU/ReLU activation, Dropout, and residual skip connections.
    3. Candidate human-object pairs are classified by InteractionHead using updated
       node embeddings and pair-level spatial relation attributes.

    Attributes:
        node_encoder (NodeEncoder): Encodes heterogeneous node attributes.
        convs (nn.ModuleList): Stack of GATConv multi-head message-passing layers.
        norms (nn.ModuleList): LayerNorm per GAT layer.
        acts (nn.ModuleList): ELU activations per layer.
        dropouts (nn.ModuleList): Dropout regularizers per layer.
        interaction_head (InteractionHead): MLP pair classifier.
        num_layers (int): Number of GATConv layers.
        hidden_dim (int): Latent dimension of node representations.
        heads (int): Number of attention heads.
    """

    def __init__(
        self,
        input_dim: int = 896,
        hidden_dim: int = 256,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        num_actions: int = 26,
        class_embed_dim: int = 64,
        spatial_embed_dim: int = 64,
        feature_subset: str = "all",
        use_pair_attr: bool = True,
        pair_attr_dim: int = 9,
        num_classes: int = 81,
        act_type: str = "elu",
        use_edge_attr: Optional[bool] = None,
        edge_dim: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})."
            )

        if use_edge_attr is not None:
            if isinstance(use_edge_attr, str):
                use_pair_attr = use_edge_attr.lower() in ("true", "1", "yes")
            else:
                use_pair_attr = bool(use_edge_attr)

        if edge_dim is not None:
            if edge_dim == 0:
                use_pair_attr = False
            pair_attr_dim = edge_dim

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.out_per_head = hidden_dim // heads

        self.node_encoder = NodeEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            class_embed_dim=class_embed_dim,
            spatial_embed_dim=spatial_embed_dim,
            dropout=dropout,
            feature_subset=feature_subset,
            num_classes=num_classes,
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=self.out_per_head,
                    heads=heads,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            if act_type.lower() == "relu":
                self.acts.append(nn.ReLU())
            else:
                self.acts.append(nn.ELU())
            self.dropouts.append(nn.Dropout(dropout))

        self.interaction_head = InteractionHead(
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            use_pair_attr=use_pair_attr,
            pair_attr_dim=pair_attr_dim,
            dropout=dropout,
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], **kwargs: Any) -> "GATHOIModel":
        """Instantiate GATHOIModel from a configuration dictionary.

        Args:
            cfg: Configuration dictionary with a 'model' section or flat keys.
            **kwargs: Explicit keyword arguments to override config values.

        Returns:
            GATHOIModel: Configured model instance.
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
            "num_layers": model_cfg.get("num_layers", 2),
            "heads": model_cfg.get("heads", 4),
            "dropout": model_cfg.get("dropout", 0.3),
            "num_actions": model_cfg.get("num_actions", 26),
            "class_embed_dim": model_cfg.get("class_embed_dim", 64),
            "spatial_embed_dim": model_cfg.get("spatial_embed_dim", 64),
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
            data: PyG Data or HOIData instance containing node features, edge_index [2, E],
                  candidate pair_index [P, 2], and optional pair_attr [P, 9].

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

        # 1. Encode node representations
        h = self.node_encoder(data)

        # If zero nodes, return empty logits
        if h.shape[0] == 0:
            return torch.zeros(
                (0, self.interaction_head.num_actions),
                dtype=torch.float32,
                device=h.device,
            )

        # 2. GAT Message Passing layers
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=h.device)

        for conv, norm, act, drop in zip(self.convs, self.norms, self.acts, self.dropouts):
            h_res = h
            h_conv = conv(h, edge_index)
            h_conv = norm(h_conv)
            h_conv = act(h_conv)
            h_conv = drop(h_conv)
            h = h_conv + h_res

        # 3. Classify candidate pairs
        pair_attr = getattr(data, "pair_attr", None)
        logits = self.interaction_head(h, pair_index, pair_attr)
        return logits
