"""Edge-Aware GATv2 for Human-Object Interaction Detection.

Key architecture incorporating 9-dimensional continuous spatial relation edge
attributes into multi-head GATv2 attention mechanisms, supporting attention weight
extraction for post-hoc explainability analysis.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from models.interaction_head import InteractionHead
from models.node_encoder import NodeEncoder


class EdgeAwareGATv2HOIModel(nn.Module):
    """Relational HOI model based on Edge-Aware GATv2 with spatial relation edge features.

    Key architecture of the project:
    1. Node features (visual appearance, semantic category, and spatial box) are
       projected into a unified latent space via NodeEncoder.
    2. Context is propagated through a stack of multi-head GATv2Conv layers conditioned
       on 9-dimensional geometric/spatial edge attributes (dx, dy, distance, log-area ratio,
       IoU, etc.).
    3. Each layer applies LayerNorm, ELU activation, Dropout, and residual skip connections.
    4. Supports `return_attention=True` to export layer-wise attention weights
       `(edge_index, alpha)` for explainability analysis (Phase 7).
    5. Candidate human-object pairs are classified by InteractionHead using updated
       node embeddings and pair-level spatial relation attributes.

    Attributes:
        node_encoder (NodeEncoder): Encodes heterogeneous node attributes.
        convs (nn.ModuleList): Stack of GATv2Conv layers accepting edge_dim.
        norms (nn.ModuleList): LayerNorm per GATv2 layer.
        acts (nn.ModuleList): ELU activations per layer.
        dropouts (nn.ModuleList): Dropout regularizers per layer.
        interaction_head (InteractionHead): MLP pair classifier.
        num_layers (int): Number of GATv2 layers.
        hidden_dim (int): Latent dimension of node representations.
        heads (int): Number of attention heads.
        edge_dim (int): Dimensionality of edge relation features (default: 9).
        use_edge_attr (bool): Whether edge attributes are incorporated in GATv2 and InteractionHead.
    """

    def __init__(
        self,
        input_dim: int = 896,
        hidden_dim: int = 256,
        num_layers: int = 2,
        heads: int = 4,
        edge_dim: Optional[int] = 9,
        dropout: float = 0.3,
        num_actions: int = 26,
        class_embed_dim: int = 64,
        spatial_embed_dim: int = 64,
        feature_subset: str = "all",
        use_edge_attr: bool = True,
        use_pair_attr: Optional[bool] = None,
        pair_attr_dim: Optional[int] = None,
        num_classes: int = 81,
        act_type: str = "elu",
        **kwargs: Any,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})."
            )

        # Parse boolean use_edge_attr
        if isinstance(use_edge_attr, str):
            use_edge_attr = use_edge_attr.lower() in ("true", "1", "yes")
        else:
            use_edge_attr = bool(use_edge_attr)

        if use_pair_attr is not None and not use_pair_attr:
            use_edge_attr = False

        # Condition: If use_edge_attr=False or edge_dim is None or edge_dim == 0:
        # - GATv2Conv layers created with edge_dim=None
        # - InteractionHead instantiated with use_pair_attr=use_edge_attr
        if not use_edge_attr or edge_dim is None or edge_dim == 0:
            self.use_edge_attr = False
            conv_edge_dim = None
            effective_edge_dim = 0
        else:
            self.use_edge_attr = True
            conv_edge_dim = int(edge_dim)
            effective_edge_dim = int(edge_dim)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.edge_dim = effective_edge_dim
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
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=self.out_per_head,
                    heads=heads,
                    edge_dim=conv_edge_dim,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            if act_type.lower() == "relu":
                self.acts.append(nn.ReLU())
            else:
                self.acts.append(nn.ELU())
            self.dropouts.append(nn.Dropout(dropout))

        # InteractionHead instantiated with use_pair_attr=use_edge_attr
        if not self.use_edge_attr:
            head_use_pair_attr = False
        else:
            head_use_pair_attr = use_pair_attr if use_pair_attr is not None else True

        p_dim = (
            pair_attr_dim
            if pair_attr_dim is not None
            else (effective_edge_dim if effective_edge_dim > 0 else 9)
        )

        self.interaction_head = InteractionHead(
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            use_pair_attr=head_use_pair_attr,
            pair_attr_dim=p_dim,
            dropout=dropout,
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], **kwargs: Any) -> "EdgeAwareGATv2HOIModel":
        """Instantiate EdgeAwareGATv2HOIModel from a configuration dictionary.

        Args:
            cfg: Configuration dictionary with a 'model' section or flat keys.
            **kwargs: Explicit keyword arguments to override config values.

        Returns:
            EdgeAwareGATv2HOIModel: Configured model instance.
        """
        model_cfg = cfg.get("model", cfg)
        raw_use_edge = model_cfg.get("use_edge_attr", True)
        if isinstance(raw_use_edge, str):
            use_edge_attr = raw_use_edge.lower() in ("true", "1", "yes")
        else:
            use_edge_attr = bool(raw_use_edge)

        raw_edge_dim = model_cfg.get("edge_dim", 9)
        edge_dim = int(raw_edge_dim) if raw_edge_dim is not None else None

        params = {
            "input_dim": model_cfg.get("input_dim", 896),
            "hidden_dim": model_cfg.get("hidden_dim", 256),
            "num_layers": model_cfg.get("num_layers", 2),
            "heads": model_cfg.get("heads", 4),
            "edge_dim": edge_dim,
            "dropout": model_cfg.get("dropout", 0.3),
            "num_actions": model_cfg.get("num_actions", 26),
            "class_embed_dim": model_cfg.get("class_embed_dim", 64),
            "spatial_embed_dim": model_cfg.get("spatial_embed_dim", 64),
            "feature_subset": model_cfg.get("feature_subset", "all"),
            "use_edge_attr": use_edge_attr,
            "use_pair_attr": use_edge_attr,
            "pair_attr_dim": edge_dim if edge_dim is not None else 9,
        }
        params.update(kwargs)
        if "use_edge_attr" in kwargs and "use_pair_attr" not in kwargs:
            params["use_pair_attr"] = kwargs["use_edge_attr"]
        return cls(**params)

    def forward(
        self,
        data: Any,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass for interaction classification with optional attention extraction.

        Args:
            data: PyG Data or HOIData instance containing node features, edge_index [2, E],
                  edge_attr [E, 9], candidate pair_index [P, 2], and optional pair_attr [P, 9].
            return_attention: If True, returns `(logits, attention_list)` where attention_list
                              contains `(edge_index, alpha)` tuples for each GATv2Conv layer.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List]]:
                - If return_attention=False: logits tensor of shape [P, num_actions].
                - If return_attention=True: tuple of (logits, attention_list).
        """
        pair_index = getattr(data, "pair_index", None)
        edge_index = getattr(data, "edge_index", None)
        edge_attr = getattr(data, "edge_attr", None)
        pair_attr = getattr(data, "pair_attr", None)

        # Determine reference device
        dev = next(self.parameters()).device
        if hasattr(data, "x") and data.x is not None and data.x.numel() > 0:
            dev = data.x.device
        elif hasattr(data, "x_visual") and data.x_visual is not None and data.x_visual.numel() > 0:
            dev = data.x_visual.device

        # Determine node count
        num_nodes = getattr(data, "num_nodes", None)
        if num_nodes is None:
            if hasattr(data, "x") and data.x is not None:
                num_nodes = data.x.shape[0]
            elif hasattr(data, "x_visual") and data.x_visual is not None:
                num_nodes = data.x_visual.shape[0]
            else:
                num_nodes = 0

        # Handle zero nodes (V = 0)
        if num_nodes == 0:
            empty_logits = torch.zeros(
                (0, self.interaction_head.num_actions),
                dtype=torch.float32,
                device=dev,
            )
            if return_attention:
                empty_att = [
                    (
                        torch.zeros((2, 0), dtype=torch.long, device=dev),
                        torch.zeros((0, self.heads), dtype=torch.float32, device=dev),
                    )
                    for _ in range(self.num_layers)
                ]
                return empty_logits, empty_att
            return empty_logits

        # 1. Project node features via NodeEncoder
        h = self.node_encoder(data)

        # Ensure edge_index is a valid tensor
        if edge_index is None:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=h.device)

        if self.use_edge_attr:
            if edge_attr is None:
                dim = self.edge_dim if self.edge_dim > 0 else 9
                edge_attr = torch.zeros(
                    (edge_index.shape[1], dim), dtype=torch.float32, device=h.device
                )
            else:
                edge_attr = edge_attr.to(torch.float32)

        # 2. GATv2 Message Passing with optional relational edge attributes
        attention_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for conv, norm, act, drop in zip(self.convs, self.norms, self.acts, self.dropouts):
            h_res = h
            if return_attention:
                if self.use_edge_attr:
                    h_conv, att = conv(
                        h, edge_index, edge_attr=edge_attr, return_attention_weights=True
                    )
                else:
                    h_conv, att = conv(
                        h, edge_index, return_attention_weights=True
                    )
                attention_list.append(att)
            else:
                if self.use_edge_attr:
                    h_conv = conv(h, edge_index, edge_attr=edge_attr)
                else:
                    h_conv = conv(h, edge_index)

            h_conv = norm(h_conv)
            h_conv = act(h_conv)
            h_conv = drop(h_conv)
            h = h_conv + h_res

        # 3. Classify candidate pairs
        if pair_index is None or pair_index.numel() == 0 or pair_index.shape[0] == 0:
            logits = torch.zeros(
                (0, self.interaction_head.num_actions),
                dtype=torch.float32,
                device=h.device,
            )
        else:
            logits = self.interaction_head(h, pair_index, pair_attr)

        if return_attention:
            return logits, attention_list
        return logits
