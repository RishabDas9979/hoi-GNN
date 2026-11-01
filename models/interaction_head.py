"""Interaction Head for Human-Object Interaction classification.

Concatenates human and object node representations with optional pair-level
spatial relation features, and predicts unnormalized action logits across
candidate pairs.
"""
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class InteractionHead(nn.Module):
    """Pair classification head predicting multi-label action logits.

    Takes node representations and candidate (human, object) pairs, constructs
    pair feature vectors by concatenating human embedding, object embedding,
    and optional 9-dimensional relative spatial relation attributes, and passes
    them through a 2-layer MLP.

    Attributes:
        hidden_dim (int): Dimensionality of input node representations.
        num_actions (int): Number of target action categories (26 for V-COCO).
        use_pair_attr (bool): Whether to incorporate 9-dim relative spatial features.
        pair_attr_dim (int): Dimensionality of pair spatial features (default: 9).
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_actions: int = 26,
        use_pair_attr: bool = True,
        pair_attr_dim: int = 9,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.use_pair_attr = use_pair_attr
        self.pair_attr_dim = pair_attr_dim

        in_dim = 2 * hidden_dim + (pair_attr_dim if use_pair_attr else 0)

        # 2-layer MLP: Linear -> LayerNorm -> ReLU -> Dropout -> Linear
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], **kwargs: Any) -> "InteractionHead":
        """Instantiate InteractionHead from configuration dictionary.

        Args:
            cfg: Dictionary containing model configuration options.
            **kwargs: Overrides for configuration parameters.

        Returns:
            InteractionHead: Configured instance.
        """
        model_cfg = cfg.get("model", cfg)
        params = {
            "hidden_dim": model_cfg.get("hidden_dim", 256),
            "num_actions": model_cfg.get("num_actions", 26),
            "use_pair_attr": model_cfg.get("use_edge_attr", True),
            "pair_attr_dim": model_cfg.get("edge_dim", 9),
            "dropout": model_cfg.get("dropout", 0.3),
        }
        params.update(kwargs)
        return cls(**params)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        pair_index: torch.Tensor,
        pair_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Classify candidate human-object pairs.

        Args:
            node_embeddings: [V, H] node representation tensor.
            pair_index: [P, 2] indices of candidate (human_node, object_node) pairs.
            pair_attr: Optional [P, 9] relative spatial relation features.

        Returns:
            torch.Tensor: [P, num_actions] unnormalized logits.
        """
        # Guard against zero candidate pairs (P = 0)
        if pair_index is None or pair_index.numel() == 0 or pair_index.shape[0] == 0:
            return torch.zeros(
                (0, self.num_actions),
                dtype=torch.float32,
                device=node_embeddings.device,
            )

        # Standardize pair_index shape to [P, 2] if transposed
        if pair_index.dim() == 2 and pair_index.shape[0] == 2 and pair_index.shape[1] != 2:
            pair_index = pair_index.t()

        h_idx = pair_index[:, 0]
        o_idx = pair_index[:, 1]

        h_emb = node_embeddings[h_idx]
        o_emb = node_embeddings[o_idx]

        if self.use_pair_attr:
            if pair_attr is None:
                # Fallback to zeros if pair_attr was omitted
                pair_attr = torch.zeros(
                    (pair_index.shape[0], self.pair_attr_dim),
                    dtype=node_embeddings.dtype,
                    device=node_embeddings.device,
                )
            pair_feat = torch.cat([h_emb, o_emb, pair_attr.to(torch.float32)], dim=-1)
        else:
            pair_feat = torch.cat([h_emb, o_emb], dim=-1)

        logits = self.mlp(pair_feat)
        return logits
