"""GNNExplainer for Human-Object Interaction Relational GNNs.

Implements graph-level post-hoc explanations for relational HOI detection models.
Given a trained HOI model and an input scene graph, identifies the most important
relational message-passing edges and contextual entity nodes that support a specific
human-object action prediction.

Note:
    Graph-level attribution (nodes and spatial edges), not pixel-level.
    Attribution highlights discrete entities (bounding boxes) and relational
    geometric connections in the scene graph, rather than dense image pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks


@dataclass
class HOIExplanation:
    """Encapsulates graph explanation masks and attribution rankings for an HOI prediction.

    Attributes:
        target_pair: (human_node_idx, object_node_idx) tuple.
        action_idx: Index of the explained action category.
        score: Model prediction probability before masking.
        edge_mask: Normalized edge importance scores [E] in [0, 1].
        node_mask: Normalized node importance scores [N] in [0, 1].
        important_edges: List of (src_node, dst_node, importance_weight) sorted descending.
        important_nodes: List of (node_idx, importance_weight) sorted descending.
        node_feat_mask: Optional node feature attribution mask [N, D].
        edge_index: Optional [2, E] edge tensor corresponding to edge_mask.
        action_name: Optional human-readable action name.
    """

    target_pair: Tuple[int, int]
    action_idx: int
    score: float
    edge_mask: torch.Tensor
    node_mask: torch.Tensor
    important_edges: List[Tuple[int, int, float]] = field(default_factory=list)
    important_nodes: List[Tuple[int, float]] = field(default_factory=list)
    node_feat_mask: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None
    action_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate shapes and value ranges."""
        if self.edge_mask is not None and not isinstance(self.edge_mask, torch.Tensor):
            self.edge_mask = torch.as_tensor(self.edge_mask, dtype=torch.float32)
        if self.node_mask is not None and not isinstance(self.node_mask, torch.Tensor):
            self.node_mask = torch.as_tensor(self.node_mask, dtype=torch.float32)


# Backward-compatible alias
GNNExplainerExplanation = HOIExplanation


class HOIModelExplainAdapter(nn.Module):
    """Adapter wrapping an HOI model to match PyG Explainer interface.

    PyG's Explainer expects forward(x, edge_index, **kwargs) -> Tensor.
    This adapter unpacks the graph context and maps to the scalar prediction logit
    of the target candidate pair and action.
    """

    def __init__(
        self,
        model: nn.Module,
        data_template: Optional[Any] = None,
        pair_idx: int = 0,
        action_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.data_template = data_template
        self.pair_idx = pair_idx
        self.action_idx = action_idx

    def set_target(self, pair_idx: int, action_idx: Optional[int] = None) -> None:
        """Update explanation target pair and action."""
        self.pair_idx = pair_idx
        self.action_idx = action_idx

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass compatible with PyG Explainer.

        Returns:
            torch.Tensor: [1, 1] raw action logit for the target pair and action.
        """
        ref = kwargs.get("data", self.data_template)
        # Create a proxy data object preserving relational attributes
        from data.graph_builder import HOIData

        data = HOIData(
            x=x,
            edge_index=edge_index,
            edge_attr=getattr(ref, "edge_attr", None) if ref is not None else None,
            pair_index=getattr(ref, "pair_index", None) if ref is not None else None,
            pair_attr=getattr(ref, "pair_attr", None) if ref is not None else None,
            class_ids=getattr(ref, "class_ids", None) if ref is not None else None,
            boxes_norm=getattr(ref, "boxes_norm", None) if ref is not None else None,
            boxes=getattr(ref, "boxes", None) if ref is not None else None,
            num_nodes=x.shape[0] if x is not None else (getattr(ref, "num_nodes", 0) if ref else 0),
        )

        # Support x_visual if model expects it
        if hasattr(ref, "x_visual") and ref.x_visual is not None:
            data.x_visual = x

        logits = self.model(data)  # [P, num_actions]

        if logits.numel() == 0 or logits.shape[0] <= self.pair_idx:
            return torch.zeros((1, 1), dtype=torch.float32, device=x.device)

        pair_logits = logits[self.pair_idx : self.pair_idx + 1]  # [1, num_actions]
        if self.action_idx is not None:
            return pair_logits[:, self.action_idx : self.action_idx + 1]  # [1, 1]
        return pair_logits


class HOIExplainer:
    """GNNExplainer tailored for Relational Human-Object Interaction Models.

    Optimizes edge masks and node feature masks via gradient descent to identify
    the minimal subgraph and node subset explaining a specific interaction prediction.

    Loss function follows Ying et al. (NeurIPS 2019):
        L = L_pred(y_hat, y) + lambda_edge_size * mean(edge_mask)
            + lambda_edge_ent * H(edge_mask)
            + lambda_node_size * mean(node_mask)
            + lambda_node_ent * H(node_mask)

    Note:
        Graph-level attribution, not pixel-level.
    """

    def __init__(
        self,
        model: nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        edge_size_coeff: float = 0.005,
        edge_ent_coeff: float = 1.0,
        node_size_coeff: float = 0.005,
        node_ent_coeff: float = 0.1,
        eps: float = 1e-7,
    ) -> None:
        """Initialize HOIExplainer.

        Args:
            model: Trained relational HOI model.
            epochs: Number of optimization iterations.
            lr: Learning rate for Adam optimizer.
            edge_size_coeff: Sparsity penalty coefficient for edge mask.
            edge_ent_coeff: Entropy penalty coefficient driving edge masks toward {0, 1}.
            node_size_coeff: Sparsity penalty coefficient for node mask.
            node_ent_coeff: Entropy penalty coefficient for node mask.
            eps: Numerical stability constant.
        """
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.edge_size_coeff = edge_size_coeff
        self.edge_ent_coeff = edge_ent_coeff
        self.node_size_coeff = node_size_coeff
        self.node_ent_coeff = node_ent_coeff
        self.eps = eps

    def explain(
        self,
        data: Any,
        pair_idx: Union[int, Tuple[int, int]] = 0,
        action_idx: Optional[int] = None,
        top_k_edges: int = 5,
        top_k_nodes: int = 3,
        epochs: Optional[int] = None,
    ) -> HOIExplanation:
        """Generate relational GNN explanation for a target candidate pair and action.

        Args:
            data: PyG HOIData or Data instance containing scene graph.
            pair_idx: Candidate pair index (int) or (human_idx, object_idx) tuple.
            action_idx: Action class index to explain. If None, uses argmax of model prediction.
            top_k_edges: Number of top important edges to extract.
            top_k_nodes: Number of top important nodes to extract.
            epochs: Optional override for optimization epochs.

        Returns:
            HOIExplanation: Object containing normalized edge and node masks and rankings.
        """
        self.model.eval()
        n_epochs = epochs if epochs is not None else self.epochs

        # 1. Resolve pair index and target nodes
        pair_index = getattr(data, "pair_index", None)
        if pair_index is None or pair_index.numel() == 0:
            raise ValueError("Input data does not contain candidate 'pair_index'.")

        if isinstance(pair_idx, tuple):
            h_idx, o_idx = pair_idx
            matches = ((pair_index[:, 0] == h_idx) & (pair_index[:, 1] == o_idx)).nonzero(
                as_tuple=False
            )
            if matches.numel() == 0:
                raise ValueError(f"Pair ({h_idx}, {o_idx}) not found in data.pair_index.")
            resolved_pair_idx = int(matches[0].item())
            target_pair = (h_idx, o_idx)
        else:
            resolved_pair_idx = int(pair_idx)
            if resolved_pair_idx < 0 or resolved_pair_idx >= pair_index.shape[0]:
                raise IndexError(
                    f"pair_idx {resolved_pair_idx} out of range [0, {pair_index.shape[0] - 1}]."
                )
            target_pair = (
                int(pair_index[resolved_pair_idx, 0].item()),
                int(pair_index[resolved_pair_idx, 1].item()),
            )

        # 2. Get unmasked reference prediction
        with torch.no_grad():
            orig_logits = self.model(data)  # [P, num_actions]
            orig_probs = torch.sigmoid(orig_logits)

        if action_idx is None:
            resolved_action_idx = int(torch.argmax(orig_probs[resolved_pair_idx]).item())
        else:
            resolved_action_idx = int(action_idx)

        orig_score = float(orig_probs[resolved_pair_idx, resolved_action_idx].item())
        orig_logit = float(orig_logits[resolved_pair_idx, resolved_action_idx].item())
        target_y = (
            torch.tensor([1.0], dtype=torch.float32, device=orig_logits.device)
            if orig_score >= 0.5
            else torch.tensor([0.0], dtype=torch.float32, device=orig_logits.device)
        )

        # 3. Setup edge and node masks
        edge_index = getattr(data, "edge_index", None)
        num_edges = edge_index.shape[1] if edge_index is not None else 0
        device = orig_logits.device

        # Get node representations tensor
        if hasattr(data, "x_visual") and data.x_visual is not None:
            x_tensor = data.x_visual
        elif hasattr(data, "x") and data.x is not None:
            x_tensor = data.x
        else:
            num_n = getattr(data, "num_nodes", 0)
            x_tensor = torch.zeros((num_n, 1), device=device)

        num_nodes = x_tensor.shape[0]

        if num_edges == 0:
            # Degenerate single-node or edgeless graph
            edge_mask = torch.zeros(0, dtype=torch.float32, device=device)
            node_mask = torch.zeros(num_nodes, dtype=torch.float32, device=device)
            return HOIExplanation(
                target_pair=target_pair,
                action_idx=resolved_action_idx,
                score=orig_score,
                edge_mask=edge_mask,
                node_mask=node_mask,
                important_edges=[],
                important_nodes=[],
                edge_index=edge_index,
            )

        # Initialize learnable masks (normal distribution around 0 => sigmoid around 0.5)
        edge_mask_param = nn.Parameter(torch.randn(num_edges, device=device) * 0.1)
        node_mask_param = nn.Parameter(torch.randn(num_nodes, device=device) * 0.1)

        optimizer = torch.optim.Adam([edge_mask_param, node_mask_param], lr=self.lr)

        # Clone data object for gradient tracking
        masked_data = data.clone()

        # 4. Optimization loop
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            m_edge = torch.sigmoid(edge_mask_param)
            m_node = torch.sigmoid(node_mask_param)

            # Apply edge mask via PyG message-passing hooks
            set_masks(self.model, m_edge, edge_index)

            # Apply node mask to node features
            if hasattr(masked_data, "x_visual") and masked_data.x_visual is not None:
                masked_data.x_visual = data.x_visual * m_node.unsqueeze(-1)
            if hasattr(masked_data, "x") and masked_data.x is not None:
                masked_data.x = data.x * m_node.unsqueeze(-1)

            # Forward pass
            out_logits = self.model(masked_data)
            pred_logit = out_logits[resolved_pair_idx, resolved_action_idx : resolved_action_idx + 1]

            # Binary cross entropy with original target direction
            loss_pred = F.binary_cross_entropy_with_logits(pred_logit, target_y)

            # Edge size (L1 sparsity) and entropy
            loss_edge_size = self.edge_size_coeff * m_edge.mean()
            ent_edge = -m_edge * torch.log(m_edge + self.eps) - (1.0 - m_edge) * torch.log(
                1.0 - m_edge + self.eps
            )
            loss_edge_ent = self.edge_ent_coeff * ent_edge.mean()

            # Node size and entropy
            loss_node_size = self.node_size_coeff * m_node.mean()
            ent_node = -m_node * torch.log(m_node + self.eps) - (1.0 - m_node) * torch.log(
                1.0 - m_node + self.eps
            )
            loss_node_ent = self.node_ent_coeff * ent_node.mean()

            loss = loss_pred + loss_edge_size + loss_edge_ent + loss_node_size + loss_node_ent

            loss.backward()
            optimizer.step()

        # Clear PyG message-passing hooks
        clear_masks(self.model)

        # 5. Extract and normalize final masks
        with torch.no_grad():
            final_edge_mask = torch.sigmoid(edge_mask_param).detach()
            final_node_mask = torch.sigmoid(node_mask_param).detach()

            # Min-max normalization into [0, 1]
            e_min, e_max = final_edge_mask.min(), final_edge_mask.max()
            if e_max > e_min:
                norm_edge_mask = (final_edge_mask - e_min) / (e_max - e_min)
            else:
                norm_edge_mask = torch.ones_like(final_edge_mask)

            n_min, n_max = final_node_mask.min(), final_node_mask.max()
            if n_max > n_min:
                norm_node_mask = (final_node_mask - n_min) / (n_max - n_min)
            else:
                norm_node_mask = torch.ones_like(final_node_mask)

        # 6. Rank top edges and nodes
        important_edges: List[Tuple[int, int, float]] = []
        if edge_index is not None and num_edges > 0:
            top_k_e = min(top_k_edges, num_edges)
            top_edge_indices = torch.topk(norm_edge_mask, k=top_k_e).indices.tolist()
            for idx in top_edge_indices:
                src = int(edge_index[0, idx].item())
                dst = int(edge_index[1, idx].item())
                weight = float(norm_edge_mask[idx].item())
                important_edges.append((src, dst, weight))

        important_nodes: List[Tuple[int, float]] = []
        top_k_n = min(top_k_nodes, num_nodes)
        top_node_indices = torch.topk(norm_node_mask, k=top_k_n).indices.tolist()
        for idx in top_node_indices:
            weight = float(norm_node_mask[idx].item())
            important_nodes.append((idx, weight))

        return HOIExplanation(
            target_pair=target_pair,
            action_idx=resolved_action_idx,
            score=orig_score,
            edge_mask=norm_edge_mask,
            node_mask=norm_node_mask,
            important_edges=important_edges,
            important_nodes=important_nodes,
            edge_index=edge_index,
        )

    def explain_pyg(
        self,
        data: Any,
        pair_idx: int = 0,
        action_idx: Optional[int] = None,
        epochs: int = 30,
    ) -> HOIExplanation:
        """Alternative explanation generator leveraging PyG's torch_geometric.explain.Explainer.

        Integrates with PyG's official Explainer framework using ModelConfig.
        """
        from torch_geometric.explain import Explainer, GNNExplainer as PyGGNNExplainer
        from torch_geometric.explain.config import ModelConfig

        adapter = HOIModelExplainAdapter(
            model=self.model,
            data_template=data,
            pair_idx=pair_idx,
            action_idx=action_idx,
        )

        model_config = ModelConfig(
            mode="binary_classification",
            task_level="graph",
            return_type="raw",
        )

        explainer = Explainer(
            model=adapter,
            algorithm=PyGGNNExplainer(epochs=epochs),
            explanation_type="model",
            node_mask_type="object",
            edge_mask_type="object",
            model_config=model_config,
        )

        if hasattr(data, "x_visual") and data.x_visual is not None:
            x_in = data.x_visual
        elif hasattr(data, "x") and data.x is not None:
            x_in = data.x
        else:
            x_in = torch.zeros((getattr(data, "num_nodes", 0), 1), device=next(self.model.parameters()).device)

        edge_index = getattr(data, "edge_index", torch.zeros((2, 0), dtype=torch.long))

        exp = explainer(x=x_in, edge_index=edge_index, data=data)

        # PyG Explainer returns edge_mask and node_mask
        edge_mask = exp.edge_mask if exp.edge_mask is not None else torch.zeros(edge_index.shape[1])
        node_mask = exp.node_mask
        if node_mask is not None and node_mask.ndim > 1:
            node_mask = node_mask.mean(dim=-1)
        if node_mask is None:
            node_mask = torch.zeros(x_in.shape[0])

        pair_index = getattr(data, "pair_index", None)
        target_pair = (
            int(pair_index[pair_idx, 0].item()),
            int(pair_index[pair_idx, 1].item()),
        ) if pair_index is not None and pair_index.shape[0] > pair_idx else (0, 1)

        with torch.no_grad():
            orig_logits = self.model(data)
            score = float(torch.sigmoid(orig_logits)[pair_idx, action_idx or 0].item())

        return HOIExplanation(
            target_pair=target_pair,
            action_idx=action_idx or 0,
            score=score,
            edge_mask=edge_mask,
            node_mask=node_mask,
            edge_index=edge_index,
        )


# Backward-compatible alias
HOIGNNExplainer = HOIExplainer
