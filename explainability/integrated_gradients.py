"""Integrated Gradients for Relational HOI Detection Models.

Computes path-integral gradients with respect to graph input features
(node visual representations, relational edge attributes, and pair spatial features)
for a specified candidate interaction pair and target action category using Captum.

Note: Graph-level attribution, not pixel-level. Attributions reflect the influence of
graph node representations and geometric edge/pair attributes on the predicted interaction,
not raw pixel intensities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients


@dataclass
class IGExplanation:
    """Container holding Integrated Gradients attribution outputs.

    Note: Graph-level attribution, not pixel-level. Attributions reflect the influence of
    graph node representations and geometric edge/pair attributes on the predicted interaction,
    not raw pixel intensities.

    Attributes:
        node_importance: Aggregated per-node attribution of shape [N].
        node_feat_importance: Detailed per-feature node attribution of shape [N, D].
        edge_importance: Aggregated per-edge attribution of shape [E], or None if edges not attributed.
        delta: Convergence delta returned by Captum approximating the path integral error.
        target_pair: Tuple of (human_node_idx, object_node_idx).
        action_idx: Target action index classified by the interaction head.
        edge_attr_importance: Detailed per-feature edge attribution of shape [E, edge_dim], or None.
        pair_attr_importance: Detailed per-feature pair attribution of shape [P, pair_attr_dim], or None.
    """

    node_importance: torch.Tensor
    node_feat_importance: torch.Tensor
    edge_importance: Optional[torch.Tensor]
    delta: float
    target_pair: Tuple[int, int]
    action_idx: int
    edge_attr_importance: Optional[torch.Tensor] = None
    pair_attr_importance: Optional[torch.Tensor] = None


class _HOIForwardWrapper(nn.Module):
    """Differentiable forward wrapper routing tensor inputs to a target pair-action scalar logit.

    Adapts PyG graph structures for Captum's IntegratedGradients by accepting tuples of
    continuous input tensors (node features, edge attributes, pair attributes) with an
    explicit batch dimension [B, ...] and evaluating the underlying HOI model.
    """

    def __init__(
        self,
        model: nn.Module,
        data: Any,
        pair_idx: int,
        action_idx: int,
        attribute_edges: bool = False,
        attribute_pair_attr: bool = False,
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.pair_idx = pair_idx
        self.action_idx = action_idx
        self.attribute_edges = attribute_edges
        self.attribute_pair_attr = attribute_pair_attr

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for Captum interpolation steps.

        Args:
            *inputs: Variable-length tuple containing:
                - x_input: [B, N, D] node visual features.
                - edge_attr_input: [B, E, edge_dim] (if attribute_edges=True).
                - pair_attr_input: [B, P, pair_attr_dim] (if attribute_pair_attr=True).

        Returns:
            torch.Tensor: [B] scalar logits for the specified (pair_idx, action_idx).
        """
        inp_idx = 0
        x_input = inputs[inp_idx]
        inp_idx += 1

        if self.attribute_edges:
            edge_attr_input = inputs[inp_idx]
            inp_idx += 1
        else:
            edge_attr_input = None

        if self.attribute_pair_attr:
            pair_attr_input = inputs[inp_idx]
            inp_idx += 1
        else:
            pair_attr_input = None

        batch_size = x_input.shape[0]
        logits_list: List[torch.Tensor] = []

        for b in range(batch_size):
            # Clone data shallowly to preserve metadata while substituting differentiated tensors
            step_data = self.data.clone()

            # Set node visual features
            x_b = x_input[b]
            if hasattr(step_data, "x_visual") and step_data.x_visual is not None:
                step_data.x_visual = x_b
            if hasattr(step_data, "x") and step_data.x is not None:
                step_data.x = x_b

            # Set edge attributes if attributed
            if self.attribute_edges and edge_attr_input is not None:
                step_data.edge_attr = edge_attr_input[b]

            # Set pair attributes if attributed
            if self.attribute_pair_attr and pair_attr_input is not None:
                step_data.pair_attr = pair_attr_input[b]

            # Evaluate underlying HOI model
            out = self.model(step_data)
            # Handle return_attention if model returns (logits, att)
            if isinstance(out, tuple):
                out = out[0]

            scalar_logit = out[self.pair_idx, self.action_idx].view(1)
            logits_list.append(scalar_logit)

        return torch.cat(logits_list, dim=0)


class HOIIntegratedGradients:
    """Integrated Gradients explainer for Human-Object Interaction (HOI) models.

    Computes path-integral attributions using Captum with respect to node visual features,
    relational edge attributes, and candidate pair spatial attributes for a specified
    interaction pair and target action category.

    Note: Graph-level attribution, not pixel-level. Attributions reflect the influence of
    graph node representations and geometric edge/pair attributes on the predicted interaction,
    not raw pixel intensities.

    Attributes:
        model (nn.Module): Pretrained or evaluation-ready HOI model.
        multiply_by_inputs (bool): Whether to multiply path gradients by (input - baseline).
    """

    def __init__(self, model: nn.Module, multiply_by_inputs: bool = True):
        self.model = model
        self.multiply_by_inputs = multiply_by_inputs

    def attribute(
        self,
        data: Any,
        pair_idx: Union[int, Tuple[int, int]],
        action_idx: int,
        attribute_edges: bool = True,
        attribute_pair_attr: bool = False,
        baseline_type: str = "zero",
        baselines: Optional[Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]] = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: int = 1,
        node_aggregation: str = "norm",
        edge_aggregation: str = "norm",
    ) -> IGExplanation:
        """Compute Integrated Gradients attributions for target (pair_idx, action_idx).

        Args:
            data: PyG Data object containing x_visual/x, edge_index, pair_index, etc.
            pair_idx: Index of candidate pair in data.pair_index, or (human_idx, object_idx) tuple.
            action_idx: Target action category index (0..num_actions - 1).
            attribute_edges: Whether to compute attributions with respect to edge attributes.
                             Automatically disabled if model or data lacks edge attributes.
            attribute_pair_attr: Whether to attribute pair spatial attributes.
            baseline_type: Baseline strategy if custom baselines not provided ('zero' or 'mean').
            baselines: Optional custom baselines matching attributed inputs.
            n_steps: Number of Riemann or Gauss-Legendre integration steps (default: 50).
            method: Integration method ('gausslegendre', 'riemann_middle', 'riemann_trapezoid').
            internal_batch_size: Internal batching for Captum steps (default: 1 for minimal memory).
            node_aggregation: Per-node feature aggregation metric ('norm', 'l2', 'abs_sum', 'sum').
            edge_aggregation: Per-edge feature aggregation metric ('norm', 'l2', 'abs_sum', 'sum').

        Returns:
            IGExplanation: Structured explanation holding node and edge importances,
                           feature-level importances, convergence delta, and target metadata.
        """
        # Ensure model is in eval mode
        was_training = self.model.training
        self.model.eval()

        try:
            # 1. Resolve pair index
            p_idx, target_pair = self._resolve_pair(data, pair_idx)

            # 2. Extract node visual features
            x = getattr(data, "x_visual", None)
            if x is None:
                x = getattr(data, "x", None)
            if x is None:
                raise ValueError("Graph data must contain 'x_visual' or 'x' node features.")

            num_nodes = x.shape[0]

            # 3. Determine if edge attributes can be attributed
            edge_attr = getattr(data, "edge_attr", None)
            model_uses_edge_attr = getattr(self.model, "use_edge_attr", True)
            can_attribute_edges = (
                attribute_edges
                and edge_attr is not None
                and edge_attr.numel() > 0
                and model_uses_edge_attr
            )

            # 4. Determine if pair attributes can be attributed
            pair_attr = getattr(data, "pair_attr", None)
            model_uses_pair_attr = getattr(self.model, "use_pair_attr", True)
            can_attribute_pair = (
                attribute_pair_attr
                and pair_attr is not None
                and pair_attr.numel() > 0
                and model_uses_pair_attr
            )

            # 5. Construct differentiable forward wrapper
            wrapper = _HOIForwardWrapper(
                model=self.model,
                data=data,
                pair_idx=p_idx,
                action_idx=action_idx,
                attribute_edges=can_attribute_edges,
                attribute_pair_attr=can_attribute_pair,
            )

            ig = IntegratedGradients(wrapper, multiply_by_inputs=self.multiply_by_inputs)

            # 6. Prepare inputs with leading batch dimension [1, ...]
            input_list: List[torch.Tensor] = [x.unsqueeze(0).float()]
            if can_attribute_edges:
                input_list.append(edge_attr.unsqueeze(0).float())
            if can_attribute_pair:
                input_list.append(pair_attr.unsqueeze(0).float())

            inputs = tuple(input_list)

            # 7. Prepare baselines
            baselines_tuple = self._prepare_baselines(
                inputs=inputs,
                baseline_type=baseline_type,
                custom_baselines=baselines,
                x=x,
                edge_attr=edge_attr if can_attribute_edges else None,
                pair_attr=pair_attr if can_attribute_pair else None,
            )

            # 8. Compute Captum Integrated Gradients
            attributions, delta_tensor = ig.attribute(
                inputs=inputs,
                baselines=baselines_tuple,
                target=None,
                return_convergence_delta=True,
                n_steps=n_steps,
                method=method,
                internal_batch_size=internal_batch_size,
            )

            delta = float(delta_tensor.mean().item())

            # 9. Extract and aggregate attributions
            out_idx = 0
            node_feat_attr = attributions[out_idx].squeeze(0)  # [N, D]
            out_idx += 1

            node_importance = self._aggregate_features(node_feat_attr, node_aggregation)

            edge_importance: Optional[torch.Tensor] = None
            edge_attr_importance: Optional[torch.Tensor] = None
            if can_attribute_edges:
                edge_attr_importance = attributions[out_idx].squeeze(0)  # [E, edge_dim]
                out_idx += 1
                edge_importance = self._aggregate_features(edge_attr_importance, edge_aggregation)

            pair_attr_importance: Optional[torch.Tensor] = None
            if can_attribute_pair:
                pair_attr_importance = attributions[out_idx].squeeze(0)  # [P, pair_attr_dim]
                out_idx += 1

            return IGExplanation(
                node_importance=node_importance,
                node_feat_importance=node_feat_attr,
                edge_importance=edge_importance,
                delta=delta,
                target_pair=target_pair,
                action_idx=action_idx,
                edge_attr_importance=edge_attr_importance,
                pair_attr_importance=pair_attr_importance,
            )

        finally:
            if was_training:
                self.model.train()

    def _resolve_pair(
        self,
        data: Any,
        pair_idx: Union[int, Tuple[int, int]],
    ) -> Tuple[int, Tuple[int, int]]:
        """Resolves pair identifier to candidate pair index and (human, object) node tuple."""
        pair_index = getattr(data, "pair_index", None)
        if pair_index is None or pair_index.numel() == 0:
            raise ValueError("Graph data contains no candidate pair_index.")

        if isinstance(pair_idx, tuple):
            h_target, o_target = pair_idx
            matched_indices = (
                (pair_index[:, 0] == h_target) & (pair_index[:, 1] == o_target)
            ).nonzero(as_tuple=False)
            if matched_indices.numel() == 0:
                raise ValueError(
                    f"Target pair ({h_target}, {o_target}) not found in data.pair_index."
                )
            resolved_p_idx = int(matched_indices[0].item())
            return resolved_p_idx, (int(h_target), int(o_target))

        resolved_p_idx = int(pair_idx)
        if resolved_p_idx < 0 or resolved_p_idx >= pair_index.shape[0]:
            raise IndexError(
                f"pair_idx {resolved_p_idx} out of range for pair_index of shape {pair_index.shape}."
            )
        target_pair = (
            int(pair_index[resolved_p_idx, 0].item()),
            int(pair_index[resolved_p_idx, 1].item()),
        )
        return resolved_p_idx, target_pair

    def _prepare_baselines(
        self,
        inputs: Tuple[torch.Tensor, ...],
        baseline_type: str,
        custom_baselines: Optional[Any],
        x: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        pair_attr: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        """Constructs baseline tensors matching input shapes."""
        if custom_baselines is not None:
            if isinstance(custom_baselines, tuple):
                return custom_baselines
            if isinstance(custom_baselines, dict):
                baselines_list = []
                x_base = custom_baselines.get("x", custom_baselines.get("x_visual"))
                if x_base is not None:
                    baselines_list.append(x_base.unsqueeze(0) if x_base.dim() == x.dim() else x_base)
                else:
                    baselines_list.append(torch.zeros_like(inputs[0]))

                if edge_attr is not None:
                    e_base = custom_baselines.get("edge_attr")
                    if e_base is not None:
                        baselines_list.append(e_base.unsqueeze(0) if e_base.dim() == edge_attr.dim() else e_base)
                    else:
                        baselines_list.append(torch.zeros_like(inputs[1]))

                if pair_attr is not None:
                    p_base = custom_baselines.get("pair_attr")
                    if p_base is not None:
                        baselines_list.append(p_base.unsqueeze(0) if p_base.dim() == pair_attr.dim() else p_base)
                    else:
                        baselines_list.append(torch.zeros_like(inputs[-1]))
                return tuple(baselines_list)

            if isinstance(custom_baselines, torch.Tensor):
                return (custom_baselines.unsqueeze(0) if custom_baselines.dim() == inputs[0].dim() - 1 else custom_baselines,)

        # Build standard baselines
        baselines_list: List[torch.Tensor] = []
        if baseline_type.lower() == "zero":
            for inp in inputs:
                baselines_list.append(torch.zeros_like(inp))
        elif baseline_type.lower() == "mean":
            # Compute mean along entity dimension and broadcast
            for inp in inputs:
                # inp is [1, num_items, feature_dim]
                mean_feat = inp.mean(dim=1, keepdim=True)
                baselines_list.append(mean_feat.expand_as(inp))
        else:
            raise ValueError(f"Unknown baseline_type: {baseline_type!r}. Use 'zero' or 'mean'.")

        return tuple(baselines_list)

    @staticmethod
    def _aggregate_features(feat_attr: torch.Tensor, aggregation: str) -> torch.Tensor:
        """Aggregates a feature attribution tensor of shape [N, D] to [N]."""
        agg = aggregation.lower()
        if agg in ("norm", "l2"):
            return torch.norm(feat_attr, p=2, dim=-1)
        elif agg in ("abs_sum", "abs"):
            return torch.abs(feat_attr).sum(dim=-1)
        elif agg == "sum":
            return feat_attr.sum(dim=-1)
        elif agg == "mean":
            return feat_attr.mean(dim=-1)
        else:
            raise ValueError(
                f"Unknown aggregation mode: {aggregation!r}. Use 'norm', 'abs_sum', 'sum', or 'mean'."
            )
