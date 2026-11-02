"""Attention Analysis for Relational HOI Graph Neural Networks.

Extracts, aggregates, and visualizes multi-head attention weights from
EdgeAwareGATv2 and GAT relational models to inspect message-passing routing.

Critical Epistemological Caveat:
    Attention weights reflect message passing routing but are NOT proof of causal
    importance (Jain & Wallace 2019, Wiegreffe & Pinter 2019). While high attention
    coefficients show which entity nodes and relational spatial edges exchange the most
    information during message passing, counterfactual necessity and sufficiency
    require perturbation methods such as Integrated Gradients (Sundararajan et al. 2017)
    and Fidelity metrics (Pope et al. 2019).

Note:
    Graph-level attribution (nodes and spatial edges), not pixel-level.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/test environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

CAUSALITY_WARNING: str = (
    "Attention weights reflect message passing routing but are NOT proof of causal "
    "importance (Jain & Wallace 2019, Wiegreffe & Pinter 2019)."
)


def extract_attention_weights(
    model: nn.Module,
    data: Any,
) -> List[Dict[str, Any]]:
    """Extract layer-wise multi-head attention weights from a relational GNN model.

    Runs forward evaluation with `return_attention=True` on models supporting attention
    export (EdgeAwareGATv2HOIModel and GATHOIModel).

    Args:
        model: Trained relational HOI model supporting `return_attention=True`.
        data: PyG HOIData or Data object representing the scene graph.

    Returns:
        List[Dict[str, Any]]: Layer-wise attention dictionaries, each containing:
            - 'layer_idx': int index of the GNN layer.
            - 'edge_index': torch.Tensor [2, E] connectivity tensor.
            - 'alpha_heads': torch.Tensor [E, heads] raw multi-head attention coefficients.
            - 'alpha_mean': torch.Tensor [E] mean attention across heads.
            - 'alpha_max': torch.Tensor [E] maximum attention across heads.
            - 'num_heads': int number of attention heads.

    Raises:
        ValueError: If model does not support attention weight extraction.
    """
    model.eval()
    with torch.no_grad():
        try:
            output = model(data, return_attention=True)
        except TypeError as exc:
            raise ValueError(
                f"Model {type(model).__name__} forward() does not support return_attention=True."
            ) from exc

    if isinstance(output, tuple) and len(output) >= 2:
        _, raw_attention_list = output[0], output[1]
    else:
        raise ValueError(
            f"Expected tuple (logits, attention_list) from model, got {type(output)}."
        )

    results: List[Dict[str, Any]] = []
    for layer_idx, att_tuple in enumerate(raw_attention_list):
        edge_index, alpha = att_tuple
        edge_index_t = edge_index.detach().cpu()
        alpha_t = alpha.detach().cpu()

        # Handle 1D alpha vs [E, heads]
        if alpha_t.ndim == 1:
            alpha_heads = alpha_t.unsqueeze(-1)
        else:
            alpha_heads = alpha_t

        alpha_mean = alpha_heads.mean(dim=-1)
        alpha_max = alpha_heads.max(dim=-1).values

        results.append(
            {
                "layer_idx": layer_idx,
                "edge_index": edge_index_t,
                "alpha_heads": alpha_heads,
                "alpha_mean": alpha_mean,
                "alpha_max": alpha_max,
                "num_heads": alpha_heads.shape[-1],
            }
        )

    return results


def analyze_pair_attention(
    model: nn.Module,
    data: Any,
    pair_idx: Union[int, Tuple[int, int]] = 0,
) -> Dict[str, Any]:
    """Analyze attention weights specifically for a target human-object pair and its context.

    Inspects:
    1. Direct interaction edges (human -> object and object -> human).
    2. Contextual attention incoming to the human node from other entities.
    3. Contextual attention incoming to the object node from other entities.

    Note:
        Attention weights reflect message passing routing but are NOT proof of causal
        importance (Jain & Wallace 2019, Wiegreffe & Pinter 2019).

    Args:
        model: GNN model with attention extraction support.
        data: PyG HOIData or Data scene graph.
        pair_idx: Index into data.pair_index or (human_idx, object_idx) tuple.

    Returns:
        Dict[str, Any]: Detailed attention routing summary containing:
            - 'target_pair': (human_node, object_node)
            - 'causality_caveat': Warning string on attention vs causality.
            - 'layer_analyses': Per-layer direct and contextual attention scores.
            - 'mean_direct_attention': Aggregate mean attention across layers for direct pair edge.
            - 'top_context_to_human': Top entities sending attention to human node.
            - 'top_context_to_object': Top entities sending attention to object node.
    """
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
        target_pair = (h_idx, o_idx)
    else:
        p_idx = int(pair_idx)
        if p_idx < 0 or p_idx >= pair_index.shape[0]:
            raise IndexError(f"pair_idx {p_idx} out of range [0, {pair_index.shape[0] - 1}].")
        target_pair = (
            int(pair_index[p_idx, 0].item()),
            int(pair_index[p_idx, 1].item()),
        )

    h_node, o_node = target_pair
    att_layers = extract_attention_weights(model, data)

    layer_analyses: List[Dict[str, Any]] = []
    direct_scores: List[float] = []

    for layer_data in att_layers:
        l_idx = layer_data["layer_idx"]
        edge_index = layer_data["edge_index"]  # [2, E]
        alpha_mean = layer_data["alpha_mean"]  # [E]

        src = edge_index[0]
        dst = edge_index[1]

        # 1. Direct edges
        h_to_o_idx = ((src == h_node) & (dst == o_node)).nonzero(as_tuple=False)
        o_to_h_idx = ((src == o_node) & (dst == h_node)).nonzero(as_tuple=False)

        h_to_o_att = float(alpha_mean[h_to_o_idx[0, 0]].item()) if h_to_o_idx.numel() > 0 else 0.0
        o_to_h_att = float(alpha_mean[o_to_h_idx[0, 0]].item()) if o_to_h_idx.numel() > 0 else 0.0

        direct_avg = 0.5 * (h_to_o_att + o_to_h_att)
        direct_scores.append(direct_avg)

        # 2. Context incoming to human (k -> h where k != o and k != h)
        incoming_h = ((dst == h_node) & (src != h_node) & (src != o_node)).nonzero(as_tuple=False)
        context_to_h: List[Tuple[int, float]] = []
        if incoming_h.numel() > 0:
            for idx in incoming_h.squeeze(-1):
                s_node = int(src[idx].item())
                w = float(alpha_mean[idx].item())
                context_to_h.append((s_node, w))
            context_to_h.sort(key=lambda item: item[1], reverse=True)

        # 3. Context incoming to object (k -> o where k != h and k != o)
        incoming_o = ((dst == o_node) & (src != o_node) & (src != h_node)).nonzero(as_tuple=False)
        context_to_o: List[Tuple[int, float]] = []
        if incoming_o.numel() > 0:
            for idx in incoming_o.squeeze(-1):
                s_node = int(src[idx].item())
                w = float(alpha_mean[idx].item())
                context_to_o.append((s_node, w))
            context_to_o.sort(key=lambda item: item[1], reverse=True)

        layer_analyses.append(
            {
                "layer_idx": l_idx,
                "human_to_object_attention": h_to_o_att,
                "object_to_human_attention": o_to_h_att,
                "direct_pair_attention": direct_avg,
                "context_to_human": context_to_h,
                "context_to_object": context_to_o,
            }
        )

    # Top overall context entities across layers
    all_context_h: Dict[int, List[float]] = {}
    all_context_o: Dict[int, List[float]] = {}
    for la in layer_analyses:
        for n, w in la["context_to_human"]:
            all_context_h.setdefault(n, []).append(w)
        for n, w in la["context_to_object"]:
            all_context_o.setdefault(n, []).append(w)

    top_context_to_human = [
        (n, float(np.mean(weights)))
        for n, weights in sorted(
            all_context_h.items(), key=lambda it: np.mean(it[1]), reverse=True
        )
    ]
    top_context_to_object = [
        (n, float(np.mean(weights)))
        for n, weights in sorted(
            all_context_o.items(), key=lambda it: np.mean(it[1]), reverse=True
        )
    ]

    return {
        "target_pair": target_pair,
        "causality_caveat": CAUSALITY_WARNING,
        "layer_analyses": layer_analyses,
        "mean_direct_attention": float(np.mean(direct_scores)) if direct_scores else 0.0,
        "top_context_to_human": top_context_to_human,
        "top_context_to_object": top_context_to_object,
    }


def plot_attention_distribution(
    attention_data: Union[List[Dict[str, Any]], Dict[str, Any]],
    out_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 4.5),
) -> plt.Figure:
    """Plot distribution of message-passing attention weights across GNN layers.

    Generates a publication-grade multi-layer attention distribution plot.

    Args:
        attention_data: Either output from `extract_attention_weights` (list of layer dicts)
                        or `analyze_pair_attention` (analysis dict).
        out_path: Optional path to save the generated figure.
        title: Optional figure title.
        figsize: Figure dimensions (width, height) in inches.

    Returns:
        plt.Figure: Matplotlib figure instance.
    """
    if isinstance(attention_data, dict) and "layer_analyses" in attention_data:
        # Visualizing pair-specific analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        analyses = attention_data["layer_analyses"]
        layers = [f"L{la['layer_idx'] + 1}" for la in analyses]
        h2o = [la["human_to_object_attention"] for la in analyses]
        o2h = [la["object_to_human_attention"] for la in analyses]

        x = np.arange(len(layers))
        width = 0.35

        ax1.bar(x - width / 2, h2o, width, label="Human -> Object", color="#1f77b4", alpha=0.85)
        ax1.bar(x + width / 2, o2h, width, label="Object -> Human", color="#ff7f0e", alpha=0.85)
        ax1.set_xlabel("GNN Layer")
        ax1.set_ylabel("Attention Weight (alpha)")
        ax1.set_title(f"Direct Interaction Pair Attention {attention_data['target_pair']}")
        ax1.set_xticks(x)
        ax1.set_xticklabels(layers)
        ax1.legend(loc="upper right")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # Contextual attention on ax2
        top_h = attention_data.get("top_context_to_human", [])[:5]
        if top_h:
            nodes, weights = zip(*top_h)
            labels = [f"Entity {n}" for n in nodes]
            ax2.barh(labels, weights, color="#2ca02c", alpha=0.85)
            ax2.set_xlabel("Mean Context Attention")
            ax2.set_title("Top Context Entities to Human")
            ax2.invert_yaxis()
            ax2.grid(axis="x", linestyle="--", alpha=0.5)
        else:
            ax2.text(
                0.5,
                0.5,
                "No Context Entities Found",
                ha="center",
                va="center",
                color="gray",
                fontsize=11,
            )
            ax2.set_axis_off()

        plt.suptitle(
            title or "Relational Message Passing Attention Analysis",
            fontsize=12,
            fontweight="bold",
        )
        # Add causality note in footer
        fig.text(
            0.5,
            0.01,
            CAUSALITY_WARNING,
            ha="center",
            fontsize=7.5,
            style="italic",
            color="#555555",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    else:
        # Visualizing general layer-wise attention list
        att_layers = attention_data if isinstance(attention_data, list) else []
        n_layers = max(len(att_layers), 1)

        fig, axes = plt.subplots(1, n_layers, figsize=figsize, sharey=True)
        if n_layers == 1:
            axes = [axes]

        for idx, layer_dict in enumerate(att_layers):
            ax = axes[idx]
            weights = layer_dict["alpha_mean"].numpy()
            ax.hist(weights, bins=25, color="#3b528b", edgecolor="black", alpha=0.75, density=True)
            ax.set_title(f"Layer {layer_dict['layer_idx'] + 1} Attention Distribution")
            ax.set_xlabel("Attention Weight")
            if idx == 0:
                ax.set_ylabel("Density")
            ax.grid(True, linestyle="--", alpha=0.4)

        plt.suptitle(
            title or "Relational GNN Edge Attention Distribution Across Layers",
            fontsize=12,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.01,
            CAUSALITY_WARNING,
            ha="center",
            fontsize=7.5,
            style="italic",
            color="#555555",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig


# Backward-compatible aliases
extract_gatv2_attention = extract_attention_weights
EdgeAttentionAnalysis = dict
