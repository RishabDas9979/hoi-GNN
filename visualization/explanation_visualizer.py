"""Publication-quality explanation visualization for HOI-GNN.

Provides visual explanations of model predictions via:
1. Relational scene graph attribution (NetworkX + Matplotlib).
2. Spatial bounding box node attribution.
3. Dual-view side-by-side comparative visualization.

Note:
All attributions represent graph-level and node/edge-level attribution on
the constructed scene graph, NOT dense pixel-level attribution (e.g. Grad-CAM).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution/testing
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

try:
    from data.vcoco_loader import COCO_CLASSES
except ImportError:
    COCO_CLASSES = (
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
    )

DISCLAIMER_TEXT = "Graph-Level Attribution (Scene Graph), not pixel-level"


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    """Convert tensor, sequence, or array to a float32 numpy array."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_01(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Safely normalize values to [0, 1] range."""
    if arr.size == 0:
        return arr
    min_v, max_v = float(np.min(arr)), float(np.max(arr))
    if max_v - min_v > eps:
        return (arr - min_v) / (max_v - min_v)
    # If all values are constant, return array of 0.5s or 1.0s if non-zero
    if max_v > eps:
        return np.ones_like(arr, dtype=np.float32)
    return np.zeros_like(arr, dtype=np.float32)


def visualize_explanation_graph(
    data: Any,
    edge_mask: Union[torch.Tensor, np.ndarray, Sequence[float]],
    node_mask: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
    target_pair: Optional[Tuple[int, int]] = None,
    action_name: str = "interaction",
    out_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10.0, 8.5),
) -> plt.Figure:
    """Visualize relational explanation as a directed scene graph.

    Nodes represent detected entities (human circles, object squares), and edges
    represent message-passing relational connections scaled by their attribution
    importance mask. The target human-object interaction pair is prominently highlighted.

    Args:
        data: PyG HOIData or Data object, or dict containing `edge_index`, optional
            `boxes`, `class_ids`, `node_type`, and `num_nodes`.
        edge_mask: Importance scores for each edge in `edge_index` [E].
        node_mask: Optional importance scores for each node [N] or [N, D].
        target_pair: Optional tuple of (human_node_idx, object_node_idx) for the target HOI.
        action_name: Name of the predicted action (e.g. 'hold', 'sit', 'ride').
        out_path: Optional file path to save the generated figure.
        title: Optional custom figure title.
        ax: Optional matplotlib axes to draw on. If None, creates a new figure.
        figsize: Figure dimensions (width, height) in inches.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        created_fig = True
    else:
        fig = ax.figure

    # Extract edge_index
    if hasattr(data, "edge_index") and data.edge_index is not None:
        edge_index_raw = data.edge_index
    elif isinstance(data, dict) and "edge_index" in data:
        edge_index_raw = data["edge_index"]
    else:
        edge_index_raw = torch.zeros((2, 0), dtype=torch.long)

    edge_index = _to_numpy(edge_index_raw).astype(np.int64)
    if edge_index.ndim == 1:
        edge_index = edge_index.reshape(2, -1)

    num_edges = edge_index.shape[1]

    # Extract node count
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        num_nodes = int(data.num_nodes)
    elif hasattr(data, "x") and data.x is not None:
        num_nodes = int(data.x.shape[0])
    elif hasattr(data, "boxes") and data.boxes is not None:
        num_nodes = int(len(data.boxes))
    elif num_edges > 0:
        num_nodes = int(np.max(edge_index) + 1)
    else:
        num_nodes = max(
            (target_pair[0] + 1 if target_pair else 0),
            (target_pair[1] + 1 if target_pair else 0),
            1,
        )

    # Extract class_ids and node_type
    class_ids = None
    if hasattr(data, "class_ids") and data.class_ids is not None:
        class_ids = _to_numpy(data.class_ids).astype(np.int64).reshape(-1)
    elif isinstance(data, dict) and "class_ids" in data:
        class_ids = _to_numpy(data["class_ids"]).astype(np.int64).reshape(-1)

    node_type = None
    if hasattr(data, "node_type") and data.node_type is not None:
        node_type = _to_numpy(data.node_type).astype(np.int64).reshape(-1)
    elif isinstance(data, dict) and "node_type" in data:
        node_type = _to_numpy(data["node_type"]).astype(np.int64).reshape(-1)

    # Edge mask preprocessing
    edge_mask_np = _to_numpy(edge_mask).astype(np.float32).reshape(-1)
    if len(edge_mask_np) < num_edges:
        # Pad with zeros if necessary
        padded = np.zeros(num_edges, dtype=np.float32)
        padded[:len(edge_mask_np)] = edge_mask_np
        edge_mask_np = padded
    elif len(edge_mask_np) > num_edges:
        edge_mask_np = edge_mask_np[:num_edges]

    norm_edge_mask = _normalize_01(edge_mask_np)

    # Node mask preprocessing
    node_importance = None
    if node_mask is not None:
        node_mask_np = _to_numpy(node_mask).astype(np.float32)
        if node_mask_np.ndim > 1:
            node_mask_np = np.mean(node_mask_np, axis=-1)
        node_mask_np = node_mask_np.reshape(-1)
        if len(node_mask_np) >= num_nodes:
            node_importance = _normalize_01(node_mask_np[:num_nodes])
        else:
            padded_n = np.zeros(num_nodes, dtype=np.float32)
            padded_n[:len(node_mask_np)] = node_mask_np
            node_importance = _normalize_01(padded_n)

    # Build NetworkX DiGraph
    G = nx.DiGraph()
    for i in range(num_nodes):
        # Determine if node is human
        is_human = False
        if node_type is not None and i < len(node_type):
            is_human = (node_type[i] == 0)
        elif class_ids is not None and i < len(class_ids):
            is_human = (class_ids[i] == 0)
        elif target_pair is not None and i == target_pair[0]:
            is_human = True

        cls_name = "person" if is_human else "object"
        if class_ids is not None and i < len(class_ids):
            cid = int(class_ids[i])
            if 0 <= cid < len(COCO_CLASSES):
                cls_name = COCO_CLASSES[cid]

        G.add_node(i, is_human=is_human, class_name=cls_name)

    # Add edges with attributes
    for e_idx in range(num_edges):
        u, v = int(edge_index[0, e_idx]), int(edge_index[1, e_idx])
        if u < num_nodes and v < num_nodes:
            raw_w = float(edge_mask_np[e_idx])
            norm_w = float(norm_edge_mask[e_idx])
            is_target = bool(
                target_pair is not None and (u, v) == (target_pair[0], target_pair[1])
            )
            G.add_edge(u, v, weight=raw_w, norm_weight=norm_w, is_target=is_target, edge_idx=e_idx)

    # Determine positions: use bounding box centers if available, else spring layout
    boxes = None
    if hasattr(data, "boxes") and data.boxes is not None:
        boxes = _to_numpy(data.boxes)
    elif isinstance(data, dict) and "boxes" in data:
        boxes = _to_numpy(data["boxes"])

    if boxes is not None and len(boxes) >= num_nodes and boxes.shape[-1] >= 4:
        cx = (boxes[:num_nodes, 0] + boxes[:num_nodes, 2]) / 2.0
        cy = (boxes[:num_nodes, 1] + boxes[:num_nodes, 3]) / 2.0
        # Invert cy so top-of-image is at top of plot
        cy_span = max(float(np.max(cy) - np.min(cy)), 1.0)
        cx_span = max(float(np.max(cx) - np.min(cx)), 1.0)
        norm_cx = 2.0 * (cx - np.mean(cx)) / cx_span
        norm_cy = -2.0 * (cy - np.mean(cy)) / cy_span
        initial_pos = {i: np.array([norm_cx[i], norm_cy[i]], dtype=np.float32) for i in range(num_nodes)}
        try:
            pos = nx.spring_layout(G, pos=initial_pos, iterations=35, k=1.4, seed=42)
        except Exception:
            pos = initial_pos
    else:
        try:
            pos = nx.spring_layout(G, k=1.3, iterations=50, seed=42)
        except Exception:
            pos = {i: np.array([np.cos(2 * np.pi * i / max(num_nodes, 1)),
                                np.sin(2 * np.pi * i / max(num_nodes, 1))]) for i in range(num_nodes)}

    # Separate human vs object nodes
    human_nodes = [n for n, attr in G.nodes(data=True) if attr.get("is_human", False)]
    object_nodes = [n for n, attr in G.nodes(data=True) if not attr.get("is_human", False)]

    # Draw Nodes: Humans (circles, blue) vs Objects (squares, green)
    base_node_size = 900
    target_h = target_pair[0] if target_pair is not None else None
    target_o = target_pair[1] if target_pair is not None else None

    # Normal human nodes
    norm_h = [n for n in human_nodes if n != target_h]
    if norm_h:
        h_sizes = [base_node_size * (1.0 + 0.4 * (node_importance[n] if node_importance is not None else 0.0)) for n in norm_h]
        nx.draw_networkx_nodes(
            G, pos, nodelist=norm_h, node_shape="o", node_color="#3498DB",
            edgecolors="#1B4F72", linewidths=2.0, node_size=h_sizes, ax=ax,
        )

    # Normal object nodes
    norm_o = [n for n in object_nodes if n != target_o]
    if norm_o:
        o_sizes = [base_node_size * (1.0 + 0.4 * (node_importance[n] if node_importance is not None else 0.0)) for n in norm_o]
        nx.draw_networkx_nodes(
            G, pos, nodelist=norm_o, node_shape="s", node_color="#2ECC71",
            edgecolors="#145A32", linewidths=2.0, node_size=o_sizes, ax=ax,
        )

    # Highlight Target Human (Golden bold halo)
    if target_h is not None and target_h in G.nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[target_h], node_shape="o", node_color="#5DADE2",
            edgecolors="#F39C12", linewidths=4.5, node_size=base_node_size * 1.5, ax=ax,
        )

    # Highlight Target Object (Golden bold halo)
    if target_o is not None and target_o in G.nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[target_o], node_shape="s", node_color="#58D68D",
            edgecolors="#F39C12", linewidths=4.5, node_size=base_node_size * 1.5, ax=ax,
        )

    # Node Labels
    labels = {}
    for n, attr in G.nodes(data=True):
        prefix = "H" if attr.get("is_human", False) else "O"
        cls = attr.get("class_name", "")
        if node_importance is not None and n < len(node_importance):
            labels[n] = f"{prefix}{n}: {cls}\n({node_importance[n]*100:.0f}%)"
        else:
            labels[n] = f"{prefix}{n}: {cls}"

    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=8.5, font_weight="bold",
        font_color="#1C2833", ax=ax,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#D5D8DC", alpha=0.85),
    )

    # Draw Edges: widths and alpha proportional to edge_mask
    target_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("is_target", False)]
    regular_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("is_target", False)]

    # Draw regular edges
    if regular_edges:
        cmap = plt.get_cmap("Blues")
        for u, v in regular_edges:
            d = G[u][v]
            nw = d.get("norm_weight", 0.5)
            # Width ranges from 0.8 to 4.5, alpha from 0.25 to 0.95
            w = 0.8 + 3.7 * nw
            alpha = 0.25 + 0.7 * nw
            edge_col = cmap(0.3 + 0.7 * nw)
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], width=w, edge_color=[edge_col],
                alpha=alpha, arrows=True, arrowsize=14, arrowstyle="-|>",
                connectionstyle="arc3,rad=0.08", ax=ax,
            )

    # Draw Target Edge with prominent color and annotation
    if target_edges:
        for u, v in target_edges:
            d = G[u][v]
            nw = d.get("norm_weight", 1.0)
            raw_w = d.get("weight", 1.0)
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], width=max(4.0, 1.5 + 4.5 * nw),
                edge_color="#E74C3C", alpha=1.0, arrows=True, arrowsize=20,
                arrowstyle="-|>", connectionstyle="arc3,rad=0.12", ax=ax,
            )
            # Draw edge label for target interaction
            p_u = pos[u]
            p_v = pos[v]
            mid_p = (p_u + p_v) / 2.0 + np.array([0.0, 0.08])
            ax.text(
                mid_p[0], mid_p[1],
                f"★ {action_name} ({raw_w:.2f})",
                color="#900C3F", fontsize=9.5, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC", edgecolor="#E74C3C", linewidth=1.5),
            )

    # Text disclaimer required: Graph-level attribution, not pixel-level
    ax.text(
        0.5, 0.02,
        DISCLAIMER_TEXT,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=9.5, fontweight="bold",
        color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EAEDED", edgecolor="#BDC3C7", alpha=0.92),
    )

    # Plot title and cleanup
    graph_title = title or (
        f"Scene Graph Attribution: {action_name.upper()} "
        + (f"(Pair {target_pair[0]} -> {target_pair[1]})" if target_pair else "")
    )
    ax.set_title(graph_title, fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")

    if created_fig:
        plt.tight_layout()

    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_p), bbox_inches="tight", dpi=300)

    return fig


def visualize_bbox_attribution(
    image_or_size: Optional[Union[np.ndarray, Tuple[int, int], Tuple[int, int, int], torch.Tensor]] = None,
    boxes: Optional[Union[torch.Tensor, np.ndarray, Sequence[Sequence[float]]]] = None,
    class_ids: Optional[Union[torch.Tensor, np.ndarray, Sequence[int]]] = None,
    node_importance: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
    target_pair: Optional[Tuple[int, int]] = None,
    out_path: Optional[Union[str, Path]] = None,
    class_names: Optional[Dict[int, str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10.0, 8.0),
    colormap: str = "viridis",
    title: Optional[str] = None,
    data: Optional[Any] = None,
    node_mask: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
    action_name: Optional[str] = None,
) -> plt.Figure:
    """Visualize node attribution intensity on spatial bounding boxes.

    Draws boxes on an image (or a clean canvas) with border thickness and colors
    reflecting their relative attribution importance.

    Args:
        image_or_size: Background RGB image [H, W, 3] or tuple/list (H, W).
        boxes: Bounding box coordinates [N, 4] in [x1, y1, x2, y2] format.
        class_ids: Category ID for each box [N].
        node_importance: Node attribution intensity [N] (or [N, D]).
        target_pair: Optional tuple of (human_idx, object_idx) for the target HOI.
        out_path: Optional file path to save the generated figure.
        class_names: Optional mapping from class_id to class string name.
        ax: Optional matplotlib axes to draw on. If None, creates a new figure.
        figsize: Figure dimensions (width, height) in inches.
        colormap: Matplotlib colormap name (e.g. 'viridis', 'YlOrRd').
        title: Optional custom figure title.
        data: Optional PyG HOIData or Data object to extract boxes and class_ids from.
        node_mask: Optional alias for node_importance.
        action_name: Optional action label name.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    if data is not None:
        if boxes is None:
            boxes = getattr(data, "boxes_norm", getattr(data, "boxes", None))
        if class_ids is None:
            class_ids = getattr(data, "class_ids", None)
    if node_importance is None and node_mask is not None:
        node_importance = node_mask
    if image_or_size is None:
        image_or_size = (600, 800)
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        created_fig = True
    else:
        fig = ax.figure

    # Determine canvas dimensions and background
    if isinstance(image_or_size, (tuple, list)):
        h, w = int(image_or_size[0]), int(image_or_size[1])
        img = np.ones((h, w, 3), dtype=np.uint8) * 250
    elif isinstance(image_or_size, torch.Tensor):
        img_np = image_or_size.detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.dtype != np.uint8 and np.max(img_np) <= 1.0:
            img = (img_np * 255).astype(np.uint8)
        else:
            img = img_np.astype(np.uint8)
    elif isinstance(image_or_size, np.ndarray):
        img = image_or_size.copy()
        if img.dtype != np.uint8 and np.max(img) <= 1.0:
            img = (img * 255).astype(np.uint8)
    else:
        img = np.ones((600, 800, 3), dtype=np.uint8) * 250

    ax.imshow(img)

    # Process boxes, class_ids, node_importance
    boxes_np = _to_numpy(boxes).astype(np.float32)
    if boxes_np.ndim == 1:
        boxes_np = boxes_np.reshape(-1, 4) if len(boxes_np) >= 4 else np.zeros((0, 4), dtype=np.float32)

    class_ids_np = _to_numpy(class_ids).astype(np.int64).reshape(-1)
    importance_np = _to_numpy(node_importance).astype(np.float32)
    if importance_np.ndim > 1:
        importance_np = np.mean(importance_np, axis=-1)
    importance_np = importance_np.reshape(-1)

    n_boxes = len(boxes_np)
    if len(class_ids_np) < n_boxes:
        padded_c = np.zeros(n_boxes, dtype=np.int64)
        padded_c[:len(class_ids_np)] = class_ids_np
        class_ids_np = padded_c

    if len(importance_np) < n_boxes:
        padded_i = np.zeros(n_boxes, dtype=np.float32)
        padded_i[:len(importance_np)] = importance_np
        importance_np = padded_i
    elif len(importance_np) > n_boxes:
        importance_np = importance_np[:n_boxes]

    # Normalize attribution to [0, 1]
    norm_importance = _normalize_01(importance_np)
    total_imp = float(np.sum(importance_np))
    if total_imp > 1e-8:
        pct_importance = (importance_np / total_imp) * 100.0
    else:
        pct_importance = norm_importance * 100.0 / max(n_boxes, 1)

    cmap = plt.get_cmap(colormap)

    # Draw boxes
    for i in range(n_boxes):
        x1, y1, x2, y2 = boxes_np[i, :4]
        box_w, box_h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        norm_val = float(norm_importance[i])
        pct_val = float(pct_importance[i])

        # Color and line width reflect node importance
        color = cmap(0.2 + 0.8 * norm_val)
        lw = 2.0 + 4.5 * norm_val

        is_target_h = bool(target_pair is not None and i == target_pair[0])
        is_target_o = bool(target_pair is not None and i == target_pair[1])

        # Target boxes get highlighted styling
        if is_target_h or is_target_o:
            lw = max(lw, 4.0)

        # Draw filled patch + border
        rect = patches.Rectangle(
            (x1, y1), box_w, box_h,
            linewidth=lw,
            edgecolor=color,
            facecolor=(*color[:3], 0.12),
        )
        ax.add_patch(rect)

        # Draw gold dashed halo for target pair entities
        if is_target_h or is_target_o:
            halo = patches.Rectangle(
                (x1 - 2, y1 - 2), box_w + 4, box_h + 4,
                linewidth=2.0,
                edgecolor="#F39C12",
                linestyle="--",
                fill=False,
            )
            ax.add_patch(halo)

        # Class label and percentage attribution
        cid = int(class_ids_np[i])
        if class_names and cid in class_names:
            cname = class_names[cid]
        elif 0 <= cid < len(COCO_CLASSES):
            cname = COCO_CLASSES[cid]
        else:
            cname = f"class_{cid}"

        tag = " [TARGET HUMAN]" if is_target_h else (" [TARGET OBJ]" if is_target_o else "")
        label_text = f"{cname}_{i}: {pct_val:.1f}%{tag}"

        # Place label above box
        label_y = max(y1 - 6, 12)
        ax.text(
            x1, label_y,
            label_text,
            fontsize=8.5,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=color, edgecolor="none", alpha=0.9),
        )

    # Colorbar
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Node Attribution Intensity [0, 1]", fontsize=9, fontweight="bold")

    # Clear text disclaimer
    ax.text(
        0.5, 0.02,
        DISCLAIMER_TEXT,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=9.0, fontweight="bold",
        color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EAEDED", edgecolor="#BDC3C7", alpha=0.92),
    )

    box_title = title or "Spatial Node Attribution (Bounding Boxes)"
    ax.set_title(box_title, fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")

    if created_fig:
        plt.tight_layout()

    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_p), bbox_inches="tight", dpi=300)

    return fig


def plot_dual_explanation_view(
    data: Any,
    edge_mask: Union[torch.Tensor, np.ndarray, Sequence[float]],
    node_mask: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]],
    target_pair: Tuple[int, int],
    action_name: str = "interaction",
    out_path: Optional[Union[str, Path]] = None,
    image: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (19.0, 8.5),
) -> plt.Figure:
    """Generate publication-ready side-by-side dual explanation visualization.

    Left Panel: Bounding box spatial node attribution view.
    Right Panel: Directed scene graph relational edge & node attribution view.

    Args:
        data: PyG HOIData or Data object, or dict.
        edge_mask: Edge attribution importance scores.
        node_mask: Node attribution importance scores.
        target_pair: (human_node_idx, object_node_idx) tuple.
        action_name: Action/interaction label.
        out_path: Optional path to save the generated figure.
        image: Optional background RGB image array.
        title: Optional overarching figure suptitle.
        figsize: Figure dimensions in inches.

    Returns:
        matplotlib.figure.Figure: The generated side-by-side figure.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize, dpi=300)

    # 1. Prepare inputs for spatial bbox view
    boxes = getattr(data, "boxes", None)
    if boxes is None and isinstance(data, dict):
        boxes = data.get("boxes", None)
    if boxes is None:
        boxes = np.array([[50, 50, 200, 350], [250, 150, 380, 300]], dtype=np.float32)

    class_ids = getattr(data, "class_ids", None)
    if class_ids is None and isinstance(data, dict):
        class_ids = data.get("class_ids", None)
    if class_ids is None:
        class_ids = np.array([0, 1], dtype=np.int64)

    image_size = getattr(data, "image_size", None)
    if image is not None:
        bg = image
    elif image_size is not None and len(image_size) >= 2:
        bg = (int(image_size[0]), int(image_size[1]))
    else:
        boxes_np = _to_numpy(boxes)
        max_x = int(np.max(boxes_np[:, [0, 2]])) + 80 if boxes_np.size > 0 else 640
        max_y = int(np.max(boxes_np[:, [1, 3]])) + 80 if boxes_np.size > 0 else 480
        bg = (max(max_y, 480), max(max_x, 640))

    # Derive node_importance if node_mask is None
    if node_mask is not None:
        node_imp = node_mask
    else:
        # Default node importance: uniform or based on edge connections
        node_imp = np.ones(len(boxes), dtype=np.float32) / max(len(boxes), 1)

    # Left: Spatial Bounding Box Attribution
    visualize_bbox_attribution(
        image_or_size=bg,
        boxes=boxes,
        class_ids=class_ids,
        node_importance=node_imp,
        target_pair=target_pair,
        ax=ax_left,
        title="(a) Spatial Node Attribution (Bounding Boxes)",
    )

    # Right: Relational Scene Graph Attribution
    visualize_explanation_graph(
        data=data,
        edge_mask=edge_mask,
        node_mask=node_mask,
        target_pair=target_pair,
        action_name=action_name,
        ax=ax_right,
        title=f"(b) Relational Scene Graph Attribution",
    )

    # Supertitle
    main_title = title or (
        f"HOI Explanation Dual-View: Target Pair ({target_pair[0]} -> {target_pair[1]}) "
        f"| Action: '{action_name.upper()}'"
    )
    fig.suptitle(main_title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_p), bbox_inches="tight", dpi=300)

    return fig
