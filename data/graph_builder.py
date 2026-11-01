"""Graph construction and spatial relation feature computation for HOI-GNN.

Produces torch_geometric.data.Data (HOIData) containing node embeddings,
directed graph edges, candidate pairs to classify, and 9-dim spatial relation features.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch_geometric.data import Data

from detection.feature_extractor import ImageFeatures


CandidateMode = Literal["all_human_object", "knn", "distance_threshold"]


class HOIData(Data):
    """PyG Data object specialized for Human-Object Interaction graphs.

    Overrides PyG's batching mechanisms so that `pair_index` [P, 2] has batch
    node offsets applied across graphs, and `pair_batch` tracks graph indices.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if "pair_batch" not in self and "pair_index" in self and self["pair_index"] is not None:
            p = self["pair_index"].shape[0]
            dev = self["pair_index"].device
            self["pair_batch"] = torch.zeros((p,), dtype=torch.long, device=dev)

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key in ("edge_index", "pair_index"):
            return self.num_nodes
        if key == "pair_batch":
            return 1
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key in ("pair_index", "pair_batch"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def compute_spatial_features(
    boxes_src: torch.Tensor,
    boxes_dst: torch.Tensor,
    image_size: Tuple[int, int],
    eps: float = 1e-6,
    clamp_log_limit: float = 10.0,
) -> torch.Tensor:
    """
    Computes the 9 normalized relative spatial relation features between
    source boxes (e.g. human) and target boxes (e.g. object):

    1. dx = (cx_dst - cx_src) / W
    2. dy = (cy_dst - cy_src) / H
    3. dist = sqrt(dx^2 + dy^2)
    4. log(w_src / w_dst)
    5. log(h_src / h_dst)
    6. log(A_src / A_dst)
    7. IoU(b_src, b_dst)
    8. (w_src * h_src) / (W * H)
    9. (w_dst * h_dst) / (W * H)

    Args:
        boxes_src: [K, 4] float32 xyxy format
        boxes_dst: [K, 4] float32 xyxy format
        image_size: (H, W) in original pixels
        eps: Epsilon for division by zero and log safety
        clamp_log_limit: Max absolute value for log ratio clamping

    Returns:
        spatial_features: [K, 9] float32 tensor
    """
    k = boxes_src.shape[0]
    if k == 0:
        return torch.zeros((0, 9), dtype=torch.float32, device=boxes_src.device)

    h_img = float(image_size[0])
    w_img = float(image_size[1])
    img_area = max(h_img * w_img, eps)
    w_img = max(w_img, eps)
    h_img = max(h_img, eps)

    # Box coordinates
    s_x1, s_y1, s_x2, s_y2 = (
        boxes_src[:, 0],
        boxes_src[:, 1],
        boxes_src[:, 2],
        boxes_src[:, 3],
    )
    d_x1, d_y1, d_x2, d_y2 = (
        boxes_dst[:, 0],
        boxes_dst[:, 1],
        boxes_dst[:, 2],
        boxes_dst[:, 3],
    )

    # Centers
    cx_s = (s_x1 + s_x2) * 0.5
    cy_s = (s_y1 + s_y2) * 0.5
    cx_d = (d_x1 + d_x2) * 0.5
    cy_d = (d_y1 + d_y2) * 0.5

    # Dimensions
    w_s = torch.clamp(s_x2 - s_x1, min=0.0)
    h_s = torch.clamp(s_y2 - s_y1, min=0.0)
    w_d = torch.clamp(d_x2 - d_x1, min=0.0)
    h_d = torch.clamp(d_y2 - d_y1, min=0.0)

    area_s = w_s * h_s
    area_d = w_d * h_d

    # 1. dx
    dx = (cx_d - cx_s) / w_img

    # 2. dy
    dy = (cy_d - cy_s) / h_img

    # 3. Euclidean distance
    dist = torch.sqrt(torch.clamp(dx * dx + dy * dy, min=0.0))

    # 4. log(w_src / w_dst)
    log_w = torch.clamp(
        torch.log((w_s + eps) / (w_d + eps)),
        min=-clamp_log_limit,
        max=clamp_log_limit,
    )

    # 5. log(h_src / h_dst)
    log_h = torch.clamp(
        torch.log((h_s + eps) / (h_d + eps)),
        min=-clamp_log_limit,
        max=clamp_log_limit,
    )

    # 6. log(A_src / A_dst)
    log_a = torch.clamp(
        torch.log((area_s + eps) / (area_d + eps)),
        min=-clamp_log_limit,
        max=clamp_log_limit,
    )

    # 7. IoU
    inter_x1 = torch.maximum(s_x1, d_x1)
    inter_y1 = torch.maximum(s_y1, d_y1)
    inter_x2 = torch.minimum(s_x2, d_x2)
    inter_y2 = torch.minimum(s_y2, d_y2)
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h
    union_area = area_s + area_d - inter_area
    iou = inter_area / (union_area + eps)

    # 8. Normalized source box area
    norm_area_s = area_s / img_area

    # 9. Normalized destination box area
    norm_area_d = area_d / img_area

    feats = torch.stack(
        [dx, dy, dist, log_w, log_h, log_a, iou, norm_area_s, norm_area_d],
        dim=-1,
    )
    return feats.to(torch.float32)


class GraphBuilder:
    """Builds PyG HOIData instances from image entities and ROI features."""

    def __init__(
        self,
        candidate_mode: CandidateMode = "all_human_object",
        person_class_id: int = 0,
        k_neighbors: int = 8,
        dist_threshold: float = 0.5,
        bidirectional: bool = False,
    ):
        """
        Args:
            candidate_mode: 'all_human_object', 'knn', or 'distance_threshold'
            person_class_id: Class ID representing persons (0 for COCO/YOLO)
            k_neighbors: For 'knn' mode, number of nearest neighbors per human
            dist_threshold: For 'distance_threshold' mode, max normalized distance
            bidirectional: If True, graph message-passing edge_index includes reverse (obj -> human) edges
        """
        self.candidate_mode = candidate_mode
        self.person_class_id = person_class_id
        self.k_neighbors = k_neighbors
        self.dist_threshold = dist_threshold
        self.bidirectional = bidirectional

    def select_candidate_pairs(
        self,
        boxes: torch.Tensor,
        class_ids: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects candidate human-object pairs and computes their 9 spatial features.

        Args:
            boxes: [N, 4] float32 xyxy format
            class_ids: [N] int64 class indices
            image_size: (H, W) in original pixels

        Returns:
            pair_index: [P, 2] int64 tensor of [human_idx, object_idx]
            pair_attr: [P, 9] float32 tensor of spatial features
        """
        n = boxes.shape[0]
        device = boxes.device
        if n < 2:
            return (
                torch.zeros((0, 2), dtype=torch.long, device=device),
                torch.zeros((0, 9), dtype=torch.float32, device=device),
            )

        human_indices = (class_ids == self.person_class_id).nonzero(as_tuple=True)[0]
        if human_indices.numel() == 0:
            return (
                torch.zeros((0, 2), dtype=torch.long, device=device),
                torch.zeros((0, 9), dtype=torch.float32, device=device),
            )

        selected_pairs: List[Tuple[int, int]] = []

        if self.candidate_mode == "all_human_object":
            for h in human_indices.tolist():
                for o in range(n):
                    if o != h:
                        selected_pairs.append((h, o))

        elif self.candidate_mode in ("knn", "distance_threshold"):
            w_img = max(float(image_size[1]), 1e-6)
            h_img = max(float(image_size[0]), 1e-6)

            # Compute centers for all boxes
            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5

            for h in human_indices.tolist():
                candidates = [o for o in range(n) if o != h]
                if not candidates:
                    continue

                cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
                dx = (cx[cand_t] - cx[h]) / w_img
                dy = (cy[cand_t] - cy[h]) / h_img
                dists = torch.sqrt(torch.clamp(dx * dx + dy * dy, min=0.0))

                if self.candidate_mode == "knn":
                    k = min(self.k_neighbors, len(candidates))
                    topk_indices = torch.topk(dists, k=k, largest=False).indices
                    for idx in topk_indices.tolist():
                        selected_pairs.append((h, candidates[idx]))
                else:  # distance_threshold
                    mask = dists <= self.dist_threshold
                    matching = cand_t[mask].tolist()
                    for o in matching:
                        selected_pairs.append((h, o))
        else:
            raise ValueError(f"Unknown candidate_mode: {self.candidate_mode}")

        if not selected_pairs:
            return (
                torch.zeros((0, 2), dtype=torch.long, device=device),
                torch.zeros((0, 9), dtype=torch.float32, device=device),
            )

        pair_index = torch.tensor(selected_pairs, dtype=torch.long, device=device)
        h_boxes = boxes[pair_index[:, 0]]
        o_boxes = boxes[pair_index[:, 1]]
        pair_attr = compute_spatial_features(h_boxes, o_boxes, image_size)
        return pair_index, pair_attr

    def build_edges(
        self,
        pair_index: torch.Tensor,
        pair_attr: torch.Tensor,
        boxes: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds directed message-passing graph edges `edge_index` [2, E] and `edge_attr` [E, 9].
        If bidirectional=True, includes reverse edges (object -> human).
        """
        p = pair_index.shape[0]
        device = pair_index.device
        if p == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros((0, 9), dtype=torch.float32, device=device),
            )

        if not self.bidirectional:
            edge_index = pair_index.t().contiguous()
            edge_attr = pair_attr
            return edge_index, edge_attr

        # Add reverse edges: target -> source
        rev_pair_index = torch.stack([pair_index[:, 1], pair_index[:, 0]], dim=1)
        src_boxes = boxes[rev_pair_index[:, 0]]
        dst_boxes = boxes[rev_pair_index[:, 1]]
        rev_pair_attr = compute_spatial_features(src_boxes, dst_boxes, image_size)

        all_edges = torch.cat([pair_index, rev_pair_index], dim=0)
        all_attrs = torch.cat([pair_attr, rev_pair_attr], dim=0)

        edge_index = all_edges.t().contiguous()
        return edge_index, all_attrs

    def build_graph(
        self,
        image_features: Optional[ImageFeatures] = None,
        boxes: Optional[torch.Tensor] = None,
        class_ids: Optional[torch.Tensor] = None,
        roi_features: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
        image_id: Optional[Union[str, int]] = None,
        scores: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        y_mask: Optional[torch.Tensor] = None,
    ) -> HOIData:
        """
        Constructs an HOIData object from features/boxes and optional ground-truth targets.

        Args:
            image_features: Optional ImageFeatures dataclass from Phase 1
            boxes: [N, 4] float32 xyxy format
            class_ids: [N] int64 class indices
            roi_features: [N, D] float32 node features
            image_size: (H, W) in original pixels
            image_id: Unique identifier for the image
            scores: [N] float32 confidence scores
            y: [P, 26] float32 multi-hot binary interaction ground truth
            y_mask: [P, 26] float32 interaction validity mask

        Returns:
            HOIData PyG graph object
        """
        if image_features is not None:
            boxes = image_features.boxes
            class_ids = image_features.class_ids
            roi_features = image_features.roi_features
            image_size = image_features.image_size
            image_id = image_features.image_id if image_id is None else image_id
            scores = image_features.scores if scores is None else scores

        if boxes is None:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        n = boxes.shape[0]
        device = boxes.device

        if class_ids is None:
            class_ids = torch.full((n,), -1, dtype=torch.long, device=device)
        else:
            class_ids = torch.as_tensor(class_ids, dtype=torch.long, device=device).reshape(-1)

        if scores is None:
            scores = torch.ones((n,), dtype=torch.float32, device=device)
        else:
            scores = torch.as_tensor(scores, dtype=torch.float32, device=device).reshape(-1)

        if roi_features is None:
            roi_features = torch.zeros((n, 56), dtype=torch.float32, device=device)
        else:
            roi_features = torch.as_tensor(roi_features, dtype=torch.float32, device=device)
            if roi_features.ndim == 1:
                roi_features = roi_features.unsqueeze(-1)
            elif roi_features.ndim > 2:
                roi_features = roi_features.flatten(1)

        if image_size is None:
            image_size = (640, 640)

        # Select candidate pairs and compute spatial features
        pair_index, pair_attr = self.select_candidate_pairs(boxes, class_ids, image_size)

        # Build message-passing graph edges
        edge_index, edge_attr = self.build_edges(pair_index, pair_attr, boxes, image_size)

        p = pair_index.shape[0]
        pair_batch = torch.zeros((p,), dtype=torch.long, device=device)

        # CONTRACTS.md compliance: node_type (0=human, 1=object), boxes_norm, x_visual
        node_type = torch.where(class_ids == 0, 0, 1).long()
        h_img, w_img = image_size
        scale = torch.tensor([w_img, h_img, w_img, h_img], dtype=torch.float32, device=device)
        boxes_norm = boxes / scale.clamp(min=1.0)

        data_kwargs: Dict[str, Any] = {
            "x": roi_features,
            "x_visual": roi_features,
            "node_type": node_type,
            "boxes": boxes,
            "boxes_norm": boxes_norm,
            "class_ids": class_ids,
            "scores": scores,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "pair_index": pair_index,
            "pair_attr": pair_attr,
            "pair_batch": pair_batch,
            "num_nodes": n,
            "image_id": image_id or "",
            "image_size": tuple(image_size),
        }

        if y is not None:
            data_kwargs["y"] = torch.as_tensor(y, dtype=torch.float32, device=device)
        if y_mask is not None:
            data_kwargs["y_mask"] = torch.as_tensor(y_mask, dtype=torch.float32, device=device)

        return HOIData(**data_kwargs)
