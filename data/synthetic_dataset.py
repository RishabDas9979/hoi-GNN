"""Synthetic HOI dataset generator and PyG Dataset for pipeline validation.

Generates deterministic synthetic human-object interaction scenes with:
- Valid bounding boxes and detection scores
- Typed nodes (person=0, object in 1..79)
- ROI appearance embeddings [N, D] (default D=56 or 896)
- Deterministic, learnable interaction labels [P, 26] based on geometric
  proximity (d < 0.35) and object category, with correlated visual features
- Strict compliance with V-COCO pair contracts (agent-only actions receive
  y = 0 and y_mask = 0; candidate actions receive y_mask = 1)
- Disk cache serialization matching CONTRACTS.md Section 1 (`ImageFeatures.to_dict()`)
- PyG HOIData construction via GraphBuilder and collation with hoi_collate
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset

from data.dataset import _find_cached_features, hoi_collate
from data.graph_builder import GraphBuilder, HOIData, compute_spatial_features
from data.vcoco_loader import (
    AGENT_ONLY_ACTION_INDICES,
    NUM_ACTIONS,
    VCOCO_ACTIONS,
)
from detection.feature_extractor import ImageFeatures


# Valid action indices for human-object pairs (excluding agent-only actions)
VALID_PAIR_ACTIONS: List[int] = [
    i for i in range(NUM_ACTIONS) if i not in AGENT_ONLY_ACTION_INDICES
]

# Canonical COCO class ID -> V-COCO action index mapping
# Creates realistic associations for well-known object classes
CANONICAL_OBJECT_ACTIONS: Dict[int, int] = {
    1: 3,    # bicycle -> ride
    3: 3,    # motorcycle -> ride
    17: 3,   # horse -> ride
    13: 2,   # bench -> sit
    56: 2,   # chair -> sit
    57: 2,   # couch -> sit
    59: 2,   # bed -> sit
    39: 21,  # bottle -> drink
    40: 21,  # wine glass -> drink
    41: 21,  # cup -> drink
    46: 7,   # banana -> eat
    47: 7,   # apple -> eat
    48: 7,   # sandwich -> eat
    49: 7,   # orange -> eat
    50: 7,   # broccoli -> eat
    51: 7,   # carrot -> eat
    52: 7,   # hot dog -> eat
    53: 7,   # pizza -> eat
    54: 7,   # donut -> eat
    55: 7,   # cake -> eat
    63: 16,  # laptop -> work_on_computer
    64: 16,  # mouse -> work_on_computer
    66: 16,  # keyboard -> work_on_computer
    67: 10,  # cell phone -> talk_on_phone
    73: 24,  # book -> read
    30: 17,  # skis -> ski
    31: 25,  # snowboard -> snowboard
    36: 19,  # skateboard -> skateboard
    37: 18,  # surfboard -> surf
    34: 6,   # baseball bat -> hit
    38: 6,   # tennis racket -> hit
    29: 12,  # frisbee -> throw
    32: 12,  # sports ball -> throw
    24: 11,  # backpack -> carry
    26: 11,  # handbag -> carry
    28: 11,  # suitcase -> carry
    43: 14,  # knife -> cut
    76: 14,  # scissors -> cut
}


class SyntheticHOIGenerator:
    """Deterministic generator of synthetic HOI scenes, features, and graphs.

    Generates synthetic scenes containing persons and objects with:
    - Bounded xyxy bounding boxes
    - Deterministic class IDs (person=0, objects in 1..79)
    - Detection scores in [0.85, 1.0]
    - ROI feature embeddings [N, D] (default D=56 or 896)
    - Deterministic, learnable interaction labels [P, 26] based on geometric
      proximity (d < 0.35) and object category, with correlated visual features
    - Strict compliance with V-COCO pair contracts
    - Serialization to cache/features_synthetic/*.pt matching ImageFeatures.to_dict()
    - PyG HOIData construction via GraphBuilder
    """

    def __init__(
        self,
        feature_dim: int = 56,
        seed: int = 42,
        min_entities: int = 2,
        max_entities: int = 6,
        image_size: Tuple[int, int] = (640, 640),
        dist_threshold: float = 0.35,
        graph_builder: Optional[GraphBuilder] = None,
    ):
        self.feature_dim = feature_dim
        self.seed = seed
        self.min_entities = max(2, min_entities)
        self.max_entities = max(self.min_entities, max_entities)
        self.image_size = image_size
        self.dist_threshold = dist_threshold
        self.graph_builder = graph_builder or GraphBuilder()

        # Fixed deterministic codebooks for category base features and action patterns
        gen = torch.Generator().manual_seed(seed)
        self._class_embeddings = F.normalize(
            torch.randn(80, feature_dim, generator=gen), dim=-1
        )
        self._action_embeddings = F.normalize(
            torch.randn(NUM_ACTIONS, feature_dim, generator=gen), dim=-1
        )

    def _get_rng(self, image_id: Union[str, int], seed: Optional[int] = None) -> torch.Generator:
        """Derive a reproducible local torch.Generator from image_id and base seed."""
        if seed is not None:
            s = int(seed)
        else:
            try:
                int_id = int(image_id)
            except (ValueError, TypeError):
                int_id = abs(hash(str(image_id))) % 10000000
            s = (self.seed * 10007 + int_id) % 2147483647
        return torch.Generator().manual_seed(s)

    def compute_targets(
        self,
        boxes: torch.Tensor,
        class_ids: torch.Tensor,
        pair_index: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ground-truth action targets (y) and masks (y_mask) for pairs.

        Deterministic rule:
        - If normalized Euclidean center distance d < dist_threshold (0.35):
          activate action index corresponding to the object's category.
        - If d < 0.20: additionally activate action 0 ('hold').
        - For all pairs: agent-only actions (stand, walk, run, smile) receive
          y = 0 and y_mask = 0. All other candidate action columns receive y_mask = 1.
        """
        p = pair_index.shape[0]
        device = boxes.device
        y = torch.zeros((p, NUM_ACTIONS), dtype=torch.float32, device=device)
        y_mask = torch.ones((p, NUM_ACTIONS), dtype=torch.float32, device=device)

        # Agent-only actions have no object role; per V-COCO contract mask them out
        for a_idx in AGENT_ONLY_ACTION_INDICES:
            y_mask[:, a_idx] = 0.0

        if p == 0:
            return y, y_mask

        h_img, w_img = float(image_size[0]), float(image_size[1])
        w_img = max(w_img, 1e-6)
        h_img = max(h_img, 1e-6)

        h_idx = pair_index[:, 0]
        o_idx = pair_index[:, 1]

        cx_h = (boxes[h_idx, 0] + boxes[h_idx, 2]) * 0.5
        cy_h = (boxes[h_idx, 1] + boxes[h_idx, 3]) * 0.5
        cx_o = (boxes[o_idx, 0] + boxes[o_idx, 2]) * 0.5
        cy_o = (boxes[o_idx, 1] + boxes[o_idx, 3]) * 0.5

        dx = (cx_o - cx_h) / w_img
        dy = (cy_o - cy_h) / h_img
        dist = torch.sqrt(torch.clamp(dx * dx + dy * dy, min=0.0))

        for i in range(p):
            d_val = float(dist[i].item())
            if d_val < self.dist_threshold:
                obj_cls = int(class_ids[o_idx[i]].item())
                if obj_cls in CANONICAL_OBJECT_ACTIONS:
                    act = CANONICAL_OBJECT_ACTIONS[obj_cls]
                else:
                    act = VALID_PAIR_ACTIONS[(abs(obj_cls) - 1) % len(VALID_PAIR_ACTIONS)]
                y[i, act] = 1.0

                # Close proximity triggers 'hold' interaction
                if d_val < 0.20:
                    y[i, 0] = 1.0

        return y, y_mask

    def generate_scene(
        self,
        image_id: Union[str, int],
        num_entities: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[ImageFeatures, torch.Tensor, torch.Tensor]:
        """Generate a complete synthetic scene: ImageFeatures, y targets, and y_mask."""
        rng = self._get_rng(image_id, seed)

        if num_entities is None:
            n = int(torch.randint(self.min_entities, self.max_entities + 1, (1,), generator=rng).item())
        else:
            n = max(2, num_entities)

        # Person count: 1 (or 2 if n >= 4 and RNG check passes)
        n_person = 1
        if n >= 4 and torch.rand((1,), generator=rng).item() > 0.6:
            n_person = 2
        n_object = n - n_person

        person_classes = torch.zeros((n_person,), dtype=torch.long)
        object_classes = torch.randint(1, 80, (n_object,), generator=rng, dtype=torch.long)
        class_ids = torch.cat([person_classes, object_classes])

        h_img, w_img = self.image_size
        boxes_list: List[List[float]] = []

        # Person 0
        w_p0 = float(torch.randint(80, 180, (1,), generator=rng).item())
        h_p0 = float(torch.randint(160, 320, (1,), generator=rng).item())
        x1_p0 = float(torch.randint(30, max(31, int(w_img - w_p0 - 30)), (1,), generator=rng).item())
        y1_p0 = float(torch.randint(30, max(31, int(h_img - h_p0 - 30)), (1,), generator=rng).item())
        boxes_list.append([x1_p0, y1_p0, x1_p0 + w_p0, y1_p0 + h_p0])
        cx_p0 = x1_p0 + w_p0 * 0.5
        cy_p0 = y1_p0 + h_p0 * 0.5

        # Person 1 (if 2 persons)
        if n_person == 2:
            w_p1 = float(torch.randint(80, 180, (1,), generator=rng).item())
            h_p1 = float(torch.randint(160, 320, (1,), generator=rng).item())
            x1_p1 = float(torch.randint(30, max(31, int(w_img - w_p1 - 30)), (1,), generator=rng).item())
            y1_p1 = float(torch.randint(30, max(31, int(h_img - h_p1 - 30)), (1,), generator=rng).item())
            boxes_list.append([x1_p1, y1_p1, x1_p1 + w_p1, y1_p1 + h_p1])

        # Object 0: placed in proximity to Person 0 (d < 0.28)
        w_o0 = float(torch.randint(40, 120, (1,), generator=rng).item())
        h_o0 = float(torch.randint(40, 120, (1,), generator=rng).item())
        off_x = (torch.rand((1,), generator=rng).item() * 0.36 - 0.18) * w_img
        off_y = (torch.rand((1,), generator=rng).item() * 0.36 - 0.18) * h_img
        cx_o0 = min(max(cx_p0 + off_x, w_o0 * 0.5 + 5.0), w_img - w_o0 * 0.5 - 5.0)
        cy_o0 = min(max(cy_p0 + off_y, h_o0 * 0.5 + 5.0), h_img - h_o0 * 0.5 - 5.0)
        boxes_list.append([cx_o0 - w_o0 * 0.5, cy_o0 - h_o0 * 0.5, cx_o0 + w_o0 * 0.5, cy_o0 + h_o0 * 0.5])

        # Remaining objects
        for o_idx in range(1, n_object):
            w_oi = float(torch.randint(40, 140, (1,), generator=rng).item())
            h_oi = float(torch.randint(40, 140, (1,), generator=rng).item())
            if o_idx == 1:
                # Object 1 placed on the opposite side of Person 0 (guaranteed d > 0.40)
                if cx_p0 < w_img * 0.5:
                    min_x = int(w_img * 0.65)
                    max_x = max(min_x + 1, int(w_img - w_oi * 0.5 - 10))
                    cx_oi = float(torch.randint(min_x, max_x, (1,), generator=rng).item())
                else:
                    max_x = int(w_img * 0.35)
                    min_x = min(int(w_oi * 0.5 + 10), max_x - 1)
                    cx_oi = float(torch.randint(min_x, max_x, (1,), generator=rng).item())
                cy_oi = float(torch.randint(int(h_oi * 0.5 + 10), max(int(h_oi * 0.5 + 11), int(h_img - h_oi * 0.5 - 10)), (1,), generator=rng).item())
            else:
                cx_oi = float(torch.randint(int(w_oi * 0.5 + 10), max(int(w_oi * 0.5 + 11), int(w_img - w_oi * 0.5 - 10)), (1,), generator=rng).item())
                cy_oi = float(torch.randint(int(h_oi * 0.5 + 10), max(int(h_oi * 0.5 + 11), int(h_img - h_oi * 0.5 - 10)), (1,), generator=rng).item())
            boxes_list.append([cx_oi - w_oi * 0.5, cy_oi - h_oi * 0.5, cx_oi + w_oi * 0.5, cy_oi + h_oi * 0.5])

        boxes = torch.tensor(boxes_list, dtype=torch.float32)
        scores = torch.empty((n,), dtype=torch.float32).uniform_(0.85, 1.0, generator=rng)

        # Candidate pairs and interaction targets
        pair_index, pair_attr = self.graph_builder.select_candidate_pairs(boxes, class_ids, self.image_size)
        y, y_mask = self.compute_targets(boxes, class_ids, pair_index, self.image_size)

        # Base ROI features from category embeddings + subtle noise
        roi_features = torch.zeros((n, self.feature_dim), dtype=torch.float32)
        for i in range(n):
            cls_id = int(class_ids[i].item())
            base_vec = self._class_embeddings[cls_id]
            noise = torch.randn((self.feature_dim,), generator=rng) * 0.05
            roi_features[i] = base_vec + noise

        # Inject interaction pattern into visual features of interacting pairs
        if pair_index.shape[0] > 0:
            for p_i in range(pair_index.shape[0]):
                h_i = int(pair_index[p_i, 0].item())
                o_i = int(pair_index[p_i, 1].item())
                active_actions = (y[p_i] > 0.5).nonzero(as_tuple=True)[0]
                for a_act in active_actions.tolist():
                    act_vec = self._action_embeddings[a_act] * 0.25
                    roi_features[h_i] = roi_features[h_i] + act_vec
                    roi_features[o_i] = roi_features[o_i] + act_vec

        roi_features = F.normalize(roi_features, dim=-1)

        img_features = ImageFeatures(
            image_id=str(image_id),
            image_size=self.image_size,
            boxes=boxes,
            class_ids=class_ids,
            scores=scores,
            roi_features=roi_features,
        )

        return img_features, y, y_mask

    def generate_image_features(
        self,
        image_id: Union[str, int],
        num_entities: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> ImageFeatures:
        """Generate ImageFeatures for a synthetic scene."""
        img_features, _, _ = self.generate_scene(image_id, num_entities, seed)
        return img_features

    def generate_graph(
        self,
        image_id: Union[str, int],
        num_entities: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> HOIData:
        """Directly construct an HOIData PyG graph using GraphBuilder."""
        img_features, y, y_mask = self.generate_scene(image_id, num_entities, seed)
        return self.graph_builder.build_graph(
            image_features=img_features,
            image_id=str(image_id),
            y=y,
            y_mask=y_mask,
        )

    def save_scene_to_cache(
        self,
        image_id: Union[str, int],
        cache_dir: Union[str, Path],
        tag: str = "synthetic",
        num_entities: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Path:
        """Serialize a synthetic scene to cache matching ImageFeatures.to_dict() format."""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        img_features, _, _ = self.generate_scene(image_id, num_entities, seed)

        file_name = f"{image_id}__{tag}.pt"
        out_file = cache_path / file_name
        torch.save(img_features.to_dict(), out_file)
        return out_file

    def save_dataset_to_cache(
        self,
        num_samples: int,
        cache_dir: Union[str, Path],
        tag: str = "synthetic",
    ) -> List[Path]:
        """Pre-populate a cache directory with synthetic scenes."""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        paths = []
        for idx in range(num_samples):
            p = self.save_scene_to_cache(idx, cache_path, tag=tag)
            paths.append(p)
        return paths


class SyntheticVCOCODataset(Dataset):
    """PyTorch Geometric Dataset for synthetic HOI graphs.

    Parameters:
        num_samples: Number of synthetic graph samples in dataset
        cache_dir: Cache directory for serialized feature .pt files
        seed: Deterministic random seed
        use_cache: If True, writes/reads ImageFeatures to/from disk cache.
                   If False, generates graphs directly in-memory.
        feature_dim: Node ROI feature dimension (default 56)
        graph_builder: Optional custom GraphBuilder instance
        generator: Optional custom SyntheticHOIGenerator instance
    """

    def __init__(
        self,
        num_samples: int = 100,
        cache_dir: Optional[Union[str, Path]] = "cache/features_synthetic",
        seed: int = 42,
        use_cache: bool = True,
        feature_dim: int = 56,
        graph_builder: Optional[GraphBuilder] = None,
        generator: Optional[SyntheticHOIGenerator] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(None, transform, pre_transform, pre_filter)
        self.num_samples = num_samples
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.seed = seed
        self.use_cache = use_cache
        self.feature_dim = feature_dim
        self.graph_builder = graph_builder or GraphBuilder()
        self.generator = generator or SyntheticHOIGenerator(
            feature_dim=self.feature_dim,
            seed=self.seed,
            graph_builder=self.graph_builder,
        )

        if self.use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def len(self) -> int:
        return self.num_samples

    def get(self, idx: int) -> HOIData:
        """Constructs and returns the HOIData graph for index idx."""
        image_id = idx

        if self.use_cache and self.cache_dir:
            cached_path = _find_cached_features(self.cache_dir, image_id)
            if cached_path is None or not cached_path.exists():
                cached_path = self.generator.save_scene_to_cache(
                    image_id=image_id,
                    cache_dir=self.cache_dir,
                    tag="synthetic",
                )

            data_dict = torch.load(cached_path, map_location="cpu", weights_only=False)
            feats = ImageFeatures.from_dict(data_dict)

            # Compute candidate pairs and targets deterministically
            pair_index, pair_attr = self.graph_builder.select_candidate_pairs(
                feats.boxes, feats.class_ids, feats.image_size
            )
            y, y_mask = self.generator.compute_targets(
                feats.boxes, feats.class_ids, pair_index, feats.image_size
            )

            graph = self.graph_builder.build_graph(
                image_features=feats,
                image_id=str(image_id),
                y=y,
                y_mask=y_mask,
            )
            return graph
        else:
            return self.generator.generate_graph(image_id=image_id)
