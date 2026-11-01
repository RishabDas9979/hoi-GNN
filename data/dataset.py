"""PyG Dataset and Batch Collation for V-COCO Human-Object Interaction Graphs.

Combines cached/extracted ImageFeatures with V-COCO annotations via GraphBuilder
to produce batched HOIData instances.
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch_geometric.data import Batch, Dataset

from data.graph_builder import GraphBuilder, HOIData
from data.vcoco_loader import VCOCOLoader, VCOCOImageRecord
from detection.feature_extractor import HOIFeatureExtractor, ImageFeatures


def _find_cached_features(
    cache_dir: Optional[Path],
    image_id: Union[int, str],
    file_name: Optional[str] = None,
) -> Optional[Path]:
    """Search for cached feature .pt file by image id or image file name."""
    if cache_dir is None or not cache_dir.exists():
        return None

    str_id = str(image_id)
    try:
        int_id = int(image_id)
        padded_id = f"{int_id:012d}"
    except ValueError:
        int_id = None
        padded_id = str_id

    patterns = [
        f"{str_id}__*.pt",
        f"{str_id}.pt",
        f"{padded_id}__*.pt",
        f"{padded_id}.pt",
    ]
    if file_name:
        stem = Path(file_name).stem
        patterns.extend([f"{stem}__*.pt", f"{stem}.pt"])

    for pat in patterns:
        matches = list(cache_dir.glob(pat))
        if matches:
            return matches[0]

    return None


def hoi_collate(data_list: Sequence[HOIData]) -> Batch:
    """
    Collate a sequence of HOIData graphs into a single mini-batch.

    Verifies and guarantees that:
    1. Node features `x`, `boxes`, `class_ids`, `scores` are stacked along dim 0.
    2. `edge_index` has batch node offsets applied.
    3. `pair_index` has batch node offsets applied according to graph node counts.
    4. `pair_attr`, `y`, `y_mask` are stacked along the pair dimension.
    5. Adds `batch.pair_batch`: [total_P] tensor assigning each pair to its graph index.
    """
    batch = Batch.from_data_list(data_list)

    # Ensure pair_batch is present and valid
    if "pair_batch" not in batch or batch["pair_batch"] is None:
        device = (
            batch.edge_index.device
            if (hasattr(batch, "edge_index") and batch.edge_index is not None)
            else torch.device("cpu")
        )
        pair_counts = [
            int(d.pair_index.shape[0])
            if hasattr(d, "pair_index") and d.pair_index is not None
            else 0
            for d in data_list
        ]
        pair_batch_list = []
        for g_idx, count in enumerate(pair_counts):
            if count > 0:
                pair_batch_list.append(torch.full((count,), g_idx, dtype=torch.long, device=device))

        if pair_batch_list:
            batch["pair_batch"] = torch.cat(pair_batch_list, dim=0)
        else:
            batch["pair_batch"] = torch.zeros((0,), dtype=torch.long, device=device)

    return batch


class VCOCODataset(Dataset):
    """
    PyTorch Geometric Dataset for V-COCO Human-Object Interaction Detection.

    Reads cached ImageFeatures (or extracts on-the-fly), applies GraphBuilder,
    and pairs candidate human-object pairs with V-COCO interaction targets.
    """

    def __init__(
        self,
        split: str = "train",
        data_dir: Union[str, Path] = "data",
        cache_dir: Optional[Union[str, Path]] = "cache/features",
        image_dir: Optional[Union[str, Path]] = "datasets/coco/images",
        feature_extractor: Optional[HOIFeatureExtractor] = None,
        graph_builder: Optional[GraphBuilder] = None,
        vcoco_loader: Optional[VCOCOLoader] = None,
        max_images: Optional[int] = None,
        use_gt_boxes: bool = False,
        fallback_to_gt_if_uncached: bool = True,
        feature_dim: int = 56,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        instances_path: Optional[Union[str, Path]] = None,
        vcoco_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__(None, transform, pre_transform, pre_filter)

        self.split = split
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.feature_extractor = feature_extractor
        self.graph_builder = graph_builder or GraphBuilder()
        self.use_gt_boxes = use_gt_boxes
        self.fallback_to_gt_if_uncached = fallback_to_gt_if_uncached
        self.feature_dim = feature_dim

        # Initialize or reuse VCOCOLoader
        if vcoco_loader is not None:
            self.loader = vcoco_loader
        else:
            self.loader = VCOCOLoader(
                data_dir=self.data_dir,
                split=self.split,
                max_images=max_images,
                instances_path=instances_path,
                vcoco_path=vcoco_path,
            )

        self.image_ids = self.loader.get_image_ids()

    def len(self) -> int:
        return len(self.image_ids)

    def _resolve_image_path(self, record: VCOCOImageRecord) -> Optional[Path]:
        """Find local image file in image_dir if it exists."""
        if not self.image_dir or not self.image_dir.exists():
            return None

        # Check by file_name
        candidate = self.image_dir / record.file_name
        if candidate.exists():
            return candidate

        # Try train2014 or val2014 pattern
        img_id = record.image_id
        for prefix in ("COCO_train2014_", "COCO_val2014_"):
            cand = self.image_dir / f"{prefix}{img_id:012d}.jpg"
            if cand.exists():
                return cand

        return None

    def _get_image_features(self, record: VCOCOImageRecord) -> ImageFeatures:
        """Obtain ImageFeatures via cache, on-the-fly extraction, or GT fallback."""
        img_id = record.image_id
        cached_path = _find_cached_features(self.cache_dir, img_id, record.file_name)

        if cached_path is not None and cached_path.exists():
            data_dict = torch.load(cached_path, map_location="cpu", weights_only=False)
            return ImageFeatures.from_dict(data_dict)

        # On-the-fly extraction if extractor and image file exist
        img_path = self._resolve_image_path(record)
        if self.feature_extractor is not None and img_path is not None and img_path.exists():
            if self.use_gt_boxes and record.num_entities > 0:
                return self.feature_extractor.extract_for_boxes(
                    img_path,
                    boxes=record.boxes,
                    class_ids=record.class_ids,
                    image_id=str(img_id),
                )
            else:
                return self.feature_extractor.extract(
                    img_path,
                    image_id=str(img_id),
                    use_cache=True,
                )

        # Fallback to GT boxes with dummy features if uncached
        if self.fallback_to_gt_if_uncached:
            n = record.num_entities
            boxes = record.boxes if n > 0 else torch.zeros((0, 4), dtype=torch.float32)
            class_ids = record.class_ids if n > 0 else torch.zeros((0,), dtype=torch.long)
            scores = torch.ones((n,), dtype=torch.float32)
            roi_features = torch.zeros((n, self.feature_dim), dtype=torch.float32)
            h, w = record.image_size
            if h == 0 or w == 0:
                h, w = (640, 640)
            return ImageFeatures(
                image_id=str(img_id),
                image_size=(h, w),
                boxes=boxes,
                class_ids=class_ids,
                scores=scores,
                roi_features=roi_features,
            )

        raise FileNotFoundError(
            f"ImageFeatures for image_id {img_id} not found in cache '{self.cache_dir}' "
            f"and image file '{record.file_name}' not available for on-the-fly extraction."
        )

    def get(self, idx: int) -> HOIData:
        """
        Builds and returns the HOIData graph for image index `idx`.
        """
        img_id = self.image_ids[idx]
        record = self.loader.get_record(img_id)

        # 1. Obtain node features and boxes
        if self.use_gt_boxes:
            n = record.num_entities
            boxes = record.boxes
            class_ids = record.class_ids
            scores = torch.ones((n,), dtype=torch.float32)
            h, w = record.image_size
            if h == 0 or w == 0:
                h, w = (640, 640)
            image_size = (h, w)
            roi_features = torch.zeros((n, self.feature_dim), dtype=torch.float32)

            # Check if cached features can supply ROI features or extraction
            img_path = self._resolve_image_path(record)
            if self.feature_extractor is not None and img_path is not None and n > 0:
                feats = self.feature_extractor.extract_for_boxes(
                    img_path, boxes=boxes, class_ids=class_ids, image_id=str(img_id)
                )
                roi_features = feats.roi_features
            else:
                cached_path = _find_cached_features(self.cache_dir, img_id, record.file_name)
                if cached_path is not None:
                    c_feats = ImageFeatures.from_dict(
                        torch.load(cached_path, map_location="cpu", weights_only=False)
                    )
                    # If cached features have matching entity count
                    if c_feats.num_entities == n:
                        roi_features = c_feats.roi_features
        else:
            feats = self._get_image_features(record)
            boxes = feats.boxes
            class_ids = feats.class_ids
            scores = feats.scores
            roi_features = feats.roi_features
            image_size = feats.image_size

        # 2. Select candidate pairs and compute spatial features
        pair_index, pair_attr = self.graph_builder.select_candidate_pairs(
            boxes, class_ids, image_size
        )

        # 3. Compute ground truth targets (y, y_mask) for the candidate pairs
        ann_ids = record.ann_ids if (self.use_gt_boxes or boxes.shape[0] == record.num_entities) else None
        y, y_mask = self.loader.get_ground_truth_for_pairs(
            image_id=img_id,
            boxes=boxes,
            class_ids=class_ids,
            pair_index=pair_index,
            ann_ids=ann_ids,
        )

        # 4. Construct HOIData graph
        graph = self.graph_builder.build_graph(
            boxes=boxes,
            class_ids=class_ids,
            roi_features=roi_features,
            image_size=image_size,
            image_id=img_id,
            scores=scores,
            y=y,
            y_mask=y_mask,
        )

        return graph
