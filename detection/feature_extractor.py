"""High-level API: image -> ImageFeatures (boxes, classes, scores, ROI embeddings).

This is the contract consumed by data/graph_builder.py in Phase 2.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from detection.detector import YOLODetector, read_image_bgr
from detection.roi_extractor import MultiScaleROIExtractor


@dataclass
class ImageFeatures:
    image_id: str
    image_size: Tuple[int, int]      # (H, W) of the original image
    boxes: torch.Tensor              # (N, 4) float32, xyxy, original pixels
    class_ids: torch.Tensor          # (N,) int64, COCO index (person = 0)
    scores: torch.Tensor             # (N,) float32
    roi_features: torch.Tensor       # (N, D) float32

    @property
    def num_entities(self) -> int:
        return int(self.boxes.shape[0])

    def validate(self) -> None:
        n = self.num_entities
        assert self.boxes.shape == (n, 4), self.boxes.shape
        assert self.class_ids.shape == (n,), self.class_ids.shape
        assert self.scores.shape == (n,), self.scores.shape
        assert self.roi_features.shape[0] == n, self.roi_features.shape

    def to_dict(self) -> Dict:
        return {
            "image_id": self.image_id,
            "image_size": tuple(self.image_size),
            "boxes": self.boxes.cpu(),
            "class_ids": self.class_ids.cpu(),
            "scores": self.scores.cpu(),
            "roi_features": self.roi_features.cpu().half(),   # float16 on disk
        }

    @staticmethod
    def from_dict(d: Dict) -> "ImageFeatures":
        return ImageFeatures(
            image_id=d["image_id"], image_size=tuple(d["image_size"]),
            boxes=d["boxes"].float(), class_ids=d["class_ids"].long(),
            scores=d["scores"].float(), roi_features=d["roi_features"].float())


class HOIFeatureExtractor:
    def __init__(self, detector: YOLODetector,
                 roi_extractor: Optional[MultiScaleROIExtractor] = None,
                 cache_dir: Optional[str] = None):
        self.detector = detector
        self.roi = roi_extractor or MultiScaleROIExtractor()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def feature_dim(self) -> int:
        return self.roi.output_dim(self.detector.feature_channels)

    def _cache_path(self, image_id: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        d = self.detector
        tag = (f"{Path(d.weights).stem}_i{d.imgsz}_c{d.conf_thres:g}"
               f"_n{d.iou_thres:g}_m{d.max_det}_p{d.max_persons}"
               f"_o{d.max_objects}_r{self.roi.output_size}")
        return self.cache_dir / f"{image_id}__{tag}.pt"

    # -- detector-generated boxes ------------------------------------------ #
    def extract(self, image_path, image_id: Optional[str] = None,
                use_cache: bool = True) -> ImageFeatures:
        image_path = Path(image_path)
        image_id = image_id or image_path.stem
        cache_file = self._cache_path(image_id)
        if use_cache and cache_file is not None and cache_file.exists():
            return ImageFeatures.from_dict(
                torch.load(cache_file, map_location="cpu", weights_only=False))

        image = read_image_bgr(image_path)
        out = self.detector.detect(image)
        roi = self.roi(out.feature_maps, out.boxes_letterbox, out.meta.input_hw)
        feats = ImageFeatures(
            image_id=image_id, image_size=tuple(out.meta.orig_hw),
            boxes=out.boxes.cpu().float(), class_ids=out.class_ids.cpu().long(),
            scores=out.scores.cpu().float(), roi_features=roi.cpu().float())
        feats.validate()
        if cache_file is not None:
            torch.save(feats.to_dict(), cache_file)
        return feats

    def extract_many(self, items, desc: str = "Extracting"):
        """items: iterable of (image_path, image_id). Skips cached images.
        Returns (num_ok, num_failed); failures are printed, not fatal."""
        from tqdm import tqdm
        ok = failed = 0
        for path, image_id in tqdm(list(items), desc=desc):
            try:
                self.extract(path, image_id, use_cache=True)
                ok += 1
            except Exception as e:  # unreadable image etc.
                failed += 1
                print(f"[skip] {image_id}: {e}")
        return ok, failed

    # -- externally supplied boxes (V-COCO ground-truth boxes, Phase 2) ----- #
    def extract_for_boxes(self, image_path, boxes: torch.Tensor,
                          class_ids: Optional[torch.Tensor] = None,
                          scores: Optional[torch.Tensor] = None,
                          image_id: Optional[str] = None) -> ImageFeatures:
        """ROI features for given boxes (xyxy, original pixels). Not cached."""
        image_path = Path(image_path)
        image_id = image_id or image_path.stem
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        n = boxes.shape[0]
        image = read_image_bgr(image_path)
        fmaps, meta = self.detector.feature_maps(image)
        roi = self.roi(fmaps, meta.to_letterbox(boxes).to(fmaps[0].device), meta.input_hw)
        feats = ImageFeatures(
            image_id=image_id, image_size=tuple(meta.orig_hw), boxes=boxes,
            class_ids=(torch.as_tensor(class_ids).long() if class_ids is not None
                       else torch.full((n,), -1, dtype=torch.long)),
            scores=(torch.as_tensor(scores).float() if scores is not None
                    else torch.ones(n)),
            roi_features=roi.cpu().float())
        feats.validate()
        return feats