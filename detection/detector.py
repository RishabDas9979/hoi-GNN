"""YOLOv8 detector that also exposes backbone feature maps.

One forward pass yields both the detections and the three multi-scale feature
maps that feed YOLO's detection head. The detector is frozen (eval mode,
no gradients).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

from utils.config import get_device


# --------------------------------------------------------------------------- #
# Image loading and letterboxing
# --------------------------------------------------------------------------- #
def read_image_bgr(path) -> np.ndarray:
    """Read an image as BGR uint8. Works with non-ASCII Windows paths."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


@dataclass
class LetterboxMeta:
    """Geometry needed to move boxes between original and letterboxed space."""
    scale: float
    pad_left: int
    pad_top: int
    orig_hw: Tuple[int, int]
    input_hw: Tuple[int, int]

    def to_letterbox(self, boxes: torch.Tensor) -> torch.Tensor:
        b = boxes.clone().float()
        b[:, [0, 2]] = b[:, [0, 2]] * self.scale + self.pad_left
        b[:, [1, 3]] = b[:, [1, 3]] * self.scale + self.pad_top
        return b

    def to_original(self, boxes: torch.Tensor) -> torch.Tensor:
        b = boxes.clone().float()
        b[:, [0, 2]] = (b[:, [0, 2]] - self.pad_left) / self.scale
        b[:, [1, 3]] = (b[:, [1, 3]] - self.pad_top) / self.scale
        h, w = self.orig_hw
        b[:, [0, 2]] = b[:, [0, 2]].clamp(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clamp(0, h)
        return b


def letterbox(img_bgr: np.ndarray, target: int = 640, stride: int = 32,
              color: int = 114) -> Tuple[np.ndarray, LetterboxMeta]:
    """Resize keeping aspect ratio, then pad to a multiple of `stride`."""
    h, w = img_bgr.shape[:2]
    r = min(target / h, target / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = img_bgr
    if (nw, nh) != (w, h):
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = (stride - nw % stride) % stride
    pad_h = (stride - nh % stride) % stride
    left, top = pad_w // 2, pad_h // 2
    out = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                             cv2.BORDER_CONSTANT, value=(color, color, color))
    meta = LetterboxMeta(scale=r, pad_left=left, pad_top=top,
                         orig_hw=(h, w), input_hw=out.shape[:2])
    return out, meta


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
@dataclass
class DetectorOutput:
    boxes: torch.Tensor              # (N, 4) xyxy, original image pixels
    boxes_letterbox: torch.Tensor    # (N, 4) xyxy, letterboxed input pixels
    class_ids: torch.Tensor          # (N,) int64
    scores: torch.Tensor             # (N,) float32
    feature_maps: List[torch.Tensor]  # 3 maps, each (1, C, H/s, W/s)
    meta: LetterboxMeta


class YOLODetector:
    def __init__(self, weights: str = "yolov8s.pt", device: str = "auto",
                 imgsz: int = 640, conf_thres: float = 0.25,
                 iou_thres: float = 0.5, max_det: int = 100):
        from ultralytics import YOLO  # imported here so tests of other modules stay light

        self.device = get_device(device)
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.weights = weights

        yolo = YOLO(weights)
        self.net = yolo.model.to(self.device).float().eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.names: Dict[int, str] = dict(self.net.names)
        self.num_classes = len(self.names)

        self._captured: Optional[List[torch.Tensor]] = None
        detect_head = self.net.model[-1]
        detect_head.register_forward_pre_hook(self._capture_features)

        # Dummy pass to learn the channel count of each feature level.
        with torch.no_grad():
            self.net(torch.zeros(1, 3, imgsz, imgsz, device=self.device))
        if not self._captured:
            raise RuntimeError("Feature hook did not fire; unsupported YOLO model layout.")
        self.feature_channels: List[int] = [int(f.shape[1]) for f in self._captured]
        self._captured = None

    def _capture_features(self, module, inputs):
        feats = inputs[0]
        if isinstance(feats, (list, tuple)):
            self._captured = [f for f in feats]

    # -- core forward ------------------------------------------------------- #
    @torch.no_grad()
    def _forward(self, image_bgr: np.ndarray):
        img, meta = letterbox(image_bgr, self.imgsz)
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        x = x.to(self.device)
        out = self.net(x)
        preds = out[0] if isinstance(out, (list, tuple)) else out
        feats = self._captured
        self._captured = None
        if feats is None:
            raise RuntimeError("No feature maps captured.")
        return preds, feats, meta

    def feature_maps(self, image_bgr: np.ndarray):
        """Backbone feature maps only (used for ground-truth boxes)."""
        _, feats, meta = self._forward(image_bgr)
        return feats, meta

    @torch.no_grad()
    def detect(self, image_bgr: np.ndarray) -> DetectorOutput:
        preds, feats, meta = self._forward(image_bgr)
        if not torch.is_tensor(preds) or preds.dim() != 3 \
                or preds.shape[1] != 4 + self.num_classes:
            raise RuntimeError(
                f"Unexpected YOLO output {getattr(preds, 'shape', type(preds))}; "
                f"expected (1, {4 + self.num_classes}, anchors).")
        boxes_lb, scores, cls = self._postprocess(preds[0])
        return DetectorOutput(
            boxes=meta.to_original(boxes_lb), boxes_letterbox=boxes_lb,
            class_ids=cls, scores=scores, feature_maps=feats, meta=meta)

    def _postprocess(self, pred: torch.Tensor):
        """pred: (4 + nc, A) with xywh boxes (letterbox pixels) and class scores."""
        pred = pred.T                                   # (A, 4 + nc)
        xywh, cls_scores = pred[:, :4], pred[:, 4:]
        conf, cls = cls_scores.max(dim=1)
        keep = conf > self.conf_thres
        if not keep.any():
            return (torch.zeros((0, 4), device=pred.device),
                    torch.zeros((0,), device=pred.device),
                    torch.zeros((0,), dtype=torch.long, device=pred.device))
        xywh, conf, cls = xywh[keep], conf[keep], cls[keep]
        if conf.numel() > 3000:                         # cap candidates before NMS
            top = conf.topk(3000).indices
            xywh, conf, cls = xywh[top], conf[top], cls[top]
        xyxy = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2,
                          xywh[:, :2] + xywh[:, 2:] / 2], dim=1)
        idx = batched_nms(xyxy, conf, cls, self.iou_thres)[: self.max_det]
        return xyxy[idx], conf[idx], cls[idx].long()