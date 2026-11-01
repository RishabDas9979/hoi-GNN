"""Draw detection boxes on an image (OpenCV)."""
from typing import Dict

import cv2
import numpy as np

from detection.feature_extractor import ImageFeatures


def draw_detections(image_bgr: np.ndarray, feats: ImageFeatures,
                    names: Dict[int, str]) -> np.ndarray:
    out = image_bgr.copy()
    for box, cid, score in zip(feats.boxes.tolist(), feats.class_ids.tolist(),
                               feats.scores.tolist()):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = (0, 200, 0) if cid == 0 else (0, 140, 255)   # person = green
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(cid, cid)} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out