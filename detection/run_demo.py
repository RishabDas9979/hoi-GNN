"""Run the Phase 1 pipeline on one image.

Usage (from project root):
    python -m detection.run_demo --image datasets/sample/bus.jpg
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from detection.detector import YOLODetector, read_image_bgr
from detection.feature_extractor import HOIFeatureExtractor
from detection.roi_extractor import MultiScaleROIExtractor
from utils.config import load_config
from visualization.detection_visualizer import draw_detections


def build_extractor(cfg) -> HOIFeatureExtractor:
    d = cfg["detection"]
    detector = YOLODetector(weights=d["weights"], device=d["device"], imgsz=d["imgsz"],
                        conf_thres=d["conf_thres"], iou_thres=d["iou_thres"],
                        max_det=d["max_det"],
                        max_persons=d.get("max_persons", 10),
                        max_objects=d.get("max_objects", 30))
    roi = MultiScaleROIExtractor(d["roi_output_size"], d["roi_sampling_ratio"])
    return HOIFeatureExtractor(detector, roi, cache_dir=cfg["paths"]["cache_dir"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    extractor = build_extractor(cfg)
    det = extractor.detector
    print(f"Device: {det.device} | weights: {det.weights} | "
          f"feature channels: {det.feature_channels} | ROI dim: {extractor.feature_dim}")

    t0 = time.time()
    feats = extractor.extract(args.image, use_cache=False)
    if det.device.type == "cuda":
        torch.cuda.synchronize()
    print(f"Uncached extraction: {time.time() - t0:.3f}s (first call includes warm-up)")

    t0 = time.time()
    feats = extractor.extract(args.image, use_cache=False)
    if det.device.type == "cuda":
        torch.cuda.synchronize()
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
    print(f"Warm extraction:     {time.time() - t0:.3f}s")

    print(f"\nImage size (H, W): {feats.image_size} | entities: {feats.num_entities}")
    print(f"roi_features shape: {tuple(feats.roi_features.shape)} | "
          f"finite: {bool(torch.isfinite(feats.roi_features).all())}")
    for i in range(feats.num_entities):
        cid = int(feats.class_ids[i])
        print(f"  [{i}] {det.names[cid]:<14} score={float(feats.scores[i]):.2f} "
              f"box={[round(v, 1) for v in feats.boxes[i].tolist()]}")

    # Cache round trip
    extractor.extract(args.image, use_cache=True)            # writes cache
    cached = extractor.extract(args.image, use_cache=True)   # reads cache
    max_diff = (cached.roi_features - feats.roi_features).abs().max().item() \
        if feats.num_entities else 0.0
    print(f"\nCache round-trip max abs diff (float16 storage): {max_diff:.5f}")

    out_dir = Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    img = read_image_bgr(args.image)
    vis = draw_detections(img, feats, det.names)
    out_path = out_dir / "phase1_demo.jpg"
    ok, buf = cv2.imencode(".jpg", vis)
    if ok:
        buf.tofile(str(out_path))
        print(f"Saved visualization: {out_path}")


if __name__ == "__main__":
    main()