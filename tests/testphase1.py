"""Phase 1 tests. Run from project root:  pytest tests/test_phase1.py -v"""
from pathlib import Path

import numpy as np
import pytest
import torch

from detection.detector import letterbox
from detection.roi_extractor import MultiScaleROIExtractor

SAMPLE = Path("datasets/sample/bus.jpg")


def test_letterbox_box_roundtrip():
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out, meta = letterbox(img, 640)
    assert out.shape[0] % 32 == 0 and out.shape[1] % 32 == 0
    boxes = torch.tensor([[10.0, 20.0, 200.0, 300.0], [0.0, 0.0, 639.0, 479.0]])
    back = meta.to_original(meta.to_letterbox(boxes))
    assert torch.allclose(back, boxes, atol=1e-3)


def test_roi_extractor_shapes_and_values():
    maps = [torch.full((1, 8, 80, 80), 3.0), torch.full((1, 16, 40, 40), 3.0),
            torch.full((1, 32, 20, 20), 3.0)]
    roi = MultiScaleROIExtractor(output_size=7)
    boxes = torch.tensor([[50.0, 60.0, 300.0, 400.0], [0.0, 0.0, 100.0, 100.0]])
    out = roi(maps, boxes, (640, 640))
    assert out.shape == (2, 8 + 16 + 32)
    assert torch.allclose(out, torch.full_like(out, 3.0), atol=1e-4)


def test_roi_extractor_empty_boxes():
    maps = [torch.zeros(1, 8, 80, 80), torch.zeros(1, 16, 40, 40), torch.zeros(1, 32, 20, 20)]
    out = MultiScaleROIExtractor()(maps, torch.zeros((0, 4)), (640, 640))
    assert out.shape == (0, 56)


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not downloaded")
def test_end_to_end_on_sample_image():
    from detection.run_demo import build_extractor
    from utils.config import load_config
    ex = build_extractor(load_config())
    feats = ex.extract(SAMPLE, use_cache=False)
    feats.validate()
    assert feats.roi_features.shape[1] == ex.feature_dim
    assert feats.num_entities > 0
    assert (feats.class_ids == 0).any(), "expected at least one person in bus.jpg"
    # ground-truth-box mode returns one embedding per supplied box
    gt = ex.extract_for_boxes(SAMPLE, feats.boxes[:2])
    assert gt.roi_features.shape == (min(2, feats.num_entities), ex.feature_dim)