"""ROIAlign over multi-scale YOLO feature maps -> one visual embedding per box."""
from typing import List, Sequence, Tuple

import torch
from torchvision.ops import roi_align


class MultiScaleROIExtractor:
    """Pools every box from every feature level and concatenates the results.

    Output dim = sum of channels over levels (896 for yolov8s: 128+256+512).
    Boxes must be in letterboxed-input pixel coordinates.
    """

    def __init__(self, output_size: int = 7, sampling_ratio: int = 2):
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio

    @staticmethod
    def output_dim(channels: Sequence[int]) -> int:
        return int(sum(channels))

    @torch.no_grad()
    def __call__(self, feature_maps: List[torch.Tensor], boxes_letterbox: torch.Tensor,
                 input_hw: Tuple[int, int]) -> torch.Tensor:
        dim = self.output_dim([f.shape[1] for f in feature_maps])
        device = feature_maps[0].device
        n = boxes_letterbox.shape[0]
        if n == 0:
            return torch.zeros((0, dim), dtype=torch.float32, device=device)
        if boxes_letterbox.dim() != 2 or boxes_letterbox.shape[1] != 4:
            raise ValueError(f"boxes must be (N, 4), got {tuple(boxes_letterbox.shape)}")

        boxes = boxes_letterbox.to(device).float()
        rois = torch.cat([torch.zeros((n, 1), device=device), boxes], dim=1)
        input_w = input_hw[1]

        pooled_levels = []
        for fmap in feature_maps:
            stride = input_w / fmap.shape[-1]
            pooled = roi_align(fmap.float(), rois,
                               output_size=(self.output_size, self.output_size),
                               spatial_scale=1.0 / stride,
                               sampling_ratio=self.sampling_ratio, aligned=True)
            pooled_levels.append(pooled.mean(dim=(2, 3)))   # (N, C)
        feats = torch.cat(pooled_levels, dim=1)
        assert feats.shape == (n, dim), f"ROI feature shape {tuple(feats.shape)} != {(n, dim)}"
        return feats