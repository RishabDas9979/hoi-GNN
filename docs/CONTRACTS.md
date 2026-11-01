# Module Contracts

Any change to this file must be agreed by both team members and pushed first.

## 1. Feature cache (Phase 1 output, one file per image)

File: `cache/features/<image_id>__<tag>.pt`, loaded with `ImageFeatures.from_dict`.

| Field | Shape / dtype | Meaning |
|---|---|---|
| image_id | str | COCO image id or file stem |
| image_size | (H, W) | original image size |
| boxes | (N, 4) float32 | xyxy, original pixels |
| class_ids | (N,) int64 | COCO index 0-79, person = 0. -1 if unknown |
| scores | (N,) float32 | detection confidence (1.0 for GT boxes) |
| roi_features | (N, 896) float32 | raw YOLOv8s ROIAlign features, stored float16 on disk |

N may be 0. The projection to a model dimension is learned inside the model (Phase 4), not cached.
Detector-box and ground-truth-box caches live in separate folders:
`cache/features` (detector) and `cache/features_gt` (GT boxes).

## 2. Graph (Phase 2 output, PyG `Data`)

| Field | Shape / dtype | Meaning |
|---|---|---|
| x_visual | (V, 896) float | ROI features |
| class_ids | (V,) long | COCO class per node |
| boxes_norm | (V, 4) float | xyxy divided by image W, H, W, H |
| scores | (V,) float | detection confidence |
| node_type | (V,) long | 0 = human, 1 = object |
| edge_index | (2, E) long | human -> object edges, plus reverse edges when bidirectional |
| edge_attr | (E, 9) float | dx, dy, distance, IoU, area ratio, rel. center x, y, rel. width, height |
| pair_index | (P, 2) long | (human node, object node) pairs to classify, local to the graph |
| y | (P, A) float | multi-hot targets, A = 26 V-COCO actions |
| y_mask | (P, A) float | 1 where the label is valid for this pair, 0 otherwise |
| num_nodes | int | V |

Batching: `pair_index` must be offset by the node count of previous graphs, so Dataset/collate must override `__inc__` for `pair_index`.
Empty graphs (V = 0 or P = 0) must still hold valid zero-length tensors.

## 3. Model interface

`model(data) -> logits` of shape (P, A). Sigmoid is applied only at inference.
Loss: BCEWithLogits multiplied by `y_mask`.