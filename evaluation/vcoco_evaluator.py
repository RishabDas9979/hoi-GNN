"""Role-AP for HOI detection, following the V-COCO definition (Phase 5.6).

A detection is a triplet: (human box, action, object box | None) with a score.
It is a true positive if it matches a not-yet-matched ground-truth triplet of the
same (action, role) in the same image with human IoU >= t AND (if the role has an
object) object IoU >= t. Each GT can be matched once, so duplicates are false
positives. AP per (action, role) is the all-point interpolated area under the
precision-recall curve; the headline number is the mean over (action, role)
entries that have at least one GT.

Scenarios for GT triplets whose role has no object in this image:
  scenario=1 : the detection must ALSO have no object box (stricter)
  scenario=2 : the object box is ignored for those GT (more lenient)

DEVIATIONS / TO VERIFY: this is my reimplementation of the definition, not the
official script. Before reporting numbers as "V-COCO role AP", diff the matching
and AP code against https://github.com/s-gupta/v-coco (vsrl_eval.py) and list any
difference in the report. Until then call it "role AP (own implementation)".
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

Box = Sequence[float]  # xyxy


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area = lambda r: max(r[2] - r[0], 0.0) * max(r[3] - r[1], 0.0)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """All-point interpolated AP (area under the monotone precision envelope)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def role_ap(detections: List[dict], ground_truth: List[dict],
            iou_thresh: float = 0.5, scenario: int = 1) -> Dict:
    """detections: dicts with image_id, action, role, human_box, object_box|None, score.
    ground_truth: same without score. Returns per-(action, role) AP and the mean."""
    if scenario not in (1, 2):
        raise ValueError("scenario must be 1 or 2")

    gts = defaultdict(lambda: defaultdict(list))   # key -> image -> [gt dicts with 'matched']
    npos: Dict[tuple, int] = defaultdict(int)
    for g in ground_truth:
        key = (g["action"], g.get("role", "obj"))
        gts[key][g["image_id"]].append({**g, "matched": False})
        npos[key] += 1

    dets = defaultdict(list)
    for d in detections:
        dets[(d["action"], d.get("role", "obj"))].append(d)

    per_role: Dict[str, float] = {}
    for key, n in npos.items():
        ds = sorted(dets.get(key, []), key=lambda d: -d["score"])
        tp = np.zeros(len(ds))
        for i, d in enumerate(ds):
            best, best_score = None, -1.0
            for g in gts[key].get(d["image_id"], []):
                if g["matched"]:
                    continue
                h_iou = box_iou(d["human_box"], g["human_box"])
                if h_iou < iou_thresh:
                    continue
                if g["object_box"] is None:
                    if scenario == 1 and d["object_box"] is not None:
                        continue
                    o_iou = 1.0
                else:
                    if d["object_box"] is None:
                        continue
                    o_iou = box_iou(d["object_box"], g["object_box"])
                    if o_iou < iou_thresh:
                        continue
                if min(h_iou, o_iou) > best_score:
                    best, best_score = g, min(h_iou, o_iou)
            if best is not None:
                best["matched"] = True
                tp[i] = 1.0
        ctp = np.cumsum(tp)
        cfp = np.cumsum(1.0 - tp)
        rec = ctp / n if len(ds) else np.zeros(0)
        prec = ctp / np.maximum(ctp + cfp, 1e-12) if len(ds) else np.zeros(0)
        per_role[f"{key[0]}:{key[1]}"] = voc_ap(rec, prec) if len(ds) else 0.0

    vals = np.array(list(per_role.values()), dtype=float)
    return {"per_role_ap": per_role, "mean_role_ap": float(vals.mean()) if vals.size else float("nan"),
            "scenario": scenario, "iou_thresh": iou_thresh}


def detections_from_predictions(image_id, human_boxes, object_boxes, pair_index, probs,
                                human_scores, object_scores,
                                action_roles: Sequence[Optional[str]]) -> List[dict]:
    """Turn model output for ONE image into detection triplets.

    pair_index [2, P] indexes into a combined node list: first the humans, then the
    objects (human i -> i, object j -> len(humans) + j). probs [P, A].
    action_roles[a] is the role name ('obj' / 'instr') or None for agent-only
    actions, in which case the object box is None.
    Score = P(action) * human detection score * object detection score, the usual
    HOI convention.
    """
    n_h = len(human_boxes)
    out = []
    for k in range(probs.shape[0]):
        h, o = int(pair_index[0][k]), int(pair_index[1][k]) - n_h
        for a, role in enumerate(action_roles):
            out.append({
                "image_id": image_id, "action": a, "role": role or "agent",
                "human_box": list(human_boxes[h]),
                "object_box": None if role is None else list(object_boxes[o]),
                "score": float(probs[k, a]) * float(human_scores[h]) * float(object_scores[o]),
            })
    return out
