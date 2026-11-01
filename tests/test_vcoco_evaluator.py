import numpy as np
import pytest

from evaluation.vcoco_evaluator import (
    box_iou, detections_from_predictions, role_ap, voc_ap,
)

H = [0, 0, 10, 10]
O = [20, 20, 30, 30]
FAR = [100, 100, 110, 110]


def gt(img, action=0, h=H, o=O, role="obj"):
    return {"image_id": img, "action": action, "role": role, "human_box": h, "object_box": o}


def det(img, score, action=0, h=H, o=O, role="obj"):
    return {**gt(img, action, h, o, role), "score": score}


def test_box_iou_basic():
    assert box_iou(H, H) == pytest.approx(1.0)
    assert box_iou(H, FAR) == 0.0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_voc_ap_perfect():
    assert voc_ap(np.array([1.0]), np.array([1.0])) == pytest.approx(1.0)


def test_perfect_detection_gives_ap_one():
    r = role_ap([det(1, 0.9)], [gt(1)])
    assert r["mean_role_ap"] == pytest.approx(1.0)


def test_wrong_object_box_is_false_positive():
    r = role_ap([det(1, 0.9, o=FAR)], [gt(1)])
    assert r["mean_role_ap"] == pytest.approx(0.0)


def test_wrong_human_box_is_false_positive():
    r = role_ap([det(1, 0.9, h=FAR)], [gt(1)])
    assert r["mean_role_ap"] == pytest.approx(0.0)


def test_duplicate_detection_counts_as_false_positive():
    # high-score TP, then a duplicate on the same GT: recall 1.0 at precision 1.0 -> AP still 1
    r = role_ap([det(1, 0.9), det(1, 0.8)], [gt(1)])
    assert r["mean_role_ap"] == pytest.approx(1.0)
    # but a high-scoring false positive ranked first lowers AP to 0.5
    r = role_ap([det(1, 0.95, o=FAR), det(1, 0.9)], [gt(1)])
    assert r["mean_role_ap"] == pytest.approx(0.5)


def test_missed_gt_lowers_recall():
    r = role_ap([det(1, 0.9)], [gt(1), gt(2)])
    assert r["mean_role_ap"] == pytest.approx(0.5)


def test_no_detections_gives_zero_and_no_gt_is_excluded():
    assert role_ap([], [gt(1)])["mean_role_ap"] == 0.0
    assert np.isnan(role_ap([det(1, 0.9)], [])["mean_role_ap"])


def test_image_id_must_match():
    assert role_ap([det(2, 0.9)], [gt(1)])["mean_role_ap"] == 0.0


def test_scenarios_for_missing_object_role():
    g = [gt(1, o=None, role="agent")]
    d_with_obj = [det(1, 0.9, o=O, role="agent")]
    d_without = [det(1, 0.9, o=None, role="agent")]
    assert role_ap(d_with_obj, g, scenario=1)["mean_role_ap"] == 0.0   # strict
    assert role_ap(d_with_obj, g, scenario=2)["mean_role_ap"] == pytest.approx(1.0)  # lenient
    assert role_ap(d_without, g, scenario=1)["mean_role_ap"] == pytest.approx(1.0)


def test_roles_are_scored_separately():
    g = [gt(1, role="obj"), gt(1, role="instr")]
    d = [det(1, 0.9, role="obj")]
    r = role_ap(d, g)
    assert r["per_role_ap"]["0:obj"] == pytest.approx(1.0)
    assert r["per_role_ap"]["0:instr"] == 0.0
    assert r["mean_role_ap"] == pytest.approx(0.5)


def test_bad_scenario_raises():
    with pytest.raises(ValueError):
        role_ap([], [], scenario=3)


def test_detections_from_predictions_shapes_and_scores():
    probs = np.array([[0.5, 0.2]])
    out = detections_from_predictions(
        "img", [H], [O], np.array([[0], [1]]), probs, [0.9], [0.8], action_roles=["obj", None])
    assert len(out) == 2
    assert out[0]["score"] == pytest.approx(0.5 * 0.9 * 0.8)
    assert out[0]["object_box"] == O and out[1]["object_box"] is None
    assert out[1]["role"] == "agent"


def test_zero_pairs_gives_no_detections():
    out = detections_from_predictions("img", [H], [], np.zeros((2, 0), dtype=int),
                                      np.zeros((0, 2)), [0.9], [], ["obj", None])
    assert out == []
