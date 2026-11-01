import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from evaluation.metrics import (
    evaluate_all, mean_ap, measure_latency, per_action_ap,
    rare_common_map, threshold_metrics,
)


def test_perfect_scores_give_ap_one():
    y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    ap = per_action_ap(y, y.astype(float))
    assert np.allclose(ap, 1.0)
    assert mean_ap(ap) == pytest.approx(1.0)


def test_matches_sklearn_on_random_data():
    rng = np.random.default_rng(0)
    y = (rng.random((200, 5)) < 0.3).astype(int)
    s = rng.random((200, 5))
    ap = per_action_ap(y, s)
    for a in range(5):
        assert ap[a] == pytest.approx(average_precision_score(y[:, a], s[:, a]))


def test_action_without_positives_is_nan_and_excluded_from_map():
    y = np.array([[1, 0], [0, 0], [1, 0]])
    s = np.array([[0.9, 0.5], [0.1, 0.5], [0.8, 0.5]])
    ap = per_action_ap(y, s)
    assert not np.isnan(ap[0]) and np.isnan(ap[1])
    assert mean_ap(ap) == pytest.approx(ap[0])


def test_mask_ignores_invalid_labels():
    # Row 2 is a wrong-looking prediction but masked out, so AP stays perfect.
    y = np.array([[1], [0], [0]])
    s = np.array([[0.9], [0.1], [0.99]])
    mask = np.array([[1], [1], [0]])
    assert per_action_ap(y, s, mask)[0] == pytest.approx(1.0)
    assert per_action_ap(y, s)[0] < 1.0


def test_all_nan_map_is_nan():
    assert np.isnan(mean_ap(np.array([np.nan, np.nan])))


def test_threshold_metrics_hand_computed():
    # action 0: tp=1 fp=1 fn=0 ; action 1: tp=0 fp=0 fn=1
    y = np.array([[1, 1], [0, 0]])
    s = np.array([[0.9, 0.2], [0.8, 0.1]])
    m = threshold_metrics(y, s, threshold=0.5)
    assert m["precision_micro"] == pytest.approx(1 / 2)
    assert m["recall_micro"] == pytest.approx(1 / 2)
    assert m["f1_micro"] == pytest.approx(1 / 2)
    assert m["precision_macro"] == pytest.approx((0.5 + 0.0) / 2)
    assert m["recall_macro"] == pytest.approx((1.0 + 0.0) / 2)


def test_threshold_metrics_respects_mask():
    y = np.array([[1], [0]])
    s = np.array([[0.9], [0.9]])
    mask = np.array([[1], [0]])  # the false positive is on an ignored label
    assert threshold_metrics(y, s, mask)["precision_micro"] == pytest.approx(1.0)


def test_rare_common_split():
    ap = np.array([1.0, 0.5, 0.2, np.nan])
    counts = [100, 50, 3, 1]
    r = rare_common_map(ap, counts, rare_below=10)
    assert r["mAP_common"] == pytest.approx(0.75)
    assert r["mAP_rare"] == pytest.approx(0.2)
    assert (r["n_rare"], r["n_common"]) == (2, 2)


def test_rare_common_length_mismatch_raises():
    with pytest.raises(ValueError):
        rare_common_map(np.zeros(3), [1, 2])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        per_action_ap(np.zeros((3, 2)), np.zeros((3, 3)))


def test_zero_pairs_does_not_crash():
    y = np.zeros((0, 4))
    out = evaluate_all(y, y, train_pos_counts=[1, 2, 3, 4])
    assert np.isnan(out["mAP"])


def test_evaluate_all_keys():
    rng = np.random.default_rng(1)
    y = (rng.random((50, 3)) < 0.4).astype(int)
    out = evaluate_all(y, rng.random((50, 3)), train_pos_counts=[20, 5, 40])
    for k in ("mAP", "f1_micro", "f1_macro", "mAP_rare", "mAP_common", "per_action_ap"):
        assert k in out


def test_latency_returns_positive_stats():
    out = measure_latency(lambda: sum(range(1000)), n_warmup=1, n_runs=5)
    assert out["latency_ms_mean"] > 0 and out["latency_ms_p95"] >= 0
