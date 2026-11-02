"""Phase 8 tests for the torch-free parts of the app: scene graph and results reading.
Run:  pytest tests/test_phase8.py -v
"""
import json

import pytest

from utils.results_io import (append_explanation_metrics, list_runs, load_explanation_metrics,
                              load_history)
from visualization.scene_graph import build_scene_graph, draw_scene_graph, interaction_table

ENTS = [
    {"idx": 0, "label": "person", "is_human": True, "score": 0.9},
    {"idx": 1, "label": "cell phone", "is_human": False, "score": 0.8},
    {"idx": 2, "label": "chair", "is_human": False, "score": 0.7},
]
INTER = [{"human": 0, "object": 1, "action": "hold", "prob": 0.9},
         {"human": 0, "object": 1, "action": "talk on phone", "prob": 0.6}]


def test_scene_graph_keeps_only_interacting_entities_by_default():
    g = build_scene_graph(ENTS, INTER)
    assert set(g.nodes) == {0, 1}
    assert g.number_of_edges() == 2
    assert set(build_scene_graph(ENTS, INTER, show_all_entities=True).nodes) == {0, 1, 2}


def test_scene_graph_unknown_entity_raises():
    with pytest.raises(ValueError):
        build_scene_graph(ENTS, [{"human": 0, "object": 9, "action": "hold", "prob": 0.5}])


def test_draw_scene_graph_handles_empty_and_normal():
    assert draw_scene_graph(build_scene_graph([], [])) is not None
    assert draw_scene_graph(build_scene_graph(ENTS, INTER)) is not None
    assert draw_scene_graph(build_scene_graph(ENTS, [], show_all_entities=True)) is not None


def test_interaction_table_sorted_by_confidence():
    rows = interaction_table(INTER[::-1], ENTS)
    assert [r["Relationship"] for r in rows] == ["hold", "talk on phone"]
    assert rows[0]["Human"] == "person #0" and rows[0]["Object"] == "cell phone #1"


def _write_run(root, name, data, mAP):
    d = root / name
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(
        {"model": "gcn", "data": data, "boxes": "gt", "seed": 0, "test": {"mAP": mAP}}))
    (d / "history.json").write_text(json.dumps([{"epoch": 1, "train_loss": 1.0}, {"epoch": 2, "train_loss": 0.5}]))
    (d / "best.pt").write_bytes(b"x")


def test_list_runs_flags_real_vs_non_real(tmp_path):
    _write_run(tmp_path, "gcn_gt_seed0", "vcoco", 0.31)
    _write_run(tmp_path, "ablations/depth/x", "synthetic", 0.99)
    runs = {r.name: r for r in list_runs(tmp_path)}
    assert runs["gcn_gt_seed0"].is_real and runs["gcn_gt_seed0"].checkpoint is not None
    assert not runs["ablations/depth/x"].is_real
    assert "NOT a benchmark" in runs["ablations/depth/x"].label
    assert len(load_history(runs["gcn_gt_seed0"].folder)) == 2


def test_list_runs_missing_dir_and_bad_json(tmp_path):
    assert list_runs(tmp_path / "nope") == []
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "metrics.json").write_text("{not json")
    assert list_runs(tmp_path) == []


def test_explanation_log_roundtrip(tmp_path):
    assert load_explanation_metrics(tmp_path).empty
    append_explanation_metrics(tmp_path, {"model": "gcn", "data": "vcoco", "fidelity_plus": 0.2})
    append_explanation_metrics(tmp_path, {"model": "gcn", "data": "vcoco", "fidelity_plus": 0.4})
    df = load_explanation_metrics(tmp_path)
    assert len(df) == 2 and abs(df["fidelity_plus"].mean() - 0.3) < 1e-9
