import json

import pytest

from evaluation.aggregate import build_summary, collect_runs, plot_comparison, plot_curves, to_markdown


def write_run(root, model, seed, mAP, boxes="gt", epochs=3):
    d = root / f"{model}_{boxes}_seed{seed}"
    d.mkdir(parents=True)
    test = {"mAP": mAP, "mAP_rare": mAP - 0.1, "mAP_common": mAP + 0.1, "f1_micro": 0.5, "f1_macro": 0.4}
    (d / "metrics.json").write_text(json.dumps({
        "model": model, "seed": seed, "boxes": boxes, "data": "fake", "n_params": 1234,
        "latency_batch_ms": {"latency_ms_mean": 2.0}, "test": test, "val": test}))
    hist = [{"epoch": e + 1, "train_loss": 1.0 / (e + 1), "val_loss": 1.1 / (e + 1), "val_mAP": 0.1 * (e + 1)}
            for e in range(epochs)]
    (d / "history.json").write_text(json.dumps(hist))


def test_missing_models_are_marked_not_run(tmp_path):
    write_run(tmp_path, "gcn", 0, 0.4)
    s = build_summary(collect_runs(tmp_path))
    assert s.loc[s["model"] == "gat", "mAP"].iloc[0] == "not run"
    assert s.loc[s["model"] == "gcn", "mAP"].iloc[0] == "0.400"


def test_multi_seed_mean_and_std(tmp_path):
    write_run(tmp_path, "gcn", 0, 0.4)
    write_run(tmp_path, "gcn", 1, 0.6)
    s = build_summary(collect_runs(tmp_path))
    cell = s.loc[s["model"] == "gcn", "mAP"].iloc[0]
    assert cell.startswith("0.500 ±")


def test_boxes_are_kept_separate(tmp_path):
    write_run(tmp_path, "gcn", 0, 0.4, boxes="gt")
    s = build_summary(collect_runs(tmp_path), boxes="detected")
    assert s.loc[s["model"] == "gcn", "mAP"].iloc[0] == "not run"


def test_empty_results_dir_does_not_crash(tmp_path):
    s = build_summary(collect_runs(tmp_path))
    assert (s["mAP"] == "not run").all()
    assert "not run" in to_markdown(s)


def test_plots_are_written(tmp_path):
    write_run(tmp_path, "gcn", 0, 0.4)
    assert len(plot_curves(tmp_path, tmp_path / "plots")) == 1
    assert plot_comparison(collect_runs(tmp_path), tmp_path / "cmp.png").exists()
