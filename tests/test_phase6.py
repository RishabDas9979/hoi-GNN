"""Tests for Phase 6 Ablation Infrastructure and Model/Data Integration.

Verifies:
6.1 Topology: GraphBuilder configuration in data hooks (bidirectional, candidate_mode, knn_k, distance_threshold).
6.2 Features: feature_subset propagation across models.
6.3 Architecture: build_model across all registry models without TypeErrors.
6.4 Depth: num_layers propagation across models.
6.5 Edge attributes: use_edge_attr and edge_dim in EdgeAwareGATv2HOIModel and InteractionHead.
"""
import pytest
import torch
from torch_geometric.data import Batch

from data.graph_builder import HOIData
from data.synthetic_dataset import SyntheticHOIGenerator
from models import build_model, MODEL_REGISTRY
from models.edge_aware_gatv2 import EdgeAwareGATv2HOIModel
from models.gat import GATHOIModel
from models.gcn import GCNHOIModel
from models.mlp_baseline import MLPBaseline
from models.prior_baseline import ObjectClassPriorBaseline
from training.data_hooks import _real_datasets


@pytest.fixture
def sample_graph():
    generator = SyntheticHOIGenerator(feature_dim=56, seed=42)
    return generator.generate_graph(image_id=101)


# ============================================================================
# 1. EdgeAwareGATv2HOIModel edge attribute ablation tests
# ============================================================================

def test_edge_aware_gatv2_use_edge_attr_false(sample_graph):
    """When use_edge_attr=False:
    - GATv2Conv layers created with edge_dim=None
    - forward() executes without error even when edge_attr is present on graph
    - return_attention=True works without edge_attr
    - InteractionHead has use_pair_attr=False
    """
    model = EdgeAwareGATv2HOIModel(
        input_dim=56,
        hidden_dim=64,
        num_layers=2,
        heads=2,
        edge_dim=9,
        use_edge_attr=False,
        num_actions=26,
    )

    assert model.use_edge_attr is False
    assert model.edge_dim == 0
    assert model.interaction_head.use_pair_attr is False
    for conv in model.convs:
        assert conv.edge_dim is None

    p = sample_graph.pair_index.shape[0]

    # Forward pass
    logits = model(sample_graph)
    assert logits.shape == (p, 26)
    assert torch.isfinite(logits).all()

    # Attention extraction forward pass
    logits_att, att_list = model(sample_graph, return_attention=True)
    assert logits_att.shape == (p, 26)
    assert len(att_list) == 2
    for edge_idx, alpha in att_list:
        assert alpha.shape[-1] == 2  # heads


def test_edge_aware_gatv2_edge_dim_none_or_zero(sample_graph):
    """When edge_dim is None or 0:
    - GATv2Conv layers created with edge_dim=None
    - use_edge_attr is False
    - InteractionHead has use_pair_attr=False
    """
    p = sample_graph.pair_index.shape[0]

    for edge_dim in (None, 0):
        model = EdgeAwareGATv2HOIModel(
            input_dim=56,
            hidden_dim=64,
            num_layers=2,
            heads=2,
            edge_dim=edge_dim,
            use_edge_attr=True,
            num_actions=26,
        )
        assert model.use_edge_attr is False
        assert model.edge_dim == 0
        assert model.interaction_head.use_pair_attr is False
        for conv in model.convs:
            assert conv.edge_dim is None

        logits = model(sample_graph)
        assert logits.shape == (p, 26)
        assert torch.isfinite(logits).all()


def test_edge_aware_gatv2_from_config_parses_use_edge_attr():
    """Verify from_config parses boolean and string values for use_edge_attr."""
    # Boolean False
    cfg_bool_false = {"model": {"hidden_dim": 64, "heads": 2, "use_edge_attr": False}}
    m1 = EdgeAwareGATv2HOIModel.from_config(cfg_bool_false)
    assert m1.use_edge_attr is False
    assert m1.interaction_head.use_pair_attr is False

    # String "false"
    cfg_str_false = {"model": {"hidden_dim": 64, "heads": 2, "use_edge_attr": "false"}}
    m2 = EdgeAwareGATv2HOIModel.from_config(cfg_str_false)
    assert m2.use_edge_attr is False
    assert m2.interaction_head.use_pair_attr is False

    # Default True
    cfg_default = {"model": {"hidden_dim": 64, "heads": 2}}
    m3 = EdgeAwareGATv2HOIModel.from_config(cfg_default)
    assert m3.use_edge_attr is True
    assert m3.interaction_head.use_pair_attr is True

    # String "true"
    cfg_str_true = {"model": {"hidden_dim": 64, "heads": 2, "use_edge_attr": "true"}}
    m4 = EdgeAwareGATv2HOIModel.from_config(cfg_str_true)
    assert m4.use_edge_attr is True
    assert m4.interaction_head.use_pair_attr is True


# ============================================================================
# 2. build_model ablation dimensions and override tests
# ============================================================================

@pytest.mark.parametrize("model_name", ["prior", "mlp", "gcn", "gat", "edge_aware_gatv2"])
def test_build_model_with_ablation_config_grid(model_name, sample_graph):
    """Test build_model across all models with full ablation config parameters without TypeError."""
    cfg = {
        "model": {
            "feature_subset": "spatial",
            "num_layers": 3,
            "hidden_dim": 64,
            "heads": 2,
            "edge_dim": 9,
            "use_edge_attr": True,
            "dropout": 0.2,
        },
        "runtime": {
            "node_dim": 56,
            "num_actions": 26,
            "edge_dim": 9,
        },
    }

    model = build_model(model_name, cfg)
    assert model is not None

    if model_name != "prior":
        assert model.node_encoder.feature_subset == "spatial"
        assert model.interaction_head.hidden_dim == 64

    if model_name in ("gcn", "gat", "edge_aware_gatv2"):
        assert model.num_layers == 3

    if model_name in ("gat", "edge_aware_gatv2"):
        assert model.heads == 2

    # Forward pass works
    p = sample_graph.pair_index.shape[0]
    logits = model(sample_graph)
    assert logits.shape == (p, 26)


def test_build_model_edge_attr_override_not_dropped():
    """Verify cfg['model']['use_edge_attr'] = False and edge_dim = 0 are NOT overwritten by runtime['edge_dim']."""
    cfg = {
        "model": {
            "hidden_dim": 64,
            "heads": 2,
            "use_edge_attr": False,
            "edge_dim": 0,
        },
        "runtime": {
            "node_dim": 56,
            "num_actions": 26,
            "edge_dim": 9,  # runtime carries 9 from dataset
        },
    }

    model = build_model("edge_aware_gatv2", cfg)
    assert model.use_edge_attr is False
    assert model.edge_dim == 0
    assert model.interaction_head.use_pair_attr is False
    for conv in model.convs:
        assert conv.edge_dim is None


def test_build_model_direct_kwargs_override():
    """Direct kwargs to build_model override cfg['model'] values."""
    cfg = {
        "model": {
            "hidden_dim": 64,
            "heads": 2,
            "num_layers": 2,
            "feature_subset": "all",
        },
        "runtime": {
            "node_dim": 56,
            "num_actions": 26,
            "edge_dim": 9,
        },
    }

    model = build_model(
        "edge_aware_gatv2",
        cfg,
        num_layers=1,
        feature_subset="appearance",
        use_edge_attr=False,
    )
    assert model.num_layers == 1
    assert model.node_encoder.feature_subset == "appearance"
    assert model.use_edge_attr is False


# ============================================================================
# 3. training.data_hooks._real_datasets GraphBuilder integration
# ============================================================================

def test_data_hooks_graph_builder_instantiation():
    """Verify _real_datasets extracts graph_cfg and model_section correctly and passes graph_builder."""
    model_cfg = {
        "graph": {
            "candidate_mode": "knn",
            "knn_k": 4,
            "distance_threshold": 0.35,
        },
        "model": {
            "input_dim": 56,
            "bidirectional": True,
        },
    }

    train_ds, val_ds, test_ds = _real_datasets("synthetic", "gt", model_cfg)

    for ds in (train_ds, val_ds, test_ds):
        gb = ds.graph_builder
        assert gb.candidate_mode == "knn"
        assert gb.k_neighbors == 4
        assert gb.dist_threshold == 0.35
        assert gb.bidirectional is True

    # Check a generated graph respects graph_builder settings
    sample = train_ds[0]
    assert isinstance(sample, HOIData)



# ============================================================================
# 4. evaluation.ablation Engine & CLI tests
# ============================================================================

import json
import pandas as pd
from evaluation.ablation import (
    ABLATION_TYPES,
    METRIC_COLUMNS,
    build_ablation_configs,
    collect_ablation_results,
    plot_ablation_results,
    run_ablation_grid,
    to_markdown,
    _parse_seeds,
    main,
)


def test_parse_seeds():
    assert _parse_seeds(42) == [42]
    assert _parse_seeds("42") == [42]
    assert _parse_seeds("42, 43, 44") == [42, 43, 44]
    assert _parse_seeds([42, "43", 44]) == [42, 43, 44]
    assert _parse_seeds(["42,43", "44"]) == [42, 43, 44]


def test_build_ablation_configs_invalid():
    with pytest.raises(ValueError, match="Unknown ablation type"):
        build_ablation_configs("unknown_grid")


def test_build_ablation_configs_topology():
    cfgs = build_ablation_configs("topology", seeds=[42, 43])
    # 4 experiments x 2 seeds = 8 configs
    assert len(cfgs) == 8
    exp_names = {c["exp_name"] for c in cfgs}
    assert exp_names == {"baseline", "directed", "knn", "distance_threshold"}

    baseline = [c for c in cfgs if c["exp_name"] == "baseline"][0]
    assert baseline["model_cfg"]["model"]["bidirectional"] is True
    assert baseline["model_cfg"]["graph"]["candidate_mode"] == "all_human_object"

    directed = [c for c in cfgs if c["exp_name"] == "directed"][0]
    assert directed["model_cfg"]["model"]["bidirectional"] is False
    assert directed["model_cfg"]["graph"]["candidate_mode"] == "all_human_object"

    knn = [c for c in cfgs if c["exp_name"] == "knn"][0]
    assert knn["model_cfg"]["graph"]["candidate_mode"] == "knn"
    assert knn["model_cfg"]["graph"]["knn_k"] == 5

    dist = [c for c in cfgs if c["exp_name"] == "distance_threshold"][0]
    assert dist["model_cfg"]["graph"]["candidate_mode"] == "distance_threshold"
    assert dist["model_cfg"]["graph"]["dist_threshold"] == 0.5


def test_build_ablation_configs_features():
    cfgs = build_ablation_configs("features", seeds=[42])
    assert len(cfgs) == 4
    subsets = {c["model_cfg"]["model"]["feature_subset"] for c in cfgs}
    assert subsets == {"all", "spatial", "appearance", "semantic"}


def test_build_ablation_configs_architecture():
    cfgs = build_ablation_configs("architecture", seeds=[42])
    assert len(cfgs) == 4
    models = {c["model_name"] for c in cfgs}
    assert models == {"mlp", "gcn", "gat", "edge_aware_gatv2"}


def test_build_ablation_configs_depth():
    cfgs = build_ablation_configs("depth", seeds=[42])
    assert len(cfgs) == 3
    depths = {c["model_cfg"]["model"]["num_layers"] for c in cfgs}
    assert depths == {1, 2, 3}


def test_build_ablation_configs_edge_attr():
    cfgs = build_ablation_configs("edge_attr", seeds=[42])
    assert len(cfgs) == 2
    attr_settings = {c["model_cfg"]["model"]["use_edge_attr"] for c in cfgs}
    assert attr_settings == {True, False}


def test_build_ablation_configs_all():
    cfgs = build_ablation_configs("all", seeds=[42])
    # 4 (topology) + 4 (features) + 4 (architecture) + 3 (depth) + 2 (edge_attr) = 17
    assert len(cfgs) == 17
    types = {c["ablation_type"] for c in cfgs}
    assert types == {"topology", "features", "architecture", "depth", "edge_attr"}


def test_dry_run_returns_planned_paths(tmp_path, capsys):
    cfgs = build_ablation_configs("depth", seeds=[42], epochs=5)
    paths = run_ablation_grid(cfgs, out_dir=tmp_path, dry_run=True)
    assert len(paths) == 3
    captured = capsys.readouterr().out
    assert "Phase 6 Ablation Experiment Plan (Dry-Run)" in captured
    assert "depth_1_seed42" in str(paths[0])
    # None of the folders should actually be created on dry run
    for p in paths:
        assert not p.exists()


def test_collect_results_and_markdown(tmp_path):
    # Simulate completed runs in tmp_path
    for ab_type, exps in [("topology", ["baseline", "directed"]), ("depth", ["depth_1"])]:
        for exp in exps:
            for seed in [42, 43]:
                run_dir = tmp_path / ab_type / f"{exp}_seed{seed}"
                run_dir.mkdir(parents=True)
                metrics = {
                    "ablation_type": ab_type,
                    "exp_name": exp,
                    "model": "edge_aware_gatv2",
                    "seed": seed,
                    "boxes": "gt",
                    "data": "synthetic",
                    "n_params": 12345,
                    "latency_batch_ms": {"latency_ms_mean": 4.5},
                    "best_epoch": 3,
                    "test": {
                        "mAP": 0.25 if seed == 42 else 0.35,
                        "mAP_rare": 0.20,
                        "mAP_common": 0.30,
                        "f1_micro": 0.40,
                        "f1_macro": 0.38,
                    },
                }
                (run_dir / "metrics.json").write_text(json.dumps(metrics))

    runs_df, summary_df = collect_ablation_results(tmp_path)
    assert len(runs_df) == 6
    assert len(summary_df) == 3

    # Check aggregation across seeds: mean should be 0.3000
    baseline_summary = summary_df[summary_df["experiment"] == "baseline"].iloc[0]
    assert baseline_summary["seeds"] == 2
    assert "0.3000" in baseline_summary["mAP"]

    md = to_markdown(summary_df)
    assert "| ablation_type | experiment | model | seeds |" in md
    assert "baseline" in md
    assert "depth_1" in md


def test_plot_ablation_results(tmp_path):
    # Setup mock runs for all 5 ablation types
    records = []
    types_and_exps = {
        "topology": ["baseline", "directed", "knn", "distance_threshold"],
        "features": ["all", "spatial", "appearance", "semantic"],
        "architecture": ["mlp", "gcn", "gat", "edge_aware_gatv2"],
        "depth": ["depth_1", "depth_2", "depth_3"],
        "edge_attr": ["with_edge_attr", "without_edge_attr"],
    }
    for ab_type, exps in types_and_exps.items():
        for exp in exps:
            records.append({
                "ablation_type": ab_type,
                "exp_name": exp,
                "model": "edge_aware_gatv2",
                "seed": 42,
                "mAP": 0.25 + 0.05 * len(exp),
                "mAP_rare": 0.2,
                "mAP_common": 0.3,
                "f1_micro": 0.4,
                "f1_macro": 0.35,
            })
    df = pd.DataFrame(records)
    plots_dir = tmp_path / "plots"
    generated = plot_ablation_results(df, out_dir=plots_dir)

    assert set(generated.keys()) == {"topology", "features", "architecture", "depth", "edge_attr"}
    for p in generated.values():
        assert p.exists()
        assert p.stat().st_size > 1000  # valid image file


def test_ablation_cli_end_to_end_smoke(tmp_path):
    """End-to-end smoke test running 1 fast experiment via CLI main()."""
    out_dir = tmp_path / "smoke_ablations"
    code = main([
        "--ablation", "depth",
        "--seeds", "42",
        "--epochs", "1",
        "--data", "synthetic",
        "--boxes", "gt",
        "--out", str(out_dir),
        "--device", "cpu",
    ])
    assert code == 0
    assert (out_dir / "depth").exists()
    assert (out_dir / "ablation_summary.csv").exists()
    assert (out_dir / "depth_summary.csv").exists()
    assert (out_dir / "plots" / "ablation_depth.png").exists()

    # Check metrics.json in the executed runs
    metrics_files = list(out_dir.rglob("metrics.json"))
    assert len(metrics_files) == 3
    m = json.loads(metrics_files[0].read_text())
    assert "val" in m and "test" in m and "mAP" in m["test"]
