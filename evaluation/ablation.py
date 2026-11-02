"""Phase 6 Ablation Studies Engine & CLI.

Systematically runs, records, aggregates, and visualizes ablation experiments:
- Plan 6.1 Topology: directed vs bidirectional vs knn vs distance threshold
- Plan 6.2 Features: spatial, appearance, semantic, combined (all)
- Plan 6.3 Architecture: mlp, gcn, gat, edge_aware_gatv2
- Plan 6.4 Depth: 1, 2, 3 layers
- Plan 6.5 Edge attributes: without vs with
- Plan 6.6 Aggregate results into CSVs, markdown tables, and comparison plots
- Plan 6.7 Repeatable across random seeds

Usage:
  python -m evaluation.ablation --ablation topology --dry-run
  python -m evaluation.ablation --ablation features --seeds 42,43,44 --epochs 5 --data synthetic
  python -m evaluation.ablation --ablation all --data synthetic --epochs 10 --out results/ablations
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

ABLATION_TYPES = ["topology", "features", "architecture", "depth", "edge_attr", "all"]
METRIC_COLUMNS = ["mAP", "mAP_rare", "mAP_common", "f1_micro", "f1_macro"]


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def get_default_model_config() -> Dict[str, Any]:
    cfg = _load_yaml("configs/model_config.yaml")
    if not cfg:
        cfg = {
            "model": {
                "input_dim": 896,
                "hidden_dim": 256,
                "num_layers": 2,
                "heads": 4,
                "dropout": 0.3,
                "edge_dim": 9,
                "num_actions": 26,
                "use_edge_attr": True,
                "bidirectional": True,
                "feature_subset": "all",
                "class_embed_dim": 64,
            },
            "graph": {
                "candidate_mode": "all_human_object",
                "knn_k": 5,
                "distance_threshold": 0.5,
                "dist_threshold": 0.5,
            },
        }
    return copy.deepcopy(cfg)


def get_default_training_config() -> Dict[str, Any]:
    cfg = _load_yaml("configs/training_config.yaml")
    defaults = {"epochs": 30, "batch_size": 32, "lr": 1e-3, "loss": {"name": "bce"}}
    tcfg = cfg.get("training", {})
    return {**defaults, **copy.deepcopy(tcfg)}


def _parse_seeds(raw_seeds: Any) -> List[int]:
    """Parse integer, comma-separated string, or list of values into a list of ints."""
    if isinstance(raw_seeds, int):
        return [raw_seeds]
    if isinstance(raw_seeds, (list, tuple)):
        result = []
        for item in raw_seeds:
            for s in str(item).split(","):
                s = s.strip()
                if s:
                    result.append(int(s))
        return result or [42]
    if isinstance(raw_seeds, str):
        result = []
        for s in raw_seeds.split(","):
            s = s.strip()
            if s:
                result.append(int(s))
        return result or [42]
    return [42]


def build_ablation_configs(
    ablation: str,
    seeds: Union[int, Sequence[Union[int, str]], str] = (42,),
    data: str = "synthetic",
    boxes: str = "gt",
    epochs: Optional[int] = None,
    base_model_cfg: Optional[Dict[str, Any]] = None,
    base_train_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build list of experiment configurations for a specified ablation study.

    Args:
        ablation: One of 'topology', 'features', 'architecture', 'depth', 'edge_attr', or 'all'.
        seeds: Random seed or list/string of seeds (e.g. 42 or [42, 43, 44] or "42,43").
        data: Data source ('synthetic', 'vcoco', or 'fake').
        boxes: Bounding box source ('gt' or 'detected').
        epochs: Optional training epoch count override.
        base_model_cfg: Base model configuration dictionary.
        base_train_cfg: Base training configuration dictionary.

    Returns:
        List of configuration dictionaries ready for execution.
    """
    ablation_type = ablation.lower().strip()
    if ablation_type not in ABLATION_TYPES:
        raise ValueError(
            f"Unknown ablation type '{ablation}'. Expected one of {ABLATION_TYPES}"
        )

    parsed_seeds = _parse_seeds(seeds)
    model_base = base_model_cfg if base_model_cfg is not None else get_default_model_config()
    train_base = base_train_cfg if base_train_cfg is not None else get_default_training_config()

    if epochs is not None:
        train_base["epochs"] = int(epochs)

    target_types = (
        ["topology", "features", "architecture", "depth", "edge_attr"]
        if ablation_type == "all"
        else [ablation_type]
    )

    configs: List[Dict[str, Any]] = []

    for t in target_types:
        experiments: List[Dict[str, Any]] = []

        if t == "topology":
            experiments = [
                {
                    "exp_name": "baseline",
                    "model_name": "edge_aware_gatv2",
                    "description": "Bidirectional edges, candidate_mode=all_human_object",
                    "model_overrides": {
                        "model": {"bidirectional": True},
                        "graph": {"candidate_mode": "all_human_object"},
                    },
                },
                {
                    "exp_name": "directed",
                    "model_name": "edge_aware_gatv2",
                    "description": "Directed edges (H->O only), candidate_mode=all_human_object",
                    "model_overrides": {
                        "model": {"bidirectional": False},
                        "graph": {"candidate_mode": "all_human_object"},
                    },
                },
                {
                    "exp_name": "knn",
                    "model_name": "edge_aware_gatv2",
                    "description": "kNN candidate edges (k=5)",
                    "model_overrides": {
                        "model": {"bidirectional": True},
                        "graph": {
                            "candidate_mode": "knn",
                            "knn_k": 5,
                            "k_neighbors": 5,
                        },
                    },
                },
                {
                    "exp_name": "distance_threshold",
                    "model_name": "edge_aware_gatv2",
                    "description": "Distance threshold candidate edges (d <= 0.5)",
                    "model_overrides": {
                        "model": {"bidirectional": True},
                        "graph": {
                            "candidate_mode": "distance_threshold",
                            "dist_threshold": 0.5,
                            "distance_threshold": 0.5,
                        },
                    },
                },
            ]

        elif t == "features":
            experiments = [
                {
                    "exp_name": "all",
                    "model_name": "edge_aware_gatv2",
                    "description": "All node modalities (spatial + appearance + semantic)",
                    "model_overrides": {
                        "model": {"feature_subset": "all"},
                    },
                },
                {
                    "exp_name": "spatial",
                    "model_name": "edge_aware_gatv2",
                    "description": "Spatial boxes only (boxes_norm)",
                    "model_overrides": {
                        "model": {"feature_subset": "spatial"},
                    },
                },
                {
                    "exp_name": "appearance",
                    "model_name": "edge_aware_gatv2",
                    "description": "Appearance visual features only (x_visual)",
                    "model_overrides": {
                        "model": {"feature_subset": "appearance"},
                    },
                },
                {
                    "exp_name": "semantic",
                    "model_name": "edge_aware_gatv2",
                    "description": "Semantic category class IDs only (class_ids)",
                    "model_overrides": {
                        "model": {"feature_subset": "semantic"},
                    },
                },
            ]

        elif t == "architecture":
            experiments = [
                {
                    "exp_name": "mlp",
                    "model_name": "mlp",
                    "description": "MLP Baseline (no message passing)",
                    "model_overrides": {},
                },
                {
                    "exp_name": "gcn",
                    "model_name": "gcn",
                    "description": "GCN message passing (isotropic)",
                    "model_overrides": {},
                },
                {
                    "exp_name": "gat",
                    "model_name": "gat",
                    "description": "Standard GAT message passing (attention without edge features)",
                    "model_overrides": {},
                },
                {
                    "exp_name": "edge_aware_gatv2",
                    "model_name": "edge_aware_gatv2",
                    "description": "Edge-Aware GATv2 (conditioned on continuous edge attributes)",
                    "model_overrides": {},
                },
            ]

        elif t == "depth":
            experiments = [
                {
                    "exp_name": "depth_1",
                    "model_name": "edge_aware_gatv2",
                    "description": "1 GATv2 layer",
                    "model_overrides": {
                        "model": {"num_layers": 1},
                    },
                },
                {
                    "exp_name": "depth_2",
                    "model_name": "edge_aware_gatv2",
                    "description": "2 GATv2 layers (standard depth)",
                    "model_overrides": {
                        "model": {"num_layers": 2},
                    },
                },
                {
                    "exp_name": "depth_3",
                    "model_name": "edge_aware_gatv2",
                    "description": "3 GATv2 layers",
                    "model_overrides": {
                        "model": {"num_layers": 3},
                    },
                },
            ]

        elif t == "edge_attr":
            experiments = [
                {
                    "exp_name": "with_edge_attr",
                    "model_name": "edge_aware_gatv2",
                    "description": "With 9-dim continuous edge attributes",
                    "model_overrides": {
                        "model": {"use_edge_attr": True},
                    },
                },
                {
                    "exp_name": "without_edge_attr",
                    "model_name": "edge_aware_gatv2",
                    "description": "Without edge attributes (topology only)",
                    "model_overrides": {
                        "model": {"use_edge_attr": False},
                    },
                },
            ]

        for exp in experiments:
            for seed in parsed_seeds:
                mcfg = copy.deepcopy(model_base)
                # Apply model and graph section overrides
                for sec, sec_vals in exp["model_overrides"].items():
                    if sec not in mcfg:
                        mcfg[sec] = {}
                    mcfg[sec].update(sec_vals)

                cfg_entry = {
                    "ablation_type": t,
                    "exp_name": exp["exp_name"],
                    "model_name": exp["model_name"],
                    "seed": seed,
                    "data": data,
                    "boxes": boxes,
                    "description": exp["description"],
                    "model_cfg": mcfg,
                    "train_cfg": copy.deepcopy(train_base),
                    "run_folder": f"{exp['exp_name']}_seed{seed}",
                }
                configs.append(cfg_entry)

    return configs


def run_single_experiment(
    cfg: Dict[str, Any],
    run_dir: Union[str, Path],
    device_pref: str = "auto",
) -> Path:
    """Execute training and evaluation for a single ablation experiment run."""
    import torch

    from evaluation.metrics import measure_latency
    from training.data_hooks import build_loaders
    from training.losses import build_loss
    from training.trainer import Trainer
    from utils.checkpoint import build_any_model, load_checkpoint
    from utils.seed import set_seed

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    device = torch.device(
        "cuda"
        if device_pref == "auto" and torch.cuda.is_available()
        else ("cpu" if device_pref == "auto" else device_pref)
    )

    tcfg = copy.deepcopy(cfg["train_cfg"])
    model_cfg = copy.deepcopy(cfg["model_cfg"])

    data = build_loaders(cfg["data"], cfg["boxes"], tcfg, cfg["seed"], model_cfg)
    runtime = {
        "num_actions": data.num_actions,
        "node_dim": data.node_dim,
        "edge_dim": data.edge_dim,
    }
    model = build_any_model(cfg["model_name"], model_cfg, runtime)
    if hasattr(model, "fit"):
        model.fit(data.train_loader.dataset)

    loss_fn = build_loss(tcfg["loss"], data.train_pos_counts, data.train_valid_counts)
    meta = {
        "model_name": cfg["model_name"],
        "ablation_type": cfg["ablation_type"],
        "exp_name": cfg["exp_name"],
        "description": cfg.get("description", ""),
        "runtime": runtime,
        "seed": cfg["seed"],
        "cfg": {
            "model": model_cfg,
            "training": copy.deepcopy(tcfg),
            "data": cfg["data"],
            "boxes": cfg["boxes"],
        },
    }
    trainer = Trainer(model, loss_fn, tcfg, device, run_dir, meta, data.train_pos_counts)
    history = trainer.fit(data.train_loader, data.val_loader)

    best = load_checkpoint(run_dir / "best.pt", map_location=device)
    trainer.model.load_state_dict(best["model_state"])
    val = trainer.evaluate(data.val_loader)
    test = trainer.evaluate(data.test_loader)

    sample = next(iter(data.val_loader)).to(device)
    sync = torch.cuda.synchronize if device.type == "cuda" else None
    with torch.no_grad():
        trainer.model.eval()
        latency = measure_latency(lambda: trainer.model(sample), sync=sync)

    metrics = {
        "ablation_type": cfg["ablation_type"],
        "exp_name": cfg["exp_name"],
        "description": cfg.get("description", ""),
        "model": cfg["model_name"],
        "seed": cfg["seed"],
        "boxes": cfg["boxes"],
        "data": cfg["data"],
        "best_epoch": best["epoch"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "latency_batch_ms": latency,
        "latency_batch_size": int(sample.num_graphs),
        "val": val,
        "test": test,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(
        f"[{cfg['ablation_type']}:{cfg['exp_name']}] seed={cfg['seed']} "
        f"val mAP={val['mAP']:.4f} test mAP={test['mAP']:.4f} -> {run_dir}"
    )
    return run_dir


def run_ablation_grid(
    configs: List[Dict[str, Any]],
    out_dir: Union[str, Path] = "results/ablations",
    device: str = "auto",
    dry_run: bool = False,
) -> List[Path]:
    """Execute or plan a grid of ablation experiments.

    Args:
        configs: List of configuration dictionaries generated by build_ablation_configs.
        out_dir: Base directory for storing experiment outputs.
        device: Device preference ('auto', 'cpu', 'cuda').
        dry_run: If True, prints experiment plan table and returns target paths without running.

    Returns:
        List of Path objects for the run directories.
    """
    base_out = Path(out_dir)
    planned_paths: List[Path] = []

    for cfg in configs:
        run_path = base_out / cfg["ablation_type"] / cfg["run_folder"]
        planned_paths.append(run_path)

    if dry_run:
        plan_rows = []
        for i, (cfg, path) in enumerate(zip(configs, planned_paths), start=1):
            plan_rows.append(
                {
                    "#": i,
                    "Ablation": cfg["ablation_type"],
                    "Experiment": cfg["exp_name"],
                    "Model": cfg["model_name"],
                    "Seed": cfg["seed"],
                    "Data": cfg["data"],
                    "Boxes": cfg["boxes"],
                    "Epochs": cfg["train_cfg"].get("epochs", "-"),
                    "Target Folder": str(path),
                }
            )
        plan_df = pd.DataFrame(plan_rows)
        print("\n=== Phase 6 Ablation Experiment Plan (Dry-Run) ===")
        print(to_markdown(plan_df))
        print(f"Total experiments planned: {len(configs)}\n")
        return planned_paths

    completed_paths: List[Path] = []
    total = len(configs)
    for idx, (cfg, run_dir) in enumerate(zip(configs, planned_paths), start=1):
        print(f"\n--- Running [{idx}/{total}] {cfg['ablation_type']}:{cfg['exp_name']} (seed={cfg['seed']}) ---")
        try:
            p = run_single_experiment(cfg, run_dir, device_pref=device)
            completed_paths.append(p)
        except Exception as e:
            print(f"ERROR running experiment {cfg['exp_name']} (seed={cfg['seed']}): {e}")
            raise e

    return completed_paths


def _fmt(vals: Sequence[Any]) -> str:
    """Format mean ± std or simple scalar."""
    clean = [float(v) for v in vals if pd.notna(v) and v is not None]
    if not clean:
        return "n/a"
    v = np.asarray(clean, dtype=float)
    if v.size == 1:
        return f"{v[0]:.4f}"
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"


def collect_ablation_results(
    out_dir: Union[str, Path],
    ablation_type: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scan out_dir for metrics.json files and aggregate into summary tables.

    Args:
        out_dir: Directory containing ablation outputs.
        ablation_type: Optional filter by ablation category.

    Returns:
        Tuple of (runs_df, summary_df):
          - runs_df: Per-run individual records
          - summary_df: Grouped summary aggregated across seeds
    """
    root = Path(out_dir)
    if not root.exists():
        empty_runs = pd.DataFrame(columns=["ablation_type", "exp_name", "model", "seed"] + METRIC_COLUMNS)
        empty_summary = pd.DataFrame(columns=["ablation_type", "experiment", "seeds"] + METRIC_COLUMNS)
        return empty_runs, empty_summary

    rows: List[Dict[str, Any]] = []
    for mpath in sorted(root.rglob("metrics.json")):
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            continue

        run_folder = mpath.parent
        parent_folder = run_folder.parent

        ab_type = m.get("ablation_type")
        if not ab_type:
            ab_type = parent_folder.name if parent_folder != root else "general"

        exp_name = m.get("exp_name")
        if not exp_name:
            folder_name = run_folder.name
            if "_seed" in folder_name:
                exp_name = folder_name.rsplit("_seed", 1)[0]
            else:
                exp_name = folder_name

        seed = m.get("seed", 0)
        test_m = m.get("test", {})
        latency_val = m.get("latency_batch_ms", {})
        latency_ms = latency_val.get("latency_ms_mean") if isinstance(latency_val, dict) else None

        row: Dict[str, Any] = {
            "ablation_type": ab_type,
            "exp_name": exp_name,
            "model": m.get("model", ""),
            "seed": seed,
            "boxes": m.get("boxes", "gt"),
            "data": m.get("data", "?"),
            "description": m.get("description", ""),
            "n_params": m.get("n_params"),
            "latency_ms": latency_ms,
            "best_epoch": m.get("best_epoch"),
            "run_dir": str(run_folder),
        }
        for c in METRIC_COLUMNS:
            row[c] = test_m.get(c, float("nan"))
        for extra in ("precision_micro", "recall_micro", "precision_macro", "recall_macro"):
            if extra in test_m:
                row[extra] = test_m[extra]

        rows.append(row)

    if not rows:
        empty_runs = pd.DataFrame(columns=["ablation_type", "exp_name", "model", "seed"] + METRIC_COLUMNS)
        empty_summary = pd.DataFrame(columns=["ablation_type", "experiment", "seeds"] + METRIC_COLUMNS)
        return empty_runs, empty_summary

    runs_df = pd.DataFrame(rows)
    if ablation_type and ablation_type != "all":
        runs_df = runs_df[runs_df["ablation_type"] == ablation_type].reset_index(drop=True)

    summary_rows: List[Dict[str, Any]] = []
    # Preserve grouping order
    grouped = runs_df.groupby(["ablation_type", "exp_name"], sort=False)
    for (ab_t, exp_n), group in grouped:
        srow: Dict[str, Any] = {
            "ablation_type": ab_t,
            "experiment": exp_n,
            "model": group["model"].iloc[0],
            "seeds": len(group),
            "description": group["description"].iloc[0] if "description" in group else "",
        }
        for c in METRIC_COLUMNS:
            srow[c] = _fmt(group[c])
            srow[f"{c}_mean"] = float(group[c].mean()) if pd.notna(group[c].mean()) else float("nan")
            srow[f"{c}_std"] = float(group[c].std(ddof=1)) if len(group) > 1 else 0.0

        p = group["n_params"].dropna()
        srow["params"] = int(p.iloc[0]) if len(p) > 0 else "n/a"
        srow["latency_ms"] = _fmt(group["latency_ms"])
        summary_rows.append(srow)

    summary_df = pd.DataFrame(summary_rows)
    return runs_df, summary_df


def to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame into GitHub-flavored Markdown table."""
    if df.empty:
        return "*(No results)*"
    display_cols = [c for c in df.columns if not c.endswith("_mean") and not c.endswith("_std")]
    sub_df = df[display_cols]
    cols = list(sub_df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in sub_df.iterrows():
        line = "| " + " | ".join(str(r[c]) for c in cols) + " |"
        lines.append(line)
    return "\n".join(lines)


def plot_ablation_results(
    runs_or_dir: Union[pd.DataFrame, str, Path],
    out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Generate publication-ready comparison bar and line plots for each ablation category.

    Args:
        runs_or_dir: Either runs DataFrame or path to ablations output directory.
        out_dir: Target directory for plot image files.

    Returns:
        Dictionary mapping ablation category name to saved plot file Path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(runs_or_dir, pd.DataFrame):
        runs_df = runs_or_dir
        plots_dir = Path(out_dir or "results/ablations/plots")
    else:
        root = Path(runs_or_dir)
        runs_df, _ = collect_ablation_results(root)
        plots_dir = Path(out_dir) if out_dir else root / "plots"

    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Path] = {}

    if runs_df.empty:
        return generated

    color_palette = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

    # 1. Topology comparison plot
    sub_top = runs_df[runs_df["ablation_type"] == "topology"]
    if not sub_top.empty:
        exp_order = ["baseline", "directed", "knn", "distance_threshold"]
        labels = ["Baseline\n(Bidir)", "Directed\n(H->O only)", "kNN\n(k=5)", "Distance\n(d <= 0.5)"]
        present_order = [e for e in exp_order if e in set(sub_top["exp_name"])]
        if present_order:
            means, stds, display_labels = [], [], []
            for e in present_order:
                g = sub_top[sub_top["exp_name"] == e]["mAP"].dropna()
                means.append(g.mean() if len(g) else 0.0)
                stds.append(g.std(ddof=1) if len(g) > 1 else 0.0)
                idx = exp_order.index(e) if e in exp_order else -1
                display_labels.append(labels[idx] if idx >= 0 else e)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            x_pos = np.arange(len(present_order))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=4, color="#2b5c8f", alpha=0.88, edgecolor="black", width=0.55)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(display_labels, fontsize=10)
            ax.set_ylabel("Test mAP", fontsize=11, fontweight="bold")
            ax.set_title("Ablation 6.1: Graph Topology & Candidate Edge Selection", fontsize=12, fontweight="bold", pad=12)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            y_max = max(means) * 1.25 if max(means) > 0 else 0.5
            ax.set_ylim(0, max(y_max, 0.1))

            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            fig.tight_layout()
            p = plots_dir / "ablation_topology.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            generated["topology"] = p

    # 2. Features comparison plot
    sub_feat = runs_df[runs_df["ablation_type"] == "features"]
    if not sub_feat.empty:
        exp_order = ["all", "spatial", "appearance", "semantic"]
        labels = ["All Modalities\n(Combined)", "Spatial Box\nOnly", "Appearance\nOnly", "Semantic Class\nOnly"]
        present_order = [e for e in exp_order if e in set(sub_feat["exp_name"])]
        if present_order:
            means, stds, display_labels = [], [], []
            for e in present_order:
                g = sub_feat[sub_feat["exp_name"] == e]["mAP"].dropna()
                means.append(g.mean() if len(g) else 0.0)
                stds.append(g.std(ddof=1) if len(g) > 1 else 0.0)
                idx = exp_order.index(e) if e in exp_order else -1
                display_labels.append(labels[idx] if idx >= 0 else e)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            x_pos = np.arange(len(present_order))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=4, color="#d95f02", alpha=0.88, edgecolor="black", width=0.55)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(display_labels, fontsize=10)
            ax.set_ylabel("Test mAP", fontsize=11, fontweight="bold")
            ax.set_title("Ablation 6.2: Node Feature Modalities", fontsize=12, fontweight="bold", pad=12)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            y_max = max(means) * 1.25 if max(means) > 0 else 0.5
            ax.set_ylim(0, max(y_max, 0.1))

            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            fig.tight_layout()
            p = plots_dir / "ablation_features.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            generated["features"] = p

    # 3. Architecture comparison plot
    sub_arch = runs_df[runs_df["ablation_type"] == "architecture"]
    if not sub_arch.empty:
        exp_order = ["mlp", "gcn", "gat", "edge_aware_gatv2"]
        labels = ["MLP Baseline", "GCN", "GAT", "Edge-Aware\nGATv2"]
        present_order = [e for e in exp_order if e in set(sub_arch["exp_name"])]
        if present_order:
            means, stds, display_labels = [], [], []
            for e in present_order:
                g = sub_arch[sub_arch["exp_name"] == e]["mAP"].dropna()
                means.append(g.mean() if len(g) else 0.0)
                stds.append(g.std(ddof=1) if len(g) > 1 else 0.0)
                idx = exp_order.index(e) if e in exp_order else -1
                display_labels.append(labels[idx] if idx >= 0 else e)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            x_pos = np.arange(len(present_order))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=4, color="#7570b3", alpha=0.88, edgecolor="black", width=0.55)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(display_labels, fontsize=10)
            ax.set_ylabel("Test mAP", fontsize=11, fontweight="bold")
            ax.set_title("Ablation 6.3: Model Architecture Comparison", fontsize=12, fontweight="bold", pad=12)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            y_max = max(means) * 1.25 if max(means) > 0 else 0.5
            ax.set_ylim(0, max(y_max, 0.1))

            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            fig.tight_layout()
            p = plots_dir / "ablation_architecture.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            generated["architecture"] = p

    # 4. Depth comparison plot
    sub_dep = runs_df[runs_df["ablation_type"] == "depth"]
    if not sub_dep.empty:
        exp_order = ["depth_1", "depth_2", "depth_3"]
        layer_nums = [1, 2, 3]
        present_order = [e for e in exp_order if e in set(sub_dep["exp_name"])]
        if present_order:
            layers, means, stds = [], [], []
            for e in present_order:
                g = sub_dep[sub_dep["exp_name"] == e]["mAP"].dropna()
                idx = exp_order.index(e)
                layers.append(layer_nums[idx])
                means.append(g.mean() if len(g) else 0.0)
                stds.append(g.std(ddof=1) if len(g) > 1 else 0.0)

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.errorbar(layers, means, yerr=stds, fmt="-o", color="#e7298a", linewidth=2.2, markersize=8, capsize=5, label="Test mAP")
            ax.set_xticks(layer_nums)
            ax.set_xlabel("Number of GNN Layers (Depth)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Test mAP", fontsize=11, fontweight="bold")
            ax.set_title("Ablation 6.4: GNN Layer Depth", fontsize=12, fontweight="bold", pad=12)
            ax.grid(True, linestyle="--", alpha=0.35)
            y_max = max(means) * 1.25 if max(means) > 0 else 0.5
            ax.set_ylim(0, max(y_max, 0.1))

            for x_val, y_val in zip(layers, means):
                ax.annotate(
                    f"{y_val:.3f}",
                    xy=(x_val, y_val),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )

            fig.tight_layout()
            p = plots_dir / "ablation_depth.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            generated["depth"] = p

    # 5. Edge Attributes comparison plot
    sub_edge = runs_df[runs_df["ablation_type"] == "edge_attr"]
    if not sub_edge.empty:
        exp_order = ["with_edge_attr", "without_edge_attr"]
        labels = ["With 9-Dim\nEdge Attributes", "Without\nEdge Attributes"]
        present_order = [e for e in exp_order if e in set(sub_edge["exp_name"])]
        if present_order:
            means, stds, display_labels = [], [], []
            for e in present_order:
                g = sub_edge[sub_edge["exp_name"] == e]["mAP"].dropna()
                means.append(g.mean() if len(g) else 0.0)
                stds.append(g.std(ddof=1) if len(g) > 1 else 0.0)
                idx = exp_order.index(e) if e in exp_order else -1
                display_labels.append(labels[idx] if idx >= 0 else e)

            fig, ax = plt.subplots(figsize=(6, 4.5))
            x_pos = np.arange(len(present_order))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=4, color=["#66a61e", "#b2df8a"], alpha=0.88, edgecolor="black", width=0.45)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(display_labels, fontsize=10)
            ax.set_ylabel("Test mAP", fontsize=11, fontweight="bold")
            ax.set_title("Ablation 6.5: Edge Attributes Impact", fontsize=12, fontweight="bold", pad=12)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            y_max = max(means) * 1.25 if max(means) > 0 else 0.5
            ax.set_ylim(0, max(y_max, 0.1))

            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            fig.tight_layout()
            p = plots_dir / "ablation_edge_attr.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            generated["edge_attr"] = p

    return generated


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Phase 6 Ablation Studies Engine & CLI")
    ap.add_argument(
        "--ablation",
        default="all",
        choices=["topology", "features", "architecture", "depth", "edge_attr", "all"],
        help="Ablation grid to execute (default: all)",
    )
    ap.add_argument(
        "--data",
        default="synthetic",
        choices=["vcoco", "synthetic", "fake"],
        help="Data source (default: synthetic)",
    )
    ap.add_argument(
        "--boxes",
        default="gt",
        choices=["gt", "detected"],
        help="Box source (default: gt)",
    )
    ap.add_argument(
        "--seeds",
        nargs="*",
        default=["42"],
        help="Random seeds, e.g. 42 or '42,43,44' or multiple numbers (default: 42)",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs (default: None, use training config)",
    )
    ap.add_argument(
        "--out",
        default="results/ablations",
        help="Output directory (default: results/ablations)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned experiments without running training",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device preference (default: auto)",
    )
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    seeds = _parse_seeds(args.seeds)
    out_path = Path(args.out)

    configs = build_ablation_configs(
        ablation=args.ablation,
        seeds=seeds,
        data=args.data,
        boxes=args.boxes,
        epochs=args.epochs,
    )

    if args.dry_run:
        run_ablation_grid(configs, out_dir=out_path, device=args.device, dry_run=True)
        return 0

    print(f"Starting Phase 6 Ablation: grid='{args.ablation}', seeds={seeds}, data='{args.data}', epochs={args.epochs}")
    run_ablation_grid(configs, out_dir=out_path, device=args.device, dry_run=False)

    runs_df, summary_df = collect_ablation_results(out_path)

    out_path.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path / "ablation_summary.csv", index=False)
    runs_df.to_csv(out_path / "ablation_runs.csv", index=False)

    # Per-ablation CSVs
    if not summary_df.empty and "ablation_type" in summary_df:
        for ab_t in summary_df["ablation_type"].unique():
            sub = summary_df[summary_df["ablation_type"] == ab_t]
            sub.to_csv(out_path / f"{ab_t}_summary.csv", index=False)

    plot_ablation_results(runs_df, out_path / "plots")

    print("\n=== Phase 6 Ablation Summary Table ===")
    print(to_markdown(summary_df))
    print(f"\nArtifacts written to: {out_path.resolve()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
