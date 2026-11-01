"""Phase 5.8 / 5.9: collect run folders into a results table and plots.

Reads results/<run>/metrics.json and history.json only. Models in EXPECTED_MODELS
with no finished run are listed as "not run", never filled in.

  python -m evaluation.aggregate --results results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

EXPECTED_MODELS = ["prior", "mlp", "gcn", "gat", "edge_aware_gatv2"]  # match Phase 4 registry names
COLUMNS = ["mAP", "mAP_rare", "mAP_common", "f1_micro", "f1_macro"]


def collect_runs(results_dir) -> pd.DataFrame:
    rows = []
    for mpath in sorted(Path(results_dir).glob("*/metrics.json")):
        m = json.loads(mpath.read_text())
        row = {"run": mpath.parent.name, "model": m["model"], "seed": m["seed"], "boxes": m["boxes"],
               "data": m.get("data", "?"), "n_params": m.get("n_params"),
               "latency_ms": m.get("latency_batch_ms", {}).get("latency_ms_mean")}
        for c in COLUMNS:
            row[c] = m["test"].get(c, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(vals: pd.Series) -> str:
    v = vals.dropna().to_numpy(dtype=float)
    if v.size == 0:
        return "n/a"
    return f"{v.mean():.3f}" if v.size == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def build_summary(runs: pd.DataFrame, expected: List[str] = None, boxes: str = "gt") -> pd.DataFrame:
    expected = expected or EXPECTED_MODELS
    rows = []
    for model in expected:
        sub = runs[(runs["model"] == model) & (runs["boxes"] == boxes)] if len(runs) else runs
        if len(sub) == 0:
            rows.append({"model": model, "seeds": 0, **{c: "not run" for c in COLUMNS},
                         "params": "not run", "latency_ms": "not run"})
            continue
        rows.append({"model": model, "seeds": len(sub), **{c: _fmt(sub[c]) for c in COLUMNS},
                     "params": int(sub["n_params"].iloc[0]) if pd.notna(sub["n_params"].iloc[0]) else "n/a",
                     "latency_ms": _fmt(sub["latency_ms"])})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)


def plot_curves(results_dir, out_dir) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for hpath in sorted(Path(results_dir).glob("*/history.json")):
        h = pd.DataFrame(json.loads(hpath.read_text()))
        if len(h) < 2:  # e.g. prior baseline: nothing to plot
            continue
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].plot(h["epoch"], h["train_loss"], label="train")
        ax[0].plot(h["epoch"], h["val_loss"], label="val")
        ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend()
        ax[1].plot(h["epoch"], h["val_mAP"])
        ax[1].set(title="Validation mAP", xlabel="epoch")
        fig.suptitle(hpath.parent.name); fig.tight_layout()
        p = Path(out_dir) / f"{hpath.parent.name}_curves.png"
        fig.savefig(p, dpi=150); plt.close(fig); out.append(p)
    return out


def plot_comparison(runs: pd.DataFrame, out_path, boxes: str = "gt"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = runs[runs["boxes"] == boxes] if len(runs) else runs
    if len(sub) == 0:
        return None
    g = sub.groupby("model")["mAP"].agg(["mean", "std"]).reindex(
        [m for m in EXPECTED_MODELS if m in set(sub["model"])])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(g.index, g["mean"], yerr=g["std"].fillna(0))
    ax.set(ylabel="test mAP", title=f"Model comparison ({boxes} boxes)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return Path(out_path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--boxes", default="gt", choices=["gt", "detected"])
    args = ap.parse_args(argv)
    runs = collect_runs(args.results)
    summary = build_summary(runs, boxes=args.boxes)
    root = Path(args.results)
    summary.to_csv(root / f"summary_{args.boxes}.csv", index=False)
    (root / f"summary_{args.boxes}.md").write_text(to_markdown(summary) + "\n")
    plot_curves(root, root / "plots")
    plot_comparison(runs, root / "plots" / f"comparison_{args.boxes}.png", args.boxes)
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
