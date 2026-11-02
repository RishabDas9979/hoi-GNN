"""Phase 5.4 CLI.

  python -m training.train --model tiny --data fake --seed 0          # smoke test
  python -m training.train --model edge_aware_gatv2 --data vcoco --seed 0 --boxes gt

Writes results/<model>_<boxes>_seed<seed>/{metrics.json, history.json, best.pt, last.pt}.
All numbers in metrics.json come from the evaluation code, never typed in by hand.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import yaml

from evaluation.metrics import measure_latency
from training.data_hooks import build_loaders
from training.losses import build_loss
from training.trainer import Trainer
from utils.checkpoint import build_any_model, load_checkpoint
from utils.seed import set_seed


def _load_yaml(path):
    p = Path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="prior | mlp | gcn | gat | edge_aware_gatv2 | tiny (names per Phase 4 registry)")
    ap.add_argument("--config", default="configs/model_config.yaml", help="Phase 4 model config")
    ap.add_argument("--train-config", default="configs/training_config.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="vcoco", choices=["vcoco", "synthetic", "fake"])
    ap.add_argument("--boxes", default="gt", choices=["gt", "detected"], help="reported separately (plan 5.7)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run-name", default=None, help="override the default folder name under --out")
    return ap.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    tcfg = _load_yaml(args.train_config).get("training", {})
    defaults = {"epochs": 30, "batch_size": 32, "lr": 1e-3, "loss": {"name": "bce"}}
    tcfg = {**defaults, **tcfg}
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    model_cfg = _load_yaml(args.config)

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))

    data = build_loaders(args.data, args.boxes, tcfg, args.seed, model_cfg)
    runtime = {"num_actions": data.num_actions, "node_dim": data.node_dim, "edge_dim": data.edge_dim}
    model = build_any_model(args.model, model_cfg, runtime)
    if hasattr(model, "fit"):  # prior baseline: counts, not gradients
        model.fit(data.train_loader.dataset)

    loss_fn = build_loss(tcfg["loss"], data.train_pos_counts, data.train_valid_counts)
    folder_name = args.run_name if args.run_name else f"{args.model}_{args.boxes}_seed{args.seed}"
    run_dir = Path(args.out) / folder_name
    meta = {"model_name": args.model, "runtime": runtime, "seed": args.seed,
            "cfg": {"model": model_cfg, "training": copy.deepcopy(tcfg), "data": args.data, "boxes": args.boxes}}
    trainer = Trainer(model, loss_fn, tcfg, device, run_dir, meta, data.train_pos_counts)

    history = trainer.fit(data.train_loader, data.val_loader)

    best = load_checkpoint(run_dir / "best.pt", map_location=device)
    trainer.model.load_state_dict(best["model_state"])
    val, test = trainer.evaluate(data.val_loader), trainer.evaluate(data.test_loader)

    sample = next(iter(data.val_loader)).to(device)
    sync = torch.cuda.synchronize if device.type == "cuda" else None
    with torch.no_grad():
        trainer.model.eval()
        latency = measure_latency(lambda: trainer.model(sample), sync=sync)

    metrics = {
        "model": args.model, "seed": args.seed, "boxes": args.boxes, "data": args.data,
        "best_epoch": best["epoch"], "n_params": sum(p.numel() for p in model.parameters()),
        "latency_batch_ms": latency, "latency_batch_size": int(sample.num_graphs),
        "val": val, "test": test,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[{args.model}] val mAP={val['mAP']:.4f} test mAP={test['mAP']:.4f} -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
