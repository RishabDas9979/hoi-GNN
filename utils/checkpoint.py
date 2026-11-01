"""Checkpoint save/load (Phase 5.3).

Deliberately free of any `training/` import so the Streamlit app can rebuild a
model from a checkpoint alone (plan item 8.8).
A checkpoint stores: weights, optimizer, config, seed, history, and the
`runtime` dict (num_actions, node_dim, edge_dim) needed to rebuild the model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def save_checkpoint(path, model, *, model_name: str, runtime: Dict[str, Any], cfg: Dict[str, Any],
                    seed: int, epoch: int, best_score: float, history: list,
                    optimizer=None, scheduler=None) -> None:
    import torch

    payload = {
        "model_state": model.state_dict(),
        "model_name": model_name,
        "runtime": runtime,
        "config": cfg,
        "seed": seed,
        "epoch": epoch,
        "best_score": best_score,
        "history": history,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu") -> Dict[str, Any]:
    import torch

    # weights_only=False: the file holds plain config dicts we wrote ourselves.
    return torch.load(path, map_location=map_location, weights_only=False)


def build_any_model(name: str, model_cfg: Optional[Dict[str, Any]], runtime: Dict[str, Any]):
    """Build a model by name. 'tiny' is the Phase 5 stub; everything else goes
    through the Phase 4 registry `models.build_model(name, cfg)`.

    CONTRACT with Phase 4: `cfg["runtime"]` carries num_actions / node_dim / edge_dim,
    inferred from the data, so model files never hardcode them.
    """
    if name == "tiny":
        from utils.tiny_model import TinyModel

        return TinyModel(runtime["node_dim"], runtime["num_actions"])
    from models import build_model  # Phase 4 (teammate)

    cfg = dict(model_cfg or {})
    cfg["runtime"] = dict(runtime)
    return build_model(name, cfg)


def load_model_from_checkpoint(path, device="cpu"):
    ckpt = load_checkpoint(path, map_location=device)
    model = build_any_model(ckpt["model_name"], ckpt.get("config", {}).get("model"), ckpt["runtime"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt
