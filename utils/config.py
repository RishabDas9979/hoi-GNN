"""Config loading and device selection."""
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "detection" not in cfg:
        raise ValueError(f"{cfg_path} is empty or missing the 'detection' section")
    return cfg


def get_device(preference: str = "auto") -> torch.device:
    """'auto' uses CUDA when available, otherwise falls back to CPU."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if preference == "cuda":
            print("WARNING: CUDA requested but unavailable, using CPU.")
        return torch.device("cpu")
    raise ValueError(f"Unknown device preference: {preference}")