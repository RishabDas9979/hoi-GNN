"""Read real experiment outputs from results/ for the app (torch-free).

Nothing here computes or invents numbers: it only reads files written by
training.train, evaluation.ablation and the app's own "save explanation metrics"
button. A run counts as a real benchmark result only when its metrics.json says
data == "vcoco". Runs on "synthetic" or "fake" data are returned separately and
must never be shown as benchmark performance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

EXPLANATION_LOG = "explanation_metrics.jsonl"


@dataclass
class RunInfo:
    name: str
    folder: Path
    checkpoint: Optional[Path]
    model: str
    data: str            # vcoco | synthetic | fake | ?
    boxes: str           # gt | detected | ?
    seed: Any
    test_mAP: Optional[float]

    @property
    def is_real(self) -> bool:
        return self.data == "vcoco"

    @property
    def label(self) -> str:
        tag = "V-COCO" if self.is_real else f"NOT a benchmark ({self.data} data)"
        m = f", test mAP {self.test_mAP:.3f}" if self.test_mAP is not None else ""
        return f"{self.name} [{tag}, {self.boxes} boxes{m}]"


def list_runs(results_dir="results") -> List[RunInfo]:
    """Every run folder that holds a metrics.json (searched recursively)."""
    root = Path(results_dir)
    runs: List[RunInfo] = []
    if not root.exists():
        return runs
    for mpath in sorted(root.rglob("metrics.json")):
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        folder = mpath.parent
        best = folder / "best.pt"
        test = m.get("test", {}) if isinstance(m.get("test"), dict) else {}
        mp = test.get("mAP")
        runs.append(RunInfo(
            name=str(folder.relative_to(root)).replace("\\", "/"), folder=folder,
            checkpoint=best if best.exists() else None, model=str(m.get("model", "?")),
            data=str(m.get("data", "?")), boxes=str(m.get("boxes", "?")), seed=m.get("seed"),
            test_mAP=float(mp) if isinstance(mp, (int, float)) else None))
    return runs


def load_history(folder) -> Optional[pd.DataFrame]:
    p = Path(folder) / "history.json"
    if not p.exists():
        return None
    try:
        df = pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
    return df if len(df) else None


def append_explanation_metrics(results_dir, record: Dict[str, Any]) -> Path:
    """Append one explanation-metrics record (computed by the app on a real model run)."""
    path = Path(results_dir) / EXPLANATION_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return path


def load_explanation_metrics(results_dir="results") -> pd.DataFrame:
    path = Path(results_dir) / EXPLANATION_LOG
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return pd.DataFrame(rows)
