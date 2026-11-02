"""Inference path for the Streamlit app (plan 8.8).

Loads a trained checkpoint and runs: image -> detector -> ROI features -> graph ->
model -> interaction probabilities. It imports NOTHING from training/ (only
utils.checkpoint, models, data, detection).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from data.graph_builder import GraphBuilder
from data.vcoco_loader import AGENT_ONLY_ACTION_INDICES, COCO_CLASSES, VCOCO_ACTIONS
from detection.feature_extractor import ImageFeatures
from utils.checkpoint import load_model_from_checkpoint


class InferenceError(RuntimeError):
    """Raised with a message that is safe to show to the user."""


@dataclass
class LoadedModel:
    model: torch.nn.Module
    model_name: str
    runtime: Dict[str, Any]
    cfg: Dict[str, Any]
    device: torch.device
    bidirectional: bool
    candidate_mode: str
    k_neighbors: int
    dist_threshold: float

    def make_builder(self) -> GraphBuilder:
        return GraphBuilder(candidate_mode=self.candidate_mode, bidirectional=self.bidirectional,
                            k_neighbors=self.k_neighbors, dist_threshold=self.dist_threshold)


def load_for_inference(checkpoint_path, device: str = "auto") -> LoadedModel:
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else ("cpu" if device == "auto" else device))
    try:
        model, ckpt = load_model_from_checkpoint(checkpoint_path, device=dev)
    except Exception as e:  # corrupt file, registry mismatch, state_dict mismatch ...
        raise InferenceError(f"Could not load checkpoint {checkpoint_path}: {e}") from e
    runtime = ckpt.get("runtime", {})
    # The checkpoint stores the whole yaml as cfg["model"]: {"model": {...}, "graph": {...}, ...}
    top = (ckpt.get("config") or {}).get("model") or {}
    graph = top.get("graph", {}) if isinstance(top, dict) else {}
    mcfg = top.get("model", {}) if isinstance(top, dict) else {}
    return LoadedModel(
        model=model, model_name=str(ckpt.get("model_name", "?")), runtime=runtime, cfg=top,
        device=dev, bidirectional=bool(mcfg.get("bidirectional", False)),
        candidate_mode=str(graph.get("candidate_mode", "all_human_object")),
        k_neighbors=int(graph.get("knn_k", 8)), dist_threshold=float(graph.get("distance_threshold", 0.5)))


@dataclass
class Analysis:
    feats: ImageFeatures
    data: Any                               # HOIData on lm.device
    probs: np.ndarray                       # (P, A) sigmoid probabilities
    entities: List[Dict[str, Any]]
    latency_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def num_pairs(self) -> int:
        return int(self.probs.shape[0])


def detect_features(extractor, image_bgr: np.ndarray, image_id: str = "upload") -> ImageFeatures:
    """Detector + ROI features for an in-memory image (no file, no cache)."""
    out = extractor.detector.detect(image_bgr)
    roi = extractor.roi(out.feature_maps, out.boxes_letterbox, out.meta.input_hw)
    feats = ImageFeatures(
        image_id=image_id, image_size=tuple(out.meta.orig_hw), boxes=out.boxes.cpu().float(),
        class_ids=out.class_ids.cpu().long(), scores=out.scores.cpu().float(),
        roi_features=roi.cpu().float())
    feats.validate()
    return feats


def entities_from_feats(feats: ImageFeatures) -> List[Dict[str, Any]]:
    ents = []
    for i in range(feats.num_entities):
        cid = int(feats.class_ids[i])
        ents.append({"idx": i, "label": COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else "unknown",
                     "is_human": cid == 0, "score": float(feats.scores[i]),
                     "box": [float(v) for v in feats.boxes[i].tolist()]})
    return ents


@torch.no_grad()
def analyze(extractor, lm: LoadedModel, image_bgr: np.ndarray, image_id: str = "upload") -> Analysis:
    t0 = time.perf_counter()
    feats = detect_features(extractor, image_bgr, image_id)
    if lm.device.type == "cuda":
        torch.cuda.synchronize()
    t_det = (time.perf_counter() - t0) * 1000

    node_dim = lm.runtime.get("node_dim")
    if node_dim is not None and feats.roi_features.shape[1] != int(node_dim):
        raise InferenceError(
            f"Feature size mismatch: the detector gives {feats.roi_features.shape[1]}-dim features but "
            f"this checkpoint was trained on {node_dim}-dim features (e.g. synthetic data). "
            f"Choose a checkpoint trained on real V-COCO features.")
    n_act = lm.runtime.get("num_actions")
    if n_act is not None and int(n_act) != len(VCOCO_ACTIONS):
        raise InferenceError(f"Checkpoint predicts {n_act} actions; expected {len(VCOCO_ACTIONS)}.")

    data = lm.make_builder().build_graph(image_features=feats).to(lm.device)
    entities = entities_from_feats(feats)
    if data.pair_index.shape[0] == 0:
        return Analysis(feats, data, np.zeros((0, len(VCOCO_ACTIONS)), dtype=np.float32), entities,
                        {"detector": t_det, "graph_model": 0.0})

    t1 = time.perf_counter()
    lm.model.eval()
    out = lm.model(data)
    logits = out[0] if isinstance(out, tuple) else out
    probs = torch.sigmoid(logits).float().cpu().numpy()
    if lm.device.type == "cuda":
        torch.cuda.synchronize()
    t_gnn = (time.perf_counter() - t1) * 1000
    return Analysis(feats, data, probs, entities, {"detector": t_det, "graph_model": t_gnn})


def interactions_from(analysis: Analysis, threshold: float = 0.5, top_k_per_pair: int = 3) -> List[Dict[str, Any]]:
    """Predicted interactions above `threshold`.

    Agent-only actions (stand, walk, run, smile) are skipped: in training they are
    masked on human-object pairs (they belong to a self-pair that the graph builder
    does not create), so their scores on object pairs carry no meaning.
    """
    keep = [a for a in range(len(VCOCO_ACTIONS)) if a not in AGENT_ONLY_ACTION_INDICES]
    pairs = analysis.data.pair_index.cpu().numpy()
    out: List[Dict[str, Any]] = []
    for p in range(analysis.num_pairs):
        scores = [(a, float(analysis.probs[p, a])) for a in keep]
        scores.sort(key=lambda t: -t[1])
        for a, s in scores[:top_k_per_pair]:
            if s >= threshold:
                out.append({"pair_idx": p, "human": int(pairs[p, 0]), "object": int(pairs[p, 1]),
                            "action_idx": a, "action": VCOCO_ACTIONS[a].replace("_", " "), "prob": s})
    return out


def explanation_text(analysis: Analysis, pair_idx: int, action_idx: int, exp, ig=None) -> str:
    """Plain-language summary built only from the computed masks. Graph-level, not causal."""
    ents = {e["idx"]: e for e in analysis.entities}
    h, o = [int(v) for v in analysis.data.pair_index[pair_idx].cpu().tolist()]
    name = lambda i: f'{ents[i]["label"]} #{i}'  # noqa: E731
    action = VCOCO_ACTIONS[action_idx].replace("_", " ")
    lines = [f"The model gives **{action}** a probability of {float(analysis.probs[pair_idx, action_idx]):.2f} "
             f"for the pair **{name(h)} → {name(o)}**."]
    nodes = [(i, w) for i, w in exp.important_nodes if i not in (h, o)][:2]
    if nodes:
        lines.append("Besides the pair itself, the explainer weights these entities most: " +
                     ", ".join(f"{name(i)} ({w:.2f})" for i, w in nodes) + ".")
    else:
        lines.append("The explainer put almost all weight on the human and the object themselves, "
                     "not on other entities in the scene.")
    edges = [(s, d, w) for s, d, w in exp.important_edges][:2]
    if edges:
        lines.append("Most influential graph edges: " +
                     ", ".join(f"{name(s)} → {name(d)} ({w:.2f})" for s, d, w in edges) + ".")
    if ig is not None:
        top = int(torch.argmax(ig.node_importance).item())
        lines.append(f"Integrated Gradients attributes most to {name(top)}.")
    lines.append("_This is graph-level attribution over entities and spatial edges. It does not say which "
                 "image pixels or body parts mattered, and it describes the model, not the true cause._")
    return "\n\n".join(lines)
