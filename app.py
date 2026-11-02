"""Streamlit demo for Explainable HOI Detection with Relational GNNs.

Run from the project root:   streamlit run app.py

Rules this app follows
- It shows only numbers read from results/ (written by training/evaluation code) or computed live
  on the uploaded image by the loaded model. Nothing is typed in.
- Runs trained on synthetic or fake data are labelled "NOT a benchmark" and never shown as results.
- Explanations are graph-level (entities and spatial edges), never pixel-level.
- The inference path imports nothing from training/.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils.results_io import (append_explanation_metrics, list_runs, load_explanation_metrics,
                              load_history)
from visualization.scene_graph import build_scene_graph, draw_scene_graph, interaction_table

RESULTS_DIR = Path("results")
EDGE_FEATURES = ["dx", "dy", "distance", "log width ratio", "log height ratio", "log area ratio",
                 "IoU", "human area / image", "object area / image"]

st.set_page_config(page_title="HOI-GNN", page_icon="🧩", layout="wide")


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading YOLOv8 detector...")
def get_extractor():
    from detection.run_demo import build_extractor
    from utils.config import load_config

    return build_extractor(load_config())


@st.cache_resource(show_spinner="Loading checkpoint...")
def get_model(path: str, mtime: float):  # mtime only invalidates the cache when the file changes
    from utils.inference import load_for_inference

    return load_for_inference(path)


def show_fig(fig):
    import matplotlib.pyplot as plt

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
PAGES = ["Home", "Image Analysis", "Scene Graph", "Explainability", "Model Evaluation"]
page = st.sidebar.radio("Page", PAGES)

runs = list_runs(RESULTS_DIR)
ckpt_runs = [r for r in runs if r.checkpoint is not None]
selected = None
if ckpt_runs:
    ckpt_runs = sorted(ckpt_runs, key=lambda r: (not r.is_real, r.name))   # real V-COCO runs first
    idx = st.sidebar.selectbox("Checkpoint", range(len(ckpt_runs)), format_func=lambda i: ckpt_runs[i].label)
    selected = ckpt_runs[idx]
    if not selected.is_real:
        st.sidebar.warning(f"This checkpoint was trained on {selected.data} data. Predictions are a "
                           "pipeline demo only, not benchmark behaviour.")
else:
    st.sidebar.info("No trained checkpoint found in results/. Detection still works; interaction "
                    "prediction needs a checkpoint from `python -m training.train`.")
threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.5, 0.05)


def get_analysis():
    """Run (or reuse) the full pipeline for the uploaded image. Returns (analysis, lm, error)."""
    up = st.session_state.get("upload")
    if up is None or selected is None:
        return None, None, None
    key = (up["name"], up["size"], str(selected.checkpoint))
    if st.session_state.get("analysis_key") == key:
        return st.session_state["analysis"], st.session_state["lm"], None
    from utils.inference import InferenceError, analyze

    try:
        lm = get_model(str(selected.checkpoint), selected.checkpoint.stat().st_mtime)
        with st.spinner("Detecting objects and predicting interactions..."):
            analysis = analyze(get_extractor(), lm, up["bgr"], image_id=up["name"])
    except InferenceError as e:
        return None, None, str(e)
    st.session_state.update(analysis_key=key, analysis=analysis, lm=lm)
    return analysis, lm, None


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_home():
    st.title("Explainable Human-Object Interaction Detection")
    st.caption("Relational Graph Neural Networks on V-COCO")
    st.markdown(
        "**Objective.** Given an image, find the people and objects in it, build a graph of their "
        "relationships, and predict what each person is doing with each object (for example "
        "*person → hold → cell phone*). The research question is whether relational graph reasoning "
        "improves HOI detection over non-graph baselines, and whether the predictions can be explained.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Pipeline")
        st.markdown(
            "1. **YOLOv8** detects people and objects (frozen).\n"
            "2. **ROIAlign** pools an appearance embedding for every box from the detector's feature maps.\n"
            "3. Each image becomes a **graph**: nodes are entities (appearance, class, box, confidence); "
            "edges carry 9 spatial relation features.\n"
            "4. An **edge-aware GATv2** passes messages between nodes, then an MLP head scores the 26 "
            "V-COCO actions for every human-object pair (multi-label, sigmoid).\n"
            "5. **GNNExplainer, attention and Integrated Gradients** explain a chosen prediction.")
    with c2:
        st.subheader("Technology")
        st.markdown("Python · PyTorch · PyTorch Geometric (GATv2Conv) · Torchvision · Ultralytics YOLOv8 · "
                    "GNNExplainer · Captum · OpenCV · NumPy · Pandas · scikit-learn · NetworkX · "
                    "Matplotlib · Streamlit")
        st.subheader("Models compared")
        st.markdown("Object-class prior · MLP (no message passing) · GCN · GAT · **edge-aware GATv2**")
    st.info("**Honest scope.** Explanations are graph-level (which entities and edges mattered to the "
            "model). They do not identify pixels or body parts. Numbers on the Model Evaluation page come "
            "only from saved runs; synthetic-data runs are never shown as benchmark results.")


def page_image_analysis():
    st.title("Image Analysis")
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file is not None:
        raw = file.getvalue()
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            st.error("Could not decode that image.")
            return
        if st.session_state.get("upload", {}).get("name") != file.name or \
                st.session_state["upload"]["size"] != len(raw):
            st.session_state["upload"] = {"name": file.name, "size": len(raw), "bgr": arr}
    up = st.session_state.get("upload")
    if up is None:
        st.write("Upload a photo with at least one person and one object.")
        return

    extractor = get_extractor()
    left, right = st.columns(2)
    left.subheader("Original")
    left.image(cv2.cvtColor(up["bgr"], cv2.COLOR_BGR2RGB), use_container_width=True)

    analysis, lm, err = get_analysis()
    if err:
        st.error(err)
    if analysis is None:
        from utils.inference import detect_features
        from visualization.detection_visualizer import draw_detections

        feats = detect_features(extractor, up["bgr"], up["name"])
        right.subheader("Detections")
        right.image(cv2.cvtColor(draw_detections(up["bgr"], feats, extractor.detector.names),
                                 cv2.COLOR_BGR2RGB), use_container_width=True)
        if selected is None:
            st.warning("No checkpoint available, so interactions are not predicted.")
        return

    from utils.inference import interactions_from
    from visualization.detection_visualizer import draw_detections

    right.subheader("Detections")
    right.image(cv2.cvtColor(draw_detections(up["bgr"], analysis.feats, extractor.detector.names),
                             cv2.COLOR_BGR2RGB), use_container_width=True)
    n_h = sum(e["is_human"] for e in analysis.entities)
    st.write(f"**{len(analysis.entities)}** entities ({n_h} people, {len(analysis.entities) - n_h} objects), "
             f"**{analysis.num_pairs}** candidate human-object pairs.")
    if n_h == 0:
        st.warning("No person detected, so there are no human-object pairs to classify.")
        return
    if analysis.num_pairs == 0:
        st.warning("A person was found but no other entity, so there is nothing to pair them with.")
        return

    inter = interactions_from(analysis, threshold)
    st.subheader(f"Predicted interactions (probability ≥ {threshold:.2f})")
    if inter:
        st.dataframe(pd.DataFrame(interaction_table(inter, analysis.entities)), use_container_width=True,
                     hide_index=True)
    else:
        st.write("No interaction reaches this threshold. Lower it in the sidebar to see weaker predictions.")
    st.caption("Scores are independent per-action sigmoid probabilities (multi-label), not a distribution. "
               "Agent-only actions (stand, walk, run, smile) are not shown: the graph has no self-pairs "
               "to score them on. "
               f"Latency: detector {analysis.latency_ms['detector']:.0f} ms, "
               f"graph model {analysis.latency_ms['graph_model']:.1f} ms.")
    with st.expander("Graph settings used (read from the checkpoint)"):
        st.write({"model": lm.model_name, "candidate_mode": lm.candidate_mode,
                  "bidirectional": lm.bidirectional, "trained_on": selected.data})


def page_scene_graph():
    st.title("Scene Graph")
    analysis, lm, err = get_analysis()
    if err:
        st.error(err)
    if analysis is None:
        st.write("Analyse an image on the **Image Analysis** page first (needs a checkpoint).")
        return
    from utils.inference import interactions_from

    inter = interactions_from(analysis, threshold)
    show_all = st.checkbox("Show entities without a predicted interaction", value=False)
    g = build_scene_graph(analysis.entities, inter, show_all_entities=show_all)
    show_fig(draw_scene_graph(g, title=f"Predicted interactions (probability ≥ {threshold:.2f})"))
    st.caption("Green: people. Orange: objects. Edge width grows with confidence.")
    if inter:
        st.dataframe(pd.DataFrame(interaction_table(inter, analysis.entities)), use_container_width=True,
                     hide_index=True)
    else:
        st.write("No interaction reaches the threshold.")


def page_explainability():
    st.title("Explainability")
    st.warning("Graph-level explanations only: they rank entities and spatial edges by how much the model "
               "used them. They do not show which pixels or body parts mattered.")
    analysis, lm, err = get_analysis()
    if err:
        st.error(err)
    if analysis is None:
        st.write("Analyse an image on the **Image Analysis** page first (needs a checkpoint).")
        return
    from utils.inference import interactions_from

    inter = interactions_from(analysis, 0.0, top_k_per_pair=1)
    inter.sort(key=lambda x: -x["prob"])
    if not inter:
        st.write("There are no human-object pairs in this image to explain.")
        return
    ents = {e["idx"]: e for e in analysis.entities}
    i = st.selectbox("Interaction to explain", range(min(len(inter), 30)), format_func=lambda k: (
        f'{ents[inter[k]["human"]]["label"]} #{inter[k]["human"]} → {inter[k]["action"]} → '
        f'{ents[inter[k]["object"]]["label"]} #{inter[k]["object"]}  (p={inter[k]["prob"]:.2f})'))
    sel = inter[i]
    epochs = st.slider("GNNExplainer optimisation steps", 20, 200, 60, 20)
    if not st.button("Explain this prediction", type="primary"):
        return

    import torch
    from data.vcoco_loader import COCO_CLASSES
    from explainability import (HOIExplainer, HOIIntegratedGradients, analyze_pair_attention,
                                evaluate_explanation, plot_attention_distribution)
    from utils.inference import explanation_text
    from visualization import visualize_bbox_attribution, visualize_explanation_graph

    data, model = analysis.data, lm.model
    p, a = sel["pair_idx"], sel["action_idx"]
    with st.spinner("Running GNNExplainer..."):
        exp = HOIExplainer(model=model, epochs=epochs, lr=0.03).explain(data, pair_idx=p, action_idx=a)
    ig = None
    try:
        with st.spinner("Running Integrated Gradients..."):
            ig = HOIIntegratedGradients(model).attribute(data, pair_idx=p, action_idx=a, n_steps=20,
                                                         attribute_pair_attr=True)
    except Exception as e:  # keep the rest of the page usable
        st.info(f"Integrated Gradients unavailable for this model: {e}")

    st.markdown(explanation_text(analysis, p, a, exp, ig))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Important nodes and edges (GNNExplainer)")
        show_fig(visualize_explanation_graph(data=data, edge_mask=exp.edge_mask, node_mask=exp.node_mask,
                                             target_pair=exp.target_pair, action_name=sel["action"]))
    with c2:
        st.subheader("Node importance on the boxes")
        rgb = cv2.cvtColor(st.session_state["upload"]["bgr"], cv2.COLOR_BGR2RGB)
        show_fig(visualize_bbox_attribution(image_or_size=rgb, data=data, node_mask=exp.node_mask,
                                            target_pair=exp.target_pair, action_name=sel["action"],
                                            class_names=dict(enumerate(COCO_CLASSES))))
        st.caption("Colour shows how important each *entity* is to the graph model. It is not a pixel heatmap.")

    st.subheader("Feature importance (Integrated Gradients)")
    if ig is not None:
        ni = ig.node_importance.detach().cpu().numpy()
        st.bar_chart(pd.DataFrame({"importance": ni},
                                  index=[f'{ents[k]["label"]} #{k}' for k in range(len(ni))]))
        if ig.pair_attr_importance is not None:
            pi = ig.pair_attr_importance.detach().cpu().numpy()
            row = pi[p] if pi.ndim == 2 and p < pi.shape[0] else pi.reshape(-1)
            if row.shape[0] == len(EDGE_FEATURES):
                st.bar_chart(pd.DataFrame({"importance": np.abs(row)}, index=EDGE_FEATURES))
                st.caption("Spatial relation features of the selected pair (absolute attribution).")
        st.caption(f"Completeness error (Captum delta): {float(ig.delta):.4f}. "
                   "Large values mean the path integral is poorly approximated.")

    st.subheader("Attention")
    try:
        att = analyze_pair_attention(model, data, pair_idx=p)
        show_fig(plot_attention_distribution(att))
        st.caption(att["causality_caveat"])
    except Exception as e:
        st.info(f"Attention is only available for attention models (gat, edge_aware_gatv2). ({e})")

    st.subheader("Explanation quality")
    with torch.no_grad():
        met = evaluate_explanation(model, data, pair_idx=p, action_idx=a, edge_mask=exp.edge_mask,
                                   node_mask=exp.node_mask, top_k_ratio=0.3)
    st.dataframe(pd.DataFrame([met]).round(4), hide_index=True, use_container_width=True)
    st.caption("Fidelity+ (higher is better): drop in probability when the top-ranked parts are removed. "
               "Fidelity− (lower is better): drop when only those parts are kept. One example is an anecdote; "
               "save several and read the averages on the Model Evaluation page.")
    if st.button("Save these metrics to results/"):
        append_explanation_metrics(RESULTS_DIR, {
            "run": selected.name, "model": selected.model, "data": selected.data, "boxes": selected.boxes,
            "image": st.session_state["upload"]["name"], "pair": [int(v) for v in exp.target_pair],
            "action": sel["action"], "explainer_steps": epochs, **{k: float(v) for k, v in met.items()}})
        st.success("Saved.")


def page_evaluation():
    from evaluation.aggregate import EXPECTED_MODELS, build_summary, collect_runs

    st.title("Model Evaluation")
    if not RESULTS_DIR.exists() or not list(RESULTS_DIR.rglob("metrics.json")):
        st.info("No results yet. Train a model with `python -m training.train --model edge_aware_gatv2 "
                "--data vcoco`, then reload this page. Nothing is shown until real runs exist.")
        return
    all_runs = collect_runs(RESULTS_DIR)
    real = all_runs[all_runs["data"] == "vcoco"] if len(all_runs) else all_runs
    other = all_runs[all_runs["data"] != "vcoco"] if len(all_runs) else all_runs

    boxes = st.radio("Box source", ["gt", "detected"], horizontal=True)
    st.caption("`gt` = ground-truth boxes (isolates the interaction model). `detected` = detector boxes "
               "(end-to-end). The two are not comparable; never quote `gt` as end-to-end performance.")

    st.subheader("Baseline and architecture comparison (V-COCO runs, test set)")
    if len(real) == 0:
        st.warning("No finished V-COCO runs yet. Existing runs used synthetic or fake data and are not benchmarks.")
    else:
        summary = build_summary(real, boxes=boxes)
        st.dataframe(summary, hide_index=True, use_container_width=True)
        num = real[real["boxes"] == boxes].groupby("model")["mAP"].mean()
        num = num.reindex([m for m in EXPECTED_MODELS if m in num.index])
        if len(num):
            st.bar_chart(num.rename("test mAP"))
        st.caption("mAP here is the repo's own multi-label AP; check evaluation/vcoco_evaluator.py notes "
                   "before calling it the official V-COCO role AP. 'not run' means no finished run exists.")

    st.subheader("Ablations")
    from evaluation.ablation import collect_ablation_results

    runs_df, _ = collect_ablation_results(RESULTS_DIR / "ablations")
    if len(runs_df) == 0:
        st.write("No ablation runs found in results/ablations.")
    else:
        real_ab = runs_df[runs_df["data"] == "vcoco"]
        if len(real_ab) == 0:
            st.warning(f"{len(runs_df)} ablation runs exist but none used V-COCO data "
                       f"({sorted(runs_df['data'].astype(str).unique())}). They are pipeline tests, not results.")
        else:
            metric_cols = ["mAP", "mAP_rare", "mAP_common", "f1_micro", "f1_macro"]
            agg = real_ab.groupby(["ablation_type", "exp_name"], sort=False)[metric_cols].mean().round(4)
            agg["seeds"] = real_ab.groupby(["ablation_type", "exp_name"], sort=False).size()
            st.dataframe(agg.reset_index(), hide_index=True, use_container_width=True)

    st.subheader("Training curves")
    with_hist = [r for r in runs if load_history(r.folder) is not None]
    if not with_hist:
        st.write("No training histories found.")
    else:
        k = st.selectbox("Run", range(len(with_hist)), format_func=lambda i: with_hist[i].label)
        h = load_history(with_hist[k].folder)
        c1, c2 = st.columns(2)
        loss_cols = [c for c in ("train_loss", "val_loss") if c in h.columns]
        if loss_cols:
            c1.line_chart(h.set_index("epoch")[loss_cols] if "epoch" in h.columns else h[loss_cols])
        if "val_mAP" in h.columns:
            c2.line_chart(h.set_index("epoch")["val_mAP"] if "epoch" in h.columns else h["val_mAP"])
        if not with_hist[k].is_real:
            st.warning("This run is not on V-COCO; the curves only show that training runs.")

    st.subheader("Explanation metrics")
    em = load_explanation_metrics(RESULTS_DIR)
    if len(em) == 0:
        st.write("None saved yet. Use **Save these metrics** on the Explainability page.")
    else:
        em_real = em[em["data"] == "vcoco"] if "data" in em.columns else em.iloc[0:0]
        if len(em_real) == 0:
            st.warning(f"{len(em)} saved explanations come from non-V-COCO checkpoints; not shown as results.")
        else:
            cols = [c for c in ("fidelity_plus", "fidelity_minus", "delta_fidelity", "edge_sparsity",
                                "node_sparsity") if c in em_real.columns]
            st.write(f"{len(em_real)} explanations on V-COCO checkpoints.")
            st.dataframe(em_real.groupby("model")[cols].agg(["mean", "std", "count"]).round(4),
                         use_container_width=True)
            st.caption("Averages over the explanations you saved, on images you chose. Small samples.")

    if len(other):
        with st.expander(f"{len(other)} runs on non-V-COCO data (pipeline tests, NOT benchmark results)"):
            st.dataframe(other[["run", "model", "data", "boxes", "seed", "mAP"]], hide_index=True)


{"Home": page_home, "Image Analysis": page_image_analysis, "Scene Graph": page_scene_graph,
 "Explainability": page_explainability, "Model Evaluation": page_evaluation}[page]()
