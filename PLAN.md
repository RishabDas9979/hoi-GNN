# Implementation Plan: Explainable HOI Detection with Relational GNNs

Contracts between modules are in `docs/CONTRACTS.md`. Config lives in `config.yaml` and `configs/model_config.yaml`.
Time figures are estimates for an RTX 3050 Ti (or a free Colab T4), not measurements.

## 0. Dependency and parallelism map

```
P0 Setup ──► P1 Detection ──┐
        └──► P2 Data+Graph ─┼─► P4 Models ─► P5 Train+Eval ─► P6 Ablations ─┐
             P3 Synthetic ──┘                     │                          ├─► P9 Docs+Report
                                                  └─► P7 Explainability ─────┤
                              P8 App (skeleton can start after P2) ──────────┘
```

| Can run in parallel | Why |
|---|---|
| P1 and P2 | Both only need the cache format in CONTRACTS.md |
| P3 and P1/P2 | Synthetic data writes the same cache format, so it needs no detector or dataset |
| P4 and P1/P2 | Models only need the graph format; test them on P3 synthetic graphs |
| P7 and P6 | Explainability needs one trained model, not the full ablation grid |
| P8 skeleton and P5/P6/P7 | UI pages can be built against a saved checkpoint and results files |
| Download of V-COCO/COCO images and everything in P1, P3, P4 | Start the download first and code while it runs |

Strictly sequential: P0 → (P1, P2) → P5 → P6, and P9 last.

---

## Phase 0: Environment and initialization (DONE)
- [x] Project skeleton, `config.yaml`, `model_config.yaml`, `requirements.txt`
- [x] `scripts/check_env.py`, `docs/CONTRACTS.md`
- [x] Run `check_env.py` on the real machine and record the output in the README
- [x] Start the V-COCO annotation and COCO image download now (long pole)

## Phase 1: Object detection and ROI features (DONE)
Output: per-image cached records (CONTRACTS section 1).
- [x] 1.1 `detection/detector.py`: load pretrained YOLOv8 (frozen), return boxes, class ids, scores; filter to `max_persons` / `max_objects`
- [x] 1.2 `detection/roi_extractor.py`: ROIAlign over a backbone feature map (7x7 pooling), project to `feature_dim`
- [x] 1.3 `detection/feature_extractor.py`: run detector + ROI, write `.pt` cache files, skip images already cached, batch extraction with `extract_many`
- [x] 1.4 Handle empty cases: no detections, no persons
- [x] 1.5 CPU fallback and AMP; batch size from config
- [x] 1.6 Optional: Faster R-CNN backend as fallback if YOLO feature hooks are awkward
- [x] 1.7 Tests: output shapes, box range, empty image, cache round-trip

## Phase 2: V-COCO loading and graph construction (DONE)
Output: graph `Data` objects (CONTRACTS section 2).
- [x] 2.1 Inspect the real V-COCO files before writing the loader (field names, action list, role lists, split files). Do not assume a format
- [x] 2.2 `data/vcoco_loader.py`: parse annotations, human boxes, object boxes, action and role labels
- [x] 2.3 Handle actions with no object role: build targets so a missing partner is valid, use `y_mask`
- [x] 2.4 Pair construction: match human-object pairs to GT labels, build multi-hot targets
- [x] 2.5 Split handling: use official train/val/test; subset via `max_images`
- [x] 2.6 `data/graph_builder.py`: nodes, typed human-to-object edges, reverse edges, 9 edge attributes, normalization
- [x] 2.7 Candidate-edge modes: `all_human_object`, `knn`, `distance_threshold`
- [x] 2.8 `data/dataset.py`: PyG Dataset reading from cache, correct batching of `pair_index`
- [x] 2.9 Tests: empty graph, one person, many people, batch offset of `pair_index` (`tests/test_phase2.py` passing 15/15)

## Phase 3: Synthetic data and pipeline validation (DONE - branch phase-3)
Output: whole pipeline runnable with no downloads. Never reported as results.
- [x] 3.1 data/synthetic_dataset.py: random boxes, features and labels written in the cache format, with a learnable signal (e.g. label depends on distance/class) so training can be sanity-checked
- [x] 3.2 End-to-end smoke test: synthetic cache → graph → forward pass → one train step
- [x] 3.3 Overfit test: a model should reach near-zero loss on 20 synthetic graphs (9/9 tests passing in tests/test_phase3.py)

## Phase 4: Models
Output: all models share the interface `model(data) -> logits [P, A]`.
- [ ] 4.1 `models/interaction_head.py`: concat human + object embedding + pair attributes → MLP → A logits
- [ ] 4.2 Object-class prior baseline (frequency of action per object class; no learning on pixels)
- [ ] 4.3 `models/mlp_baseline.py`: same inputs and head, no message passing
- [ ] 4.4 `models/gcn.py`: GCNConv stack
- [ ] 4.5 `models/gat.py`: GATConv stack
- [ ] 4.6 `models/edge_aware_gatv2.py`: GATv2Conv with `edge_dim`, LayerNorm, dropout, residuals, configurable depth, `return_attention`
- [ ] 4.7 Feature-subset switch: spatial / appearance / semantic / all
- [ ] 4.8 Tests: forward pass shapes for every model, zero-pair graphs, gradient flows

## Phase 5: Training and evaluation
- [ ] 5.1 `training/losses.py`: BCEWithLogits, focal, weighted BCE; apply `y_mask`
- [ ] 5.2 `training/trainer.py`: AdamW, cosine schedule, AMP, early stopping, grad accumulation
- [ ] 5.3 Checkpoint saving: weights, optimizer, config, history, best model
- [ ] 5.4 `training/train.py`: CLI `--model --config --seed`
- [ ] 5.5 `evaluation/metrics.py`: per-action AP, mAP, precision, recall, micro/macro F1, rare vs common split, latency
- [ ] 5.6 `evaluation/vcoco_evaluator.py`: role AP following the official definition where feasible. Document every deviation. Study the official `s-gupta/v-coco` evaluator for the convention
- [ ] 5.7 Report GT-box and detector-box results separately
- [ ] 5.8 Train all 5 models on the same split and seed; save results to `results/`
- [ ] 5.9 Plots: training curves, baseline comparison table

## Phase 6: Ablations
Run only after Phase 5 works. Reduce the grid if time is short.
- [ ] 6.1 Topology: directed vs bidirectional vs alternative candidate edges
- [ ] 6.2 Features: spatial only, appearance only, semantic only, combined
- [ ] 6.3 Architecture: GCN vs GAT vs GATv2
- [ ] 6.4 Depth: 1, 2, 3 layers
- [ ] 6.5 Edge attributes: without vs with
- [ ] 6.6 `evaluation/ablation.py`: loop, write CSV, make tables and plots
- [ ] 6.7 If time allows, repeat the key comparison across 3 seeds

## Phase 7: Explainability
- [ ] 7.1 `explainability/gnn_explainer.py`: PyG Explainer + GNNExplainer for one pair; verify the model interface works with PyG 2.6.1 explainers (may need a thin wrapper)
- [ ] 7.2 `explainability/attention_analysis.py`: extract GATv2 attention, plot per edge; note attention is not proof of importance
- [ ] 7.3 `explainability/integrated_gradients.py`: Captum IG on node and edge features
- [ ] 7.4 `explainability/explanation_metrics.py`: fidelity, sparsity, stability across seeds, edge-removal and feature-perturbation tests, randomized-model sanity check
- [ ] 7.5 `visualization/explanation_visualizer.py`: node/edge importance on the scene graph
- [ ] 7.6 State clearly that this is graph-level attribution, not pixel-level

## Phase 8: Streamlit application
- [ ] 8.1 `visualization/detection_visualizer.py`: boxes and labels on the image
- [ ] 8.2 `visualization/scene_graph.py`: NetworkX graph with predicted action edges and confidences
- [ ] 8.3 `app.py` Home page
- [ ] 8.4 Image Analysis page: upload, detect, build graph, predict, show scores
- [ ] 8.5 Scene Graph page
- [ ] 8.6 Explainability page: pick an interaction, show nodes, edges, feature importance, attention, text explanation
- [ ] 8.7 Model Evaluation page: reads the real files in `results/` only; shows a clear message if none exist
- [ ] 8.8 Inference path loads a checkpoint and does not import training code
- [ ] 8.9 Handle no-person images and missing checkpoint gracefully

## Phase 9: Testing, documentation, report
- [ ] 9.1 Run the full test suite; fix failures
- [ ] 9.2 README: install, data setup, train, evaluate, run app, memory tips
- [ ] 9.3 Final report: problem, method, dataset, setup, results, ablations, explainability, limitations
- [ ] 9.4 State clearly what is engineering (graph representation, edge-aware GATv2, explainability evaluation) and what is existing work
- [ ] 9.5 Deployment note: ship the inference app with a small checkpoint; do not train on a free host
- [ ] 9.6 Viva preparation: likely questions and honest answers about limits
