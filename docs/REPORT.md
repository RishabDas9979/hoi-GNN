# Report skeleton: Explainable HOI Detection with Relational GNNs

Rule: every number below is copied from `results/` after the run exists. Leave "TODO" until then. State the
data (V-COCO / synthetic), box source (gt / detected), seed(s) and number of images for each table.

## 1. Problem and research question
Does relational graph reasoning improve HOI detection over non-graph baselines, and can predictions be explained reliably?

## 2. Dataset
V-COCO (26 actions, COCO 2014 images). Official train/val/test splits. Actions with no object partner
(stand, walk, run, smile) have only an agent role; positives with no annotated object are masked, not negatives
(`y_mask`). TODO: image counts per split actually used (`python -m data.vcoco_loader --split train`).

## 3. Method
- Detector: YOLOv8s, frozen. ROIAlign 7x7 over 3 feature levels -> 896-dim embedding per box.
- Graph: nodes = entities; node input = appearance + learned class embedding + normalised box (+ confidence/type in the graph data);
  edges carry 9 spatial features (dx, dy, distance, log width/height/area ratios, IoU, two area fractions).
- Models: object-class prior, MLP, GCN, GAT, edge-aware GATv2 (LayerNorm, dropout, residuals, configurable depth).
- Head: concat(human, object, pair attributes) -> MLP -> 26 logits; masked BCE (or focal / weighted BCE); sigmoid at inference.
- Training: AdamW, warmup + cosine schedule, AMP, early stopping on validation mAP, fixed seeds.

## 4. Experimental setup
Hardware (RTX 3050 Laptop GPU), library versions (`pip freeze`), config files, seeds. TODO.

## 5. Results
### 5.1 Baselines and architectures (test set, gt boxes)
TODO: table from `results/summary_gt.md` (`python -m evaluation.aggregate --results results`).
### 5.2 Detector boxes (end-to-end)
TODO: `--boxes detected` runs, reported separately from 5.1.
### 5.3 Ablations
TODO: `results/ablations/ablation_summary.csv` for features, architecture, depth, edge attributes, topology. Report seeds and spread.

## 6. Explainability
Methods: GNNExplainer (edge/node masks), GATv2 attention, Integrated Gradients (Captum).
Metrics: fidelity+, fidelity-, sparsity, stability, randomised-model sanity check.
TODO: averages from `results/explanation_metrics.jsonl` (state how many explanations and how they were chosen).
**Scope:** graph-level attribution over entities and spatial edges. It does not identify pixels or body parts,
and attention is not evidence of causal importance.

## 7. Limitations
- Own AP implementation; not yet diffed against the official V-COCO evaluator. TODO.
- Self-pairs for agent-only actions are not created by the graph builder, so those four actions are not predicted in the app.
- Role identity is lost for multi-role actions (all roles count as positive).
- Zero-feature fallback in `VCOCODataset` if images are missing (see README).
- A model may not beat the MLP on a small subset: report what you get.
- Small explanation sample; no user study.

## 8. What is ours and what is existing work
- Ours (engineering): graph representation and 9 edge features, edge-aware GATv2 integration, masked multi-label targets, explainability evaluation, app.
- Existing: YOLOv8 (Ultralytics), GATv2 (Brody et al.), GCN, GAT, GNNExplainer (Ying et al.), Integrated Gradients (Sundararajan et al.), V-COCO and COCO datasets.

## 9. References
V-COCO (Gupta and Malik); COCO (Lin et al.); GATv2 (Brody et al., 2022); GNNExplainer (Ying et al., 2019);
Integrated Gradients (Sundararajan et al., 2017); PyTorch Geometric; Captum; Ultralytics YOLOv8.
