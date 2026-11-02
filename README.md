# HOI-GNN: Explainable Human-Object Interaction Detection with Relational GNNs

Detects people and objects in an image, builds a graph of their relationships, predicts V-COCO
interactions (for example *person -> hold -> cell phone*) with an edge-aware GATv2, and explains the
prediction at graph level (GNNExplainer, attention, Integrated Gradients).

**Research question.** Does relational graph reasoning improve HOI detection over non-graph
baselines, and can the predictions be explained reliably?

> **Results policy.** This README contains no performance numbers. Every number shown by the app or
> in the report must come from files under `results/` written by the training and evaluation code.
> Runs on synthetic or fake data are pipeline tests, never benchmark results.

## Pipeline

1. YOLOv8s detects people and objects (frozen).
2. ROIAlign pools an 896-dim appearance embedding per box from the detector's three feature maps.
3. Each image becomes a graph: nodes = entities (appearance, class, box, confidence), edges = 9 spatial relation features.
4. A GNN passes messages; an MLP head scores 26 V-COCO actions per human-object pair (multi-label, sigmoid, masked BCE).
5. Explainability: GNNExplainer, attention extraction, Captum Integrated Gradients, fidelity/sparsity metrics.

Models compared: object-class prior, MLP (no message passing), GCN, GAT, edge-aware GATv2.

## 1. Install (Windows, conda)

```cmd
conda create -n hoi-gnn python=3.11 -y
conda activate hoi-gnn
pip install -r requirements.txt
python -m scripts.check_env
```

`requirements.txt` pins PyTorch 2.5.1 (CUDA 12.1 build), PyG 2.6.1 and numpy 1.26.4. Use
`num_workers: 0` on Windows (already the default). CPU works; it is just slow. Record your own
`check_env` output for the report, do not copy someone else's.

## 2. Data

V-COCO annotates COCO 2014 images and has no images of its own.

```cmd
git clone https://github.com/s-gupta/v-coco.git datasets\vcoco
python -m scripts.download_data --split train --max-images 50
```

Run `python -m scripts.download_data --help` for options. COCO images are about 19 GB (train2014 + val2014);
extract one zip at a time if disk is tight. The dataset class looks for images under `datasets/coco/images`
by default (`image_dir` in `data/dataset.py`); put or link the extracted images there.

### Warning: silent dummy features

If an image is neither in `cache/features*` nor readable from `image_dir`, `VCOCODataset` falls back to
**all-zero appearance features** (`fallback_to_gt_if_uncached=True`). Training then still runs, and
`metrics.json` still says `data: vcoco`, but the model only sees boxes and classes. Before any real
experiment, check that images are found, and report appearance-feature results only for runs where they were.

## 3. Train and evaluate

```cmd
python -m training.train --model prior            --data vcoco --boxes gt --seed 42
python -m training.train --model mlp              --data vcoco --boxes gt --seed 42
python -m training.train --model gcn              --data vcoco --boxes gt --seed 42
python -m training.train --model gat              --data vcoco --boxes gt --seed 42
python -m training.train --model edge_aware_gatv2 --data vcoco --boxes gt --seed 42
python -m evaluation.aggregate --results results
```

Each run writes `results/<model>_<boxes>_seed<seed>/` with `metrics.json`, `history.json`, `best.pt`, `last.pt`.
Settings: `configs/model_config.yaml` (model, graph) and `configs/training_config.yaml` (optimiser, loss).
Use `--data synthetic` or `--data fake` for a smoke test without any download.

- `--boxes gt`: ground-truth boxes, isolates the interaction model.
- `--boxes detected`: detector boxes, end-to-end. The two are not comparable. Never quote `gt` as end-to-end performance.
- The repo's `mAP` is its own reimplementation (`evaluation/vcoco_evaluator.py`); diff it against the official `s-gupta/v-coco` evaluator before calling it official role AP.

### Ablations

```cmd
python -m evaluation.ablation --ablation all --data vcoco --boxes gt --seeds 42,43,44 --out results/ablations
```

Covers features, architecture, depth, edge attributes and topology. Use `--dry-run` to list the runs first.

## 4. Run the app

```cmd
streamlit run app.py
```

Pages: Home, Image Analysis, Scene Graph, Explainability, Model Evaluation. The sidebar lists every
checkpoint in `results/`; checkpoints trained on non-V-COCO data are labelled "NOT a benchmark".
Without a checkpoint, detection still works. The inference path imports nothing from `training/`.

## 5. Tests

```cmd
pytest tests -q
```

## Memory tips (4-6 GB GPU)

Detector peak memory was about 85 MB on one image. Cache features once (`cache/`), keep the detector frozen,
use `batch_size` 16-32, close other GPU programs (game overlays), and use AMP (on by default on CUDA).

## Project layout

See `docs/CONTRACTS.md` for the data contracts between modules, `PLAN.md` for the phase checklist,
`docs/REPORT.md` for the report skeleton, `docs/DEPLOYMENT.md` and `docs/VIVA.md`.

```
config.yaml, configs/        settings
detection/                   YOLOv8, ROI features, feature cache
data/                        V-COCO loader, graph builder, PyG dataset, synthetic data
models/                      prior, MLP, GCN, GAT, edge-aware GATv2
training/                    losses, trainer, train CLI
evaluation/                  metrics, V-COCO evaluator, ablations, aggregation
explainability/              GNNExplainer, attention, IG, explanation metrics
visualization/               detections, scene graph, explanation plots
utils/                       checkpoints, seeds, inference path, results reader
app.py                       Streamlit app
```
