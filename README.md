# HOI-GNN: Explainable Human-Object Interaction Detection with Relational GNNs

A modular, explainable framework for detecting Human-Object Interactions (HOI) on the V-COCO dataset using edge-aware Graph Attention Networks (GATv2), baseline GNN architectures (GCN, GAT), and MLP baselines.

---

## Environment Verification (Local Machine)

Verified on local system using `python -m scripts.check_env`:

```text
Python        : 3.11.18
torch         : 2.5.1+cu121
torchvision   : 0.20.1+cu121
CUDA available: True
GPU           : NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB)
CUDA runtime  : 12.1
torch_geometric: 2.6.1
GATv2Conv batch forward OK, output shape (5, 32)
ultralytics   : 8.4.197
captum        : 0.7.0
opencv        : 4.11.0
numpy         : 1.26.4
pandas        : 3.0.6
scikit-learn  : 1.9.1
networkx      : 3.6.1
matplotlib    : 3.11.2
streamlit     : 1.66.1
pycocotools   : imported OK
CUDA matmul test OK (0.04s)

ALL CHECKS PASSED
```

---

## Quickstart

### 1. Environment Setup
```bash
conda activate hoi-gnn
python -m scripts.check_env
```

### 2. Download Data & Annotations
```bash
python -m scripts.download_data
```

### 3. Run Phase 2 Tests
```bash
pytest tests/test_phase2.py -v
```

### 4. Run Single Image Detection Demo
```bash
python -m detection.run_demo --image datasets/sample/bus.jpg
```

---

## Project Structure

```
hoi-GNN/
├── configs/
│   └── model_config.yaml       # GNN and training hyperparameters
├── config.yaml                 # Detection and path configurations
├── data/                       # V-COCO dataset loader, graph builder, PyG dataset
│   ├── dataset.py              # PyG Dataset with mini-batch collation
│   ├── graph_builder.py        # 9 normalized spatial edge features and HOIData
│   ├── vcoco_loader.py         # V-COCO parsing, role handling and y_mask
│   └── splits/                 # V-COCO split IDs (train, val, test)
├── detection/                  # YOLOv8 detector, multi-scale ROI feature extraction
├── docs/
│   └── CONTRACTS.md            # Data contracts between pipeline phases
├── evaluation/                 # Metrics, VCOCO evaluator, ablation studies
├── explainability/             # GNNExplainer, attention analysis, Captum IG
├── models/                     # GNN models (MLP, GCN, GAT, Edge-Aware GATv2)
├── scripts/                    # Environment check and data preparation scripts
├── tests/                      # Unit and integration test suites
│   └── test_phase2.py          # 15 unit tests covering spatial features & PyG batching
├── training/                   # Trainer, loss functions, checkpointing
├── utils/                      # Helper utilities and inspection scripts
├── visualization/              # Detections and scene graph visualizers
├── PLAN.md                     # Phase roadmap and tracking
└── app.py                      # Interactive Streamlit application
```
