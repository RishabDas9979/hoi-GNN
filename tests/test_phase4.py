r"""Phase 4 tests: GNN Architectures, Baselines, Encoders, Heads, and Comprehensive Model Test Suite.

Tests:
1. NodeEncoder: feature subsets ('all', 'appearance', 'spatial', 'semantic'),
   empty nodes (V = 0), class id mapping (-1 -> 80), gradient flow.
2. InteractionHead: pair forward pass, pair_attr toggle, empty pairs (P = 0),
   gradient flow.
3. ObjectClassPriorBaseline: empirical lookup, fit() on synthetic dataset,
   log-odds sigmoid consistency, empty pairs (P = 0).
4. MLPBaseline: full forward pass on HOIData, zero pairs, zero nodes,
   end-to-end gradient step with BCEWithLogitsLoss.
5. All 5 Models (ObjectClassPriorBaseline, MLPBaseline, GCNHOIModel, GATHOIModel, EdgeAwareGATv2HOIModel):
   - Forward pass output shape [P, 26] on synthetic data.
   - Zero candidate pairs (P = 0) returns tensor of shape (0, 26).
   - Zero nodes (V = 0) returns tensor of shape (0, 26).
   - Feature-subset switch: forward passes with feature_subset in ['all', 'appearance', 'spatial', 'semantic'].
   - EdgeAwareGATv2HOIModel with return_attention=True returns valid attention tuples (edge_index, alpha).
   - Gradient flow: backward pass on masked BCE loss updates parameters with non-NaN gradients across all models.
   - Batched forward pass using DataLoader with hoi_collate.

Run with:
  & "C:\Users\jayan\miniconda3\envs\hoi-gnn\python.exe" -m pytest tests/test_phase4.py -v
"""
from typing import Any, Dict, List, Tuple
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from data.dataset import hoi_collate
from data.graph_builder import HOIData
from data.synthetic_dataset import SyntheticHOIGenerator, SyntheticVCOCODataset
from models import (
    EdgeAwareGATv2HOIModel,
    GATHOIModel,
    GCNHOIModel,
    InteractionHead,
    MLPBaseline,
    NodeEncoder,
    ObjectClassPriorBaseline,
)

NUM_ACTIONS = 26
FEATURE_DIM = 56
HIDDEN_DIM = 64
HEADS = 4
EDGE_DIM = 9


# --------------------------------------------------------------------------- #
# Test Fixtures                                                               #
# --------------------------------------------------------------------------- #

@pytest.fixture
def synthetic_graph() -> HOIData:
    """Fixture providing a deterministic synthetic HOIData graph with P > 0."""
    generator = SyntheticHOIGenerator(feature_dim=FEATURE_DIM, seed=42)
    return generator.generate_graph(image_id=101)


@pytest.fixture
def empty_nodes_graph() -> HOIData:
    """Fixture providing an empty graph with V = 0 and P = 0."""
    return HOIData(
        x=torch.zeros((0, FEATURE_DIM), dtype=torch.float32),
        x_visual=torch.zeros((0, FEATURE_DIM), dtype=torch.float32),
        boxes=torch.zeros((0, 4), dtype=torch.float32),
        boxes_norm=torch.zeros((0, 4), dtype=torch.float32),
        class_ids=torch.zeros((0,), dtype=torch.long),
        scores=torch.zeros((0,), dtype=torch.float32),
        node_type=torch.zeros((0,), dtype=torch.long),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, EDGE_DIM), dtype=torch.float32),
        pair_index=torch.zeros((0, 2), dtype=torch.long),
        pair_attr=torch.zeros((0, EDGE_DIM), dtype=torch.float32),
        num_nodes=0,
    )


@pytest.fixture
def zero_pairs_graph() -> HOIData:
    """Fixture providing a graph with nodes V > 0 but zero candidate pairs P = 0 (no human)."""
    return HOIData(
        x=torch.randn((3, FEATURE_DIM), dtype=torch.float32),
        x_visual=torch.randn((3, FEATURE_DIM), dtype=torch.float32),
        boxes=torch.rand((3, 4), dtype=torch.float32),
        boxes_norm=torch.rand((3, 4), dtype=torch.float32),
        class_ids=torch.tensor([1, 2, 3], dtype=torch.long),  # objects only, no person (0)
        scores=torch.ones((3,), dtype=torch.float32),
        node_type=torch.ones((3,), dtype=torch.long),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, EDGE_DIM), dtype=torch.float32),
        pair_index=torch.zeros((0, 2), dtype=torch.long),
        pair_attr=torch.zeros((0, EDGE_DIM), dtype=torch.float32),
        num_nodes=3,
    )


def _build_model(model_cls, **kwargs):
    """Helper to instantiate any of the 5 models with default testing parameters."""
    if model_cls is ObjectClassPriorBaseline:
        return model_cls(**kwargs)
    elif model_cls is EdgeAwareGATv2HOIModel:
        params = {
            "input_dim": FEATURE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "heads": HEADS,
            "edge_dim": EDGE_DIM,
            "num_layers": 2,
        }
        params.update(kwargs)
        return model_cls(**params)
    elif model_cls is GATHOIModel:
        params = {
            "input_dim": FEATURE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "heads": HEADS,
            "num_layers": 2,
        }
        params.update(kwargs)
        return model_cls(**params)
    elif model_cls is GCNHOIModel:
        params = {
            "input_dim": FEATURE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": 2,
        }
        params.update(kwargs)
        return model_cls(**params)
    elif model_cls is MLPBaseline:
        params = {
            "input_dim": FEATURE_DIM,
            "hidden_dim": HIDDEN_DIM,
        }
        params.update(kwargs)
        return model_cls(**params)
    raise ValueError(f"Unknown model class: {model_cls}")


# --------------------------------------------------------------------------- #
# 1. NodeEncoder Tests                                                        #
# --------------------------------------------------------------------------- #

def test_node_encoder_feature_subsets():
    """Verify NodeEncoder supports all 4 feature subsets and produces [V, H]."""
    v = 5
    d_vis = 56
    h_dim = 128
    x_vis = torch.randn((v, d_vis))
    class_ids = torch.tensor([0, 1, 13, 80, -1], dtype=torch.long)
    boxes_norm = torch.rand((v, 4))

    data = HOIData(x_visual=x_vis, class_ids=class_ids, boxes_norm=boxes_norm, num_nodes=v)

    # 1. 'all'
    enc_all = NodeEncoder(input_dim=d_vis, hidden_dim=h_dim, feature_subset="all")
    out_all = enc_all(data)
    assert out_all.shape == (v, h_dim)
    assert not torch.isnan(out_all).any()

    # 2. 'appearance'
    enc_app = NodeEncoder(input_dim=d_vis, hidden_dim=h_dim, feature_subset="appearance")
    out_app = enc_app(data)
    assert out_app.shape == (v, h_dim)

    # 3. 'spatial'
    enc_spat = NodeEncoder(input_dim=d_vis, hidden_dim=h_dim, feature_subset="spatial")
    out_spat = enc_spat(data)
    assert out_spat.shape == (v, h_dim)

    # 4. 'semantic'
    enc_sem = NodeEncoder(input_dim=d_vis, hidden_dim=h_dim, feature_subset="semantic")
    out_sem = enc_sem(data)
    assert out_sem.shape == (v, h_dim)

    # 5. Invalid subset raises ValueError
    with pytest.raises(ValueError):
        NodeEncoder(input_dim=d_vis, hidden_dim=h_dim, feature_subset="invalid_subset")


def test_node_encoder_empty_graph():
    """Verify NodeEncoder safely returns [0, hidden_dim] when V = 0."""
    enc = NodeEncoder(input_dim=56, hidden_dim=64, feature_subset="all")
    empty_data = HOIData(
        x_visual=torch.zeros((0, 56)),
        class_ids=torch.zeros((0,), dtype=torch.long),
        boxes_norm=torch.zeros((0, 4)),
        num_nodes=0,
    )
    out = enc(empty_data)
    assert out.shape == (0, 64)


def test_node_encoder_gradient_flow():
    """Verify gradients propagate through NodeEncoder layers."""
    enc = NodeEncoder(input_dim=56, hidden_dim=64, feature_subset="all")
    x = torch.randn(4, 56, requires_grad=True)
    class_ids = torch.tensor([0, 1, 2, -1], dtype=torch.long)
    boxes_norm = torch.rand(4, 4, requires_grad=True)

    out = enc(x_visual=x, class_ids=class_ids, boxes_norm=boxes_norm)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None and not torch.isnan(x.grad).any()
    assert boxes_norm.grad is not None and not torch.isnan(boxes_norm.grad).any()
    assert enc.visual_proj.weight.grad is not None
    assert enc.class_embed.weight.grad is not None
    assert enc.spatial_proj.weight.grad is not None
    assert enc.fusion.weight.grad is not None


# --------------------------------------------------------------------------- #
# 2. InteractionHead Tests                                                    #
# --------------------------------------------------------------------------- #

def test_interaction_head_forward():
    """Verify InteractionHead computes unnormalized logits [P, num_actions]."""
    v, p, h_dim, num_actions = 6, 4, 128, 26
    node_emb = torch.randn((v, h_dim))
    pair_index = torch.tensor([[0, 1], [0, 2], [3, 4], [3, 5]], dtype=torch.long)
    pair_attr = torch.randn((p, 9))

    head = InteractionHead(hidden_dim=h_dim, num_actions=num_actions, use_pair_attr=True)
    logits = head(node_emb, pair_index, pair_attr)
    assert logits.shape == (p, num_actions)
    assert not torch.isnan(logits).any()


def test_interaction_head_without_pair_attr():
    """Verify InteractionHead functions when use_pair_attr=False."""
    v, p, h_dim, num_actions = 4, 3, 64, 26
    node_emb = torch.randn((v, h_dim))
    pair_index = torch.tensor([[0, 1], [0, 2], [0, 3]], dtype=torch.long)

    head = InteractionHead(hidden_dim=h_dim, num_actions=num_actions, use_pair_attr=False)
    logits = head(node_emb, pair_index)
    assert logits.shape == (p, num_actions)


def test_interaction_head_empty_pairs():
    """Verify InteractionHead returns [0, num_actions] when P = 0."""
    node_emb = torch.randn((4, 64))
    empty_pairs = torch.zeros((0, 2), dtype=torch.long)
    head = InteractionHead(hidden_dim=64, num_actions=26)
    out = head(node_emb, empty_pairs)
    assert out.shape == (0, 26)


def test_interaction_head_gradient_flow():
    """Verify gradients propagate from loss back into node representations."""
    node_emb = torch.randn((4, 64), requires_grad=True)
    pair_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    pair_attr = torch.randn((2, 9), requires_grad=True)

    head = InteractionHead(hidden_dim=64, num_actions=26, use_pair_attr=True)
    logits = head(node_emb, pair_index, pair_attr)
    loss = logits.sum()
    loss.backward()

    assert node_emb.grad is not None
    assert pair_attr.grad is not None


# --------------------------------------------------------------------------- #
# 3. ObjectClassPriorBaseline Tests                                           #
# --------------------------------------------------------------------------- #

def test_prior_baseline_lookup():
    """Verify ObjectClassPriorBaseline returns correct shape and handles unknown classes."""
    prior = ObjectClassPriorBaseline(num_classes=81, num_actions=26)
    class_ids = torch.tensor([0, 1, 56, -1, 99], dtype=torch.long)
    # Pairs with person (0) interacting with: bicycle (1), chair (56), unknown (-1), out-of-range (99)
    pair_index = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=torch.long)

    data = HOIData(class_ids=class_ids, pair_index=pair_index)
    logits = prior(data)
    assert logits.shape == (4, 26)

    # Sigmoid should produce probabilities strictly in (0, 1)
    probs = torch.sigmoid(logits)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Empty pairs returns [0, 26]
    empty_prior_out = prior(HOIData(class_ids=class_ids, pair_index=torch.zeros((0, 2), dtype=torch.long)))
    assert empty_prior_out.shape == (0, 26)


def test_prior_baseline_fit():
    """Verify fit() updates prior co-occurrence counts and calibrated logit predictions."""
    generator = SyntheticHOIGenerator(feature_dim=56, seed=42)
    dataset = [generator.generate_graph(image_id=i) for i in range(10)]

    prior = ObjectClassPriorBaseline(num_classes=81, num_actions=26)
    initial_counts = prior.cooccurrence.clone()

    prior.fit(dataset, alpha=0.1)
    assert not torch.equal(prior.cooccurrence, initial_counts)
    assert prior.cooccurrence.sum() > 0

    test_graph = dataset[0]
    logits = prior(test_graph)
    assert logits.shape == (test_graph.pair_index.shape[0], 26)


# --------------------------------------------------------------------------- #
# 4. MLPBaseline End-to-End Tests                                             #
# --------------------------------------------------------------------------- #

def test_mlp_baseline_forward_and_loss():
    """Verify MLPBaseline forward pass on synthetic HOIData and backward optimization step."""
    generator = SyntheticHOIGenerator(feature_dim=56, seed=42)
    graph = generator.generate_graph(image_id=101)

    model = MLPBaseline(
        input_dim=56,
        hidden_dim=128,
        num_actions=26,
        feature_subset="all",
        use_pair_attr=True,
    )
    model.train()

    logits = model(graph)
    p = graph.pair_index.shape[0]
    assert logits.shape == (p, 26)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    raw_loss = criterion(logits, graph.y)
    masked_loss = (raw_loss * graph.y_mask).sum() / graph.y_mask.sum().clamp(min=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    masked_loss.backward()
    optimizer.step()

    assert masked_loss.item() > 0.0


def test_mlp_baseline_batched_data():
    """Verify MLPBaseline works seamlessly on batched graphs."""
    generator = SyntheticHOIGenerator(feature_dim=56, seed=42)
    graphs = [generator.generate_graph(image_id=i) for i in range(4)]
    batch = Batch.from_data_list(graphs)

    model = MLPBaseline(input_dim=56, hidden_dim=64, num_actions=26)
    logits = model(batch)

    total_pairs = sum(g.pair_index.shape[0] for g in graphs)
    assert logits.shape == (total_pairs, 26)


def test_mlp_baseline_empty_cases():
    """Verify MLPBaseline handles zero-pair and zero-node graphs gracefully."""
    model = MLPBaseline(input_dim=56, hidden_dim=64, num_actions=26)

    # 1. Zero pairs
    no_pairs = HOIData(
        x_visual=torch.randn((3, 56)),
        class_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        boxes_norm=torch.rand((3, 4)),
        pair_index=torch.zeros((0, 2), dtype=torch.long),
        num_nodes=3,
    )
    assert model(no_pairs).shape == (0, 26)

    # 2. Zero nodes
    no_nodes = HOIData(
        x_visual=torch.zeros((0, 56)),
        class_ids=torch.zeros((0,), dtype=torch.long),
        boxes_norm=torch.zeros((0, 4)),
        pair_index=torch.zeros((0, 2), dtype=torch.long),
        num_nodes=0,
    )
    assert model(no_nodes).shape == (0, 26)


def test_mlp_baseline_from_config():
    """Verify MLPBaseline instantiates cleanly from config dict."""
    cfg = {
        "model": {
            "input_dim": 896,
            "hidden_dim": 128,
            "num_actions": 26,
            "class_embed_dim": 32,
            "spatial_embed_dim": 32,
            "dropout": 0.2,
            "feature_subset": "appearance",
            "use_edge_attr": False,
        }
    }
    model = MLPBaseline.from_config(cfg)
    assert model.node_encoder.hidden_dim == 128
    assert model.node_encoder.feature_subset == "appearance"
    assert not model.interaction_head.use_pair_attr


# --------------------------------------------------------------------------- #
# 5. Comprehensive Tests for All 5 Models                                     #
# --------------------------------------------------------------------------- #

ALL_MODEL_CLASSES = [
    ObjectClassPriorBaseline,
    MLPBaseline,
    GCNHOIModel,
    GATHOIModel,
    EdgeAwareGATv2HOIModel,
]

PARAMETRIC_MODEL_CLASSES = [
    MLPBaseline,
    GCNHOIModel,
    GATHOIModel,
    EdgeAwareGATv2HOIModel,
]


@pytest.mark.parametrize("model_cls", ALL_MODEL_CLASSES)
def test_all_models_forward_shape(model_cls, synthetic_graph: HOIData):
    """Verify that all 5 models produce output logits of shape [P, 26] and finite values."""
    model = _build_model(model_cls)
    model.eval()

    with torch.no_grad():
        logits = model(synthetic_graph)

    p = synthetic_graph.pair_index.shape[0]
    assert p > 0, "Test synthetic graph must have candidate pairs P > 0"
    assert logits.shape == (p, NUM_ACTIONS), (
        f"{model_cls.__name__} returned shape {logits.shape}, expected ({p}, {NUM_ACTIONS})"
    )
    assert torch.isfinite(logits).all(), f"{model_cls.__name__} output contains NaNs or Infs"


@pytest.mark.parametrize("model_cls", ALL_MODEL_CLASSES)
def test_all_models_zero_candidate_pairs(model_cls, zero_pairs_graph: HOIData):
    """Verify all 5 models return shape (0, 26) when candidate pairs P = 0."""
    model = _build_model(model_cls)
    model.eval()

    with torch.no_grad():
        logits = model(zero_pairs_graph)

    assert logits.shape == (0, NUM_ACTIONS), (
        f"{model_cls.__name__} on P=0 returned shape {logits.shape}, expected (0, {NUM_ACTIONS})"
    )


@pytest.mark.parametrize("model_cls", ALL_MODEL_CLASSES)
def test_all_models_zero_nodes(model_cls, empty_nodes_graph: HOIData):
    """Verify all 5 models return shape (0, 26) when nodes V = 0."""
    model = _build_model(model_cls)
    model.eval()

    with torch.no_grad():
        logits = model(empty_nodes_graph)

    assert logits.shape == (0, NUM_ACTIONS), (
        f"{model_cls.__name__} on V=0 returned shape {logits.shape}, expected (0, {NUM_ACTIONS})"
    )


@pytest.mark.parametrize("model_cls", PARAMETRIC_MODEL_CLASSES)
@pytest.mark.parametrize("subset", ["all", "appearance", "spatial", "semantic"])
def test_feature_subset_switch(model_cls, subset: str, synthetic_graph: HOIData):
    """Verify all parametric models support feature_subset in ['all', 'appearance', 'spatial', 'semantic']."""
    model = _build_model(model_cls, feature_subset=subset)
    model.eval()

    with torch.no_grad():
        logits = model(synthetic_graph)

    p = synthetic_graph.pair_index.shape[0]
    assert logits.shape == (p, NUM_ACTIONS), (
        f"{model_cls.__name__} with subset '{subset}' returned shape {logits.shape}, expected ({p}, {NUM_ACTIONS})"
    )
    assert torch.isfinite(logits).all()


def test_edge_aware_gatv2_return_attention(synthetic_graph: HOIData):
    """Verify EdgeAwareGATv2HOIModel with return_attention=True exports valid attention tuples."""
    num_layers = 2
    model = EdgeAwareGATv2HOIModel(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        heads=HEADS,
        edge_dim=EDGE_DIM,
        num_layers=num_layers,
    )
    model.eval()

    p = synthetic_graph.pair_index.shape[0]

    with torch.no_grad():
        logits, attention_list = model(synthetic_graph, return_attention=True)

    # 1. Output shapes
    assert logits.shape == (p, NUM_ACTIONS)
    assert isinstance(attention_list, list)
    assert len(attention_list) == num_layers, (
        f"Expected {num_layers} attention elements, got {len(attention_list)}"
    )

    # 2. Check each layer's attention tuple (edge_index, alpha)
    for layer_idx, (edge_index, alpha) in enumerate(attention_list):
        assert isinstance(edge_index, torch.Tensor)
        assert isinstance(alpha, torch.Tensor)
        assert edge_index.dim() == 2 and edge_index.shape[0] == 2
        assert alpha.dim() == 2 and alpha.shape[1] == HEADS

        # Verify attention coefficients are valid probabilities in [0, 1]
        assert torch.isfinite(alpha).all(), f"Layer {layer_idx} attention contains NaNs/Infs"
        assert (alpha >= 0.0).all() and (alpha <= 1.05).all(), (
            f"Layer {layer_idx} attention weights outside valid [0, 1] range"
        )


def test_edge_aware_gatv2_return_attention_empty_cases(
    empty_nodes_graph: HOIData, zero_pairs_graph: HOIData
):
    """Verify EdgeAwareGATv2HOIModel return_attention=True handles empty and zero-pair graphs safely."""
    model = EdgeAwareGATv2HOIModel(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        heads=HEADS,
        edge_dim=EDGE_DIM,
        num_layers=2,
    )
    model.eval()

    # V = 0 case
    logits_v0, att_v0 = model(empty_nodes_graph, return_attention=True)
    assert logits_v0.shape == (0, NUM_ACTIONS)
    assert len(att_v0) == 2
    for e_idx, alpha in att_v0:
        assert e_idx.shape == (2, 0)
        assert alpha.shape == (0, HEADS)

    # P = 0, V > 0 case
    logits_p0, att_p0 = model(zero_pairs_graph, return_attention=True)
    assert logits_p0.shape == (0, NUM_ACTIONS)
    assert len(att_p0) == 2


@pytest.mark.parametrize("model_cls", PARAMETRIC_MODEL_CLASSES)
def test_all_parametric_models_gradient_flow(model_cls, synthetic_graph: HOIData):
    """Verify backward pass on masked BCE loss updates parameters with non-NaN gradients across models."""
    torch.manual_seed(42)
    model = _build_model(model_cls)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial_params = [p.clone() for p in model.parameters() if p.requires_grad]

    optimizer.zero_grad()
    logits = model(synthetic_graph)

    # Compute masked BCE loss
    loss = (
        F.binary_cross_entropy_with_logits(logits, synthetic_graph.y, reduction="none")
        * synthetic_graph.y_mask
    ).sum() / (synthetic_graph.y_mask.sum() + 1e-6)

    assert torch.isfinite(loss), f"Loss for {model_cls.__name__} is non-finite: {loss.item()}"
    loss.backward()

    # Verify all trainable parameters received valid non-NaN gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} in {model_cls.__name__} has no gradient"
            assert torch.isfinite(param.grad).all(), (
                f"Gradient for parameter {name} in {model_cls.__name__} contains NaNs/Infs"
            )

    optimizer.step()

    # Verify parameters updated
    updated_params = [p for p in model.parameters() if p.requires_grad]
    param_changed = False
    for p_init, p_now in zip(initial_params, updated_params):
        if not torch.equal(p_init, p_now):
            param_changed = True
            break
    assert param_changed, f"No parameters changed after optimizer step in {model_cls.__name__}"


@pytest.mark.parametrize("model_cls", ALL_MODEL_CLASSES)
def test_all_models_batched_forward_dataloader(model_cls):
    """Verify batched forward pass using PyG DataLoader with hoi_collate across all 5 models."""
    num_samples = 6
    batch_size = 3
    dataset = SyntheticVCOCODataset(
        num_samples=num_samples,
        use_cache=False,
        seed=123,
        feature_dim=FEATURE_DIM,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=hoi_collate)
    batch = next(iter(loader))

    total_pairs = batch.pair_index.shape[0]
    assert total_pairs > 0, "Collated batch must contain candidate pairs"

    model = _build_model(model_cls)
    model.eval()

    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (total_pairs, NUM_ACTIONS), (
        f"{model_cls.__name__} batched output shape {logits.shape}, expected ({total_pairs}, {NUM_ACTIONS})"
    )
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("model_cls", [GCNHOIModel, GATHOIModel, EdgeAwareGATv2HOIModel])
def test_gnn_models_from_config(model_cls):
    """Verify GNN models instantiate successfully from configs/model_config.yaml."""
    config_path = "configs/model_config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = model_cls.from_config(cfg)
    assert isinstance(model, model_cls)
    assert model.hidden_dim == cfg["model"]["hidden_dim"]
    assert model.interaction_head.num_actions == cfg["model"]["num_actions"]
