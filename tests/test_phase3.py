r"""Phase 3 tests: Synthetic HOI Dataset, Pipeline Smoke Test, and Overfit Validation.

Run from project root:
  & "C:\Users\jayan\miniconda3\envs\hoi-gnn\python.exe" -m pytest tests/test_phase3.py -v
"""
from pathlib import Path
from typing import List, Tuple
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from data.dataset import hoi_collate
from data.graph_builder import GraphBuilder, HOIData
from data.synthetic_dataset import (
    SyntheticHOIGenerator,
    SyntheticVCOCODataset,
    VALID_PAIR_ACTIONS,
    CANONICAL_OBJECT_ACTIONS,
)
from data.vcoco_loader import (
    AGENT_ONLY_ACTIONS,
    AGENT_ONLY_ACTION_INDICES,
    ACTION_TO_IDX,
    NUM_ACTIONS,
    VCOCO_ACTIONS,
)
from detection.feature_extractor import ImageFeatures


# -------------------------------------------------------------------------- #
# Helper Interaction Models for Testing                                      #
# -------------------------------------------------------------------------- #

class SimpleInteractionMLP(nn.Module):
    """Concatenates human ROI embedding, object ROI embedding, and pair_attr [9] -> MLP -> 26 logits."""

    def __init__(self, node_dim: int = 56, pair_attr_dim: int = 9, num_actions: int = 26):
        super().__init__()
        in_dim = 2 * node_dim + pair_attr_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, batch: HOIData) -> torch.Tensor:
        h_idx = batch.pair_index[:, 0]
        o_idx = batch.pair_index[:, 1]
        h_feat = batch.x[h_idx]
        o_feat = batch.x[o_idx]
        p_attr = batch.pair_attr
        pair_feat = torch.cat([h_feat, o_feat, p_attr], dim=-1)
        return self.mlp(pair_feat)


class OverfitHOIModel(nn.Module):
    """Capacity-adequate MLP to test overfitting on small synthetic graphs."""

    def __init__(self, in_dim: int = 121, hidden_dim: int = 256, out_dim: int = 26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch: HOIData) -> torch.Tensor:
        h_idx = batch.pair_index[:, 0]
        o_idx = batch.pair_index[:, 1]
        h_feat = batch.x[h_idx]
        o_feat = batch.x[o_idx]
        p_attr = batch.pair_attr
        pair_feat = torch.cat([h_feat, o_feat, p_attr], dim=-1)
        return self.net(pair_feat)


# -------------------------------------------------------------------------- #
# Task 3.1 & 3.2: Synthetic Data Shapes and Invariants                        #
# -------------------------------------------------------------------------- #

def test_synthetic_data_shapes_and_invariants():
    """
    Verify generated HOIData has all required fields matching CONTRACTS.md Section 2:
    - Fields: x, x_visual, boxes, boxes_norm, class_ids, scores, node_type,
      edge_index, edge_attr, pair_index, pair_attr, y, y_mask
    - pair_index is non-empty for scenes with persons and objects
    - Agent-only actions (stand, walk, run, smile) have y=0 and y_mask=0
    - All other actions have y_mask=1
    - node_type is 0 for humans and 1 for objects
    """
    generator = SyntheticHOIGenerator(feature_dim=56, seed=42)

    required_fields = [
        "x",
        "x_visual",
        "boxes",
        "boxes_norm",
        "class_ids",
        "scores",
        "node_type",
        "edge_index",
        "edge_attr",
        "pair_index",
        "pair_attr",
        "y",
        "y_mask",
    ]

    for img_id in range(10):
        graph = generator.generate_graph(image_id=img_id)

        # 1. Check graph type and required fields presence
        assert isinstance(graph, HOIData), f"Expected HOIData, got {type(graph)}"
        for field in required_fields:
            assert hasattr(graph, field), f"Missing field '{field}' in HOIData for img_id={img_id}"
            tensor_val = getattr(graph, field)
            assert isinstance(tensor_val, torch.Tensor), f"Field '{field}' must be a torch.Tensor"

        v = graph.num_nodes
        assert v >= 2, f"Expected at least 2 entities, got {v}"
        assert graph.x.shape == (v, 56)
        assert graph.x_visual.shape == (v, 56)
        assert graph.boxes.shape == (v, 4)
        assert graph.boxes_norm.shape == (v, 4)
        assert graph.class_ids.shape == (v,)
        assert graph.scores.shape == (v,)
        assert graph.node_type.shape == (v,)

        # 2. Node type and class invariant
        # class_id 0 = person -> node_type 0; class_id > 0 -> node_type 1
        expected_node_type = torch.where(graph.class_ids == 0, 0, 1).long()
        assert torch.equal(graph.node_type, expected_node_type)

        # Check at least 1 person and 1 object exists
        n_persons = int((graph.node_type == 0).sum().item())
        n_objects = int((graph.node_type == 1).sum().item())
        assert n_persons >= 1, f"Expected >= 1 person, got {n_persons}"
        assert n_objects >= 1, f"Expected >= 1 object, got {n_objects}"

        # 3. Normalized boxes in valid bounds [0, 1]
        assert (graph.boxes_norm >= 0.0).all() and (graph.boxes_norm <= 1.05).all()

        # 4. Pair index and spatial attributes
        p = graph.pair_index.shape[0]
        # Since scene has >= 1 person and >= 1 object, candidate pair_index must be non-empty
        assert p >= 1, f"Expected non-empty pair_index for scene with persons & objects, got {p}"
        assert graph.pair_index.shape == (p, 2)
        assert graph.pair_attr.shape == (p, 9)
        assert graph.y.shape == (p, 26)
        assert graph.y_mask.shape == (p, 26)

        # Check pair index nodes: source must be person (node_type 0), destination must be a distinct entity
        for pair_i in range(p):
            src = int(graph.pair_index[pair_i, 0].item())
            dst = int(graph.pair_index[pair_i, 1].item())
            assert 0 <= src < v and 0 <= dst < v
            assert src != dst, f"Source and destination must be distinct, got {src} == {dst}"
            assert graph.node_type[src].item() == 0, f"Source node {src} is not human"

        # 5. Agent-only action mask & target invariant
        # Agent-only actions (stand, walk, run, smile) must have y=0 and y_mask=0 for all pairs
        for a_idx in AGENT_ONLY_ACTION_INDICES:
            assert (graph.y[:, a_idx] == 0.0).all(), (
                f"Agent-only action {VCOCO_ACTIONS[a_idx]} (idx={a_idx}) has nonzero y"
            )
            assert (graph.y_mask[:, a_idx] == 0.0).all(), (
                f"Agent-only action {VCOCO_ACTIONS[a_idx]} (idx={a_idx}) has nonzero y_mask"
            )

        # All non-agent actions must have y_mask=1
        for a_idx in VALID_PAIR_ACTIONS:
            assert (graph.y_mask[:, a_idx] == 1.0).all(), (
                f"Non-agent action {VCOCO_ACTIONS[a_idx]} (idx={a_idx}) must have y_mask=1"
            )


def test_deterministic_reproducibility():
    """Verify identical random seed generates identical graphs and features."""
    gen1 = SyntheticHOIGenerator(feature_dim=56, seed=777)
    gen2 = SyntheticHOIGenerator(feature_dim=56, seed=777)

    g1 = gen1.generate_graph(image_id=42)
    g2 = gen2.generate_graph(image_id=42)

    assert torch.equal(g1.boxes, g2.boxes)
    assert torch.equal(g1.class_ids, g2.class_ids)
    assert torch.allclose(g1.scores, g2.scores, atol=1e-6)
    assert torch.allclose(g1.x, g2.x, atol=1e-6)
    assert torch.equal(g1.pair_index, g2.pair_index)
    assert torch.allclose(g1.pair_attr, g2.pair_attr, atol=1e-5)
    assert torch.equal(g1.y, g2.y)
    assert torch.equal(g1.y_mask, g2.y_mask)


# -------------------------------------------------------------------------- #
# Task 3.2: Synthetic Cache Serialization Round-trip                         #
# -------------------------------------------------------------------------- #

def test_synthetic_cache_roundtrip(tmp_path: Path):
    """
    Test saving synthetic ImageFeatures to disk in cache directory and reloading.
    Verifies CONTRACTS.md Section 1 compliance:
    - ImageFeatures to_dict and from_dict
    - float16 on-disk storage of roi_features correctly converts back to float32
    - Exact match on image_id, image_size, boxes, class_ids, scores
    """
    cache_dir = tmp_path / "cache_synthetic"
    generator = SyntheticHOIGenerator(feature_dim=56, seed=123)

    for img_id in [10, 42, 99]:
        img_features_orig = generator.generate_image_features(image_id=img_id)
        saved_path = generator.save_scene_to_cache(
            image_id=img_id, cache_dir=cache_dir, tag="synthetic"
        )

        assert saved_path.exists()
        assert saved_path.name == f"{img_id}__synthetic.pt"

        # Reload from disk
        raw_dict = torch.load(saved_path, map_location="cpu", weights_only=False)
        assert isinstance(raw_dict, dict)
        assert raw_dict["roi_features"].dtype == torch.float16, "roi_features must be float16 on disk"

        loaded_feats = ImageFeatures.from_dict(raw_dict)
        loaded_feats.validate()

        # Invariant checks
        assert loaded_feats.image_id == str(img_id)
        assert loaded_feats.image_size == img_features_orig.image_size
        assert torch.equal(loaded_feats.boxes, img_features_orig.boxes)
        assert torch.equal(loaded_feats.class_ids, img_features_orig.class_ids)
        assert torch.allclose(loaded_feats.scores, img_features_orig.scores, atol=1e-5)

        # roi_features precision: float32 -> float16 -> float32 gives max error ~ 1e-3
        assert torch.allclose(loaded_feats.roi_features, img_features_orig.roi_features, atol=2e-3)


# -------------------------------------------------------------------------- #
# Task 3.2: Smoke Forward & Backward Step with PyG DataLoader                #
# -------------------------------------------------------------------------- #

def test_smoke_forward_backward_step():
    """
    End-to-end smoke test of DataLoader batching and training step:
    - PyG DataLoader batching with hoi_collate
    - batch.pair_index offset verification
    - Simple interaction model: concatenates human ROI, object ROI, and pair_attr [9] -> MLP -> 26 logits
    - Masked BCE loss calculation
    - Forward + backward pass + optimizer step
    - Parameter update verification
    """
    dataset = SyntheticVCOCODataset(num_samples=8, use_cache=False, seed=100, feature_dim=56)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=hoi_collate)

    batch: HOIData = next(iter(loader))

    # 1. Invariant verification on batched HOIData
    expected_total_nodes = sum(dataset.get(i).num_nodes for i in range(4))
    expected_total_pairs = sum(dataset.get(i).pair_index.shape[0] for i in range(4))

    assert batch.num_nodes == expected_total_nodes
    assert batch.x.shape == (expected_total_nodes, 56)
    assert batch.pair_index.shape == (expected_total_pairs, 2)
    assert batch.pair_attr.shape == (expected_total_pairs, 9)
    assert batch.pair_batch.shape == (expected_total_pairs,)
    assert batch.y.shape == (expected_total_pairs, NUM_ACTIONS)
    assert batch.y_mask.shape == (expected_total_pairs, NUM_ACTIONS)

    # Verify pair_index batch node offsets
    node_offset = 0
    pair_offset = 0
    for g_idx in range(4):
        g = dataset.get(g_idx)
        g_pairs = g.pair_index.shape[0]
        if g_pairs > 0:
            slice_pairs = batch.pair_index[pair_offset : pair_offset + g_pairs]
            expected_pairs = g.pair_index + node_offset
            assert torch.equal(slice_pairs, expected_pairs)
            slice_batch = batch.pair_batch[pair_offset : pair_offset + g_pairs]
            assert (slice_batch == g_idx).all()
        node_offset += g.num_nodes
        pair_offset += g_pairs

    # 2. Model initialization and forward pass
    model = SimpleInteractionMLP(node_dim=56, pair_attr_dim=9, num_actions=NUM_ACTIONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_params = [p.clone() for p in model.parameters()]

    optimizer.zero_grad()
    logits = model(batch)
    assert logits.shape == (expected_total_pairs, NUM_ACTIONS)

    # 3. Masked BCE Loss
    loss = (
        F.binary_cross_entropy_with_logits(logits, batch.y, reduction="none") * batch.y_mask
    ).sum() / (batch.y_mask.sum() + 1e-6)

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0.0

    # 4. Backward pass
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"Gradient of {name} has NaNs/Infs"

    # 5. Optimizer step and parameter change
    optimizer.step()

    params_changed = False
    for p_before, p_after in zip(initial_params, model.parameters()):
        if not torch.equal(p_before, p_after):
            params_changed = True
            break
    assert params_changed, "Model parameters did not update after optimizer.step()"


# -------------------------------------------------------------------------- #
# Task 3.3: Overfit 20 Synthetic Graphs                                      #
# -------------------------------------------------------------------------- #

def test_overfit_20_synthetic_graphs():
    """
    Overfit test on 20 fixed synthetic graphs:
    - Train OverfitHOIModel for 70 epochs
    - Verify initial loss ~0.7 drops to < 0.05
    - Verify masked accuracy reaches > 98%
    """
    torch.manual_seed(42)

    # Generate 20 fixed synthetic graphs
    num_samples = 20
    dataset = SyntheticVCOCODataset(
        num_samples=num_samples, use_cache=False, seed=42, feature_dim=56
    )
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=False, collate_fn=hoi_collate)
    batch: HOIData = next(iter(loader))

    total_pairs = batch.pair_index.shape[0]
    assert total_pairs > 0, "No pairs in 20 synthetic graphs"

    # Model: in_dim = 56 (human) + 56 (object) + 9 (pair_attr) = 121
    model = OverfitHOIModel(in_dim=121, hidden_dim=256, out_dim=NUM_ACTIONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial_loss = None
    final_loss = None
    final_acc = None

    num_epochs = 70
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = model(batch)
        loss = (
            F.binary_cross_entropy_with_logits(logits, batch.y, reduction="none") * batch.y_mask
        ).sum() / (batch.y_mask.sum() + 1e-6)

        if epoch == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (logits > 0.0).float()
            correct = ((preds == batch.y) * batch.y_mask).sum()
            total_valid = batch.y_mask.sum()
            acc = (correct / total_valid).item()

        final_loss = loss.item()
        final_acc = acc

    # Overfit criteria:
    # 1. Initial loss should be around random binary log-loss (~0.65 - 0.75)
    assert initial_loss is not None
    assert 0.50 <= initial_loss <= 0.85, f"Initial loss {initial_loss:.4f} outside expected ~0.7 range"

    # 2. Final loss drops to < 0.05
    assert final_loss is not None
    assert final_loss < 0.05, f"Final loss was {final_loss:.4f}, expected < 0.05"

    # 3. Final masked accuracy reaches > 98%
    assert final_acc is not None
    assert final_acc >= 0.98, f"Final masked accuracy was {final_acc * 100:.2f}%, expected >= 98%"


# -------------------------------------------------------------------------- #
# Synthetic Dataset Cache Mode Integration Test                              #
# -------------------------------------------------------------------------- #

def test_synthetic_dataset_cached_mode(tmp_path: Path):
    """Verify SyntheticVCOCODataset operates seamlessly with disk caching enabled."""
    cache_dir = tmp_path / "cache_synthetic_ds"
    ds = SyntheticVCOCODataset(
        num_samples=5,
        cache_dir=cache_dir,
        seed=42,
        use_cache=True,
        feature_dim=56,
    )

    assert len(ds) == 5

    # First access populates cache
    g0 = ds[0]
    cached_files = list(cache_dir.glob("*.pt"))
    assert len(cached_files) == 1

    # Second access reads from cache
    g0_reload = ds[0]
    assert torch.equal(g0.pair_index, g0_reload.pair_index)
    assert torch.equal(g0.y, g0_reload.y)
    assert torch.allclose(g0.x, g0_reload.x, atol=2e-3)


def test_synthetic_generation_high_dim():
    """Verify synthetic generator supports feature_dim=896 (CONTRACTS.md backbone dimension)."""
    generator = SyntheticHOIGenerator(feature_dim=896, seed=100)
    graph = generator.generate_graph(image_id="img_896", num_entities=3)

    assert graph.x.shape == (3, 896)
    assert graph.x_visual.shape == (3, 896)
    assert graph.pair_index.shape[0] >= 1


def test_save_dataset_to_cache(tmp_path: Path):
    """Verify pre-populating dataset cache saves all sample files properly."""
    generator = SyntheticHOIGenerator(feature_dim=56, seed=42)
    saved_paths = generator.save_dataset_to_cache(num_samples=5, cache_dir=tmp_path)
    assert len(saved_paths) == 5
    for p in saved_paths:
        assert p.exists()
        assert p.suffix == ".pt"


def test_synthetic_generator_candidate_modes():
    """Verify synthetic generator works with different GraphBuilder candidate modes."""
    for mode in ("all_human_object", "knn", "distance_threshold"):
        builder = GraphBuilder(candidate_mode=mode)
        generator = SyntheticHOIGenerator(graph_builder=builder, seed=777)
        graph = generator.generate_graph(image_id=mode, num_entities=5)
        assert isinstance(graph, HOIData)
        assert graph.pair_index.shape[1] == 2
        assert graph.pair_attr.shape[1] == 9

