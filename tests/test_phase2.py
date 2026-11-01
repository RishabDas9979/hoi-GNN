r"""Phase 2 tests: V-COCO Loading, Graph Construction, PyG Dataset, and Batch Collation.

Run from project root:
  & "C:\Users\jayan\miniconda3\envs\hoi-gnn\python.exe" -m pytest tests/test_phase2.py -v
"""
import math
from pathlib import Path
import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from data.graph_builder import (
    GraphBuilder,
    HOIData,
    compute_spatial_features,
)
from data.vcoco_loader import (
    VCOCOLoader,
    VCOCOImageRecord,
    VCOCOEntity,
    VCOCO_ACTIONS,
    NUM_ACTIONS,
    ACTION_TO_IDX,
    AGENT_ONLY_ACTIONS,
    compute_box_iou,
)
from data.dataset import (
    VCOCODataset,
    hoi_collate,
)


INSTANCES_JSON = Path("data/instances_vcoco_all_2014.json")


# -------------------------------------------------------------------------- #
# 1. Spatial Relation Edge Attribute Calculations                            #
# -------------------------------------------------------------------------- #

def test_spatial_features_analytical_math():
    """Verify exact formula calculations for the 9 spatial relation features."""
    # Box A: [100, 100, 200, 300] -> cx=150, cy=200, w=100, h=200, area=20000
    b_src = torch.tensor([[100.0, 100.0, 200.0, 300.0]])
    # Box B: [200, 100, 400, 300] -> cx=300, cy=200, w=200, h=200, area=40000
    b_dst = torch.tensor([[200.0, 100.0, 400.0, 300.0]])
    img_size = (400, 600)  # H=400, W=600

    feats = compute_spatial_features(b_src, b_dst, img_size)
    assert feats.shape == (1, 9)

    f = feats[0]
    # 1. dx = (300 - 150) / 600 = 150 / 600 = 0.25
    assert math.isclose(f[0].item(), 0.25, abs_tol=1e-5)

    # 2. dy = (200 - 200) / 400 = 0.0
    assert math.isclose(f[1].item(), 0.0, abs_tol=1e-5)

    # 3. dist = sqrt(0.25^2 + 0.0^2) = 0.25
    assert math.isclose(f[2].item(), 0.25, abs_tol=1e-5)

    # 4. log(w_src / w_dst) = log(100 / 200) = log(0.5) ≈ -0.693147
    assert math.isclose(f[3].item(), math.log(0.5), abs_tol=1e-4)

    # 5. log(h_src / h_dst) = log(200 / 200) = log(1.0) = 0.0
    assert math.isclose(f[4].item(), 0.0, abs_tol=1e-4)

    # 6. log(A_src / A_dst) = log(20000 / 40000) = log(0.5) ≈ -0.693147
    assert math.isclose(f[5].item(), math.log(0.5), abs_tol=1e-4)

    # 7. IoU: boxes touch at x=200 but do not overlap -> inter area = 0
    assert math.isclose(f[6].item(), 0.0, abs_tol=1e-4)

    # 8. norm_area_src = 20000 / (400 * 600) = 20000 / 240000 = 1/12
    assert math.isclose(f[7].item(), 20000.0 / 240000.0, abs_tol=1e-4)

    # 9. norm_area_dst = 40000 / (400 * 600) = 40000 / 240000 = 1/6
    assert math.isclose(f[8].item(), 40000.0 / 240000.0, abs_tol=1e-4)


def test_spatial_features_identical_boxes():
    """Identical source and destination boxes produce expected identity values."""
    box = torch.tensor([[50.0, 60.0, 250.0, 360.0]])
    feats = compute_spatial_features(box, box, (500, 500))

    f = feats[0]
    # dx=0, dy=0, dist=0
    assert torch.allclose(f[:3], torch.zeros(3), atol=1e-5)
    # log ratios are 0
    assert torch.allclose(f[3:6], torch.zeros(3), atol=1e-3)
    # IoU is 1.0
    assert math.isclose(f[6].item(), 1.0, abs_tol=1e-3)
    # Both normalized areas must match
    assert math.isclose(f[7].item(), f[8].item(), abs_tol=1e-5)


def test_spatial_features_zero_division_and_bounds():
    """Verify numerical stability on zero-sized, degenerate, and extreme boxes."""
    zero_box = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    point_box = torch.tensor([[100.0, 100.0, 100.0, 100.0]])
    normal_box = torch.tensor([[10.0, 10.0, 100.0, 100.0]])
    huge_box = torch.tensor([[0.0, 0.0, 10000.0, 10000.0]])

    src_boxes = torch.cat([zero_box, point_box, normal_box, normal_box], dim=0)
    dst_boxes = torch.cat([normal_box, zero_box, huge_box, normal_box], dim=0)

    feats = compute_spatial_features(src_boxes, dst_boxes, (640, 640))

    # Must contain no NaN and no Inf
    assert not torch.isnan(feats).any(), "Found NaN in spatial features"
    assert not torch.isinf(feats).any(), "Found Inf in spatial features"

    # Clamping limits respected for log features (cols 3, 4, 5)
    assert (feats[:, 3:6] >= -10.0).all()
    assert (feats[:, 3:6] <= 10.0).all()

    # IoU (col 6) within [0, 1]
    assert (feats[:, 6] >= 0.0).all()
    assert (feats[:, 6] <= 1.0).all()

    # Distance non-negative
    assert (feats[:, 2] >= 0.0).all()


# -------------------------------------------------------------------------- #
# 2. GraphBuilder Output Shapes & Candidate Modes                           #
# -------------------------------------------------------------------------- #

def test_graph_builder_all_human_object_shapes():
    """Verify GraphBuilder shapes with candidate_mode='all_human_object'."""
    # 2 humans (0, 3) and 2 objects (1, 2)
    boxes = torch.tensor([
        [10.0, 10.0, 50.0, 120.0],   # human 0
        [60.0, 60.0, 120.0, 140.0],  # object 1
        [150.0, 150.0, 200.0, 200.0],# object 2
        [20.0, 20.0, 70.0, 150.0],   # human 3
    ])
    class_ids = torch.tensor([0, 2, 56, 0])
    x = torch.randn(4, 56)

    gb = GraphBuilder(candidate_mode="all_human_object")
    g = gb.build_graph(boxes=boxes, class_ids=class_ids, roi_features=x, image_size=(480, 640))

    assert g.num_nodes == 4
    assert g.x.shape == (4, 56)
    assert g.boxes.shape == (4, 4)
    assert g.class_ids.shape == (4,)
    assert g.scores.shape == (4,)

    # Each of 2 humans pairs with 3 other nodes -> 6 pairs
    assert g.pair_index.shape == (6, 2)
    assert g.pair_attr.shape == (6, 9)
    assert g.edge_index.shape == (2, 6)
    assert g.edge_attr.shape == (6, 9)

    # Directed edges: edge_index is pair_index.t()
    assert torch.equal(g.edge_index, g.pair_index.t())


def test_graph_builder_knn_mode():
    """Verify GraphBuilder k-nearest-neighbors candidate selection."""
    boxes = torch.tensor([
        [0.0, 0.0, 50.0, 50.0],      # human 0 at origin
        [10.0, 10.0, 60.0, 60.0],    # object 1 (very close)
        [100.0, 100.0, 150.0, 150.0],# object 2 (medium)
        [500.0, 500.0, 550.0, 550.0],# object 3 (far)
    ])
    class_ids = torch.tensor([0, 1, 2, 3])

    gb_k1 = GraphBuilder(candidate_mode="knn", k_neighbors=1)
    g_k1 = gb_k1.build_graph(boxes=boxes, class_ids=class_ids, image_size=(600, 600))
    # Human 0 selects only 1 neighbor (the closest, object 1)
    assert g_k1.pair_index.shape == (1, 2)
    assert g_k1.pair_index[0, 0].item() == 0
    assert g_k1.pair_index[0, 1].item() == 1

    gb_k2 = GraphBuilder(candidate_mode="knn", k_neighbors=2)
    g_k2 = gb_k2.build_graph(boxes=boxes, class_ids=class_ids, image_size=(600, 600))
    # Human 0 selects 2 closest (objects 1 and 2)
    assert g_k2.pair_index.shape == (2, 2)
    assert set(g_k2.pair_index[:, 1].tolist()) == {1, 2}


def test_graph_builder_distance_threshold_mode():
    """Verify GraphBuilder distance-threshold filtering."""
    # Normalized image size (1000, 1000)
    boxes = torch.tensor([
        [0.0, 0.0, 50.0, 50.0],      # human 0 at (25, 25)
        [30.0, 30.0, 80.0, 80.0],    # object 1 at (55, 55), dist ≈ 42/1000 = 0.042
        [800.0, 800.0, 850.0, 850.0],# object 2 at (825, 825), dist ≈ 1131/1000 = 1.131
    ])
    class_ids = torch.tensor([0, 1, 2])

    gb = GraphBuilder(candidate_mode="distance_threshold", dist_threshold=0.1)
    g = gb.build_graph(boxes=boxes, class_ids=class_ids, image_size=(1000, 1000))

    # Only object 1 is within dist 0.1
    assert g.pair_index.shape == (1, 2)
    assert g.pair_index[0, 1].item() == 1


def test_graph_builder_bidirectional():
    """Verify bidirectional mode adds reverse message-passing edges."""
    boxes = torch.tensor([
        [10.0, 10.0, 50.0, 100.0],   # human 0
        [100.0, 100.0, 150.0, 150.0],# object 1
    ])
    class_ids = torch.tensor([0, 1])

    gb = GraphBuilder(candidate_mode="all_human_object", bidirectional=True)
    g = gb.build_graph(boxes=boxes, class_ids=class_ids, image_size=(500, 500))

    # 1 pair to classify (human -> object)
    assert g.pair_index.shape == (1, 2)
    assert torch.equal(g.pair_index[0], torch.tensor([0, 1]))

    # 2 message passing edges: 0->1 and 1->0
    assert g.edge_index.shape == (2, 2)
    assert g.edge_attr.shape == (2, 9)
    assert set(map(tuple, g.edge_index.t().tolist())) == {(0, 1), (1, 0)}


# -------------------------------------------------------------------------- #
# 3. Edge Cases: Empty Detections, Person Only, Objects Only                 #
# -------------------------------------------------------------------------- #

def test_edge_case_empty_detections():
    """Empty detections (N=0) produce well-formed empty graph tensors."""
    gb = GraphBuilder()
    g = gb.build_graph(
        boxes=torch.zeros((0, 4)),
        class_ids=torch.zeros((0,), dtype=torch.long),
        roi_features=torch.zeros((0, 56)),
    )

    assert g.num_nodes == 0
    assert g.x.shape == (0, 56)
    assert g.boxes.shape == (0, 4)
    assert g.class_ids.shape == (0,)
    assert g.scores.shape == (0,)
    assert g.pair_index.shape == (0, 2)
    assert g.pair_attr.shape == (0, 9)
    assert g.edge_index.shape == (2, 0)
    assert g.edge_attr.shape == (0, 9)


def test_edge_case_person_only():
    """Image with only 1 person (and no other entities) produces 0 candidate pairs."""
    gb = GraphBuilder()
    g = gb.build_graph(
        boxes=torch.tensor([[10.0, 20.0, 100.0, 200.0]]),
        class_ids=torch.tensor([0]),
        roi_features=torch.randn(1, 56),
    )

    assert g.num_nodes == 1
    assert g.pair_index.shape == (0, 2)
    assert g.pair_attr.shape == (0, 9)
    assert g.edge_index.shape == (2, 0)
    assert g.edge_attr.shape == (0, 9)


def test_edge_case_objects_without_person():
    """Image with only non-person objects produces 0 candidate pairs."""
    gb = GraphBuilder()
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 120.0, 120.0]])
    class_ids = torch.tensor([1, 2])  # bicycle, car (no person)

    g = gb.build_graph(boxes=boxes, class_ids=class_ids, roi_features=torch.randn(2, 56))

    assert g.num_nodes == 2
    assert g.pair_index.shape == (0, 2)
    assert g.pair_attr.shape == (0, 9)
    assert g.edge_index.shape == (2, 0)


# -------------------------------------------------------------------------- #
# 4. PyG DataLoader Mini-Batch Collation & Offsetting                        #
# -------------------------------------------------------------------------- #

def test_pyg_batch_pair_index_offsetting():
    """Verify that pair_index in batched data correctly offsets indices according to batch.ptr."""
    # Graph 1: 3 nodes (0, 1, 2). Pairs: (0, 1), (0, 2)
    g1 = HOIData(
        x=torch.randn(3, 16),
        boxes=torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]], dtype=torch.float32),
        class_ids=torch.tensor([0, 1, 2]),
        scores=torch.ones(3),
        edge_index=torch.tensor([[0, 0], [1, 2]]),
        edge_attr=torch.randn(2, 9),
        pair_index=torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
        pair_attr=torch.randn(2, 9),
        y=torch.zeros((2, NUM_ACTIONS)),
        y_mask=torch.ones((2, NUM_ACTIONS)),
        num_nodes=3,
        image_id="g1",
    )

    # Graph 2: 4 nodes (0, 1, 2, 3). Pairs: (0, 1), (0, 2), (0, 3)
    g2 = HOIData(
        x=torch.randn(4, 16),
        boxes=torch.tensor([[0, 0, 5, 5], [5, 5, 10, 10], [10, 10, 15, 15], [15, 15, 20, 20]], dtype=torch.float32),
        class_ids=torch.tensor([0, 3, 4, 5]),
        scores=torch.ones(4),
        edge_index=torch.tensor([[0, 0, 0], [1, 2, 3]]),
        edge_attr=torch.randn(3, 9),
        pair_index=torch.tensor([[0, 1], [0, 2], [0, 3]], dtype=torch.long),
        pair_attr=torch.randn(3, 9),
        y=torch.ones((3, NUM_ACTIONS)),
        y_mask=torch.ones((3, NUM_ACTIONS)),
        num_nodes=4,
        image_id="g2",
    )

    batch = hoi_collate([g1, g2])

    assert batch.num_graphs == 2
    assert batch.num_nodes == 7
    assert torch.equal(batch.ptr, torch.tensor([0, 3, 7]))

    # Expected batched pair_index:
    # First 2 rows: [0, 1], [0, 2] (offset 0)
    # Next 3 rows: [3+0, 3+1], [3+0, 3+2], [3+0, 3+3] = [3, 4], [3, 5], [3, 6] (offset 3)
    expected_pairs = torch.tensor([
        [0, 1],
        [0, 2],
        [3, 4],
        [3, 5],
        [3, 6],
    ], dtype=torch.long)
    assert torch.equal(batch.pair_index, expected_pairs)

    # Verify pair_batch assignment [0, 0, 1, 1, 1]
    expected_pair_batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    assert torch.equal(batch.pair_batch, expected_pair_batch)

    # Verify targets stacked along dim 0
    assert batch.y.shape == (5, NUM_ACTIONS)
    assert batch.y_mask.shape == (5, NUM_ACTIONS)
    assert torch.equal(batch.y[:2], torch.zeros(2, NUM_ACTIONS))
    assert torch.equal(batch.y[2:], torch.ones(3, NUM_ACTIONS))


def test_pyg_dataloader_with_empty_graphs():
    """Verify PyG DataLoader collates graphs even when some have 0 candidate pairs."""
    g1 = HOIData(
        x=torch.randn(2, 8),
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.randn(1, 9),
        pair_index=torch.tensor([[0, 1]]),
        pair_attr=torch.randn(1, 9),
        y=torch.zeros((1, NUM_ACTIONS)),
        y_mask=torch.ones((1, NUM_ACTIONS)),
    )
    # g2 has no candidate pairs
    g2 = HOIData(
        x=torch.randn(3, 8),
        num_nodes=3,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 9)),
        pair_index=torch.zeros((0, 2), dtype=torch.long),
        pair_attr=torch.zeros((0, 9)),
        y=torch.zeros((0, NUM_ACTIONS)),
        y_mask=torch.zeros((0, NUM_ACTIONS)),
    )
    # g3 has 1 pair
    g3 = HOIData(
        x=torch.randn(2, 8),
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.randn(1, 9),
        pair_index=torch.tensor([[0, 1]]),
        pair_attr=torch.randn(1, 9),
        y=torch.zeros((1, NUM_ACTIONS)),
        y_mask=torch.ones((1, NUM_ACTIONS)),
    )

    loader = DataLoader([g1, g2, g3], batch_size=3, shuffle=False, collate_fn=hoi_collate)
    for batch in loader:
        assert batch.num_graphs == 3
        assert batch.num_nodes == 7
        assert torch.equal(batch.ptr, torch.tensor([0, 2, 5, 7]))
        # Total pairs: 1 + 0 + 1 = 2
        assert batch.pair_index.shape == (2, 2)
        # Pair from g1 has offset 0: [0, 1]
        assert torch.equal(batch.pair_index[0], torch.tensor([0, 1]))
        # Pair from g3 has offset g1_nodes + g2_nodes = 2 + 3 = 5: [5+0, 5+1] = [5, 6]
        assert torch.equal(batch.pair_index[1], torch.tensor([5, 6]))
        assert torch.equal(batch.pair_batch, torch.tensor([0, 2]))


# -------------------------------------------------------------------------- #
# 5. V-COCO Loader & Dataset Parsing Sanity Tests                            #
# -------------------------------------------------------------------------- #

def test_vcoco_loader_parsing_and_masking_mock():
    """Verify V-COCO loader parses action labels, roles, and generates correct y_mask."""
    loader = VCOCOLoader.create_mock_loader(num_images=2)
    assert len(loader) == 2

    img_id = loader.get_image_ids()[0]
    rec = loader.get_record(img_id)
    assert rec.num_entities == 3
    assert rec.boxes.shape == (3, 4)

    # Candidate pairs: (0, 1) person->cup, (0, 2) person->chair
    pairs = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    y, y_mask = loader.get_ground_truth_for_pairs(
        img_id, rec.boxes, rec.class_ids, pairs, ann_ids=rec.ann_ids
    )

    assert y.shape == (2, NUM_ACTIONS)
    assert y_mask.shape == (2, NUM_ACTIONS)

    # 1. Agent-only actions (stand, walk, run, smile) must be masked out (0.0)
    for act_name in AGENT_ONLY_ACTIONS:
        idx = ACTION_TO_IDX[act_name]
        assert (y_mask[:, idx] == 0.0).all(), f"Agent-only action '{act_name}' not masked"

    # 2. Object actions (hold, drink, sit) must have y_mask == 1.0
    for act_name in ("hold", "drink", "sit"):
        idx = ACTION_TO_IDX[act_name]
        assert (y_mask[:, idx] == 1.0).all()

    # 3. Pair 0 (person, cup): positive for hold and drink, negative for sit
    assert y[0, ACTION_TO_IDX["hold"]] == 1.0
    assert y[0, ACTION_TO_IDX["drink"]] == 1.0
    assert y[0, ACTION_TO_IDX["sit"]] == 0.0

    # 4. Pair 1 (person, chair): positive for sit, negative for hold and drink
    assert y[1, ACTION_TO_IDX["sit"]] == 1.0
    assert y[1, ACTION_TO_IDX["hold"]] == 0.0
    assert y[1, ACTION_TO_IDX["drink"]] == 0.0


@pytest.mark.skipif(not INSTANCES_JSON.exists(), reason="Real annotations JSON not present")
def test_vcoco_loader_real_annotations():
    """Verify loading real annotations from data/instances_vcoco_all_2014.json and vcoco_train.json."""
    loader = VCOCOLoader(split="train", max_images=10)
    assert len(loader) == 10

    img_ids = loader.get_image_ids()
    assert len(img_ids) == 10

    rec = loader.get_record(img_ids[0])
    assert rec.image_id == img_ids[0]
    assert rec.num_entities > 0
    assert rec.boxes.shape == (rec.num_entities, 4)
    # All boxes should have non-negative width and height
    w = rec.boxes[:, 2] - rec.boxes[:, 0]
    h = rec.boxes[:, 3] - rec.boxes[:, 1]
    assert (w >= 0).all()
    assert (h >= 0).all()


@pytest.mark.skipif(not INSTANCES_JSON.exists(), reason="Real annotations JSON not present")
def test_vcoco_dataset_end_to_end_real():
    """End-to-end integration test: VCOCODataset -> HOIData -> PyG DataLoader Batch."""
    dataset = VCOCODataset(split="train", max_images=6, use_gt_boxes=True)
    assert len(dataset) == 6

    # Test single item indexing
    item0 = dataset[0]
    assert isinstance(item0, HOIData)
    assert item0.num_nodes > 0
    assert item0.x.shape[0] == item0.num_nodes
    assert item0.boxes.shape == (item0.num_nodes, 4)
    assert item0.y.shape[1] == NUM_ACTIONS
    assert item0.y_mask.shape[1] == NUM_ACTIONS

    # Test mini-batch collation with DataLoader
    loader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=hoi_collate)
    batches = list(loader)
    assert len(batches) == 2

    b0 = batches[0]
    assert b0.num_graphs == 3
    assert b0.num_nodes == sum(dataset[i].num_nodes for i in range(3))
    assert b0.pair_index.shape[0] == sum(dataset[i].pair_index.shape[0] for i in range(3))
    assert b0.pair_batch.shape[0] == b0.pair_index.shape[0]
    assert b0.y.shape == (b0.pair_index.shape[0], NUM_ACTIONS)
    assert b0.y_mask.shape == (b0.pair_index.shape[0], NUM_ACTIONS)
