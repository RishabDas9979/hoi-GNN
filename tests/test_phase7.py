"""Tests for Phase 7: Integrated Gradients and Explanation Metrics.

Validates:
- HOIIntegratedGradients attribution generation on relational HOI models.
- Handling of node visual features, edge relation attributes, and pair spatial features.
- Baseline strategies ('zero', 'mean') and aggregation methods ('norm', 'abs_sum', 'sum').
- Quantitative explanation metrics: Fidelity+, Fidelity-, Sparsity, Stability,
  Randomized-Model Sanity Check (Adebayo et al.), and Unified Evaluator.
"""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from explainability import (
    HOIIntegratedGradients,
    IGExplanation,
    HOIExplainer,
    HOIExplanation,
    HOIModelExplainAdapter,
    extract_attention_weights,
    analyze_pair_attention,
    plot_attention_distribution,
    CAUSALITY_WARNING,
    compute_fidelity_plus,
    compute_fidelity_minus,
    compute_sparsity,
    compute_seed_stability,
    randomized_model_sanity_check,
    evaluate_explanation,
)
from models.edge_aware_gatv2 import EdgeAwareGATv2HOIModel
from models.gat import GATHOIModel
from visualization import (
    visualize_explanation_graph,
    visualize_bbox_attribution,
    plot_dual_explanation_view,
)


def create_mock_graph(
    num_nodes: int = 4,
    node_dim: int = 16,
    num_actions: int = 5,
) -> Data:
    """Creates a deterministic synthetic graph for explainability unit tests."""
    torch.manual_seed(42)
    x = torch.randn(num_nodes, node_dim)
    # Fully connect directed edges
    src = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.long)
    dst = torch.tensor([1, 2, 0, 3, 3, 0], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(edge_index.shape[1], 9)

    pair_index = torch.tensor([[0, 1], [0, 2], [1, 3]], dtype=torch.long)
    pair_attr = torch.randn(pair_index.shape[0], 9)
    class_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    boxes_norm = torch.tensor(
        [
            [0.1, 0.1, 0.4, 0.5],
            [0.2, 0.2, 0.6, 0.7],
            [0.5, 0.5, 0.8, 0.9],
            [0.0, 0.0, 0.3, 0.3],
        ],
        dtype=torch.float32,
    )

    return Data(
        x_visual=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pair_index=pair_index,
        pair_attr=pair_attr,
        class_ids=class_ids,
        boxes_norm=boxes_norm,
        num_nodes=num_nodes,
    )


def test_ig_explanation_dataclass():
    """Verify IGExplanation dataclass attributes and documentation."""
    exp = IGExplanation(
        node_importance=torch.ones(4),
        node_feat_importance=torch.ones(4, 16),
        edge_importance=torch.ones(6),
        delta=0.005,
        target_pair=(0, 1),
        action_idx=2,
    )
    assert exp.node_importance.shape == (4,)
    assert exp.node_feat_importance.shape == (4, 16)
    assert exp.edge_importance is not None and exp.edge_importance.shape == (6,)
    assert exp.delta == 0.005
    assert exp.target_pair == (0, 1)
    assert exp.action_idx == 2
    assert exp.edge_attr_importance is None
    assert exp.pair_attr_importance is None
    assert "Graph-level attribution, not pixel-level" in IGExplanation.__doc__


def test_ig_attribution_edge_aware_gatv2():
    """Verify HOIIntegratedGradients on EdgeAwareGATv2HOIModel."""
    model = EdgeAwareGATv2HOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=2,
        heads=2,
        edge_dim=9,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()

    explainer = HOIIntegratedGradients(model)
    explanation = explainer.attribute(
        data=data,
        pair_idx=0,
        action_idx=1,
        attribute_edges=True,
        attribute_pair_attr=True,
        n_steps=15,
        method="gausslegendre",
    )

    assert isinstance(explanation, IGExplanation)
    assert explanation.node_importance.shape == (4,)
    assert explanation.node_feat_importance.shape == (4, 16)
    assert explanation.edge_importance is not None
    assert explanation.edge_importance.shape == (6,)
    assert explanation.edge_attr_importance is not None
    assert explanation.edge_attr_importance.shape == (6, 9)
    assert explanation.pair_attr_importance is not None
    assert explanation.pair_attr_importance.shape == (3, 9)
    assert explanation.target_pair == (0, 1)
    assert explanation.action_idx == 1
    assert abs(explanation.delta) < 0.1


def test_ig_attribution_by_tuple_pair():
    """Verify pair resolution when passing a (human, object) tuple."""
    model = EdgeAwareGATv2HOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=1,
        heads=2,
        edge_dim=9,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()

    explainer = HOIIntegratedGradients(model)
    explanation = explainer.attribute(
        data=data,
        pair_idx=(1, 3),  # third pair in mock graph
        action_idx=3,
        n_steps=10,
    )

    assert explanation.target_pair == (1, 3)
    assert explanation.action_idx == 3


def test_ig_baselines_zero_and_mean():
    """Verify baseline strategies 'zero' and 'mean' compute valid attributions."""
    model = EdgeAwareGATv2HOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=1,
        heads=2,
        edge_dim=9,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()
    explainer = HOIIntegratedGradients(model)

    exp_zero = explainer.attribute(
        data=data, pair_idx=0, action_idx=0, baseline_type="zero", n_steps=10
    )
    exp_mean = explainer.attribute(
        data=data, pair_idx=0, action_idx=0, baseline_type="mean", n_steps=10
    )

    assert exp_zero.node_importance.shape == (4,)
    assert exp_mean.node_importance.shape == (4,)
    assert not torch.allclose(exp_zero.node_feat_importance, exp_mean.node_feat_importance)


def test_ig_aggregation_modes():
    """Verify per-node and per-edge aggregation options."""
    model = EdgeAwareGATv2HOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=1,
        heads=2,
        edge_dim=9,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()
    explainer = HOIIntegratedGradients(model)

    exp_norm = explainer.attribute(
        data=data, pair_idx=0, action_idx=0, node_aggregation="norm", n_steps=5
    )
    exp_sum = explainer.attribute(
        data=data, pair_idx=0, action_idx=0, node_aggregation="sum", n_steps=5
    )
    exp_abs = explainer.attribute(
        data=data, pair_idx=0, action_idx=0, node_aggregation="abs_sum", n_steps=5
    )

    assert exp_norm.node_importance.shape == (4,)
    assert exp_sum.node_importance.shape == (4,)
    assert exp_abs.node_importance.shape == (4,)
    # Norm values must be non-negative
    assert (exp_norm.node_importance >= 0).all()
    assert (exp_abs.node_importance >= 0).all()


def test_ig_graceful_handling_without_edge_attr():
    """Verify explainer works when model does not use edge attributes."""
    model = EdgeAwareGATv2HOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=1,
        heads=2,
        use_edge_attr=False,
        edge_dim=None,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()

    explainer = HOIIntegratedGradients(model)
    explanation = explainer.attribute(
        data=data,
        pair_idx=0,
        action_idx=1,
        attribute_edges=True,
        n_steps=10,
    )

    assert explanation.node_importance.shape == (4,)
    # Model does not use edge_attr, so edge_importance is None
    assert explanation.edge_importance is None


def test_compute_fidelity_plus_and_minus():
    """Verify Fidelity+ and Fidelity- metrics."""
    model = EdgeAwareGATv2HOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=1,
        heads=2,
        edge_dim=9,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()

    edge_mask = torch.tensor([0.9, 0.8, 0.1, 0.05, 0.02, 0.01])
    node_mask = torch.tensor([0.8, 0.7, 0.2, 0.1])

    fid_plus = compute_fidelity_plus(
        model=model,
        data=data,
        pair_idx=0,
        action_idx=1,
        edge_mask=edge_mask,
        node_mask=node_mask,
        top_k_ratio=0.33,
    )
    assert isinstance(fid_plus, float)

    fid_minus = compute_fidelity_minus(
        model=model,
        data=data,
        pair_idx=0,
        action_idx=1,
        edge_mask=edge_mask,
        node_mask=node_mask,
        top_k_ratio=0.33,
    )
    assert isinstance(fid_minus, float)


def test_compute_sparsity():
    """Verify Sparsity metric under various mask distributions."""
    # All active (all 1.0) -> sparsity = 0.0
    all_active = torch.ones(10)
    assert compute_sparsity(all_active, threshold=0.5, normalize=False) == 0.0

    # All zeros -> sparsity = 1.0
    all_zero = torch.zeros(10)
    assert compute_sparsity(all_zero, threshold=0.5, normalize=False) == 1.0

    # Half active
    half_active = torch.tensor([1.0] * 5 + [0.0] * 5)
    assert compute_sparsity(half_active, threshold=0.5, normalize=False) == 0.5

    # Normalized mask
    unnormalized = torch.tensor([0.0, 0.1, 0.2, 0.3, 10.0])
    sp = compute_sparsity(unnormalized, threshold=0.5, normalize=True)
    assert sp == 0.8  # only 10.0 is > 0.5 when normalized


def test_compute_seed_stability():
    """Verify stability across seeds for Jaccard, Spearman, Pearson, and Cosine."""
    mask_a = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    mask_b = torch.tensor([0.9, 0.85, 0.55, 0.35, 0.25, 0.05])
    mask_c = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.8, 1.0])  # inverted

    # Identical ranking
    jaccard = compute_seed_stability(mask_a, mask_b, top_k=2, metric="jaccard")
    assert jaccard == 1.0

    # All metrics dict
    all_metrics = compute_seed_stability(mask_a, mask_b, metric="all")
    assert isinstance(all_metrics, dict)
    assert all_metrics["spearman"] > 0.9
    assert all_metrics["pearson"] > 0.9
    assert all_metrics["cosine"] > 0.9

    # Inverted ranking
    s_inv = compute_seed_stability(mask_a, mask_c, metric="spearman")
    assert s_inv < -0.9


def test_randomized_model_sanity_check():
    """Verify randomized-model sanity check execution."""
    model = EdgeAwareGATv2HOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=1,
        heads=2,
        edge_dim=9,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()

    def mock_explainer(m, d, p, a):
        ig = HOIIntegratedGradients(m)
        return ig.attribute(d, pair_idx=p, action_idx=a, n_steps=5)

    sanity_results = randomized_model_sanity_check(
        trained_model=model,
        explainer_fn=mock_explainer,
        data=data,
        pair_idx=0,
        action_idx=1,
    )

    assert "spearman_correlation" in sanity_results
    assert "pearson_correlation" in sanity_results
    assert "cosine_similarity" in sanity_results
    assert "passed_sanity_check" in sanity_results


def test_unified_evaluate_explanation():
    """Verify evaluate_explanation returns unified metrics dictionary."""
    model = EdgeAwareGATv2HOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=1,
        heads=2,
        edge_dim=9,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()

    edge_mask = torch.tensor([0.9, 0.5, 0.2, 0.1, 0.05, 0.0])
    node_mask = torch.tensor([0.8, 0.3, 0.1, 0.0])

    summary = evaluate_explanation(
        model=model,
        data=data,
        pair_idx=0,
        action_idx=1,
        edge_mask=edge_mask,
        node_mask=node_mask,
        top_k_ratio=0.33,
    )

    expected_keys = {
        "orig_probability",
        "fidelity_plus",
        "fidelity_minus",
        "delta_fidelity",
        "edge_sparsity",
        "node_sparsity",
    }
    assert expected_keys.issubset(summary.keys())
    assert 0.0 <= summary["orig_probability"] <= 1.0
    assert 0.0 <= summary["edge_sparsity"] <= 1.0
    assert 0.0 <= summary["node_sparsity"] <= 1.0


# ==============================================================================
# GNNExplainer Tests (Task 7.1)
# ==============================================================================


def test_hoi_explainer_edge_aware_gatv2():
    """Verify HOIExplainer optimization and mask generation on EdgeAwareGATv2HOIModel."""
    model = EdgeAwareGATv2HOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=2,
        heads=2,
        edge_dim=9,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()

    explainer = HOIExplainer(model=model, epochs=15, lr=0.05)
    exp = explainer.explain(data, pair_idx=0, action_idx=1, top_k_edges=3, top_k_nodes=2)

    assert isinstance(exp, HOIExplanation)
    assert exp.target_pair == (0, 1)
    assert exp.action_idx == 1
    assert 0.0 <= exp.score <= 1.0
    assert exp.edge_mask.shape == (data.edge_index.shape[1],)
    assert exp.node_mask.shape == (data.num_nodes,)
    assert exp.edge_mask.min() >= 0.0 and exp.edge_mask.max() <= 1.0
    assert exp.node_mask.min() >= 0.0 and exp.node_mask.max() <= 1.0
    assert len(exp.important_edges) <= 3
    assert len(exp.important_nodes) <= 2
    # Check top edge tuple structure: (src, dst, weight)
    for src, dst, w in exp.important_edges:
        assert 0 <= src < data.num_nodes
        assert 0 <= dst < data.num_nodes
        assert 0.0 <= w <= 1.0


def test_hoi_explainer_gat_and_tuple_target():
    """Verify HOIExplainer with GATHOIModel and tuple target pair specification."""
    model = GATHOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=1,
        heads=2,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()

    explainer = HOIExplainer(model=model, epochs=10)
    exp = explainer.explain(data, pair_idx=(0, 2), action_idx=None)

    assert exp.target_pair == (0, 2)
    assert 0 <= exp.action_idx < 5
    assert exp.edge_mask.shape == (data.edge_index.shape[1],)
    assert exp.node_mask.shape == (data.num_nodes,)


def test_hoi_explainer_pyg_adapter():
    """Verify HOIModelExplainAdapter forward pass output shape and PyG compatibility."""
    model = EdgeAwareGATv2HOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=1,
        heads=2,
        edge_dim=9,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()

    adapter = HOIModelExplainAdapter(model, data_template=data, pair_idx=0, action_idx=1)
    out = adapter(data.x_visual, data.edge_index, data=data)
    assert out.shape == (1, 1)


# ==============================================================================
# Attention Analysis Tests (Task 7.2)
# ==============================================================================


def test_extract_attention_weights_edge_aware_gatv2():
    """Verify multi-head attention weight extraction on EdgeAwareGATv2HOIModel."""
    model = EdgeAwareGATv2HOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=2,
        heads=4,
        edge_dim=9,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()

    att_layers = extract_attention_weights(model, data)
    assert len(att_layers) == 2

    for l_idx, layer in enumerate(att_layers):
        assert layer["layer_idx"] == l_idx
        assert layer["edge_index"].shape[0] == 2
        assert layer["edge_index"].shape[1] >= data.edge_index.shape[1]
        assert layer["alpha_heads"].shape == (layer["edge_index"].shape[1], 4)
        assert layer["alpha_mean"].shape == (layer["edge_index"].shape[1],)
        assert layer["alpha_max"].shape == (layer["edge_index"].shape[1],)
        assert layer["num_heads"] == 4


def test_extract_attention_weights_gat():
    """Verify multi-head attention weight extraction on GATHOIModel."""
    model = GATHOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=1,
        heads=2,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()

    att_layers = extract_attention_weights(model, data)
    assert len(att_layers) == 1
    assert att_layers[0]["edge_index"].shape[0] == 2
    assert att_layers[0]["alpha_heads"].shape == (att_layers[0]["edge_index"].shape[1], 2)


def test_analyze_pair_attention_and_causality_caveat():
    """Verify pair-specific attention routing analysis and causality documentation."""
    model = EdgeAwareGATv2HOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=2,
        heads=2,
        edge_dim=9,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()

    analysis = analyze_pair_attention(model, data, pair_idx=0)

    assert analysis["target_pair"] == (0, 1)
    assert "causality_caveat" in analysis
    assert "NOT proof of causal importance" in analysis["causality_caveat"]
    assert "NOT proof of causal importance" in CAUSALITY_WARNING
    assert len(analysis["layer_analyses"]) == 2
    assert "mean_direct_attention" in analysis
    assert isinstance(analysis["top_context_to_human"], list)
    assert isinstance(analysis["top_context_to_object"], list)


def test_plot_attention_distribution(tmp_path):
    """Verify attention distribution plotting saves publication-ready figures."""
    model = EdgeAwareGATv2HOIModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=2,
        heads=2,
        edge_dim=9,
        num_actions=5,
    )
    model.eval()
    data = create_mock_graph()

    # 1. Test plotting pair analysis dict
    pair_analysis = analyze_pair_attention(model, data, pair_idx=0)
    out_file1 = str(tmp_path / "pair_att.png")
    fig1 = plot_attention_distribution(pair_analysis, out_path=out_file1)
    assert fig1 is not None
    assert (tmp_path / "pair_att.png").exists()

    # 2. Test plotting raw attention list
    raw_att = extract_attention_weights(model, data)
    out_file2 = str(tmp_path / "raw_att.png")
    fig2 = plot_attention_distribution(raw_att, out_path=out_file2)
    assert fig2 is not None
    assert (tmp_path / "raw_att.png").exists()


# ==============================================================================
# Explanation Visualizer Tests (Task 7.5)
# ==============================================================================


def test_visualize_explanation_graph(tmp_path):
    """Verify scene graph attribution visualization."""
    data = create_mock_graph()
    edge_mask = torch.tensor([0.9, 0.7, 0.4, 0.2, 0.1, 0.05])
    node_mask = torch.tensor([0.8, 0.6, 0.3, 0.1])

    out_file = str(tmp_path / "scene_graph_exp.png")
    fig = visualize_explanation_graph(
        data=data,
        edge_mask=edge_mask,
        node_mask=node_mask,
        target_pair=(0, 1),
        action_name="hold",
        out_path=out_file,
    )
    assert fig is not None
    assert (tmp_path / "scene_graph_exp.png").exists()


def test_visualize_bbox_attribution(tmp_path):
    """Verify bounding box node attribution visualization."""
    data = create_mock_graph()
    node_mask = torch.tensor([0.9, 0.6, 0.3, 0.1])

    out_file = str(tmp_path / "bbox_exp.png")
    fig = visualize_bbox_attribution(
        data=data,
        node_mask=node_mask,
        target_pair=(0, 1),
        action_name="sit",
        out_path=out_file,
    )
    assert fig is not None
    assert (tmp_path / "bbox_exp.png").exists()


def test_plot_dual_explanation_view(tmp_path):
    """Verify dual-view comparative visualization."""
    data = create_mock_graph()
    edge_mask = torch.tensor([0.9, 0.7, 0.4, 0.2, 0.1, 0.05])
    node_mask = torch.tensor([0.8, 0.6, 0.3, 0.1])

    out_file = str(tmp_path / "dual_exp.png")
    fig = plot_dual_explanation_view(
        data=data,
        edge_mask=edge_mask,
        node_mask=node_mask,
        target_pair=(0, 1),
        action_name="ride",
        out_path=out_file,
    )
    assert fig is not None
    assert (tmp_path / "dual_exp.png").exists()

