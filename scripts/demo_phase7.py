"""Phase 7 Demo & Smoke Test.

Demonstrates end-to-end explainability pipeline on synthetic graph:
1. GNNExplainer optimization and top edge/node ranking.
2. Captum Integrated Gradients attribution on node and edge attributes.
3. Attention weight extraction and message-passing routing analysis.
4. Quantitative explanation metrics (Fidelity+, Fidelity-, Sparsity, Adebayo Sanity Check).
5. Visualizations export (Scene Graph, Bounding Box, Dual-View, Attention plots).
"""
import sys
from pathlib import Path
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from data.synthetic_dataset import SyntheticHOIGenerator
from explainability import (
    HOIExplainer,
    HOIIntegratedGradients,
    analyze_pair_attention,
    evaluate_explanation,
    extract_attention_weights,
    plot_attention_distribution,
    randomized_model_sanity_check,
)
from models.edge_aware_gatv2 import EdgeAwareGATv2HOIModel
from visualization import (
    plot_dual_explanation_view,
    visualize_bbox_attribution,
    visualize_explanation_graph,
)


def run_demo():
    print("=" * 60)
    print("  HOI-GNN Phase 7: Post-Hoc Graph Explainability Demo")
    print("=" * 60)

    # 1. Generate Synthetic Scene Graph
    print("\n[1/5] Generating synthetic scene graph...")
    generator = SyntheticHOIGenerator(feature_dim=64, seed=42)
    data = generator.generate_graph(image_id=0)
    print(f"  Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}, Candidate Pairs: {data.pair_index.shape[0]}")

    # 2. Instantiate Relational Model
    print("\n[2/5] Instantiating Edge-Aware GATv2 model...")
    model = EdgeAwareGATv2HOIModel(
        input_dim=64,
        hidden_dim=64,
        num_layers=2,
        heads=4,
        edge_dim=9,
        num_actions=26,
    )
    model.eval()

    # 3. GNNExplainer
    print("\n[3/5] Running HOIExplainer optimization...")
    explainer = HOIExplainer(model=model, epochs=25, lr=0.03)
    exp = explainer.explain(data, pair_idx=0, top_k_edges=5, top_k_nodes=3)
    print(f"  Target Pair: {exp.target_pair}, Action Idx: {exp.action_idx}, Score: {exp.score:.4f}")
    print(f"  Top Important Edges: {exp.important_edges[:3]}")
    print(f"  Top Important Nodes: {exp.important_nodes[:3]}")

    # 4. Integrated Gradients & Metrics
    print("\n[4/5] Computing Integrated Gradients and Quantitative Metrics...")
    ig = HOIIntegratedGradients(model)
    ig_exp = ig.attribute(data, pair_idx=0, action_idx=exp.action_idx, n_steps=10)
    print(f"  IG Delta (completeness error): {ig_exp.delta:.6f}")

    metrics = evaluate_explanation(
        model=model,
        data=data,
        pair_idx=0,
        action_idx=exp.action_idx,
        edge_mask=exp.edge_mask,
        node_mask=exp.node_mask,
        top_k_ratio=0.4,
    )
    print(f"  Fidelity+ (Comprehensiveness): {metrics['fidelity_plus']:.4f}")
    print(f"  Fidelity- (Sufficiency):       {metrics['fidelity_minus']:.4f}")
    print(f"  Delta Fidelity (+ - -):        {metrics['delta_fidelity']:.4f}")
    print(f"  Edge Sparsity:                 {metrics['edge_sparsity']:.4f}")
    print(f"  Node Sparsity:                 {metrics['node_sparsity']:.4f}")

    # 5. Attention & Visualizations
    print("\n[5/5] Generating Visualizations and Attention Plots...")
    out_dir = Path("results/explainability_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Attention
    pair_att = analyze_pair_attention(model, data, pair_idx=0)
    att_path = out_dir / "attention_distribution.png"
    plot_attention_distribution(pair_att, out_path=att_path)
    print(f"  Saved: {att_path}")

    # Scene Graph Attribution
    sg_path = out_dir / "scene_graph_explanation.png"
    visualize_explanation_graph(
        data=data,
        edge_mask=exp.edge_mask,
        node_mask=exp.node_mask,
        target_pair=exp.target_pair,
        action_name=f"action_{exp.action_idx}",
        out_path=sg_path,
    )
    print(f"  Saved: {sg_path}")

    # Bounding Box Attribution
    bbox_path = out_dir / "bbox_explanation.png"
    visualize_bbox_attribution(
        data=data,
        node_mask=exp.node_mask,
        target_pair=exp.target_pair,
        action_name=f"action_{exp.action_idx}",
        out_path=bbox_path,
    )
    print(f"  Saved: {bbox_path}")

    # Dual View
    dual_path = out_dir / "dual_view_explanation.png"
    plot_dual_explanation_view(
        data=data,
        edge_mask=exp.edge_mask,
        node_mask=exp.node_mask,
        target_pair=exp.target_pair,
        action_name=f"action_{exp.action_idx}",
        out_path=dual_path,
    )
    print(f"  Saved: {dual_path}")

    print("\n[SUCCESS] Phase 7 Demo completed without errors.")


if __name__ == "__main__":
    run_demo()
