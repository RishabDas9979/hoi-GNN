"""Explainability package for Relational HOI Detection Models.

Provides post-hoc graph-level explainability methods, attention analysis, and benchmark evaluation metrics:
- GNNExplainer (Ying et al. NeurIPS 2019): Post-hoc relational edge and node mask optimization
- Attention Analysis: Multi-head attention extraction, pair-specific routing, distribution plotting
- Integrated Gradients (Sundararajan et al. ICML 2017): Captum path-integral attribution over graph features
- Quantitative explanation metrics: Fidelity+, Fidelity-, Sparsity, Stability, Adebayo Sanity Check

Note:
    All attributions represent graph-level attribution (discrete entities and spatial edges),
    NOT dense pixel-level attribution.
"""
from __future__ import annotations

from explainability.attention_analysis import (
    CAUSALITY_WARNING,
    EdgeAttentionAnalysis,
    analyze_pair_attention,
    extract_attention_weights,
    extract_gatv2_attention,
    plot_attention_distribution,
)
from explainability.explanation_metrics import (
    compute_fidelity_minus,
    compute_fidelity_plus,
    compute_seed_stability,
    compute_sparsity,
    evaluate_explanation,
    randomized_model_sanity_check,
)
from explainability.gnn_explainer import (
    GNNExplainerExplanation,
    HOIExplainer,
    HOIExplanation,
    HOIGNNExplainer,
    HOIModelExplainAdapter,
)
from explainability.integrated_gradients import (
    HOIIntegratedGradients,
    IGExplanation,
)

__all__ = [
    # GNNExplainer
    "HOIExplainer",
    "HOIGNNExplainer",
    "HOIExplanation",
    "GNNExplainerExplanation",
    "HOIModelExplainAdapter",
    # Attention Analysis
    "extract_attention_weights",
    "extract_gatv2_attention",
    "analyze_pair_attention",
    "plot_attention_distribution",
    "EdgeAttentionAnalysis",
    "CAUSALITY_WARNING",
    # Integrated Gradients
    "HOIIntegratedGradients",
    "IGExplanation",
    # Explanation Metrics
    "compute_fidelity_plus",
    "compute_fidelity_minus",
    "compute_sparsity",
    "compute_seed_stability",
    "randomized_model_sanity_check",
    "evaluate_explanation",
]
