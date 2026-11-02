"""Quantitative Explanation Evaluation Metrics for Relational HOI Models.

Implements benchmark evaluation metrics following graph explainability literature
(GraphExplainer, Yuan et al. 2022, Sanchez-Lengeling et al. 2020, Adebayo et al. 2018):
1. Fidelity+ (Comprehensiveness): tests probability drop upon removing essential components.
2. Fidelity- (Sufficiency): tests probability retention using only the isolated explanation subgraph.
3. Sparsity: measures explanation compactness (fraction of inactive graph elements).
4. Stability Across Seeds: measures agreement between explanations generated across multiple training seeds.
5. Randomized-Model Sanity Check: verifies explanations depend on learned weights rather than graph priors.
6. Comprehensive Evaluator: unified evaluation reporting all metrics.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy import stats


def _to_tensor(x: Union[torch.Tensor, np.ndarray, List[float]]) -> torch.Tensor:
    """Converts array-like input to a 1D float torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().clone().view(-1).float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).view(-1).float()
    return torch.tensor(x, dtype=torch.float32).view(-1)


def _resolve_pair_index(
    pair_index: torch.Tensor,
    pair_idx: Union[int, Tuple[int, int]],
) -> int:
    """Resolves pair identifier into an integer index into pair_index [P, 2]."""
    if isinstance(pair_idx, tuple):
        h_target, o_target = pair_idx
        matches = ((pair_index[:, 0] == h_target) & (pair_index[:, 1] == o_target)).nonzero(
            as_tuple=False
        )
        if matches.numel() == 0:
            raise ValueError(f"Pair ({h_target}, {o_target}) not found in candidate pair_index.")
        return int(matches[0].item())

    p_idx = int(pair_idx)
    if p_idx < 0 or p_idx >= pair_index.shape[0]:
        raise IndexError(
            f"pair_idx {p_idx} out of range for pair_index of shape {pair_index.shape}."
        )
    return p_idx


@torch.no_grad()
def compute_fidelity_plus(
    model: nn.Module,
    data: Any,
    pair_idx: Union[int, Tuple[int, int]],
    action_idx: int,
    edge_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    node_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    top_k_ratio: float = 0.3,
) -> float:
    """Compute Fidelity+ (Comprehensiveness) metric.

    Measures the drop in model predicted probability when the top-k% most important
    graph elements (edges or nodes) identified by the explanation are removed:
        Fid+ = sigma(logit_orig) - sigma(logit_masked)

    Higher Fidelity+ indicates that the removed elements were truly essential for
    the model's initial prediction.

    Args:
        model: HOI detection model implementing model(data) -> logits [P, A].
        data: PyG Data object.
        pair_idx: Candidate pair index or (human_idx, object_idx) tuple.
        action_idx: Target action category index.
        edge_mask: Optional 1D importance tensor/array over edges [E].
        node_mask: Optional 1D importance tensor/array over nodes [N].
        top_k_ratio: Fraction of top components to remove in range (0.0, 1.0].

    Returns:
        float: Probability drop sigma(logit_orig) - sigma(logit_masked).
    """
    model.eval()
    p_idx = _resolve_pair_index(data.pair_index, pair_idx)

    # Original model prediction
    orig_out = model(data)
    if isinstance(orig_out, tuple):
        orig_out = orig_out[0]
    orig_prob = float(torch.sigmoid(orig_out[p_idx, action_idx]).item())

    if edge_mask is None and node_mask is None:
        return 0.0

    masked_data = data.clone()

    # 1. Mask edges if edge_mask provided
    if edge_mask is not None:
        e_mask = _to_tensor(edge_mask)
        num_edges = e_mask.numel()
        if num_edges > 0:
            k = max(1, min(num_edges, int(round(top_k_ratio * num_edges))))
            top_edge_indices = torch.topk(e_mask, k=k, largest=True).indices

            keep_edges = torch.ones(num_edges, dtype=torch.bool)
            keep_edges[top_edge_indices] = False

            if hasattr(masked_data, "edge_index") and masked_data.edge_index is not None:
                masked_data.edge_index = masked_data.edge_index[:, keep_edges]
            if hasattr(masked_data, "edge_attr") and masked_data.edge_attr is not None:
                masked_data.edge_attr = masked_data.edge_attr[keep_edges]

    # 2. Mask nodes if node_mask provided
    if node_mask is not None:
        n_mask = _to_tensor(node_mask)
        num_nodes = n_mask.numel()
        if num_nodes > 0:
            k = max(1, min(num_nodes, int(round(top_k_ratio * num_nodes))))
            top_node_indices = torch.topk(n_mask, k=k, largest=True).indices

            # Zero out visual features of top nodes
            if hasattr(masked_data, "x_visual") and masked_data.x_visual is not None:
                masked_x_vis = masked_data.x_visual.clone()
                masked_x_vis[top_node_indices] = 0.0
                masked_data.x_visual = masked_x_vis
            if hasattr(masked_data, "x") and masked_data.x is not None:
                masked_x = masked_data.x.clone()
                masked_x[top_node_indices] = 0.0
                masked_data.x = masked_x

    # Masked prediction
    masked_out = model(masked_data)
    if isinstance(masked_out, tuple):
        masked_out = masked_out[0]
    masked_prob = float(torch.sigmoid(masked_out[p_idx, action_idx]).item())

    return float(orig_prob - masked_prob)


@torch.no_grad()
def compute_fidelity_minus(
    model: nn.Module,
    data: Any,
    pair_idx: Union[int, Tuple[int, int]],
    action_idx: int,
    edge_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    node_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    top_k_ratio: float = 0.3,
) -> float:
    """Compute Fidelity- (Sufficiency) metric.

    Measures probability retention when ONLY the top-k% most important graph
    elements identified by the explanation are kept (all non-salient elements removed):
        Fid- = sigma(logit_orig) - sigma(logit_retained)

    Lower Fidelity- indicates that the explanation subgraph alone is sufficient
    to preserve the model's prediction.

    Args:
        model: HOI detection model implementing model(data) -> logits [P, A].
        data: PyG Data object.
        pair_idx: Candidate pair index or (human_idx, object_idx) tuple.
        action_idx: Target action category index.
        edge_mask: Optional 1D importance tensor/array over edges [E].
        node_mask: Optional 1D importance tensor/array over nodes [N].
        top_k_ratio: Fraction of top components to retain in range (0.0, 1.0].

    Returns:
        float: Difference sigma(logit_orig) - sigma(logit_retained). Lower is better.
    """
    model.eval()
    p_idx = _resolve_pair_index(data.pair_index, pair_idx)

    # Original model prediction
    orig_out = model(data)
    if isinstance(orig_out, tuple):
        orig_out = orig_out[0]
    orig_prob = float(torch.sigmoid(orig_out[p_idx, action_idx]).item())

    if edge_mask is None and node_mask is None:
        return 0.0

    retained_data = data.clone()

    # 1. Retain only top edges if edge_mask provided
    if edge_mask is not None:
        e_mask = _to_tensor(edge_mask)
        num_edges = e_mask.numel()
        if num_edges > 0:
            k = max(1, min(num_edges, int(round(top_k_ratio * num_edges))))
            top_edge_indices = torch.topk(e_mask, k=k, largest=True).indices

            retain_edges = torch.zeros(num_edges, dtype=torch.bool)
            retain_edges[top_edge_indices] = True

            if hasattr(retained_data, "edge_index") and retained_data.edge_index is not None:
                retained_data.edge_index = retained_data.edge_index[:, retain_edges]
            if hasattr(retained_data, "edge_attr") and retained_data.edge_attr is not None:
                retained_data.edge_attr = retained_data.edge_attr[retain_edges]

    # 2. Retain only top nodes if node_mask provided
    if node_mask is not None:
        n_mask = _to_tensor(node_mask)
        num_nodes = n_mask.numel()
        if num_nodes > 0:
            k = max(1, min(num_nodes, int(round(top_k_ratio * num_nodes))))
            top_node_indices = torch.topk(n_mask, k=k, largest=True).indices

            remove_nodes = torch.ones(num_nodes, dtype=torch.bool)
            remove_nodes[top_node_indices] = False

            # Zero out non-top nodes
            if hasattr(retained_data, "x_visual") and retained_data.x_visual is not None:
                retained_x_vis = retained_data.x_visual.clone()
                retained_x_vis[remove_nodes] = 0.0
                retained_data.x_visual = retained_x_vis
            if hasattr(retained_data, "x") and retained_data.x is not None:
                retained_x = retained_data.x.clone()
                retained_x[remove_nodes] = 0.0
                retained_data.x = retained_x

    # Retained prediction
    retained_out = model(retained_data)
    if isinstance(retained_out, tuple):
        retained_out = retained_out[0]
    retained_prob = float(torch.sigmoid(retained_out[p_idx, action_idx]).item())

    return float(orig_prob - retained_prob)


def compute_sparsity(
    mask: Union[torch.Tensor, np.ndarray, List[float]],
    threshold: float = 0.5,
    normalize: bool = True,
) -> float:
    """Compute Sparsity of an importance mask.

    Measures the fraction of elements considered inactive or negligible:
        Sparsity = 1 - (count(mask > threshold) / total)

    Higher sparsity indicates a more concise, focused explanation.

    Args:
        mask: Importance tensor/array for edges or nodes.
        threshold: Decision threshold for active components (default: 0.5).
        normalize: If True and max > min, normalizes mask linearly to [0, 1] first.

    Returns:
        float: Sparsity in range [0.0, 1.0].
    """
    t = _to_tensor(mask)
    if t.numel() == 0:
        return 1.0

    if normalize:
        min_val = t.min().item()
        max_val = t.max().item()
        diff = max_val - min_val
        if diff > 1e-8:
            t = (t - min_val) / diff
        else:
            t = torch.zeros_like(t)

    active_count = float((t > threshold).sum().item())
    total_count = float(t.numel())
    sparsity = 1.0 - (active_count / total_count)
    return float(np.clip(sparsity, 0.0, 1.0))


def compute_seed_stability(
    mask_seed1: Union[torch.Tensor, np.ndarray, List[float]],
    mask_seed2: Union[torch.Tensor, np.ndarray, List[float]],
    top_k: int = 5,
    metric: str = "jaccard",
) -> Union[float, Dict[str, float]]:
    """Compute explanation stability across models trained with different random seeds.

    Args:
        mask_seed1: Importance mask from model seed 1.
        mask_seed2: Importance mask from model seed 2.
        top_k: Number of top elements to inspect for rank/Jaccard stability.
        metric: Metric to compute ('jaccard', 'spearman', 'pearson', 'cosine', or 'all').

    Returns:
        Union[float, Dict[str, float]]: Stability score or dictionary of scores.
    """
    m1 = _to_tensor(mask_seed1)
    m2 = _to_tensor(mask_seed2)

    if m1.numel() != m2.numel():
        raise ValueError(f"Mask length mismatch: {m1.numel()} vs {m2.numel()}.")

    num_elements = m1.numel()
    if num_elements == 0:
        if metric.lower() == "all":
            return {"jaccard": 1.0, "spearman": 1.0, "pearson": 1.0, "cosine": 1.0}
        return 1.0

    k = max(1, min(top_k, num_elements))
    top1 = set(torch.topk(m1, k=k, largest=True).indices.tolist())
    top2 = set(torch.topk(m2, k=k, largest=True).indices.tolist())

    union_len = len(top1 | top2)
    jaccard = float(len(top1 & top2) / union_len) if union_len > 0 else 1.0

    # Spearman rank correlation
    s_corr, _ = stats.spearmanr(m1.numpy(), m2.numpy())
    spearman = float(0.0 if np.isnan(s_corr) else s_corr)

    # Pearson linear correlation
    p_corr, _ = stats.pearsonr(m1.numpy(), m2.numpy())
    pearson = float(0.0 if np.isnan(p_corr) else p_corr)

    # Cosine similarity
    denom = (torch.norm(m1, p=2) * torch.norm(m2, p=2)).item()
    cosine = float(torch.dot(m1, m2).item() / denom) if denom > 1e-8 else 1.0

    scores = {
        "jaccard": float(jaccard),
        "spearman": float(spearman),
        "pearson": float(pearson),
        "cosine": float(cosine),
    }

    m = metric.lower()
    if m == "all":
        return scores
    if m in scores:
        return scores[m]
    raise ValueError(f"Unknown stability metric: {metric!r}. Use 'jaccard', 'spearman', 'pearson', 'cosine', or 'all'.")


def randomized_model_sanity_check(
    trained_model: nn.Module,
    explainer_fn: Callable[[nn.Module, Any, Union[int, Tuple[int, int]], int], Any],
    data: Any,
    pair_idx: Union[int, Tuple[int, int]],
    action_idx: int,
    correlation_threshold: float = 0.6,
) -> Dict[str, float]:
    """Perform Adebayo et al. randomized-model sanity check on graph explanations.

    Compares the explanation produced by the trained model with that produced by an
    untrained/reinitialized copy of the same architecture. A valid explanation must
    produce low correlation (<< 1.0), proving it reflects learned weights rather than
    graph structural artifacts or edge density alone.

    Args:
        trained_model: The trained HOI model.
        explainer_fn: Callable taking (model, data, pair_idx, action_idx) and returning
                      an explanation object or tensor mask.
        data: Input graph data.
        pair_idx: Candidate pair index or tuple.
        action_idx: Target action index.
        correlation_threshold: Upper threshold on correlation to pass sanity check (default: 0.6).

    Returns:
        Dict[str, float]: Evaluation metrics:
            - spearman_correlation: Rank correlation between trained and random explanations.
            - pearson_correlation: Linear correlation between explanations.
            - cosine_similarity: Cosine similarity between explanations.
            - passed_sanity_check: 1.0 if spearman_correlation < threshold, 0.0 otherwise.
    """
    trained_model.eval()

    # 1. Generate explanation from trained model
    trained_res = explainer_fn(trained_model, data, pair_idx, action_idx)
    trained_mask = _extract_explanation_mask(trained_res)

    # 2. Instantiate and reinitialize randomized model
    random_model = copy.deepcopy(trained_model)
    _reinitialize_parameters(random_model)
    random_model.eval()

    # 3. Generate explanation from randomized model
    random_res = explainer_fn(random_model, data, pair_idx, action_idx)
    random_mask = _extract_explanation_mask(random_res)

    # 4. Measure alignment between trained and randomized explanations
    m_trained = _to_tensor(trained_mask)
    m_random = _to_tensor(random_mask)

    stability = compute_seed_stability(m_trained, m_random, metric="all")
    assert isinstance(stability, dict)

    spearman = stability["spearman"]
    pearson = stability["pearson"]
    cosine = stability["cosine"]

    # Sanity check passes if correlation is below the threshold
    passed = 1.0 if abs(spearman) < correlation_threshold else 0.0

    return {
        "spearman_correlation": spearman,
        "pearson_correlation": pearson,
        "cosine_similarity": cosine,
        "passed_sanity_check": passed,
    }


def _extract_explanation_mask(explanation: Any) -> torch.Tensor:
    """Extracts a 1D tensor mask from various explanation return types."""
    if hasattr(explanation, "edge_importance") and explanation.edge_importance is not None:
        return explanation.edge_importance
    if hasattr(explanation, "node_importance") and explanation.node_importance is not None:
        return explanation.node_importance
    if isinstance(explanation, tuple):
        # Prefer edge mask (second) if present, else node mask (first)
        if len(explanation) > 1 and explanation[1] is not None:
            return explanation[1]
        return explanation[0]
    if isinstance(explanation, dict):
        if "edge_mask" in explanation:
            return explanation["edge_mask"]
        if "edge_importance" in explanation:
            return explanation["edge_importance"]
        if "node_mask" in explanation:
            return explanation["node_mask"]
        if "node_importance" in explanation:
            return explanation["node_importance"]
    if isinstance(explanation, (torch.Tensor, np.ndarray, list)):
        return _to_tensor(explanation)
    raise ValueError(f"Could not extract importance mask from explanation of type {type(explanation)}.")


def _reinitialize_parameters(model: nn.Module) -> None:
    """Reinitializes all parameters of a model to pseudo-random weights."""
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def evaluate_explanation(
    model: nn.Module,
    data: Any,
    pair_idx: Union[int, Tuple[int, int]],
    action_idx: int,
    edge_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    node_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    top_k_ratio: float = 0.3,
) -> Dict[str, float]:
    """Unified evaluator reporting comprehensive quantitative explanation metrics.

    Args:
        model: HOI detection model.
        data: Graph data instance.
        pair_idx: Candidate pair index or tuple.
        action_idx: Target action category.
        edge_mask: Optional edge importance mask [E].
        node_mask: Optional node importance mask [N].
        top_k_ratio: Fraction of top components used in fidelity tests (default: 0.3).

    Returns:
        Dict[str, float]: Evaluation summary dictionary with:
            - orig_probability: Model baseline prediction probability.
            - fidelity_plus: Fidelity+ comprehensiveness score (higher is better).
            - fidelity_minus: Fidelity- sufficiency score (lower is better).
            - delta_fidelity: fidelity_plus - fidelity_minus (higher is better).
            - edge_sparsity: Sparsity of edge explanation (if edge_mask provided).
            - node_sparsity: Sparsity of node explanation (if node_mask provided).
    """
    model.eval()
    p_idx = _resolve_pair_index(data.pair_index, pair_idx)

    # Original probability
    orig_out = model(data)
    if isinstance(orig_out, tuple):
        orig_out = orig_out[0]
    orig_prob = float(torch.sigmoid(orig_out[p_idx, action_idx]).item())

    fid_plus = compute_fidelity_plus(
        model=model,
        data=data,
        pair_idx=p_idx,
        action_idx=action_idx,
        edge_mask=edge_mask,
        node_mask=node_mask,
        top_k_ratio=top_k_ratio,
    )

    fid_minus = compute_fidelity_minus(
        model=model,
        data=data,
        pair_idx=p_idx,
        action_idx=action_idx,
        edge_mask=edge_mask,
        node_mask=node_mask,
        top_k_ratio=top_k_ratio,
    )

    results: Dict[str, float] = {
        "orig_probability": orig_prob,
        "fidelity_plus": fid_plus,
        "fidelity_minus": fid_minus,
        "delta_fidelity": fid_plus - fid_minus,
    }

    if edge_mask is not None:
        results["edge_sparsity"] = compute_sparsity(edge_mask)

    if node_mask is not None:
        results["node_sparsity"] = compute_sparsity(node_mask)

    return results
