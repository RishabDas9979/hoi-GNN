"""Object-Class Prior Baseline for Human-Object Interaction classification.

Non-parametric baseline that predicts action logits purely from the object class
frequency distribution P(action | object_class) without visual or spatial learning.
"""
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union
import torch
import torch.nn as nn


# Default canonical action associations for fallback when no data file exists
CANONICAL_ACTIONS: Dict[int, int] = {
    1: 3,    # bicycle -> ride
    3: 3,    # motorcycle -> ride
    17: 3,   # horse -> ride
    13: 2,   # bench -> sit
    56: 2,   # chair -> sit
    57: 2,   # couch -> sit
    59: 2,   # bed -> sit
    39: 21,  # bottle -> drink
    40: 21,  # wine glass -> drink
    41: 21,  # cup -> drink
    46: 7,   # banana -> eat
    47: 7,   # apple -> eat
    48: 7,   # sandwich -> eat
    49: 7,   # orange -> eat
    50: 7,   # broccoli -> eat
    51: 7,   # carrot -> eat
    52: 7,   # hot dog -> eat
    53: 7,   # pizza -> eat
    54: 7,   # donut -> eat
    55: 7,   # cake -> eat
    63: 16,  # laptop -> work_on_computer
    64: 16,  # mouse -> work_on_computer
    66: 16,  # keyboard -> work_on_computer
    67: 10,  # cell phone -> talk_on_phone
    73: 24,  # book -> read
    30: 17,  # skis -> ski
    31: 25,  # snowboard -> snowboard
    36: 19,  # skateboard -> skateboard
    37: 18,  # surfboard -> surf
    34: 6,   # baseball bat -> hit
    38: 6,   # tennis racket -> hit
    29: 12,  # frisbee -> throw
    32: 12,  # sports ball -> throw
    24: 11,  # backpack -> carry
    26: 11,  # handbag -> carry
    28: 11,  # suitcase -> carry
    43: 14,  # knife -> cut
    76: 14,  # scissors -> cut
}


def _build_default_prior_matrix(num_classes: int = 81, num_actions: int = 26) -> torch.Tensor:
    """Builds a default co-occurrence frequency matrix."""
    matrix = torch.ones((num_classes, num_actions), dtype=torch.float32) * 0.1
    for cat_id, act_id in CANONICAL_ACTIONS.items():
        if 0 <= cat_id < num_classes and 0 <= act_id < num_actions:
            matrix[cat_id, act_id] += 50.0
            matrix[cat_id, 0] += 20.0  # hold
    return matrix


class ObjectClassPriorBaseline(nn.Module):
    """Non-parametric baseline predicting interactions from object category frequencies.

    Given candidate (human, object) pairs, looks up the object's class ID in a
    precomputed or fitted co-occurrence matrix [81, 26] and outputs log-probability
    (or log-odds) action logits.

    Attributes:
        num_classes (int): Number of object categories (80 COCO classes + 1 unknown at 80).
        num_actions (int): Number of actions (26 for V-COCO).
        cooccurrence (torch.Tensor): Registered buffer of action co-occurrence counts [81, 26].
        prior_logits (torch.Tensor): Registered buffer of calibrated logit table [81, 26].
        log_odds (bool): If True, returns log-odds logits such that sigmoid(logits) == P(a|c).
    """

    def __init__(
        self,
        num_classes: int = 81,
        num_actions: int = 26,
        cooccurrence_matrix: Optional[Union[torch.Tensor, str, Path]] = None,
        alpha: float = 0.1,
        log_odds: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_actions = num_actions
        self.alpha = alpha
        self.log_odds = log_odds

        # Load or initialize co-occurrence counts
        if cooccurrence_matrix is None:
            precomputed_file = Path("data/vcoco/precomputed_cooccurrence_train.pt")
            if precomputed_file.exists():
                matrix = torch.load(precomputed_file, weights_only=True)
            else:
                matrix = _build_default_prior_matrix(num_classes, num_actions)
        elif isinstance(cooccurrence_matrix, (str, Path)):
            matrix = torch.load(cooccurrence_matrix, weights_only=True)
        elif isinstance(cooccurrence_matrix, torch.Tensor):
            matrix = cooccurrence_matrix
        else:
            raise TypeError(f"Unsupported cooccurrence_matrix type: {type(cooccurrence_matrix)}")

        if matrix.shape != (num_classes, num_actions):
            raise ValueError(
                f"Expected matrix shape ({num_classes}, {num_actions}), got {tuple(matrix.shape)}"
            )

        self.register_buffer("cooccurrence", matrix.clone().to(torch.float32))
        self.register_buffer("prior_logits", torch.zeros((num_classes, num_actions), dtype=torch.float32))
        self._update_logits(self.alpha)

    def _update_logits(self, alpha: float) -> None:
        """Computes logit table from the co-occurrence matrix with Laplace smoothing."""
        counts = self.cooccurrence
        row_sums = counts.sum(dim=-1, keepdim=True)  # [81, 1]
        global_counts = counts.sum(dim=0, keepdim=True)  # [1, 26]
        global_total = counts.sum().clamp(min=1.0)
        global_probs = (global_counts + alpha) / (global_total + self.num_actions * alpha)

        # Compute empirical probability per class
        smooth_denom = row_sums + (self.num_actions * alpha)
        cond_probs = (counts + alpha) / smooth_denom.clamp(min=1e-6)

        # Fall back to global distribution for unseen classes with row_sums == 0
        unseen_mask = (row_sums == 0).expand(-1, self.num_actions)
        probs = torch.where(unseen_mask, global_probs.expand(self.num_classes, -1), cond_probs)

        # Clamp probabilities away from exact 0 and 1
        eps = 1e-5
        probs = probs.clamp(min=eps, max=1.0 - eps)

        if self.log_odds:
            # log-odds: log(p / (1 - p)), so that sigmoid(logits) == p
            logits = torch.logit(probs)
        else:
            # log-probabilities: log(p)
            logits = torch.log(probs)

        self.prior_logits.copy_(logits)

    def fit(self, dataset: Iterable[Any], alpha: Optional[float] = None) -> "ObjectClassPriorBaseline":
        """Fits the co-occurrence matrix empirically from a dataset of graphs.

        Args:
            dataset: Iterable of PyG Data or HOIData objects with 'class_ids', 'pair_index', and 'y'.
            alpha: Optional smoothing factor. If None, uses self.alpha.

        Returns:
            self: The fitted baseline model.
        """
        if alpha is not None:
            self.alpha = alpha

        new_counts = torch.zeros((self.num_classes, self.num_actions), dtype=torch.float32)

        for graph in dataset:
            pair_index = getattr(graph, "pair_index", None)
            if pair_index is None or pair_index.numel() == 0 or pair_index.shape[0] == 0:
                continue

            class_ids = getattr(graph, "class_ids", None)
            y = getattr(graph, "y", None)
            y_mask = getattr(graph, "y_mask", None)

            if class_ids is None or y is None:
                continue

            # Standardize pair_index to [P, 2]
            if pair_index.dim() == 2 and pair_index.shape[0] == 2 and pair_index.shape[1] != 2:
                pair_index = pair_index.t()

            obj_nodes = pair_index[:, 1]
            obj_classes = class_ids[obj_nodes]

            # Map negative or out-of-range classes to class index 80 (unknown)
            clamped_classes = torch.where(
                (obj_classes < 0) | (obj_classes >= 80),
                torch.full_like(obj_classes, 80),
                obj_classes,
            )

            for p_idx in range(pair_index.shape[0]):
                c = int(clamped_classes[p_idx].item())
                if y_mask is not None:
                    valid_acts = (y[p_idx] > 0.5) & (y_mask[p_idx] > 0.5)
                else:
                    valid_acts = y[p_idx] > 0.5

                new_counts[c, valid_acts] += 1.0

        self.cooccurrence.copy_(new_counts)
        self._update_logits(self.alpha)
        return self

    def forward(
        self,
        data: Optional[Any] = None,
        class_ids: Optional[torch.Tensor] = None,
        pair_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Look up prior action logits for all candidate pairs.

        Args:
            data: PyG Data or HOIData containing 'class_ids' and 'pair_index'.
            class_ids: Optional direct [V] class ID tensor.
            pair_index: Optional direct [P, 2] candidate pair index tensor.

        Returns:
            torch.Tensor: [P, 26] action logits.
        """
        if data is not None:
            if class_ids is None:
                class_ids = getattr(data, "class_ids", None)
            if pair_index is None:
                pair_index = getattr(data, "pair_index", None)

        if pair_index is None or pair_index.numel() == 0 or pair_index.shape[0] == 0:
            dev = self.prior_logits.device if class_ids is None else class_ids.device
            return torch.zeros((0, self.num_actions), dtype=torch.float32, device=dev)

        if class_ids is None:
            raise ValueError("class_ids must be provided either via 'data' or directly.")

        # Standardize pair_index to [P, 2]
        if pair_index.dim() == 2 and pair_index.shape[0] == 2 and pair_index.shape[1] != 2:
            pair_index = pair_index.t()

        obj_indices = pair_index[:, 1]
        obj_classes = class_ids[obj_indices]

        # Map negative (-1) or out-of-range COCO ids to index 80
        clamped_classes = torch.where(
            (obj_classes < 0) | (obj_classes >= 80),
            torch.full_like(obj_classes, 80),
            obj_classes,
        )

        # Lookup from registered buffer on correct device
        return self.prior_logits[clamped_classes]
