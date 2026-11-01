"""V-COCO annotation loader and ground-truth interaction target generator.

Parses V-COCO annotations (26 action classes) and COCO instances (bounding boxes,
categories) to construct ground-truth human-object interaction targets.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch


# 26 standard V-COCO action classes in canonical order
VCOCO_ACTIONS: Tuple[str, ...] = (
    "hold",
    "stand",
    "sit",
    "ride",
    "walk",
    "look",
    "hit",
    "eat",
    "jump",
    "lay",
    "talk_on_phone",
    "carry",
    "throw",
    "catch",
    "cut",
    "run",
    "work_on_computer",
    "ski",
    "surf",
    "skateboard",
    "smile",
    "drink",
    "kick",
    "point",
    "read",
    "snowboard",
)

NUM_ACTIONS: int = len(VCOCO_ACTIONS)
ACTION_TO_IDX: Dict[str, int] = {act: i for i, act in enumerate(VCOCO_ACTIONS)}
IDX_TO_ACTION: Dict[int, str] = {i: act for i, act in enumerate(VCOCO_ACTIONS)}

# Actions that only have the 'agent' role (no interacting object)
AGENT_ONLY_ACTIONS: Set[str] = {"stand", "walk", "run", "smile"}
AGENT_ONLY_ACTION_INDICES: Set[int] = {
    ACTION_TO_IDX[act] for act in AGENT_ONLY_ACTIONS
}

# Standard 80 COCO category names matching YOLO class index (0..79, person=0)
COCO_CLASSES: Tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)

COCO_NAME_TO_YOLO_ID: Dict[str, int] = {name: i for i, name in enumerate(COCO_CLASSES)}

# COCO original dataset category_id (1..90) to 0-indexed YOLO category id (0..79)
COCO_CAT_ID_TO_YOLO_ID: Dict[int, int] = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18,
    21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27,
    33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36,
    42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45,
    52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54,
    61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63,
    74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72,
    84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}


@dataclass
class VCOCOEntity:
    """A detected or ground-truth entity (person or object) in an image."""
    ann_id: int
    category_id: int          # COCO category id (1 = person)
    class_id: int             # 0-indexed YOLO category id (0 = person)
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in original pixels
    is_person: bool = False

    def __post_init__(self):
        self.is_person = (self.class_id == 0 or self.category_id == 1)


@dataclass
class VCOCOImageRecord:
    """All annotations and ground-truth interactions for a single image."""
    image_id: int
    file_name: str
    image_size: Tuple[int, int]  # (H, W) in original pixels
    entities: List[VCOCOEntity] = field(default_factory=list)
    # Mapping from (person_ann_id, object_ann_id) -> set of action indices (0..25)
    pair_actions: Dict[Tuple[int, int], Set[int]] = field(default_factory=dict)
    # Mapping from person_ann_id -> set of action indices performed by that person
    person_actions: Dict[int, Set[int]] = field(default_factory=dict)

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def boxes(self) -> torch.Tensor:
        if not self.entities:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.tensor([e.bbox for e in self.entities], dtype=torch.float32)

    @property
    def class_ids(self) -> torch.Tensor:
        if not self.entities:
            return torch.zeros((0,), dtype=torch.long)
        return torch.tensor([e.class_id for e in self.entities], dtype=torch.long)

    @property
    def ann_ids(self) -> List[int]:
        return [e.ann_id for e in self.entities]


def compute_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute IoU matrix between boxes1 [M, 4] and boxes2 [N, 4] in xyxy format."""
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0:1].t(), boxes2[:, 1:2].t(), boxes2[:, 2:3].t(), boxes2[:, 3:4].t()

    inter_x1 = torch.maximum(b1_x1, b2_x1)
    inter_y1 = torch.maximum(b1_y1, b2_y1)
    inter_x2 = torch.minimum(b1_x2, b2_x2)
    inter_y2 = torch.minimum(b1_y2, b2_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    area1 = torch.clamp(b1_x2 - b1_x1, min=0.0) * torch.clamp(b1_y2 - b1_y1, min=0.0)
    area2 = torch.clamp(b2_x2 - b2_x1, min=0.0) * torch.clamp(b2_y2 - b2_y1, min=0.0)
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + eps)


class VCOCOLoader:
    """Loads V-COCO dataset splits, COCO annotations, and ground-truth action targets."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        split: str = "train",
        max_images: Optional[int] = None,
        instances_path: Optional[Union[str, Path]] = None,
        vcoco_path: Optional[Union[str, Path]] = None,
        mock_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            data_dir: Path to data directory containing splits/ and vcoco/
            split: 'train', 'val', 'test', or 'trainval'
            max_images: Optional maximum number of images to load from split
            instances_path: Optional custom path to instances_vcoco_all_2014.json
            vcoco_path: Optional custom path to vcoco_{split}.json
            mock_data: Optional mock data dict for isolated unit testing
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_images = max_images

        # Internal index structures
        self.image_ids: List[int] = []
        self.image_records: Dict[int, VCOCOImageRecord] = {}

        if mock_data is not None:
            self._load_from_mock(mock_data)
        else:
            self._load_from_disk(instances_path, vcoco_path)

    def _load_from_mock(self, mock_data: Dict[str, Any]) -> None:
        """Populate records directly from structured mock data."""
        self.image_ids = list(mock_data.get("image_ids", []))
        if self.max_images is not None and self.max_images > 0:
            self.image_ids = self.image_ids[:self.max_images]
        self.image_records = mock_data.get("image_records", {})

    def _load_from_disk(
        self,
        instances_path: Optional[Union[str, Path]],
        vcoco_path: Optional[Union[str, Path]],
    ) -> None:
        """Load split IDs, instances JSON, and V-COCO action JSON from disk."""
        # 1. Load split image IDs
        split_ids_file = self.data_dir / "splits" / f"vcoco_{self.split}.ids"
        if split_ids_file.exists():
            with open(split_ids_file, "r") as f:
                self.image_ids = [int(line.strip()) for line in f if line.strip()]
        else:
            # Fallback: if split ids file not found, will infer from vcoco JSON
            self.image_ids = []

        if self.max_images is not None and self.max_images > 0:
            self.image_ids = self.image_ids[:self.max_images]

        target_image_ids = set(self.image_ids)

        # 2. Load COCO instance annotations
        inst_file = (
            Path(instances_path)
            if instances_path
            else self.data_dir / "instances_vcoco_all_2014.json"
        )

        coco_images: Dict[int, Dict[str, Any]] = {}
        coco_annotations: Dict[int, List[Dict[str, Any]]] = {}

        if inst_file.exists():
            with open(inst_file, "r") as f:
                coco_data = json.load(f)
            for img in coco_data.get("images", []):
                img_id = img["id"]
                if not target_image_ids or img_id in target_image_ids:
                    coco_images[img_id] = img
            for ann in coco_data.get("annotations", []):
                img_id = ann["image_id"]
                if not target_image_ids or img_id in target_image_ids:
                    coco_annotations.setdefault(img_id, []).append(ann)

        # 3. Load V-COCO action annotations
        v_file = (
            Path(vcoco_path)
            if vcoco_path
            else self.data_dir / "vcoco" / f"vcoco_{self.split}.json"
        )

        vcoco_actions_data: List[Dict[str, Any]] = []
        if v_file.exists():
            with open(v_file, "r") as f:
                vcoco_actions_data = json.load(f)
            if not self.image_ids:
                # Infer image_ids from vcoco json if split file was missing
                inferred = set(vcoco_actions_data[0].get("image_id", []))
                self.image_ids = sorted(list(inferred))
                if self.max_images is not None and self.max_images > 0:
                    self.image_ids = self.image_ids[:self.max_images]
                target_image_ids = set(self.image_ids)

        # 4. Build VCOCOImageRecord for each image
        for img_id in self.image_ids:
            img_info = coco_images.get(img_id, {})
            file_name = img_info.get("file_name", f"COCO_train2014_{img_id:012d}.jpg")
            h = img_info.get("height", 0)
            w = img_info.get("width", 0)

            # Build entities from COCO annotations
            entities: List[VCOCOEntity] = []
            for ann in coco_annotations.get(img_id, []):
                ann_id = ann["id"]
                cat_id = ann["category_id"]
                class_id = COCO_CAT_ID_TO_YOLO_ID.get(cat_id, -1)
                bbox_coco = ann["bbox"]  # [x, y, w, h]
                bbox_xyxy = (
                    float(bbox_coco[0]),
                    float(bbox_coco[1]),
                    float(bbox_coco[0] + bbox_coco[2]),
                    float(bbox_coco[1] + bbox_coco[3]),
                )
                entities.append(
                    VCOCOEntity(
                        ann_id=ann_id,
                        category_id=cat_id,
                        class_id=class_id,
                        bbox=bbox_xyxy,
                    )
                )

            self.image_records[img_id] = VCOCOImageRecord(
                image_id=img_id,
                file_name=file_name,
                image_size=(h, w),
                entities=entities,
            )

        # 5. Populate interaction targets from V-COCO actions
        for act in vcoco_actions_data:
            action_name = act.get("action_name", "")
            if action_name not in ACTION_TO_IDX:
                continue
            act_idx = ACTION_TO_IDX[action_name]
            role_names = act.get("role_name", [])
            num_roles = len(role_names)
            ann_ids = act.get("ann_id", [])
            num_anns = len(ann_ids)
            role_objs = act.get("role_object_id", [])
            labels = act.get("label", [])
            img_ids = act.get("image_id", [])

            for i in range(num_anns):
                if labels[i] != 1:
                    continue
                img_id = img_ids[i]
                if img_id not in self.image_records:
                    continue
                person_ann = ann_ids[i]
                record = self.image_records[img_id]

                # Record person action
                record.person_actions.setdefault(person_ann, set()).add(act_idx)

                # If action has object/instrument roles (role index >= 1)
                for r_idx in range(1, num_roles):
                    obj_idx = r_idx * num_anns + i
                    if obj_idx < len(role_objs):
                        obj_ann = role_objs[obj_idx]
                        if obj_ann > 0:
                            record.pair_actions.setdefault(
                                (person_ann, obj_ann), set()
                            ).add(act_idx)

    def __len__(self) -> int:
        return len(self.image_ids)

    def get_image_ids(self) -> List[int]:
        return list(self.image_ids)

    def get_record(self, image_id: int) -> VCOCOImageRecord:
        if image_id not in self.image_records:
            raise KeyError(f"Image ID {image_id} not found in loaded split '{self.split}'")
        return self.image_records[image_id]

    def get_ground_truth_for_pairs(
        self,
        image_id: int,
        boxes: torch.Tensor,
        class_ids: torch.Tensor,
        pair_index: torch.Tensor,
        iou_threshold: float = 0.5,
        ann_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ground-truth action targets (y) and action validity masks (y_mask)
        for candidate pairs [P, 2] in an image.

        Args:
            image_id: Image identifier
            boxes: [N, 4] float32 tensor of entity boxes (xyxy format)
            class_ids: [N] int64 tensor of class IDs (person = 0)
            pair_index: [P, 2] int64 tensor of [human_node_idx, object_node_idx]
            iou_threshold: IoU threshold for matching detected boxes to GT annotations
            ann_ids: Optional [N] list of COCO annotation IDs if oracle GT boxes are used

        Returns:
            y: [P, 26] float32 multi-hot binary indicators
            y_mask: [P, 26] float32 mask (0.0 for agent-only actions e.g. stand/walk/run/smile;
                    1.0 for valid object-role actions)
        """
        num_pairs = pair_index.shape[0]
        y = torch.zeros((num_pairs, NUM_ACTIONS), dtype=torch.float32)
        y_mask = torch.ones((num_pairs, NUM_ACTIONS), dtype=torch.float32)

        # Agent-only actions have no object role; mask them out for human-object pairs
        for a_idx in AGENT_ONLY_ACTION_INDICES:
            y_mask[:, a_idx] = 0.0

        if num_pairs == 0 or image_id not in self.image_records:
            return y, y_mask

        record = self.image_records[image_id]
        if not record.entities:
            return y, y_mask

        gt_boxes = record.boxes
        gt_ann_ids = record.ann_ids
        gt_class_ids = record.class_ids

        # If explicit annotation IDs are provided (oracle mode)
        if ann_ids is not None and len(ann_ids) == boxes.shape[0]:
            node_to_ann: Dict[int, int] = {i: ann_ids[i] for i in range(len(ann_ids))}
        else:
            # Match candidate boxes to GT entities via IoU
            iou_matrix = compute_box_iou(boxes, gt_boxes)
            node_to_ann = {}
            for i in range(boxes.shape[0]):
                if gt_boxes.shape[0] == 0:
                    continue
                best_iou, best_gt_idx = torch.max(iou_matrix[i], dim=0)
                if best_iou.item() >= iou_threshold:
                    # Require class match if class_ids is provided and not dummy -1
                    c_det = int(class_ids[i].item())
                    c_gt = int(gt_class_ids[best_gt_idx].item())
                    if c_det == -1 or c_det == c_gt:
                        node_to_ann[i] = gt_ann_ids[int(best_gt_idx.item())]

        # For each candidate pair, assign labels
        for p_idx in range(num_pairs):
            h_node = int(pair_index[p_idx, 0].item())
            o_node = int(pair_index[p_idx, 1].item())

            h_ann = node_to_ann.get(h_node)
            o_ann = node_to_ann.get(o_node)

            if h_ann is not None and o_ann is not None:
                actions = record.pair_actions.get((h_ann, o_ann), set())
                for a_idx in actions:
                    y[p_idx, a_idx] = 1.0

        return y, y_mask

    @staticmethod
    def create_mock_loader(num_images: int = 3) -> "VCOCOLoader":
        """Construct a lightweight mock VCOCOLoader with synthetic annotations."""
        image_ids = [1000 + i for i in range(num_images)]
        records: Dict[int, VCOCOImageRecord] = {}

        for i, img_id in enumerate(image_ids):
            # Entity 0: Person
            p_ann = img_id * 10 + 1
            # Entity 1: Object (e.g. cup)
            o_ann = img_id * 10 + 2
            # Entity 2: Another object (e.g. chair)
            o2_ann = img_id * 10 + 3

            entities = [
                VCOCOEntity(
                    ann_id=p_ann,
                    category_id=1,
                    class_id=0,
                    bbox=(50.0, 50.0, 200.0, 400.0),
                ),
                VCOCOEntity(
                    ann_id=o_ann,
                    category_id=47,
                    class_id=41,  # cup
                    bbox=(120.0, 150.0, 180.0, 220.0),
                ),
                VCOCOEntity(
                    ann_id=o2_ann,
                    category_id=62,
                    class_id=56,  # chair
                    bbox=(80.0, 250.0, 250.0, 450.0),
                ),
            ]

            record = VCOCOImageRecord(
                image_id=img_id,
                file_name=f"mock_{img_id}.jpg",
                image_size=(480, 640),
                entities=entities,
            )

            # Person holds and drinks from cup
            hold_idx = ACTION_TO_IDX["hold"]
            drink_idx = ACTION_TO_IDX["drink"]
            sit_idx = ACTION_TO_IDX["sit"]
            stand_idx = ACTION_TO_IDX["stand"]

            record.pair_actions[(p_ann, o_ann)] = {hold_idx, drink_idx}
            record.pair_actions[(p_ann, o2_ann)] = {sit_idx}
            record.person_actions[p_ann] = {hold_idx, drink_idx, sit_idx, stand_idx}

            records[img_id] = record

        mock_data = {
            "image_ids": image_ids,
            "image_records": records,
        }
        return VCOCOLoader(mock_data=mock_data)
