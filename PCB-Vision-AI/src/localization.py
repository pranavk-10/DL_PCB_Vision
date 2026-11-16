"""
PCB Vision AI - Sliding Window Visual Defect Localization & Inspection
Implements:
1. Coarse sliding-window scanning across full PCB test images.
2. Batched patch extraction and parallel CNN inference for computational efficiency.
3. Class probability filtering against threshold and non-normal classes.
4. Non-Maximum Suppression (NMS) for overlapping detection consolidation.
5. Overlay renderer for bounding boxes, class names, and confidence scores.
6. Quantitative localization evaluation against ground truth using IoU (Intersection over Union).
"""

import logging
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from config.config import LOCALIZATION_CONFIG, DATASET_CONFIG, LocalizationConfig, DatasetConfig
from src.dataset_loader import BoundingBox

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = float(boxA_area + boxB_area - inter_area)
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def apply_nms(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.30
) -> List[Dict[str, Any]]:
    """
    Performs Non-Maximum Suppression (NMS) to eliminate redundant overlapping boxes.
    Expects list of dicts with keys: 'box' (x1, y1, x2, y2), 'confidence', 'class_idx', 'class_name'.
    """
    if not detections:
        return []

    # Sort descending by confidence score
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept_dets: List[Dict[str, Any]] = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        kept_dets.append(best)
        remaining = []

        for candidate in sorted_dets:
            # Calculate IoU
            iou = compute_iou(best["box"], candidate["box"])
            # Keep if low overlap or different defect class
            if iou < iou_threshold:
                remaining.append(candidate)

        sorted_dets = remaining

    return kept_dets


class SlidingWindowInspector:
    """Performs visual defect localization using a trained patch-classification CNN."""

    def __init__(
        self,
        model: Any,
        config: LocalizationConfig = LOCALIZATION_CONFIG,
        class_labels: List[str] = DATASET_CONFIG.class_labels
    ):
        self.model = model
        self.config = config
        self.class_labels = class_labels

    def inspect_image(
        self,
        image_input: Any,
        stride: Optional[int] = None,
        conf_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes sliding-window inspection on a full PCB image:
        1. Tiles image into 128x128 patches
        2. Batches patches through CNN
        3. Filters predictions above confidence threshold and != Normal (index 6)
        4. Applies Non-Maximum Suppression (NMS)
        5. Renders annotated result with bounding boxes and defect labels
        """
        # Load image if file path or numpy array
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not load image from {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            raise TypeError("image_input must be a file path or numpy ndarray")

        h, w = img.shape[:2]
        win_w, win_h = self.config.window_size
        step = stride or self.config.stride
        threshold = conf_threshold or self.config.confidence_threshold
        normal_idx = self.config.normal_class_idx

        # Convert to single-channel grayscale normalized to [0, 1]
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Generate window coordinates
        windows: List[Tuple[int, int, int, int]] = []
        patches: List[np.ndarray] = []

        for y in range(0, h - win_h + 1, step):
            for x in range(0, w - win_w + 1, step):
                crop = gray[y:y + win_h, x:x + win_w]
                crop_norm = (crop.astype(np.float32) / 255.0)[..., np.newaxis]
                windows.append((x, y, x + win_w, y + win_h))
                patches.append(crop_norm)

        # Edge handling for remaining border pixels
        if (h % step) != 0 and (h > win_h):
            y = h - win_h
            for x in range(0, w - win_w + 1, step):
                crop = gray[y:y + win_h, x:x + win_w]
                windows.append((x, y, x + win_w, y + win_h))
                patches.append((crop.astype(np.float32) / 255.0)[..., np.newaxis])

        raw_detections: List[Dict[str, Any]] = []

        if patches:
            batch_x = np.array(patches, dtype=np.float32)
            # Batch model inference
            probs = self.model.predict(batch_x, batch_size=64, verbose=0)
            
            for box, prob in zip(windows, probs):
                top_idx = int(np.argmax(prob))
                conf = float(prob[top_idx])

                # Discard if predicted class is Normal (6) or below confidence threshold
                if top_idx != normal_idx and conf >= threshold:
                    raw_detections.append({
                        "box": box,
                        "confidence": conf,
                        "class_idx": top_idx,
                        "class_name": self.class_labels[top_idx],
                        "probabilities": prob.tolist()
                    })

        # Apply Non-Maximum Suppression
        final_detections = apply_nms(raw_detections, iou_threshold=self.config.iou_nms_threshold)

        # Render annotated overlay
        annotated_image = self.render_overlay(img, final_detections)

        return {
            "total_windows_scanned": len(patches),
            "raw_candidates_found": len(raw_detections),
            "detections": final_detections,
            "annotated_image": annotated_image,
            "original_image": img
        }

    def render_overlay(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Draws bounding boxes and labels with semi-transparent banners."""
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # Palette
        palette = [
            (50, 50, 220),   # Open (Red)
            (180, 50, 180),  # Short (Purple)
            (30, 180, 220),  # Mousebite (Orange)
            (180, 180, 30),  # Spur (Cyan)
            (220, 120, 30),  # Pinhole (Blue)
            (100, 100, 200), # Spurious copper
            (50, 200, 50),   # Normal
        ]

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_idx = det["class_idx"]
            cls_name = det["class_name"]
            conf = det["confidence"]

            color = palette[cls_idx % len(palette)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name}: {conf * 100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(y1 - 6, th + 4)
            cv2.rectangle(vis, (x1, ty - th - 3), (x1 + tw + 4, ty + 2), color, -1)
            cv2.putText(vis, label, (x1 + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return vis

    def evaluate_localization(
        self,
        predicted_dets: List[Dict[str, Any]],
        ground_truth_boxes: List[BoundingBox],
        iou_match_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Quantitatively evaluates sliding-window detections against ground-truth defect boxes:
        Calculates True Positives, False Positives, False Negatives, Mean IoU,
        Localization Precision, and Localization Recall.
        """
        gt_matched = [False] * len(ground_truth_boxes)
        pred_matched = [False] * len(predicted_dets)
        ious = []

        for p_idx, p_det in enumerate(predicted_dets):
            p_box = p_det["box"]
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, g_box in enumerate(ground_truth_boxes):
                gt_coords = (g_box.x1, g_box.y1, g_box.x2, g_box.y2)
                iou = compute_iou(p_box, gt_coords)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_match_threshold and best_gt_idx != -1 and not gt_matched[best_gt_idx]:
                gt_matched[best_gt_idx] = True
                pred_matched[p_idx] = True
                ious.append(best_iou)

        tp = sum(pred_matched)
        fp = len(predicted_dets) - tp
        fn = len(ground_truth_boxes) - sum(gt_matched)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        mean_iou = float(np.mean(ious)) if ious else 0.0

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "localization_precision": precision,
            "localization_recall": recall,
            "mean_matched_iou": mean_iou,
            "iou_threshold_used": iou_match_threshold
        }
