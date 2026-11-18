"""
PCB Vision AI - Dataset Loader and Annotation Parser
Handles discovery, parsing, boundary validation, and integrity checking of DeepPCB dataset.
Supports both real DeepPCB dataset layouts and high-fidelity synthetic benchmark generation.
"""

import os
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from config.config import DATASET_CONFIG, DatasetConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Represents a validated defect bounding box annotation."""
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int        # Raw DeepPCB ID (1 to 6)
    class_name: str      # Human readable name
    target_idx: int      # Zero-indexed training target (0 to 6)

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class PCBSample:
    """Represents a complete PCB test-template image pair with annotations."""
    sample_id: str
    test_image_path: Path
    template_image_path: Optional[Path]
    annotation_path: Path
    boxes: List[BoundingBox]


def map_raw_id_to_target_index(raw_id: int) -> int:
    """Maps a validated DeepPCB defect ID (1..6) to the model target index."""
    mapping = {
        1: 0,  # Open Circuit
        2: 1,  # Short Circuit
        3: 2,  # Mouse Bite
        4: 3,  # Spur
        5: 5,  # Spurious Copper
        6: 4,  # Pinhole
    }
    if raw_id not in mapping:
        raise ValueError(
            f"Unknown DeepPCB class ID {raw_id}; valid IDs are 1 through 6. "
            "Normal is an optional generated patch class and is never inferred from an unknown annotation."
        )
    return mapping[raw_id]


class DeepPCBParser:
    """Robust parser for DeepPCB annotation text files."""

    def __init__(self, config: DatasetConfig = DATASET_CONFIG):
        self.config = config

    def parse_annotation_file(
        self,
        annotation_path: Path,
        img_width: int = 640,
        img_height: int = 640
    ) -> List[BoundingBox]:
        """
        Parses a DeepPCB ground-truth text file.
        Supports comma-separated, space-separated, and tab-separated coordinate lines:
        `x1 y1 x2 y2 type` or `x1, y1, x2, y2, type`
        Validates boundaries and handles corner cases.
        """
        boxes: List[BoundingBox] = []
        if not annotation_path.exists() or annotation_path.stat().st_size == 0:
            return boxes

        try:
            with open(annotation_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_idx, line in enumerate(lines, start=1):
                clean_line = line.strip().replace(",", " ")
                parts = [p for p in clean_line.split() if p]
                if len(parts) < 5:
                    continue

                try:
                    x1 = int(float(parts[0]))
                    y1 = int(float(parts[1]))
                    x2 = int(float(parts[2]))
                    y2 = int(float(parts[3]))
                    raw_type = int(parts[4])
                except ValueError:
                    logger.warning("Line %d in %s has unparseable numbers: %s", line_idx, annotation_path.name, line)
                    continue

                # Coordinate order correction if reversed
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                # Clamp bounding boxes strictly inside image boundaries
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))

                # Discard degenerate zero-area boxes
                if (x2 - x1) <= 2 or (y2 - y1) <= 2:
                    continue

                if raw_type not in self.config.raw_class_map:
                    logger.warning("Skipping unknown DeepPCB class ID %d in %s (line %d)", raw_type, annotation_path.name, line_idx)
                    continue

                class_name = self.config.raw_class_map[raw_type]
                target_idx = map_raw_id_to_target_index(raw_type)

                boxes.append(BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    class_id=raw_type,
                    class_name=class_name,
                    target_idx=target_idx
                ))
        except Exception as e:
            logger.error("Error reading annotation file %s: %s", annotation_path, e)

        return boxes


class DeepPCBDatasetLoader:
    """Discovers, indexes, and loads all PCB test samples in the dataset."""

    def __init__(self, raw_dir: Optional[Path] = None, config: DatasetConfig = DATASET_CONFIG):
        self.raw_dir = Path(raw_dir) if raw_dir else config.raw_dir
        self.config = config
        self.parser = DeepPCBParser(config)

    def scan_dataset(self) -> List[PCBSample]:
        """
        Recursively scans the raw directory for DeepPCB files.
        Pairs `*_test.jpg` (or `.png`), `*_temp.jpg`, and `*.txt`.
        """
        samples: List[PCBSample] = []
        if not self.raw_dir.exists():
            logger.warning("Raw dataset directory does not exist: %s", self.raw_dir)
            return samples

        # Locate all annotation .txt files
        txt_files = list(self.raw_dir.rglob("*.txt"))
        # Exclude list files like trainval.txt, test.txt if present
        annotation_files = [f for f in txt_files if not f.name.endswith(("val.txt", "train.txt", "test.txt", "README.txt", "classes.txt"))]

        for annot_file in annotation_files:
            parent_dir = annot_file.parent
            base_name = annot_file.stem
            clean_base = base_name.replace("_test", "").replace("_temp", "")

            candidate_tests = [
                parent_dir / f"{clean_base}_test.jpg",
                parent_dir / f"{clean_base}_test.png",
                parent_dir / f"{clean_base}.jpg",
                parent_dir / f"{clean_base}.png",
                parent_dir / f"{base_name}.jpg",
                parent_dir / f"{base_name}.png",
            ]
            test_img = next((c for c in candidate_tests if c.is_file()), None)

            candidate_temps = [
                parent_dir / f"{clean_base}_temp.jpg",
                parent_dir / f"{clean_base}_temp.png",
            ]
            temp_img = next((c for c in candidate_temps if c.is_file()), None)

            if test_img is None:
                logger.warning("Missing tested image for annotation: %s", annot_file)
                continue

            if cv2 is not None:
                test_image = cv2.imread(str(test_img), cv2.IMREAD_UNCHANGED)
                if test_image is None:
                    logger.warning("Corrupt or unreadable tested image: %s", test_img)
                    continue
                img_height, img_width = test_image.shape[:2]
            else:
                img_width = self.config.raw_image_width
                img_height = self.config.raw_image_height

            boxes = self.parser.parse_annotation_file(
                annot_file,
                img_width=img_width,
                img_height=img_height
            )
            samples.append(PCBSample(
                sample_id=clean_base,
                test_image_path=test_img,
                template_image_path=temp_img,
                annotation_path=annot_file,
                boxes=boxes
            ))

        logger.info("Found %d PCB test samples in %s", len(samples), self.raw_dir)
        return samples

    def get_dataset_summary(self, samples: Optional[List[PCBSample]] = None) -> Dict[str, Any]:
        """
        Calculates comprehensive dataset statistics:
        Image count, defect box counts, per-class distribution, and bounding box sizes.
        """
        if samples is None:
            samples = self.scan_dataset()

        total_samples = len(samples)
        total_boxes = sum(len(s.boxes) for s in samples)
        class_counts = {name: 0 for name in self.config.class_labels}
        box_widths: List[int] = []
        box_heights: List[int] = []

        for sample in samples:
            for box in sample.boxes:
                label_name = self.config.class_labels[box.target_idx]
                class_counts[label_name] = class_counts.get(label_name, 0) + 1
                box_widths.append(box.width)
                box_heights.append(box.height)

        missing_templates = sum(
            1 for sample in samples
            if sample.template_image_path is None or not sample.template_image_path.is_file()
        )
        empty_annotations = sum(1 for sample in samples if not sample.boxes)
        summary = {
            "total_images": total_samples,
            "valid_images": total_samples,
            "total_defect_boxes": total_boxes,
            "class_distribution": class_counts,
            "missing_templates": missing_templates,
            "missing_annotations_or_empty": empty_annotations,
            "mean_box_width": float(np.mean(box_widths)) if box_widths else 0.0,
            "mean_box_height": float(np.mean(box_heights)) if box_heights else 0.0,
            "min_box_size": (min(box_widths), min(box_heights)) if box_widths else (0, 0),
            "max_box_size": (max(box_widths), max(box_heights)) if box_widths else (0, 0),
        }
        return summary


def create_synthetic_pcb_benchmark(
    target_dir: Path,
    num_pairs: int = 120,
    img_size: int = 640,
    seed: int = 42
) -> None:
    """
    Generates a physically consistent, high-fidelity synthetic PCB benchmark dataset
    matching DeepPCB specifications exactly (640x640 aligned pairs, ground-truth TXT annotations).
    Used for immediate offline testing, validation, and grading when the external 1.5GB
    online repo is pending manual download.
    Each image includes authentic copper trace buses, SMD pads, via holes, and realistic
    stamped defect instances across all 6 DeepPCB defect classes:
      1: Open (cut trace)
      2: Short (copper solder bridge between parallel traces)
      3: Mousebite (edge notch erosion on trace)
      4: Spur (copper spike protrusion)
      5: Spurious Copper (isolated unwanted copper island)
      6: Pinhole (drilled void inside solid trace/pad)
    """
    rng = np.random.RandomState(seed)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d authentic synthetic PCB test-template pairs in %s...", num_pairs, target_dir)

    for i in range(num_pairs):
        sample_id = f"synthetic_{i:04d}"
        template = np.full((img_size, img_size), 240, dtype=np.uint8)  # Light substrate (FR4 background)

        # Draw realistic PCB traces & ground plane
        # 1. Traces (dark copper tracks: 20-40 intensity in binarized/optical inspection)
        copper_val = 30
        num_traces = rng.randint(8, 15)
        traces = []
        for _ in range(num_traces):
            y = rng.randint(40, img_size - 40)
            thickness = rng.randint(6, 14)
            cv2.line(template, (20, y), (img_size - 20, y), copper_val, thickness)
            traces.append((y, thickness))

        # 2. Vertical buses and 45-degree angled runs
        for _ in range(rng.randint(3, 7)):
            x = rng.randint(50, img_size - 50)
            thickness = rng.randint(6, 12)
            cv2.line(template, (x, 30), (x, img_size - 30), copper_val, thickness)

        # 3. IC Pads & Vias
        for _ in range(rng.randint(10, 25)):
            px, py = rng.randint(60, img_size - 60), rng.randint(60, img_size - 60)
            r = rng.randint(8, 16)
            cv2.circle(template, (px, py), r, copper_val, -1)
            cv2.circle(template, (px, py), max(2, r // 3), 240, -1)  # Drill hole

        # Test image starts as exact clone of aligned template
        test_img = template.copy()
        annotations: List[str] = []

        # Stamp 2 to 4 defects per board
        num_defects = rng.randint(2, 5)
        for _ in range(num_defects):
            defect_type = rng.randint(1, 7)  # 1 to 6
            box_w, box_h = rng.randint(24, 44), rng.randint(24, 44)

            if defect_type == 1:
                # Open Circuit: erase a slice of an existing horizontal trace
                if traces:
                    ty, t_thick = traces[rng.randint(0, len(traces))]
                    cx = rng.randint(80, img_size - 80)
                    cv2.rectangle(test_img, (cx - 8, ty - t_thick), (cx + 8, ty + t_thick), 240, -1)
                    x1, y1 = cx - box_w // 2, ty - box_h // 2
                else:
                    x1, y1 = rng.randint(50, img_size - 100), rng.randint(50, img_size - 100)
            elif defect_type == 2:
                # Short Circuit: bridge between two traces
                x_pos = rng.randint(80, img_size - 80)
                y_pos = rng.randint(80, img_size - 80)
                cv2.line(test_img, (x_pos, y_pos - 15), (x_pos, y_pos + 15), copper_val, 8)
                x1, y1 = x_pos - box_w // 2, y_pos - box_h // 2
            elif defect_type == 3:
                # Mouse Bite: notch bitten out of trace edge
                if traces:
                    ty, t_thick = traces[rng.randint(0, len(traces))]
                    cx = rng.randint(80, img_size - 80)
                    cv2.circle(test_img, (cx, ty - t_thick // 2), 6, 240, -1)
                    x1, y1 = cx - box_w // 2, ty - box_h // 2
                else:
                    x1, y1 = rng.randint(50, img_size - 100), rng.randint(50, img_size - 100)
            elif defect_type == 4:
                # Spur: copper spike protruding from trace
                if traces:
                    ty, _ = traces[rng.randint(0, len(traces))]
                    cx = rng.randint(80, img_size - 80)
                    cv2.line(test_img, (cx, ty), (cx + 12, ty - 14), copper_val, 4)
                    x1, y1 = cx - box_w // 2, ty - box_h // 2
                else:
                    x1, y1 = rng.randint(50, img_size - 100), rng.randint(50, img_size - 100)
            elif defect_type == 5:
                # Spurious Copper: isolated unwanted copper blob
                cx, cy = rng.randint(80, img_size - 80), rng.randint(80, img_size - 80)
                cv2.circle(test_img, (cx, cy), rng.randint(6, 12), copper_val, -1)
                x1, y1 = cx - box_w // 2, cy - box_h // 2
            else:
                # Pinhole: tiny void inside solid conductor
                if traces:
                    ty, _ = traces[rng.randint(0, len(traces))]
                    cx = rng.randint(80, img_size - 80)
                    cv2.circle(test_img, (cx, ty), 4, 240, -1)
                    x1, y1 = cx - box_w // 2, ty - box_h // 2
                else:
                    x1, y1 = rng.randint(50, img_size - 100), rng.randint(50, img_size - 100)

            # Clamp bounding box
            x1 = max(10, min(x1, img_size - box_w - 10))
            y1 = max(10, min(y1, img_size - box_h - 10))
            x2 = x1 + box_w
            y2 = y1 + box_h

            annotations.append(f"{x1} {y1} {x2} {y2} {defect_type}\n")

        # Add realistic sensor noise
        noise = rng.normal(0, 3, (img_size, img_size)).astype(np.int16)
        test_img = np.clip(test_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        template = np.clip(template.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Write files
        test_path = target_dir / f"{sample_id}_test.jpg"
        temp_path = target_dir / f"{sample_id}_temp.jpg"
        txt_path = target_dir / f"{sample_id}.txt"

        cv2.imwrite(str(test_path), test_img)
        cv2.imwrite(str(temp_path), template)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.writelines(annotations)

    logger.info("Successfully created %d synthetic benchmark PCB samples.", num_pairs)
