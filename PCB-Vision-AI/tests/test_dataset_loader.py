"""
Unit tests for DeepPCB dataset loader and annotation parser.
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import cv2

from config.config import DATASET_CONFIG
from src.dataset_loader import (
    DeepPCBParser,
    DeepPCBDatasetLoader,
    BoundingBox,
    PCBSample,
    map_raw_id_to_target_index,
    create_synthetic_pcb_benchmark
)


def test_map_raw_id_to_target_index():
    """Validates mapping of DeepPCB IDs 1..6 to target classes 0..6."""
    assert map_raw_id_to_target_index(1) == 0  # Open Circuit
    assert map_raw_id_to_target_index(2) == 1  # Short Circuit
    assert map_raw_id_to_target_index(3) == 2  # Mouse Bite
    assert map_raw_id_to_target_index(4) == 3  # Spur
    assert map_raw_id_to_target_index(5) == 5  # Spurious Copper
    assert map_raw_id_to_target_index(6) == 4  # Pinhole
    with pytest.raises(ValueError, match="Unknown DeepPCB class ID"):
        map_raw_id_to_target_index(99)


def test_parse_annotation_file_valid_and_comma_formats():
    """Tests annotation parsing with space, comma, and inverted coordinates."""
    parser = DeepPCBParser()
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tf:
        # Line 1: Space-separated
        tf.write("100 120 140 160 1\n")
        # Line 2: Comma-separated
        tf.write("200, 210, 240, 250, 2\n")
        # Line 3: Inverted coordinates (x1 > x2)
        tf.write("350 300 310 260 3\n")
        # Line 4: Degenerate zero-area box (should be ignored)
        tf.write("50 50 50 50 4\n")
        temp_path = Path(tf.name)

    try:
        boxes = parser.parse_annotation_file(temp_path, img_width=640, img_height=640)
        assert len(boxes) == 3, f"Expected 3 valid boxes, got {len(boxes)}"

        # Box 1
        assert boxes[0].x1 == 100 and boxes[0].y1 == 120
        assert boxes[0].x2 == 140 and boxes[0].y2 == 160
        assert boxes[0].class_id == 1
        assert boxes[0].target_idx == 0

        # Box 2
        assert boxes[1].x1 == 200 and boxes[1].y1 == 210
        assert boxes[1].class_id == 2
        assert boxes[1].target_idx == 1

        # Box 3 (corrected inversion)
        assert boxes[2].x1 == 310 and boxes[2].x2 == 350
        assert boxes[2].y1 == 260 and boxes[2].y2 == 300
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_boundary_clamping():
    """Tests that out-of-bounds coordinates are clamped to image boundaries."""
    parser = DeepPCBParser()
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tf:
        tf.write("-20 -30 700 800 1\n")
        temp_path = Path(tf.name)

    try:
        boxes = parser.parse_annotation_file(temp_path, img_width=640, img_height=640)
        assert len(boxes) == 1
        assert boxes[0].x1 == 0
        assert boxes[0].y1 == 0
        assert boxes[0].x2 == 640
        assert boxes[0].y2 == 640
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_synthetic_benchmark_creation_and_loading():
    """Tests synthetic dataset generation and discovery."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        create_synthetic_pcb_benchmark(tmp_path, num_pairs=5, seed=123)

        loader = DeepPCBDatasetLoader(raw_dir=tmp_path)
        samples = loader.scan_dataset()

        assert len(samples) == 5, f"Expected 5 samples, found {len(samples)}"
        summary = loader.get_dataset_summary(samples)
        assert summary["total_images"] == 5
        assert summary["total_defect_boxes"] > 0


def test_unknown_annotation_ids_are_not_treated_as_normal():
    parser = DeepPCBParser()
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tf:
        tf.write("10 10 30 30 99\n")
        temp_path = Path(tf.name)

    try:
        boxes = parser.parse_annotation_file(temp_path, img_width=640, img_height=640)
        assert boxes == []
    finally:
        temp_path.unlink()
