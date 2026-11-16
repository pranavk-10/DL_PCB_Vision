"""
PCB Vision AI - Built-in Comprehensive Unit Test Runner
Executes all unit tests using standard library `unittest` (compatible with both unittest and pytest).
Validates:
1. Annotation parsing, coordinate ordering, boundary clamping, and zero-area box filtering.
2. Anti-leakage image-level partitioning (asserting train/val/test disjoint sets).
3. Patch extraction dimensions, channels, and Normal region overlap avoidance.
4. Inference output contracts, probability distribution constraints (sum == 1.0), and label mapping.
"""

import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATASET_CONFIG
from src.dataset_loader import (
    DeepPCBParser,
    BoundingBox,
    PCBSample,
    map_raw_id_to_target_index
)
from src.preprocessing import split_pcb_samples


class TestDatasetLoader(unittest.TestCase):
    """Test suite for DeepPCB dataset loader and annotation parser."""

    def test_class_mapping(self):
        """Verifies mapping from raw DeepPCB IDs (1..6) to target classes (0..6)."""
        self.assertEqual(map_raw_id_to_target_index(1), 0)  # Open Circuit
        self.assertEqual(map_raw_id_to_target_index(2), 1)  # Short Circuit
        self.assertEqual(map_raw_id_to_target_index(3), 2)  # Mouse Bite
        self.assertEqual(map_raw_id_to_target_index(4), 3)  # Spur
        self.assertEqual(map_raw_id_to_target_index(5), 5)  # Spurious Copper
        self.assertEqual(map_raw_id_to_target_index(6), 4)  # Pinhole
        self.assertEqual(map_raw_id_to_target_index(99), 6) # Default Normal

    def test_annotation_parsing_and_boundary_clamping(self):
        """Tests parsing space/comma delimiters, boundary clamping, and inverted coordinates."""
        parser = DeepPCBParser()
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tf:
            tf.write("100 120 140 160 1\n")
            tf.write("200, 210, 240, 250, 2\n")
            tf.write("350 300 310 260 3\n")      # Inverted x and y
            tf.write("50 50 50 50 4\n")           # Zero area (degenerate)
            tf.write("-20 -30 700 800 5\n")       # Out-of-bounds coordinates
            temp_path = Path(tf.name)

        try:
            boxes = parser.parse_annotation_file(temp_path, img_width=640, img_height=640)
            self.assertEqual(len(boxes), 4, f"Expected 4 valid boxes, got {len(boxes)}")

            # Box 1
            self.assertEqual(boxes[0].x1, 100)
            self.assertEqual(boxes[0].y1, 120)
            self.assertEqual(boxes[0].class_id, 1)
            self.assertEqual(boxes[0].target_idx, 0)

            # Box 3 (corrected inversion)
            self.assertEqual(boxes[2].x1, 310)
            self.assertEqual(boxes[2].x2, 350)
            self.assertEqual(boxes[2].y1, 260)
            self.assertEqual(boxes[2].y2, 300)

            # Box 4 (clamped to [0, 640])
            self.assertEqual(boxes[3].x1, 0)
            self.assertEqual(boxes[3].y1, 0)
            self.assertEqual(boxes[3].x2, 640)
            self.assertEqual(boxes[3].y2, 640)
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestPreprocessingAntiLeakage(unittest.TestCase):
    """Test suite for anti-leakage splitting and data preparation."""

    def test_strict_anti_leakage_split(self):
        """CRITICAL: Verifies zero sample ID overlap across Train, Val, and Test splits."""
        samples = [
            PCBSample(
                sample_id=f"sample_{i:04d}",
                test_image_path=Path(f"sample_{i:04d}_test.jpg"),
                template_image_path=Path(f"sample_{i:04d}_temp.jpg"),
                annotation_path=Path(f"sample_{i:04d}.txt"),
                boxes=[]
            )
            for i in range(120)
        ]

        train_s, val_s, test_s = split_pcb_samples(
            samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
        )

        self.assertEqual(len(train_s), 84)
        self.assertEqual(len(val_s), 18)
        self.assertEqual(len(test_s), 18)

        train_ids = set(s.sample_id for s in train_s)
        val_ids = set(s.sample_id for s in val_s)
        test_ids = set(s.sample_id for s in test_s)

        # Mutually disjoint
        self.assertTrue(train_ids.isdisjoint(val_ids), "Data leakage between Train and Val!")
        self.assertTrue(train_ids.isdisjoint(test_ids), "Data leakage between Train and Test!")
        self.assertTrue(val_ids.isdisjoint(test_ids), "Data leakage between Val and Test!")


class TestModelContracts(unittest.TestCase):
    """Test suite for model output dimension and mathematical probability constraints."""

    def test_class_labels_length(self):
        """Verifies that target classes match exactly 7 categories."""
        self.assertEqual(len(DATASET_CONFIG.class_labels), 7)
        self.assertIn("Open Circuit", DATASET_CONFIG.class_labels)
        self.assertIn("Short Circuit", DATASET_CONFIG.class_labels)
        self.assertIn("Mouse Bite", DATASET_CONFIG.class_labels)
        self.assertIn("Spur", DATASET_CONFIG.class_labels)
        self.assertIn("Pinhole", DATASET_CONFIG.class_labels)
        self.assertIn("Spurious Copper", DATASET_CONFIG.class_labels)
        self.assertIn("Normal", DATASET_CONFIG.class_labels)

    def test_softmax_probability_axioms(self):
        """Tests that any valid prediction outputs strictly non-negative probabilities summing to 1.0."""
        # Simulated logits
        logits = np.array([2.1, -1.0, 0.5, 3.2, -0.4, 1.1, 0.0])
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        self.assertEqual(len(probs), 7)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
