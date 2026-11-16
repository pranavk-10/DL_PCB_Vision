"""
Unit tests for preprocessing and inference pipeline.
"""

import numpy as np
import pytest

from src.inference import preprocess_input_patch, PCBDefectClassifier
from config.config import DATASET_CONFIG


def test_preprocess_input_patch():
    """Validates preprocessing normalization and dimension expansion."""
    # Test with 2D array
    arr_2d = np.full((100, 100), 200, dtype=np.uint8)
    tensor = preprocess_input_patch(arr_2d, target_size=(128, 128))

    assert tensor.shape == (1, 128, 128, 1)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0


def test_classifier_prediction_output_structure():
    """Validates structure and probability constraints of prediction outputs."""
    classifier = PCBDefectClassifier()
    dummy_patch = np.random.randint(0, 255, (128, 128), dtype=np.uint8)

    res = classifier.predict_patch(dummy_patch)

    assert "predicted_class" in res
    assert "confidence" in res
    assert "class_index" in res
    assert "probabilities" in res

    assert res["predicted_class"] in DATASET_CONFIG.class_labels
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 7

    # Check sum of probabilities equals 1.0
    prob_sum = sum(res["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-4


def test_batch_prediction():
    """Validates batch prediction across multiple patches."""
    classifier = PCBDefectClassifier()
    patches = [np.full((128, 128), val, dtype=np.uint8) for val in [50, 150, 240]]

    batch_res = classifier.predict_batch(patches)
    assert len(batch_res) == 3
    for r in batch_res:
        assert r["predicted_class"] in DATASET_CONFIG.class_labels
