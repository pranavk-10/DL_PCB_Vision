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


def test_classifier_requires_trained_model_by_default(tmp_path):
    """Prevents untrained heuristics from being presented as trained CNN predictions."""
    classifier = PCBDefectClassifier(model_path=tmp_path / "missing.keras")
    dummy_patch = np.random.randint(0, 255, (128, 128), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="No compatible trained model"):
        classifier.predict_patch(dummy_patch)


def test_explicit_heuristic_fallback_output_structure():
    """Validates the explicitly opt-in heuristic fallback contract."""
    classifier = PCBDefectClassifier(allow_heuristic_fallback=True)
    dummy_patch = np.random.randint(0, 255, (128, 128), dtype=np.uint8)

    res = classifier.predict_patch(dummy_patch)

    assert res["predicted_class"] in DATASET_CONFIG.class_labels
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 7
    prob_sum = sum(res["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-4


def test_batch_prediction_explicit_fallback():
    """Validates explicit fallback batch prediction without implying trained inference."""
    classifier = PCBDefectClassifier(allow_heuristic_fallback=True)
    patches = [np.full((128, 128), val, dtype=np.uint8) for val in [50, 150, 240]]

    batch_res = classifier.predict_batch(patches)
    assert len(batch_res) == 3
    for r in batch_res:
        assert r["predicted_class"] in DATASET_CONFIG.class_labels
