"""
Unit tests for CNN and ANN model architectures.
"""

import pytest
import numpy as np

from src.model import build_custom_cnn, build_ann_baseline, TF_AVAILABLE

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")


def test_custom_cnn_architecture():
    """Validates layer shapes, output dimensions, and parameter existence of the CNN."""
    model = build_custom_cnn(input_shape=(128, 128, 1), num_classes=7)

    # Test output shape
    assert model.output_shape == (None, 7)
    assert model.input_shape == (None, 128, 128, 1)

    # Test feed-forward with dummy batch
    dummy_input = np.random.uniform(0.0, 1.0, size=(2, 128, 128, 1)).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)

    assert preds.shape == (2, 7)
    # Check probabilities sum approximately to 1.0 (Softmax guarantee)
    np.testing.assert_allclose(np.sum(preds, axis=1), [1.0, 1.0], atol=1e-5)
    # Check no NaNs
    assert not np.isnan(preds).any()


def test_ann_baseline_architecture():
    """Validates output shape and functionality of the ANN Baseline MLP."""
    ann = build_ann_baseline(input_shape=(128, 128, 1), num_classes=7)

    assert ann.output_shape == (None, 7)
    dummy_input = np.random.uniform(0.0, 1.0, size=(2, 128, 128, 1)).astype(np.float32)
    preds = ann.predict(dummy_input, verbose=0)

    assert preds.shape == (2, 7)
    np.testing.assert_allclose(np.sum(preds, axis=1), [1.0, 1.0], atol=1e-5)
    assert not np.isnan(preds).any()
