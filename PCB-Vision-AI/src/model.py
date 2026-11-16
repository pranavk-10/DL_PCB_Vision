"""
PCB Vision AI - Model Architecture Definitions
Implements:
1. Primary Custom Convolutional Neural Network (CNN) as specified:
   - Input: 128x128x1
   - Block 1: Conv2D(32, 3x3, relu) -> MaxPool2D(2x2)
   - Block 2: Conv2D(64, 3x3, relu) -> MaxPool2D(2x2)
   - Block 3: Conv2D(128, 3x3, relu) -> MaxPool2D(2x2)
   - Head: Flatten -> Dense(128, relu) -> Dropout(0.3) -> Dense(7, softmax)
2. Baseline Artificial Neural Network (ANN / MLP) for comparative analysis:
   - Input: Flatten (16,384) -> Dense(256, relu) -> Dropout(0.3) -> Dense(128, relu) -> Dense(7, softmax)
3. Model inspection, parameter count analysis, and visual summary exporter.
"""

import logging
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from config.config import MODEL_CONFIG, ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Attempt to import TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, regularizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow is not yet installed in the current environment.")


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (128, 128, 1),
    num_classes: int = 7,
    dropout_rate: float = 0.30,
    learning_rate: float = 0.001,
    optimizer_name: str = "adam",
    momentum: float = 0.90
) -> Any:
    """
    Constructs and compiles the primary Custom CNN for PCB defect classification.
    
    Architecture:
      Input: (128, 128, 1)
      Feature Extractor:
        [Conv2D 32 (3x3)] -> ReLU -> [MaxPool2D (2x2)]  => Output: (64, 64, 32)
        [Conv2D 64 (3x3)] -> ReLU -> [MaxPool2D (2x2)]  => Output: (32, 32, 64)
        [Conv2D 128 (3x3)] -> ReLU -> [MaxPool2D (2x2)] => Output: (16, 16, 128)
      Classification Head:
        [Flatten]                                        => Output: (32768)
        [Dense 128] -> ReLU                              => Output: (128)
        [Dropout (rate=0.30)]                            => Output: (128)
        [Dense 7] -> Softmax                             => Output: (7)
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required to build the Keras CNN model.")

    inputs = layers.Input(shape=input_shape, name="pcb_patch_input")

    # Block 1: Low-level edge and copper trace feature extraction
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        name="block1_conv"
    )(inputs)
    x = layers.Activation("relu", name="block1_relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2: Intermediate geometry and defect motif detection (notches, protrusions)
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        name="block2_conv"
    )(x)
    x = layers.Activation("relu", name="block2_relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3: High-level abstract structural defect representation
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        name="block3_conv"
    )(x)
    x = layers.Activation("relu", name="block3_relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool")(x)

    # Classification Head
    x = layers.Flatten(name="flatten_layer")(x)
    x = layers.Dense(128, kernel_initializer="he_normal", name="dense_features")(x)
    x = layers.Activation("relu", name="dense_relu")(x)
    x = layers.Dropout(dropout_rate, name="dropout_regularization")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="defect_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PCB_Vision_CNN")

    # Optimizer selection
    if optimizer_name.lower() == "adam":
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
        ]
    )

    logger.info("Built custom CNN: %d total parameters", model.count_params())
    return model


def build_ann_baseline(
    input_shape: Tuple[int, int, int] = (128, 128, 1),
    num_classes: int = 7,
    dropout_rate: float = 0.30,
    learning_rate: float = 0.001
) -> Any:
    """
    Constructs a baseline Artificial Neural Network (Dense MLP).
    Used for Experiment 1 (ANN vs CNN) to demonstrate the spatial inductive bias
    superiority of convolutions over flat matrix multiplications for computer vision.
    
    Architecture:
      Flatten (128*128*1 = 16,384) -> Dense(256, relu) -> Dropout(0.3) ->
      Dense(128, relu) -> Dropout(0.3) -> Dense(7, softmax)
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required to build the ANN baseline model.")

    inputs = layers.Input(shape=input_shape, name="ann_input")
    x = layers.Flatten(name="flatten_pixels")(inputs)
    x = layers.Dense(256, activation="relu", name="hidden_layer_1")(x)
    x = layers.Dropout(dropout_rate, name="ann_dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="hidden_layer_2")(x)
    x = layers.Dropout(dropout_rate, name="ann_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="ann_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PCB_Vision_ANN_Baseline")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    logger.info("Built ANN Baseline: %d total parameters", model.count_params())
    return model


def get_layer_details(model: Any) -> Dict[str, Any]:
    """
    Extracts detailed structural metrics for academic viva and reports:
    Layer names, output shapes, parameter counts, kernel shapes, and activations.
    """
    details = []
    total_params = 0
    trainable_params = 0

    for layer in model.layers:
        params = layer.count_params()
        total_params += params
        trainable = layer.trainable_weights
        if trainable:
            trainable_params += sum(tf.size(w).numpy() for w in trainable)

        details.append({
            "name": layer.name,
            "class": layer.__class__.__name__,
            "output_shape": layer.output.shape.as_list(),
            "param_count": params,
        })

    return {
        "model_name": model.name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "layers": details
    }
