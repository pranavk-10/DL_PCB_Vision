"""Custom CNN/ANN from scratch, sparse multiclass metrics, and architecture diagrams."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import tensorflow as tf
from config.config import CONFIG, Settings

@tf.keras.utils.register_keras_serializable(package="PCBVision")
class SparseMacroPrecision(tf.keras.metrics.Metric):
    """Macro precision based on argmax classes, rather than seven independent binary labels."""
    def __init__(self, num_classes: int = 7, name: str = "precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion = self.add_weight(name="confusion", shape=(num_classes, num_classes), initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        predicted = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        weights = None if sample_weight is None else tf.cast(tf.reshape(sample_weight, [-1]), self.dtype)
        self.confusion.assign_add(tf.math.confusion_matrix(labels, predicted, num_classes=self.num_classes, weights=weights, dtype=self.dtype))
    def result(self):
        return tf.reduce_mean(tf.math.divide_no_nan(tf.linalg.diag_part(self.confusion), tf.reduce_sum(self.confusion, axis=0)))
    def reset_state(self):
        self.confusion.assign(tf.zeros_like(self.confusion))
    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes}

@tf.keras.utils.register_keras_serializable(package="PCBVision")
class SparseMacroRecall(SparseMacroPrecision):
    def __init__(self, num_classes: int = 7, name: str = "recall", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)
    def result(self):
        return tf.reduce_mean(tf.math.divide_no_nan(tf.linalg.diag_part(self.confusion), tf.reduce_sum(self.confusion, axis=1)))

def set_reproducibility(seed: int = CONFIG.seed) -> None:
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

def augmentation_layers(seed: int = CONFIG.seed):
    """Mild geometry/brightness changes, active only when Keras training=True."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed, name="flip"),
        tf.keras.layers.RandomRotation(0.035, fill_mode="reflect", seed=seed+1, name="rotation"),
        tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="reflect", seed=seed+2, name="translation"),
        tf.keras.layers.RandomZoom((-0.05, 0.05), (-0.05, 0.05), fill_mode="reflect", seed=seed+3, name="zoom"),
        tf.keras.layers.RandomBrightness(0.08, value_range=(0, 1), seed=seed+4, name="brightness"),
    ], name="training_augmentation")

def build_cnn(settings: Settings = CONFIG, augmentation: bool | None = None) -> tf.keras.Model:
    """128x128x1 -> (Conv/ReLU/Pool)x3 -> Flatten -> Dense128 -> Dropout -> Softmax7."""
    settings.validate()
    set_reproducibility(settings.seed)
    inputs = tf.keras.Input((settings.image_size, settings.image_size, settings.channels), name="pcb_patch")
    use_augmentation = settings.augmentation if augmentation is None else augmentation
    features = augmentation_layers(settings.seed)(inputs) if use_augmentation else inputs
    for block, filters in enumerate((32, 64, 128), 1):
        features = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=settings.seed+10+block), name=f"conv_{block}")(features)
        features = tf.keras.layers.MaxPooling2D(2, name=f"pool_{block}")(features)
    features = tf.keras.layers.Flatten(name="flatten")(features)
    features = tf.keras.layers.Dense(128, activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=settings.seed+20), name="dense_features")(features)
    features = tf.keras.layers.Dropout(settings.dropout, seed=settings.seed+21, name="dropout")(features)
    probabilities = tf.keras.layers.Dense(7, activation="softmax", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=settings.seed+22), name="defect_probabilities")(features)
    return tf.keras.Model(inputs, probabilities, name="pcb_custom_cnn")

def build_ann(settings: Settings = CONFIG, augmentation: bool = False) -> tf.keras.Model:
    """A parameter-comparable flattened-image baseline with Dense256/Dense128 layers."""
    settings.validate()
    set_reproducibility(settings.seed)
    inputs = tf.keras.Input((128, 128, 1), name="pcb_patch")
    features = augmentation_layers(settings.seed)(inputs) if augmentation else inputs
    features = tf.keras.layers.Flatten(name="flatten")(features)
    features = tf.keras.layers.Dense(256, activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=settings.seed+11), name="ann_dense_1")(features)
    features = tf.keras.layers.Dense(128, activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=settings.seed+20), name="ann_dense_2")(features)
    features = tf.keras.layers.Dropout(settings.dropout, seed=settings.seed+21, name="dropout")(features)
    outputs = tf.keras.layers.Dense(7, activation="softmax", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=settings.seed+22), name="defect_probabilities")(features)
    return tf.keras.Model(inputs, outputs, name="pcb_ann_baseline")

def compile_model(model: tf.keras.Model, settings: Settings = CONFIG) -> tf.keras.Model:
    if settings.optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=settings.learning_rate)
    elif settings.optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=settings.learning_rate, momentum=0.0)
    else:
        raise ValueError("Optimizer must be adam or sgd.")
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"), SparseMacroPrecision(), SparseMacroRecall()])
    return model

def load_keras_model(path: str | Path, compile: bool = False) -> tf.keras.Model:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}. Run python -m src.train first.")
    return tf.keras.models.load_model(path, compile=compile,
        custom_objects={"SparseMacroPrecision": SparseMacroPrecision, "SparseMacroRecall": SparseMacroRecall})

def architecture_details(model: tf.keras.Model) -> list[dict]:
    return [{"layer": layer.name, "type": type(layer).__name__, "output_shape": [None if dim is None else int(dim) for dim in layer.output.shape],
             "parameters": layer.count_params()} for layer in model.layers]

def plot_architecture(model: tf.keras.Model, path: Path) -> Path:
    """Render the actual model graph as a readable layer stack without Graphviz."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = architecture_details(model)
    figure, axis = plt.subplots(figsize=(9, max(6, len(layers) * 0.65)))
    axis.axis("off")
    for index, layer in enumerate(layers):
        y = len(layers) - index
        text = f"{layer['layer']} | {layer['type']}\n{layer['output_shape']} | {layer['parameters']:,} parameters"
        axis.text(0.5, y, text, ha="center", va="center", fontsize=9, bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eff4f1", "edgecolor": "#b7c8c0"})
        if index:
            axis.annotate("", xy=(0.5, y+0.35), xytext=(0.5, y+0.65), arrowprops={"arrowstyle": "->", "color": "#597369"})
    axis.set_xlim(0, 1)
    axis.set_ylim(0.3, len(layers)+0.7)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
