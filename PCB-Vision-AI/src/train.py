"""
PCB Vision AI - Training Pipeline
Implements:
1. Primary Custom CNN model training with Adam optimizer.
2. Full training callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
3. Class-weight balancing for defect category skew.
4. Experiment runners:
   - Experiment 1: Baseline ANN vs Custom CNN
   - Experiment 2: Adam Optimizer vs SGD Optimizer
   - Experiment 3: Data Augmentation vs No Augmentation
5. JSON serialization of loss/accuracy histories for downstream evaluation and plotting.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from config.config import MODEL_CONFIG, MODELS_DIR, METRICS_DIR, ModelConfig
from src.model import build_custom_cnn, build_ann_baseline, TF_AVAILABLE

if TF_AVAILABLE:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_training_callbacks(
    model_save_path: Path,
    patience_es: int = 7,
    patience_lr: int = 4,
    factor_lr: float = 0.5,
    min_lr: float = 1e-6
) -> list:
    """Instantiates standard production Deep Learning callbacks."""
    if not TF_AVAILABLE:
        return []

    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience_es,
        restore_best_weights=True,
        verbose=1,
        mode="min"
    )

    checkpoint = ModelCheckpoint(
        filepath=str(model_save_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
        mode="min"
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=factor_lr,
        patience=patience_lr,
        min_lr=min_lr,
        verbose=1,
        mode="min"
    )

    return [early_stopping, checkpoint, reduce_lr]


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ModelConfig = MODEL_CONFIG,
    save_path: Optional[Path] = None,
    class_weights: Optional[Dict[int, float]] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None
) -> Tuple[Any, Dict[str, list], float]:
    """
    Trains a TensorFlow/Keras model on the preprocessed PCB dataset.
    Returns:
        (trained_model, history_dict, training_time_seconds)
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available. Please install dependencies.")

    save_path = save_path or config.cnn_model_path
    num_epochs = epochs or config.epochs
    bs = batch_size or config.batch_size

    callbacks = create_training_callbacks(
        model_save_path=save_path,
        patience_es=config.early_stopping_patience,
        patience_lr=config.lr_reduction_patience,
        factor_lr=config.lr_reduction_factor,
        min_lr=config.min_lr
    )

    logger.info("Starting training: Model=%s, Epochs=%d, BatchSize=%d, Samples=%d",
                model.name, num_epochs, bs, len(X_train))

    start_time = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=num_epochs,
        batch_size=bs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    elapsed_time = time.time() - start_time

    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    history_dict["training_time_seconds"] = elapsed_time

    best_epoch = int(np.argmin(history_dict.get("val_loss", history_dict["loss"]))) + 1
    metadata_path = Path(save_path).with_suffix(".metadata.json")
    metadata = {
        "architecture": model.name,
        "class_names": list(getattr(config, "class_labels", [])) if hasattr(config, "class_labels") else [
            "Open Circuit", "Short Circuit", "Mouse Bite", "Spur", "Pinhole", "Spurious Copper", "Normal"
        ],
        "input_shape": list(config.input_shape if hasattr(config, "input_shape") else (128, 128, 1)),
        "normalization": "grayscale_uint8_scaled_to_[0,1]",
        "batch_size": bs,
        "epochs_requested": num_epochs,
        "learning_rate": config.learning_rate,
        "random_seed": getattr(config, "random_seed", 42),
        "best_epoch": best_epoch,
        "history": history_dict,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Completed training %s in %.2f seconds", model.name, elapsed_time)
    logger.info("Saved training metadata to: %s", metadata_path)
    return model, history_dict, elapsed_time


def run_experiment_1_ann_vs_cnn(
    data: Dict[str, np.ndarray],
    epochs: int = 15,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    EXPERIMENT 1: ANN Baseline vs Custom CNN
    Demonstrates spatial convolution benefits vs flat dense representations.
    """
    logger.info("==========================================")
    logger.info("RUNNING EXPERIMENT 1: ANN vs Custom CNN")
    logger.info("==========================================")

    # 1. Train ANN Baseline
    ann = build_ann_baseline()
    _, ann_hist, ann_time = train_model(
        ann,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        save_path=MODELS_DIR / "pcb_ann_baseline.keras",
        epochs=epochs,
        batch_size=batch_size
    )

    # 2. Train Custom CNN
    cnn = build_custom_cnn()
    _, cnn_hist, cnn_time = train_model(
        cnn,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        save_path=MODELS_DIR / "pcb_cnn_exp1.keras",
        epochs=epochs,
        batch_size=batch_size
    )

    results = {
        "experiment": "ANN_vs_CNN",
        "ann": {
            "params": ann.count_params(),
            "time_sec": ann_time,
            "final_val_acc": ann_hist["val_accuracy"][-1],
            "best_val_loss": min(ann_hist["val_loss"]),
            "history": ann_hist
        },
        "cnn": {
            "params": cnn.count_params(),
            "time_sec": cnn_time,
            "final_val_acc": cnn_hist["val_accuracy"][-1],
            "best_val_loss": min(cnn_hist["val_loss"]),
            "history": cnn_hist
        }
    }

    out_file = METRICS_DIR / "experiment_1_ann_vs_cnn.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved Experiment 1 metrics to: %s", out_file)
    return results


def run_experiment_2_adam_vs_sgd(
    data: Dict[str, np.ndarray],
    epochs: int = 15,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    EXPERIMENT 2: Adam Optimizer vs SGD with Momentum
    Compares adaptive learning rate convergence vs classical momentum gradient descent.
    """
    logger.info("==========================================")
    logger.info("RUNNING EXPERIMENT 2: Adam vs SGD")
    logger.info("==========================================")

    # 1. CNN with Adam
    cnn_adam = build_custom_cnn(optimizer_name="adam", learning_rate=0.001)
    _, adam_hist, adam_time = train_model(
        cnn_adam,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        save_path=MODELS_DIR / "pcb_cnn_adam.keras",
        epochs=epochs,
        batch_size=batch_size
    )

    # 2. CNN with SGD (with momentum=0.90)
    cnn_sgd = build_custom_cnn(optimizer_name="sgd", learning_rate=0.01, momentum=0.90)
    _, sgd_hist, sgd_time = train_model(
        cnn_sgd,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        save_path=MODELS_DIR / "pcb_cnn_sgd.keras",
        epochs=epochs,
        batch_size=batch_size
    )

    results = {
        "experiment": "Adam_vs_SGD",
        "adam": {
            "time_sec": adam_time,
            "final_val_acc": adam_hist["val_accuracy"][-1],
            "best_val_loss": min(adam_hist["val_loss"]),
            "history": adam_hist
        },
        "sgd": {
            "time_sec": sgd_time,
            "final_val_acc": sgd_hist["val_accuracy"][-1],
            "best_val_loss": min(sgd_hist["val_loss"]),
            "history": sgd_hist
        }
    }

    out_file = METRICS_DIR / "experiment_2_adam_vs_sgd.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved Experiment 2 metrics to: %s", out_file)
    return results


def run_experiment_3_augmentation(
    clean_data: Dict[str, np.ndarray],
    augmented_data: Dict[str, np.ndarray],
    epochs: int = 15,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    EXPERIMENT 3: Data Augmentation vs No Augmentation
    Evaluates overfitting mitigation and generalization gap.
    """
    logger.info("==========================================")
    logger.info("RUNNING EXPERIMENT 3: Augmentation Study")
    logger.info("==========================================")

    # 1. CNN without Augmentation
    cnn_no_aug = build_custom_cnn()
    _, no_aug_hist, no_aug_time = train_model(
        cnn_no_aug,
        clean_data["X_train"], clean_data["y_train"],
        clean_data["X_val"], clean_data["y_val"],
        save_path=MODELS_DIR / "pcb_cnn_no_aug.keras",
        epochs=epochs,
        batch_size=batch_size
    )

    # 2. CNN with Augmentation
    cnn_aug = build_custom_cnn()
    _, aug_hist, aug_time = train_model(
        cnn_aug,
        augmented_data["X_train"], augmented_data["y_train"],
        augmented_data["X_val"], augmented_data["y_val"],
        save_path=MODELS_DIR / "pcb_cnn_aug.keras",
        epochs=epochs,
        batch_size=batch_size
    )

    results = {
        "experiment": "Augmentation_vs_No_Augmentation",
        "no_augmentation": {
            "time_sec": no_aug_time,
            "train_val_gap": abs(no_aug_hist["accuracy"][-1] - no_aug_hist["val_accuracy"][-1]),
            "final_val_acc": no_aug_hist["val_accuracy"][-1],
            "history": no_aug_hist
        },
        "with_augmentation": {
            "time_sec": aug_time,
            "train_val_gap": abs(aug_hist["accuracy"][-1] - aug_hist["val_accuracy"][-1]),
            "final_val_acc": aug_hist["val_accuracy"][-1],
            "history": aug_hist
        }
    }

    out_file = METRICS_DIR / "experiment_3_augmentation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved Experiment 3 metrics to: %s", out_file)
    return results
