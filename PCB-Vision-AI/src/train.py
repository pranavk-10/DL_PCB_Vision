"""Reproducible training with class weights, validation-only callbacks, and provenance."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import platform
import re
import time
import numpy as np
import tensorflow as tf
from config.config import CONFIG, Settings
from src.model import architecture_details, build_ann, build_cnn, compile_model, load_keras_model, plot_architecture, set_reproducibility
from src.preprocessing import load_manifest, make_tf_dataset

LOGGER = logging.getLogger(__name__)

def class_weights(labels) -> dict[int, float]:
    counts = Counter(int(label) for label in labels)
    if set(counts) != set(range(7)):
        raise ValueError("Training requires all seven classes; class weights cannot repair missing classes.")
    total = sum(counts.values())
    return {class_id: total / (7 * count) for class_id, count in counts.items()}

def train_model(settings: Settings = CONFIG, architecture: str = "cnn", run_name: str | None = None,
                overwrite: bool = False, evaluate: bool = True) -> dict:
    settings.validate()
    if architecture not in {"cnn", "ann"}:
        raise ValueError("Architecture must be cnn or ann.")
    run_name = run_name or f"{architecture}_{settings.optimizer}_{'aug' if settings.augmentation else 'plain'}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Run name must contain only letters, numbers, hyphens, or underscores.")
    model_dir, metrics_dir = settings.models_dir / run_name, settings.metrics_dir / run_name
    model_path = model_dir / "best.keras"
    if model_path.exists() and not overwrite:
        raise FileExistsError(f"Model already exists: {model_path}. Use a new run name or --overwrite explicitly.")
    frame, preparation = load_manifest(settings)
    train_frame = frame[frame["split"] == "train"]
    validation_frame = frame[frame["split"] == "validation"]
    if validation_frame.empty:
        raise ValueError("Validation split is empty.")
    set_reproducibility(settings.seed)
    tf.keras.backend.clear_session()
    model = build_cnn(settings) if architecture == "cnn" else build_ann(settings, settings.augmentation)
    compile_model(model, settings)
    weights = class_weights(train_frame["class_id"])
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = []
    model.summary(print_fn=lambda line, **kwargs: summary_lines.append(line))
    (metrics_dir / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    plot_architecture(model, settings.figures_dir / run_name / "architecture.png")
    metadata = {"run_name": run_name, "status": "training", "architecture": architecture,
                "settings": settings.to_dict(), "class_names": list(settings.class_names),
                "manifest_sha256": preparation["manifest_sha256"], "dataset_fingerprint": preparation["dataset_fingerprint"],
                "groups_by_split": preparation["groups_by_split"], "class_weights": weights,
                "parameter_count": model.count_params(), "trainable_parameter_count": int(sum(np.prod(variable.shape) for variable in model.trainable_weights)),
                "layers": architecture_details(model), "python_version": platform.python_version(),
                "tensorflow_version": tf.__version__, "sample_counts": {name: int((frame['split'] == name).sum()) for name in ('train', 'validation', 'test')},
                "monitor": "val_loss", "preprocessing": "grayscale, configured interpolation to 128x128, float32 /255; augmentation only training"}
    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=settings.early_stopping_patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=settings.reduce_lr_patience, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(str(metrics_dir / "epoch_log.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    start = time.perf_counter()
    try:
        history = model.fit(make_tf_dataset(train_frame, settings, training=True),
                            validation_data=make_tf_dataset(validation_frame, settings),
                            epochs=settings.epochs, class_weight=weights, callbacks=callbacks, verbose=2)
        elapsed = time.perf_counter() - start
        values = {key: [float(value) for value in data] for key, data in history.history.items()}
        if not model_path.is_file() or not values.get("val_loss") or not all(np.isfinite(values["val_loss"])):
            raise RuntimeError("Training produced no finite validation checkpoint.")
        (metrics_dir / "history.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
        metadata.update({"status": "trained", "training_seconds": elapsed, "epochs_completed": len(values['loss']),
                         "best_epoch": int(np.argmin(values['val_loss'])) + 1,
                         "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                         "history_last_training_accuracy": values['accuracy'][-1],
                         "history_last_validation_accuracy": values['val_accuracy'][-1]})
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        from src.evaluate import evaluate_model, plot_training_history
        plot_training_history(values, settings.figures_dir / run_name)
        if evaluate:
            evaluate_model(model_path, settings, run_name)
        LOGGER.info("Saved validation-selected model: %s (%.1f seconds)", model_path, elapsed)
        return metadata
    except Exception as error:
        if metadata["status"] != "trained":
            metadata.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("cnn", "ann"), default="cnn")
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default=CONFIG.optimizer)
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    learning_rate = args.learning_rate if args.learning_rate is not None else (CONFIG.learning_rate if args.optimizer == "adam" else CONFIG.sgd_learning_rate)
    settings = replace(CONFIG, optimizer=args.optimizer, epochs=args.epochs, batch_size=args.batch_size,
                       learning_rate=learning_rate, augmentation=not args.no_augmentation)
    try:
        train_model(settings, args.architecture, args.run_name, args.overwrite)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Training failed: {error}\n")

if __name__ == "__main__":
    main()
