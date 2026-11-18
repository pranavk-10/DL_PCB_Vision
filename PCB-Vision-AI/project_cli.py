"""Windows-friendly command-line entry point for PCB Vision AI workflows."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATASET_CONFIG, MODEL_CONFIG, METRICS_DIR
from src.dataset_loader import DeepPCBDatasetLoader, create_synthetic_pcb_benchmark
from src.preprocessing import PCBDataPipeline
from src.model import build_custom_cnn, TF_AVAILABLE


def command_download(args):
    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and any(target.iterdir()):
        print(f"Dataset target already exists and is not empty: {target}")
        return 0
    subprocess.run(["git", "clone", "https://github.com/tangsanli5201/DeepPCB.git", str(target)], check=True)
    print(f"Downloaded DeepPCB repository to {target}")
    return 0


def command_validate(args):
    raw_dir = Path(args.raw_dir)
    loader = DeepPCBDatasetLoader(raw_dir=raw_dir)
    samples = loader.scan_dataset()
    summary = loader.get_dataset_summary(samples)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not samples:
        print("ERROR: No valid DeepPCB sample pairs were found.", file=sys.stderr)
        print("Place the real dataset under data\\raw or pass --raw-dir.", file=sys.stderr)
        return 2
    return 0


def command_synthetic(args):
    create_synthetic_pcb_benchmark(Path(args.target), num_pairs=args.num_pairs, seed=args.seed)
    print("Synthetic data is for testing only and is not a substitute for DeepPCB training.")
    return 0


def command_preprocess(args):
    raw_dir = Path(args.raw_dir)
    loader = DeepPCBDatasetLoader(raw_dir=raw_dir)
    samples = loader.scan_dataset()
    if not samples:
        raise SystemExit("No valid real or synthetic PCB samples found. Run `project_cli.py synthetic` only for testing.")
    pipeline = PCBDataPipeline()
    pipeline.dataset_config.raw_dir = raw_dir
    bundle = pipeline.run_full_pipeline(samples=samples, save_to_disk=True)
    print(json.dumps({
        "train_patches": len(bundle["y_train"]),
        "val_patches": len(bundle["y_val"]),
        "test_patches": len(bundle["y_test"]),
        "train_board_count": len(bundle["split_metadata"]["train_samples"]),
        "val_board_count": len(bundle["split_metadata"]["val_samples"]),
        "test_board_count": len(bundle["split_metadata"]["test_samples"]),
    }, indent=2))
    return 0


def command_train(args):
    if not TF_AVAILABLE:
        raise SystemExit("TensorFlow is unavailable; install requirements.txt before training.")
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit(
            "No TensorFlow-compatible GPU is available. Training was not started. "
            "Use the GPU-only DeepPCB Colab notebook."
        )
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(DATASET_CONFIG.random_seed)
    processed_path = Path(args.processed)
    if not processed_path.exists():
        raise SystemExit("Processed dataset not found. Run `python project_cli.py preprocess` first.")
    from src.preprocessing import PCBDataPipeline
    data = PCBDataPipeline.load_processed_dataset(processed_path)
    if len(data["y_train"]) == 0:
        raise SystemExit("The processed training split is empty.")
    model = build_custom_cnn(
        input_shape=MODEL_CONFIG.input_shape,
        num_classes=MODEL_CONFIG.num_classes,
        dropout_rate=MODEL_CONFIG.dropout_rate,
        learning_rate=args.learning_rate,
    )
    from src.train import train_model
    _, history, elapsed = train_model(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        save_path=Path(args.save),
        class_weights=data.get("class_weights"),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    metadata_path = Path(args.save).with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "training_dataset": "DeepPCB",
        "split_method": "official DeepPCB trainval/test lists; validation held out from trainval",
        "train_board_count": len(data.get("train_sample_ids", [])),
        "validation_board_count": len(data.get("val_sample_ids", [])),
        "test_board_count": len(data.get("test_sample_ids", [])),
        "class_weights": {str(key): value for key, value in data.get("class_weights", {}).items()},
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(args.save), "elapsed_seconds": elapsed, "epochs_trained": len(history["loss"])}, indent=2))
    return 0


def command_evaluate(args):
    if not TF_AVAILABLE:
        raise SystemExit("TensorFlow is unavailable.")
    processed_path = Path(args.processed)
    model_path = Path(args.model)
    if not processed_path.exists():
        raise SystemExit("Processed dataset not found.")
    if not model_path.exists():
        raise SystemExit("Model checkpoint not found.")
    from tensorflow.keras.models import load_model
    from src.preprocessing import PCBDataPipeline
    from src.evaluate import evaluate_model_predictions, plot_confusion_matrix
    data = PCBDataPipeline.load_processed_dataset(processed_path)
    model = load_model(str(model_path))
    probabilities = model.predict(data["X_test"], batch_size=args.batch_size, verbose=0)
    report = evaluate_model_predictions(data["y_test"], probabilities, DATASET_CONFIG.class_labels)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output = METRICS_DIR / "test_evaluation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cm = np.asarray(report["confusion_matrix"])
    plot_confusion_matrix(cm, DATASET_CONFIG.class_labels, normalize=False, save_path=METRICS_DIR.parent / "figures" / "confusion_matrix.png")
    plot_confusion_matrix(cm, DATASET_CONFIG.class_labels, normalize=True, save_path=METRICS_DIR.parent / "figures" / "confusion_matrix_normalized.png")
    print(json.dumps(report["overall"], indent=2))
    print(f"Saved report to {output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="PCB Vision AI project workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="Explicitly clone the official DeepPCB repository")
    download.add_argument("--target", default=str(DATASET_CONFIG.raw_dir / "DeepPCB"))
    download.set_defaults(func=command_download)

    validate = sub.add_parser("validate", help="Validate a real or synthetic DeepPCB dataset")
    validate.add_argument("--raw-dir", default=str(DATASET_CONFIG.raw_dir))
    validate.add_argument("--output", default=str(METRICS_DIR / "dataset_validation.json"))
    validate.set_defaults(func=command_validate)

    synthetic = sub.add_parser("synthetic", help="Explicitly generate testing-only synthetic data")
    synthetic.add_argument("--target", default=str(DATASET_CONFIG.raw_dir / "synthetic"))
    synthetic.add_argument("--num-pairs", type=int, default=120)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.set_defaults(func=command_synthetic)

    preprocess = sub.add_parser("preprocess", help="Build leakage-safe processed arrays")
    preprocess.add_argument("--raw-dir", default=str(DATASET_CONFIG.raw_dir))
    preprocess.set_defaults(func=command_preprocess)

    train = sub.add_parser("train", help="Train the custom CNN")
    train.add_argument("--processed", default=str(DATASET_CONFIG.processed_dir / "pcb_dataset_processed.npz"))
    train.add_argument("--save", default=str(MODEL_CONFIG.cnn_model_path))
    train.add_argument("--epochs", type=int, default=MODEL_CONFIG.epochs)
    train.add_argument("--batch-size", type=int, default=MODEL_CONFIG.batch_size)
    train.add_argument("--learning-rate", type=float, default=MODEL_CONFIG.learning_rate)
    train.set_defaults(func=command_train)

    evaluate = sub.add_parser("evaluate", help="Evaluate the final test split")
    evaluate.add_argument("--processed", default=str(DATASET_CONFIG.processed_dir / "pcb_dataset_processed.npz"))
    evaluate.add_argument("--model", default=str(MODEL_CONFIG.cnn_model_path))
    evaluate.add_argument("--batch-size", type=int, default=MODEL_CONFIG.batch_size)
    evaluate.set_defaults(func=command_evaluate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
