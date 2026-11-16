"""
PCB Vision AI - Comprehensive Model Evaluation Module
Calculates:
- Accuracy, Precision, Recall, F1-Score (Macro and Weighted averages).
- Class-wise performance reports.
- Confusion Matrix (raw and normalized).
Generates publication-quality diagnostic plots saved to results/figures/:
1. Training and Validation Loss Curves.
2. Training and Validation Accuracy Curves.
3. Confusion Matrix Heatmap.
4. Per-Class Precision, Recall, and F1-Score Bar Chart.
5. Multi-Model Experimental Comparison Charts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None
try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_recall_fscore_support
    )
except ImportError:
    classification_report = None
    confusion_matrix = None
    accuracy_score = None
    precision_recall_fscore_support = None

from config.config import DATASET_CONFIG, FIGURES_DIR, METRICS_DIR, DatasetConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_labels: List[str] = DATASET_CONFIG.class_labels
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics from predicted probabilities.
    Returns structured dictionary with overall and per-class metrics.
    """
    y_pred = np.argmax(y_pred_probs, axis=-1)
    
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Class-wise metrics
    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_labels))), zero_division=0
    )

    per_class_metrics = {}
    for idx, name in enumerate(class_labels):
        per_class_metrics[name] = {
            "precision": float(p_per_class[idx]),
            "recall": float(r_per_class[idx]),
            "f1_score": float(f1_per_class[idx]),
            "support": int(support_per_class[idx])
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))

    report = {
        "overall": {
            "accuracy": acc,
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f1_macro),
            "weighted_precision": float(p_weighted),
            "weighted_recall": float(r_weighted),
            "weighted_f1": float(f1_weighted),
            "total_test_samples": int(len(y_true))
        },
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "class_labels": class_labels
    }

    return report


def plot_learning_curves(
    history: Dict[str, list],
    save_path_loss: Optional[Path] = None,
    save_path_acc: Optional[Path] = None
) -> Tuple[Any, Any]:
    """Plots and saves separate loss and accuracy training/validation curves."""
    epochs = range(1, len(history["loss"]) + 1)

    # 1. Loss Curve
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5), dpi=150)
    ax_loss.plot(epochs, history["loss"], "o-", color="#dc322f", label="Training Loss", linewidth=2, markersize=5)
    if "val_loss" in history:
        ax_loss.plot(epochs, history["val_loss"], "s--", color="#268bd2", label="Validation Loss", linewidth=2, markersize=5)
    ax_loss.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax_loss.set_ylabel("Loss", fontsize=10, fontweight="bold")
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend(fontsize=10)
    plt.tight_layout()

    if save_path_loss:
        save_path_loss = Path(save_path_loss)
        save_path_loss.parent.mkdir(parents=True, exist_ok=True)
        fig_loss.savefig(save_path_loss, bbox_inches="tight")
        logger.info("Saved loss curve: %s", save_path_loss)

    # 2. Accuracy Curve
    fig_acc, ax_acc = plt.subplots(figsize=(8, 5), dpi=150)
    ax_acc.plot(epochs, [a * 100 for a in history["accuracy"]], "o-", color="#859900", label="Training Accuracy", linewidth=2, markersize=5)
    if "val_accuracy" in history:
        ax_acc.plot(epochs, [a * 100 for a in history["val_accuracy"]], "s--", color="#2aa198", label="Validation Accuracy", linewidth=2, markersize=5)
    ax_acc.set_title("Classification Accuracy (%) vs. Epochs", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax_acc.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_acc.legend(fontsize=10)
    plt.tight_layout()

    if save_path_acc:
        save_path_acc = Path(save_path_acc)
        save_path_acc.parent.mkdir(parents=True, exist_ok=True)
        fig_acc.savefig(save_path_acc, bbox_inches="tight")
        logger.info("Saved accuracy curve: %s", save_path_acc)

    return fig_loss, fig_acc


def plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: List[str] = DATASET_CONFIG.class_labels,
    normalize: bool = False,
    save_path: Optional[Path] = None
) -> Any:
    """Plots a clear annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)
    
    if normalize:
        cm_display = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
        fmt = ".2f"
        title = "Normalized Confusion Matrix (Recall per Class)"
    else:
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix (Sample Counts)"

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        cbar=True,
        linewidths=0.5,
        linecolor="#dddddd"
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=10, fontweight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved confusion matrix: %s", save_path)

    return fig


def plot_per_class_metrics(
    per_class_report: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None
) -> Any:
    """Plots a grouped horizontal or vertical bar chart of Precision, Recall, and F1."""
    classes = list(per_class_report.keys())
    precisions = [per_class_report[c]["precision"] * 100 for c in classes]
    recalls = [per_class_report[c]["recall"] * 100 for c in classes]
    f1s = [per_class_report[c]["f1_score"] * 100 for c in classes]

    x = np.arange(len(classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    ax.bar(x - width, precisions, width, label="Precision", color="#268bd2", edgecolor="black", alpha=0.9)
    ax.bar(x, recalls, width, label="Recall", color="#859900", edgecolor="black", alpha=0.9)
    ax.bar(x + width, f1s, width, label="F1-Score", color="#b58900", edgecolor="black", alpha=0.9)

    ax.set_title("Per-Class Classification Performance (%)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved per-class metrics chart: %s", save_path)

    return fig


def plot_model_comparison(
    models_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None
) -> Any:
    """Plots comparative metrics across experimental configurations."""
    names = list(models_metrics.keys())
    accuracies = [models_metrics[m].get("accuracy", 0.0) * 100 for m in names]
    f1_scores = [models_metrics[m].get("macro_f1", 0.0) * 100 for m in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.bar(x - width / 2, accuracies, width, label="Accuracy (%)", color="#2aa198", edgecolor="black", alpha=0.85)
    ax.bar(x + width / 2, f1_scores, width, label="Macro F1-Score (%)", color="#6c71c4", edgecolor="black", alpha=0.85)

    ax.set_title("Model Architecture & Optimizer Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved model comparison plot: %s", save_path)

    return fig


def run_full_evaluation(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    history: Optional[Dict[str, list]] = None,
    save_dir_figures: Path = FIGURES_DIR,
    save_dir_metrics: Path = METRICS_DIR
) -> Dict[str, Any]:
    """Orchestrates comprehensive metric evaluation and chart generation."""
    report = evaluate_model_predictions(y_true, y_pred_probs)

    cm = np.array(report["confusion_matrix"])
    plot_confusion_matrix(cm, save_path=save_dir_figures / "confusion_matrix.png")
    plot_confusion_matrix(cm, normalize=True, save_path=save_dir_figures / "confusion_matrix_normalized.png")
    plot_per_class_metrics(report["per_class"], save_path=save_dir_figures / "per_class_metrics.png")

    if history:
        plot_learning_curves(
            history,
            save_path_loss=save_dir_figures / "loss_curves.png",
            save_path_acc=save_dir_figures / "accuracy_curves.png"
        )

    # Save evaluation report JSON
    metrics_file = save_dir_metrics / "final_evaluation_report.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved final evaluation report to %s", metrics_file)

    return report
