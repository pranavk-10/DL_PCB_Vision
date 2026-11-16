"""
PCB Vision AI - Exploratory Data Analysis & Visualization
Generates academic-quality visualizations and statistical reports:
1. Class distribution bar chart.
2. Bounding box aspect-ratio and size distribution.
3. PCB full-image visualizer with color-coded ground-truth annotations.
4. Gallery of extracted 128x128 patches across all 7 classes.
5. Train / Val / Test split class-distribution consistency chart.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
try:
    import matplotlib
    matplotlib.use("Agg")  # Headless backend for reliable server/script plotting
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

from config.config import (
    DATASET_CONFIG, FIGURES_DIR, METRICS_DIR,
    DatasetConfig
)
from src.dataset_loader import PCBSample, BoundingBox

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Color palette for defect classes (BGR for OpenCV, RGB for Matplotlib)
DEFECT_COLORS = {
    "Open Circuit": (220, 50, 47),      # Red
    "Short Circuit": (211, 54, 130),    # Magenta
    "Mouse Bite": (181, 137, 0),        # Yellow/Gold
    "Spur": (42, 161, 152),             # Cyan
    "Pinhole": (38, 139, 210),          # Blue
    "Spurious Copper": (108, 113, 196), # Purple
    "Normal": (133, 153, 0),            # Green
}


def plot_class_distribution(
    class_counts: Dict[str, int],
    save_path: Optional[Path] = None,
    title: str = "DeepPCB Defect Class Distribution"
) -> Any:
    """Creates a stylized horizontal bar plot of class counts with percentage annotations."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    total = max(1, sum(counts))
    
    colors = [
        [c / 255.0 for c in DEFECT_COLORS.get(cls_name, (70, 130, 180))]
        for cls_name in classes
    ]
    
    bars = ax.barh(classes, counts, color=colors, edgecolor="black", alpha=0.85, height=0.6)
    
    # Annotate bar values
    for bar in bars:
        w = bar.get_width()
        pct = (w / total) * 100
        ax.text(
            w + max(total * 0.01, 1),
            bar.get_y() + bar.get_height() / 2,
            f"{int(w)} ({pct:.1f}%)",
            va="center", ha="left", fontsize=9, fontweight="bold", color="#333333"
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Annotated Defect Instances", fontsize=10, fontweight="bold")
    ax.set_ylabel("Defect Category", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlim(0, max(counts) * 1.18 if counts else 10)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved class distribution plot to: %s", save_path)
        
    return fig


def plot_bbox_dimensions(
    samples: List[PCBSample],
    save_path: Optional[Path] = None
) -> Any:
    """Plots scatter and histograms of bounding box width vs height."""
    widths = [b.width for s in samples for b in s.boxes]
    heights = [b.height for s in samples for b in s.boxes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    if widths and heights:
        ax1.scatter(widths, heights, color="#2b5c8f", alpha=0.6, edgecolors="none", s=25)
        ax1.plot([0, 80], [0, 80], "r--", label="Square Aspect Ratio (1:1)")
        ax1.set_title("Bounding Box Dimensions (Width vs Height)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Width (pixels)", fontsize=10)
        ax1.set_ylabel("Height (pixels)", fontsize=10)
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2.hist(widths, bins=15, color="#268bd2", alpha=0.7, label="Width", edgecolor="black")
        ax2.hist(heights, bins=15, color="#dc322f", alpha=0.5, label="Height", edgecolor="black")
        ax2.set_title("Distribution of Bounding Box Sizes", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Dimension (pixels)", fontsize=10)
        ax2.set_ylabel("Frequency", fontsize=10)
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.4)
    else:
        ax1.text(0.5, 0.5, "No bounding boxes available", ha="center")
        ax2.text(0.5, 0.5, "No bounding boxes available", ha="center")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved bbox dimensions plot to: %s", save_path)

    return fig


def visualize_pcb_sample_with_boxes(
    sample: PCBSample,
    save_path: Optional[Path] = None
) -> np.ndarray:
    """
    Renders the test PCB image and aligned template side-by-side with
    color-coded bounding boxes and label tags.
    """
    img_bgr = cv2.imread(str(sample.test_image_path))
    if img_bgr is None:
        raise ValueError(f"Could not load image: {sample.test_image_path}")

    annotated = img_bgr.copy()
    for box in sample.boxes:
        rgb_col = DEFECT_COLORS.get(box.class_name, (0, 0, 255))
        bgr_col = (rgb_col[2], rgb_col[1], rgb_col[0])

        cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), bgr_col, 2)
        label_text = f"{box.class_name}"
        
        # Draw text background pill
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(box.y1 - 4, th + 4)
        cv2.rectangle(annotated, (box.x1, ty - th - 3), (box.x1 + tw + 4, ty + 2), bgr_col, -1)
        cv2.putText(annotated, label_text, (box.x1 + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # If template image exists, stitch side by side
    if sample.template_image_path and sample.template_image_path.exists():
        temp_bgr = cv2.imread(str(sample.template_image_path))
        if temp_bgr is not None:
            # Header labels
            h, w = img_bgr.shape[:2]
            canvas = np.zeros((h + 40, w * 2 + 10, 3), dtype=np.uint8) + 245
            canvas[40:40 + h, :w] = img_bgr
            canvas[40:40 + h, w + 10:w * 2 + 10] = annotated

            cv2.putText(canvas, "Raw Test PCB Image", (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
            cv2.putText(canvas, f"Annotated ({len(sample.boxes)} Defects)", (w + 25, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 150), 2)

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), canvas)
                logger.info("Saved side-by-side sample visualization: %s", save_path)
            return canvas

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), annotated)
        logger.info("Saved sample visualization: %s", save_path)

    return annotated


def plot_patch_gallery(
    X: np.ndarray,
    y: np.ndarray,
    class_labels: List[str] = DATASET_CONFIG.class_labels,
    num_per_class: int = 4,
    save_path: Optional[Path] = None
) -> Any:
    """
    Renders an illustrative gallery showing representative 128x128 patches
    for each of the 7 defect/normal classes.
    """
    num_classes = len(class_labels)
    fig, axes = plt.subplots(num_classes, num_per_class, figsize=(num_per_class * 2.2, num_classes * 2.0), dpi=150)
    
    if num_classes == 1:
        axes = np.expand_dims(axes, 0)
    if num_per_class == 1:
        axes = np.expand_dims(axes, 1)

    for c_idx, label_name in enumerate(class_labels):
        indices = np.where(y == c_idx)[0]
        for col_idx in range(num_per_class):
            ax = axes[c_idx, col_idx]
            if col_idx < len(indices):
                patch = X[indices[col_idx]].squeeze()
                ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
                if col_idx == 0:
                    ax.set_ylabel(label_name, fontsize=9, fontweight="bold", labelpad=8)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", color="gray")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Extracted 128x128 PCB Defect Patches by Class", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved patch gallery to: %s", save_path)

    return fig


def run_full_eda(
    samples: List[PCBSample],
    save_figures: bool = True
) -> Dict[str, Any]:
    """Runs complete Exploratory Data Analysis and saves plots and metadata."""
    class_counts = {name: 0 for name in DATASET_CONFIG.class_labels}
    for s in samples:
        for b in s.boxes:
            name = DATASET_CONFIG.class_labels[b.target_idx]
            class_counts[name] = class_counts.get(name, 0) + 1

    summary = {
        "total_images": len(samples),
        "total_annotations": sum(len(s.boxes) for s in samples),
        "class_distribution": class_counts
    }

    if save_figures:
        plot_class_distribution(
            class_counts,
            save_path=FIGURES_DIR / "class_distribution.png"
        )
        plot_bbox_dimensions(
            samples,
            save_path=FIGURES_DIR / "bbox_dimensions.png"
        )
        if samples:
            visualize_pcb_sample_with_boxes(
                samples[0],
                save_path=FIGURES_DIR / "sample_annotated_pcb.png"
            )

        # Save metrics json
        metrics_file = METRICS_DIR / "dataset_eda_summary.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved dataset EDA summary to %s", metrics_file)

    return summary
