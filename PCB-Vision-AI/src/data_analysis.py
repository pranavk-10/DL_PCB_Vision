"""Generate honest dataset summaries, class distributions, and annotated image plots."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from config.config import CLASS_NAMES, CONFIG
from src.dataset_loader import inspect_dataset, load_dataset
from src.preprocessing import load_manifest, read_image

COLORS = ["#15836b", "#ca463a", "#3476aa", "#cf862a", "#835aa3", "#576a72", "#718079"]

def plot_annotations(records, output: Path, limit: int = 6) -> Path:
    selected = records[:limit]
    if not selected:
        raise ValueError("No records available for annotation visualization.")
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, record in zip(axes.flat, selected):
        try:
            image = read_image(record.test_path)
            axis.imshow(image, cmap="gray")
            for annotation in record.annotations:
                x1, y1, x2, y2 = annotation.box
                color = COLORS[annotation.class_id]
                axis.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=1.5))
                axis.text(x1, y1, CLASS_NAMES[annotation.class_id], fontsize=7, color="white", backgroundcolor=color)
            axis.set_title(record.image_id, fontsize=9)
        except ValueError as error:
            axis.text(0.5, 0.5, f"Cannot decode sample\n{error}", ha="center", wrap=True)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output

def plot_class_distribution(frame: pd.DataFrame, output: Path) -> Path:
    counts = pd.crosstab(frame["class_id"], frame["split"]).reindex(range(7), fill_value=0)
    counts.index = CLASS_NAMES
    figure, axis = plt.subplots(figsize=(12, 5))
    counts.reindex(columns=("train", "validation", "test"), fill_value=0).plot.bar(ax=axis, color=["#15836b", "#597b92", "#c48b47"], rot=25)
    axis.set_ylabel("Prepared patches")
    axis.set_xlabel("")
    axis.set_title("Class distribution after splitting source PCB groups")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output

def analyze_dataset(settings=CONFIG) -> dict:
    inspection = inspect_dataset(settings.dataset_dir, settings.annotation_origin, settings.group_by_board)
    records = load_dataset(settings.dataset_dir, settings.annotation_origin, settings.group_by_board)
    settings.metrics_dir.mkdir(parents=True, exist_ok=True)
    (settings.metrics_dir / "dataset_summary.json").write_text(json.dumps(inspection, indent=2), encoding="utf-8")
    plot_annotations(records, settings.figures_dir / "sample_annotations.png")
    # Raw defect distribution is available even before normal patch generation.
    figure, axis = plt.subplots(figsize=(10, 5))
    names, counts = zip(*inspection["counts_by_class"].items())
    axis.bar(names, counts, color="#15836b")
    axis.set_ylabel("Annotated defects")
    axis.tick_params(axis="x", rotation=25)
    axis.set_title("Observed DeepPCB annotation distribution")
    figure.tight_layout()
    figure.savefig(settings.figures_dir / "annotation_distribution.png", dpi=150)
    plt.close(figure)
    if (settings.processed_dir / "manifest.csv").is_file():
        frame, metadata = load_manifest(settings)
        plot_class_distribution(frame, settings.figures_dir / "class_distribution.png")
        inspection["prepared_counts"] = metadata["counts_by_split"]
    return inspection

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=CONFIG.dataset_dir)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        summary = analyze_dataset(CONFIG.with_dataset(args.dataset))
        logging.info("Observed %s image pairs; summaries and figures saved under results.", summary["image_pairs"])
    except (OSError, ValueError) as error:
        parser.exit(1, f"Dataset analysis failed: {error}\n")

if __name__ == "__main__":
    main()
