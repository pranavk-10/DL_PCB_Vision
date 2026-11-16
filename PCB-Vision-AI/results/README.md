# PCB Vision AI - Results, Figures, and Metrics

## Contents
- `results/figures/`:
  - `class_distribution.png`: Horizontal bar chart of defect counts.
  - `bbox_dimensions.png`: Scatter plot and histogram of bounding box width vs. height.
  - `sample_annotated_pcb.png`: Side-by-side test image and ground-truth bounding box visualization.
  - `patch_gallery.png`: Representative $128 \times 128$ patches across all 7 classes.
  - `loss_curves.png`: Training vs. Validation Cross-Entropy Loss.
  - `accuracy_curves.png`: Training vs. Validation Classification Accuracy.
  - `confusion_matrix.png`: Test set confusion matrix (raw counts).
  - `confusion_matrix_normalized.png`: Normalized confusion matrix (per-class recall).
  - `per_class_metrics.png`: Grouped bar chart of Precision, Recall, and F1-score.
  - `model_comparison.png`: Architectural and optimizer comparison.
- `results/metrics/`:
  - `dataset_eda_summary.json`: Statistical breakdown of raw annotations.
  - `final_evaluation_report.json`: Overall and per-class test metrics.
  - `experiment_1_ann_vs_cnn.json`: Quantitative metrics for Experiment 1.
  - `experiment_2_adam_vs_sgd.json`: Quantitative metrics for Experiment 2.
  - `experiment_3_augmentation.json`: Quantitative metrics for Experiment 3.
