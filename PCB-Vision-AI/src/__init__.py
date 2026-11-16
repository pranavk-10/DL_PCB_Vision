"""
PCB Vision AI - Core Source Package
"""

from .dataset_loader import DeepPCBParser, DeepPCBDatasetLoader, PCBSample, BoundingBox
from .preprocessing import PCBDataPipeline, split_pcb_samples, extract_defect_patch, extract_normal_patches
from .data_analysis import run_full_eda, plot_class_distribution, plot_patch_gallery
from .model import build_custom_cnn, build_ann_baseline, TF_AVAILABLE
from .train import train_model, run_experiment_1_ann_vs_cnn, run_experiment_2_adam_vs_sgd, run_experiment_3_augmentation
from .evaluate import evaluate_model_predictions, run_full_evaluation
from .localization import SlidingWindowInspector
from .inference import PCBDefectClassifier

__all__ = [
    "DeepPCBParser",
    "DeepPCBDatasetLoader",
    "PCBSample",
    "BoundingBox",
    "PCBDataPipeline",
    "split_pcb_samples",
    "extract_defect_patch",
    "extract_normal_patches",
    "run_full_eda",
    "plot_class_distribution",
    "plot_patch_gallery",
    "build_custom_cnn",
    "build_ann_baseline",
    "TF_AVAILABLE",
    "train_model",
    "run_experiment_1_ann_vs_cnn",
    "run_experiment_2_adam_vs_sgd",
    "run_experiment_3_augmentation",
    "evaluate_model_predictions",
    "run_full_evaluation",
    "SlidingWindowInspector",
    "PCBDefectClassifier",
]
