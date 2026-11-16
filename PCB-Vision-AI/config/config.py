"""
PCB Vision AI - Central Configuration Module
Defines all directory paths, model hyperparameters, dataset constants,
augmentation parameters, and localization settings.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# Ensure runtime directories exist
for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetConfig:
    """Dataset and annotation parsing configuration."""
    raw_dir: Path = DATA_RAW_DIR
    processed_dir: Path = DATA_PROCESSED_DIR
    
    # DeepPCB standard raw image dimensions
    raw_image_width: int = 640
    raw_image_height: int = 640
    
    # Defect patch dimensions
    patch_size: Tuple[int, int] = (128, 128)
    channels: int = 1  # Grayscale
    
    # Margin padding around bounding box (in pixels)
    patch_padding: int = 12
    
    # Max normal patches extracted per defect-free region / image
    normal_patches_per_image: int = 3
    
    # DeepPCB raw annotation mapping: {integer_id: defect_name}
    # In DeepPCB annotation files: 1=open, 2=short, 3=mousebite, 4=spur, 5=copper, 6=pin-hole
    raw_class_map: Dict[int, str] = field(default_factory=lambda: {
        1: "Open Circuit",
        2: "Short Circuit",
        3: "Mouse Bite",
        4: "Spur",
        5: "Spurious Copper",
        6: "Pinhole",
    })
    
    # 7 Target classification classes (Index 0 through 6)
    class_labels: List[str] = field(default_factory=lambda: [
        "Open Circuit",      # Index 0 (DeepPCB id 1)
        "Short Circuit",     # Index 1 (DeepPCB id 2)
        "Mouse Bite",        # Index 2 (DeepPCB id 3)
        "Spur",              # Index 3 (DeepPCB id 4)
        "Pinhole",           # Index 4 (DeepPCB id 6)
        "Spurious Copper",   # Index 5 (DeepPCB id 5)
        "Normal",            # Index 6 (Generated Defect-Free)
    ])
    
    num_classes: int = 7
    
    # Train / Validation / Test split ratios (Applied at the original PCB image level!)
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42


@dataclass
class AugmentationConfig:
    """Data augmentation hyperparameters for training."""
    rotation_range: float = 15.0       # degrees
    width_shift_range: float = 0.08    # fractional translation
    height_shift_range: float = 0.08   # fractional translation
    horizontal_flip: bool = True
    vertical_flip: bool = True
    brightness_range: Tuple[float, float] = (0.85, 1.15)
    zoom_range: Tuple[float, float] = (0.92, 1.08)
    fill_mode: str = "reflect"


@dataclass
class ModelConfig:
    """CNN and ANN model architecture & training hyperparameters."""
    input_shape: Tuple[int, int, int] = (128, 128, 1)
    num_classes: int = 7
    dropout_rate: float = 0.30
    
    # Optimization
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 0.001       # Adam default
    sgd_learning_rate: float = 0.01   # SGD baseline
    sgd_momentum: float = 0.90
    
    # Early Stopping & LR Scheduler
    early_stopping_patience: int = 7
    lr_reduction_patience: int = 4
    lr_reduction_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Model Artifact Paths
    cnn_model_path: Path = MODELS_DIR / "pcb_cnn_best.keras"
    cnn_final_path: Path = MODELS_DIR / "pcb_cnn_final.keras"
    ann_model_path: Path = MODELS_DIR / "pcb_ann_baseline.keras"
    sgd_model_path: Path = MODELS_DIR / "pcb_cnn_sgd.keras"
    no_aug_model_path: Path = MODELS_DIR / "pcb_cnn_no_aug.keras"


@dataclass
class LocalizationConfig:
    """Sliding-window visual defect localization parameters."""
    window_size: Tuple[int, int] = (128, 128)
    stride: int = 48                    # Step size in pixels
    confidence_threshold: float = 0.70  # Minimum probability to trigger defect box
    iou_nms_threshold: float = 0.30     # Non-Maximum Suppression IoU threshold
    normal_class_idx: int = 6           # Index of 'Normal' class


# Instantiate global singleton configurations for easy import
DATASET_CONFIG = DatasetConfig()
AUGMENTATION_CONFIG = AugmentationConfig()
MODEL_CONFIG = ModelConfig()
LOCALIZATION_CONFIG = LocalizationConfig()
