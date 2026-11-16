"""Reproducible paths, label mapping, preprocessing, training, and inspection settings."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("Open Circuit", "Short Circuit", "Mouse Bite", "Spur", "Pinhole", "Spurious Copper", "Normal / Defect-Free")
# DeepPCB IDs are not the same order as our display classes: 5=copper, 6=pin-hole.
DEEPPCB_TO_CLASS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 5, 6: 4}

@dataclass(frozen=True)
class Settings:
    root: Path = PROJECT_ROOT
    dataset_dir: Path = PROJECT_ROOT / "data" / "raw" / "DeepPCB-master" / "PCBData"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    figures_dir: Path = PROJECT_ROOT / "results" / "figures"
    metrics_dir: Path = PROJECT_ROOT / "results" / "metrics"
    image_size: int = 128
    channels: int = 1
    class_names: tuple[str, ...] = CLASS_NAMES
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 0.001
    sgd_learning_rate: float = 0.001
    optimizer: str = "adam"
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    padding: int = 12
    # Normal windows use the same physical scale as inspection windows.
    normal_window_size: int = 128
    normal_patches_per_image: int = 4
    normal_clearance: int = 12
    normal_max_attempts: int = 400
    annotation_origin: str = "zero"  # Explicitly choose zero or one after repository inspection.
    group_by_board: bool = True
    augmentation: bool = True
    dropout: float = 0.3
    early_stopping_patience: int = 7
    reduce_lr_patience: int = 3
    confidence_threshold: float = 0.80
    window_size: int = 128
    stride: int = 64
    nms_iou_threshold: float = 0.30
    localization_iou_threshold: float = 0.50
    max_image_pixels: int = 16_000_000

    def validate(self) -> None:
        if self.image_size != 128 or self.channels != 1 or len(self.class_names) != 7:
            raise ValueError("This architecture requires 128 x 128 x 1 inputs and seven classes.")
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1) > 1e-8:
            raise ValueError("Split fractions must sum to one.")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("All three split fractions must be positive.")
        if self.annotation_origin not in {"zero", "one"}:
            raise ValueError("annotation_origin must be zero or one.")
        if self.batch_size < 1 or self.epochs < 1 or self.learning_rate <= 0:
            raise ValueError("Batch size, epochs, and learning rate must be positive.")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be adam or sgd.")
        if min(self.padding, self.normal_clearance, self.normal_patches_per_image) < 0:
            raise ValueError("Padding, clearance, and sample count cannot be negative.")
        if self.normal_window_size < 1 or self.window_size < 1 or not 1 <= self.stride <= self.window_size:
            raise ValueError("Window sizes must be positive; stride must be within window size.")
        for value in (self.confidence_threshold, self.nms_iou_threshold, self.localization_iou_threshold):
            if not 0 <= value <= 1:
                raise ValueError("Confidence and IoU thresholds must lie within [0, 1].")

    @property
    def model_path(self) -> Path:
        return self.models_dir / "cnn_adam_aug" / "best.keras"

    def with_dataset(self, path: str | Path) -> Settings:
        return replace(self, dataset_dir=Path(path).expanduser().resolve())

    def to_dict(self) -> dict:
        values = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}
        values["model_output_path"] = str(self.model_path)
        return values

CONFIG = Settings()
