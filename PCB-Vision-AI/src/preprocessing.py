"""
PCB Vision AI - Data Preprocessing and Patch Extraction Pipeline
Performs:
1. Leakage-free dataset splitting at the original PCB image level.
2. Defect patch extraction with configurable margin padding.
3. Defect-free (Normal) background patch extraction from non-annotated regions.
4. Resizing to 128x128, grayscale conversion, and [0, 1] normalization.
5. Training-only data augmentation (rotation, flips, translation, brightness).
6. Class weight calculation for class imbalance mitigation.
7. Serialization of processed arrays into numpy datasets for fast reproducible loading.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from sklearn.utils.class_weight import compute_class_weight
except ImportError:
    compute_class_weight = None

from config.config import DATASET_CONFIG, AUGMENTATION_CONFIG, DatasetConfig, AugmentationConfig
from src.dataset_loader import PCBSample, BoundingBox, DeepPCBDatasetLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def split_pcb_samples(
    samples: List[PCBSample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[PCBSample], List[PCBSample], List[PCBSample]]:
    """
    CRITICAL ANTI-LEAKAGE REQUIREMENT:
    Splits the original PCB image pairs BEFORE extracting patches.
    Ensures no patches from the same PCB board or pair exist across train/val/test splits.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(len(samples))
    
    n_train = int(len(samples) * train_ratio)
    n_val = int(len(samples) * val_ratio)
    
    train_idx = shuffled_indices[:n_train]
    val_idx = shuffled_indices[n_train:n_train + n_val]
    test_idx = shuffled_indices[n_train + n_val:]
    
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    
    logger.info(
        "Image-Level Split completed: Train=%d, Val=%d, Test=%d (Total images=%d)",
        len(train_samples), len(val_samples), len(test_samples), len(samples)
    )
    return train_samples, val_samples, test_samples


def extract_defect_patch(
    image: np.ndarray,
    box: BoundingBox,
    target_size: Tuple[int, int] = (128, 128),
    padding: int = 12
) -> np.ndarray:
    """
    Extracts a defect patch centered on the bounding box with configurable padding,
    resizes to target_size (128, 128), and converts to grayscale single channel.
    """
    h, w = image.shape[:2]
    
    # Calculate padded bounding box
    px1 = max(0, box.x1 - padding)
    py1 = max(0, box.y1 - padding)
    px2 = min(w, box.x2 + padding)
    py2 = min(h, box.y2 + padding)
    
    # Ensure square crop or minimum dimension
    crop = image[py1:py2, px1:px2]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        # Fallback to center-crop around defect center
        cx = (box.x1 + box.x2) // 2
        cy = (box.y1 + box.y2) // 2
        half_w = target_size[0] // 2
        half_h = target_size[1] // 2
        x1 = max(0, min(cx - half_w, w - target_size[0]))
        y1 = max(0, min(cy - half_h, h - target_size[1]))
        crop = image[y1:y1 + target_size[1], x1:x1 + target_size[0]]

    # Convert to grayscale if 3 channels
    if len(crop.shape) == 3 and crop.shape[2] == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
    # Resize to exactly target_size
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    
    # Expand channel dimension (128, 128, 1)
    if len(resized.shape) == 2:
        resized = np.expand_dims(resized, axis=-1)
        
    return resized


def extract_normal_patches(
    image: np.ndarray,
    defect_boxes: List[BoundingBox],
    num_patches: int = 3,
    target_size: Tuple[int, int] = (128, 128),
    seed: int = 42
) -> List[np.ndarray]:
    """
    Extracts defect-free patches from regions outside annotated defect bounding boxes.
    Validates that candidate normal patches do not overlap any ground-truth defect box.
    """
    h, w = image.shape[:2]
    pw, ph = target_size
    normal_patches: List[np.ndarray] = []
    
    if h <= ph or w <= pw:
        return normal_patches

    rng = np.random.RandomState(seed + int(np.sum(image[:10, :10])))
    attempts = 0
    max_attempts = 40

    while len(normal_patches) < num_patches and attempts < max_attempts:
        attempts += 1
        rx1 = rng.randint(0, w - pw)
        ry1 = rng.randint(0, h - ph)
        rx2 = rx1 + pw
        ry2 = ry1 + ph

        # Check overlap with any defect box
        overlap = False
        for box in defect_boxes:
            # Check bounding box intersection
            ix1 = max(rx1, box.x1)
            iy1 = max(ry1, box.y1)
            ix2 = min(rx2, box.x2)
            iy2 = min(ry2, box.y2)
            if ix1 < ix2 and iy1 < iy2:
                overlap = True
                break

        if not overlap:
            crop = image[ry1:ry2, rx1:rx2]
            if len(crop.shape) == 3 and crop.shape[2] == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if len(crop.shape) == 2:
                crop = np.expand_dims(crop, axis=-1)
            normal_patches.append(crop)

    return normal_patches


def augment_patch(
    patch: np.ndarray,
    config: AugmentationConfig = AUGMENTATION_CONFIG,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Applies realistic, defect-preserving photometric and spatial augmentations:
    - Random rotation (-15 to +15 deg)
    - Random horizontal & vertical flips
    - Subtle brightness perturbation (0.85 to 1.15)
    - Small translation (shift up to 8%)
    Avoids destructive distortions (e.g., severe elastic deformation or shearing)
    that could fabricate false trace bridges or breaks.
    """
    if rng is None:
        rng = np.random.RandomState()

    img = patch.squeeze().copy()  # (128, 128)
    h, w = img.shape[:2]

    # 1. Random horizontal flip
    if config.horizontal_flip and rng.rand() > 0.5:
        img = cv2.flip(img, 1)

    # 2. Random vertical flip
    if config.vertical_flip and rng.rand() > 0.5:
        img = cv2.flip(img, 0)

    # 3. Small rotation
    angle = rng.uniform(-config.rotation_range, config.rotation_range)
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 4. Small translation
    dx = rng.uniform(-config.width_shift_range * w, config.width_shift_range * w)
    dy = rng.uniform(-config.height_shift_range * h, config.height_shift_range * h)
    trans_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, trans_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 5. Brightness adjustment
    scale = rng.uniform(config.brightness_range[0], config.brightness_range[1])
    img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    return np.expand_dims(img, axis=-1)


class PCBDataPipeline:
    """
    Complete end-to-end dataset extraction and preprocessing engine.
    Extracts all patches, applies anti-leakage splitting, normalizes to [0, 1],
    computes class weights, and saves processed arrays.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig = DATASET_CONFIG,
        aug_config: AugmentationConfig = AUGMENTATION_CONFIG
    ):
        self.dataset_config = dataset_config
        self.aug_config = aug_config

    def process_samples_to_dataset(
        self,
        samples: List[PCBSample],
        apply_augmentation: bool = False,
        target_size: Tuple[int, int] = (128, 128)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts defect patches and normal patches from a list of PCB samples.
        Normalizes pixel values from [0, 255] to [0.0, 1.0].
        Returns:
            X: np.ndarray of shape (N, 128, 128, 1) float32
            y: np.ndarray of shape (N,) int64
        """
        x_list: List[np.ndarray] = []
        y_list: List[int] = []

        rng = np.random.RandomState(self.dataset_config.random_seed)

        for sample in samples:
            if not sample.test_image_path.exists():
                continue
            
            # Read image in grayscale
            img = cv2.imread(str(sample.test_image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # 1. Extract defect patches
            for box in sample.boxes:
                patch = extract_defect_patch(
                    img, box, target_size=target_size, padding=self.dataset_config.patch_padding
                )
                x_list.append(patch)
                y_list.append(box.target_idx)

                # If training split, optionally add augmented copies for minority classes
                if apply_augmentation:
                    aug_patch = augment_patch(patch, config=self.aug_config, rng=rng)
                    x_list.append(aug_patch)
                    y_list.append(box.target_idx)

            # 2. Extract Normal (defect-free) patches (Class index 6)
            normal_patches = extract_normal_patches(
                img,
                sample.boxes,
                num_patches=self.dataset_config.normal_patches_per_image,
                target_size=target_size,
                seed=self.dataset_config.random_seed
            )
            for n_patch in normal_patches:
                x_list.append(n_patch)
                y_list.append(6)  # Index 6 is Normal

        if not x_list:
            # Handle empty corner case gracefully
            logger.warning("No patches extracted from provided samples.")
            return np.zeros((0, *target_size, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        X = np.array(x_list, dtype=np.float32) / 255.0  # Normalize to [0.0, 1.0]
        y = np.array(y_list, dtype=np.int64)

        logger.info("Processed %d patches (X shape: %s, y shape: %s)", len(y), X.shape, y.shape)
        return X, y

    def run_full_pipeline(
        self,
        samples: Optional[List[PCBSample]] = None,
        save_to_disk: bool = True
    ) -> Dict[str, Any]:
        """
        Executes complete pipeline:
        1. Discover samples
        2. Anti-leakage split
        3. Train / Val / Test patch extraction (Train with augmentation)
        4. Class weights computation
        5. Save processed data to data/processed/
        """
        if samples is None:
            loader = DeepPCBDatasetLoader(self.dataset_config.raw_dir, self.dataset_config)
            samples = loader.scan_dataset()

        if not samples:
            raise ValueError(f"No PCB samples found in {self.dataset_config.raw_dir}")

        train_s, val_s, test_s = split_pcb_samples(
            samples,
            train_ratio=self.dataset_config.train_split,
            val_ratio=self.dataset_config.val_split,
            test_ratio=self.dataset_config.test_split,
            seed=self.dataset_config.random_seed
        )

        logger.info("Extracting training patches (with augmentation)...")
        X_train, y_train = self.process_samples_to_dataset(train_s, apply_augmentation=True)

        logger.info("Extracting validation patches (clean, no augmentation)...")
        X_val, y_val = self.process_samples_to_dataset(val_s, apply_augmentation=False)

        logger.info("Extracting test patches (clean, no augmentation)...")
        X_test, y_test = self.process_samples_to_dataset(test_s, apply_augmentation=False)

        # Compute balanced class weights to mitigate class distribution skew
        classes = np.unique(y_train)
        if compute_class_weight is not None:
            weights_arr = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_train
            )
            class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights_arr)}
        else:
            counts = np.bincount(y_train, minlength=len(self.dataset_config.class_labels))
            total = max(1, int(counts.sum()))
            class_weight_dict = {
                int(c): float(total / (len(classes) * max(1, int(counts[c]))))
                for c in classes
            }

        data_bundle = {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "class_weights": class_weight_dict,
            "class_labels": self.dataset_config.class_labels,
            "split_metadata": {
                "train_samples": [s.sample_id for s in train_s],
                "val_samples": [s.sample_id for s in val_s],
                "test_samples": [s.sample_id for s in test_s],
                "seed": self.dataset_config.random_seed,
            }
        }

        if save_to_disk:
            out_file = self.dataset_config.processed_dir / "pcb_dataset_processed.npz"
            np.savez_compressed(
                out_file,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
                class_labels=np.asarray(self.dataset_config.class_labels, dtype=np.str_),
                train_sample_ids=np.asarray([s.sample_id for s in train_s], dtype=np.str_),
                val_sample_ids=np.asarray([s.sample_id for s in val_s], dtype=np.str_),
                test_sample_ids=np.asarray([s.sample_id for s in test_s], dtype=np.str_),
            )
            logger.info("Saved processed dataset to: %s", out_file)

        return data_bundle

    @staticmethod
    def load_processed_dataset(npz_path: Path) -> Dict[str, np.ndarray]:
        """Loads pre-extracted numpy arrays from disk."""
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Processed dataset not found at {npz_path}")
        with np.load(npz_path, allow_pickle=False) as data:
            result = {
                "X_train": data["X_train"],
                "y_train": data["y_train"],
                "X_val": data["X_val"],
                "y_val": data["y_val"],
                "X_test": data["X_test"],
                "y_test": data["y_test"]
            }
            if "class_labels" in data:
                result["class_labels"] = data["class_labels"].tolist()
            for key in ("train_sample_ids", "val_sample_ids", "test_sample_ids"):
                if key in data:
                    result[key] = data[key].tolist()
            return result
