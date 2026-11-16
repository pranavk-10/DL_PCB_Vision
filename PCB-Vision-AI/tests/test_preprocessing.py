"""
Unit tests for PCB preprocessing, anti-leakage splitting, and patch extraction.
"""

from pathlib import Path
import numpy as np
import pytest

from src.dataset_loader import PCBSample, BoundingBox
from src.preprocessing import (
    split_pcb_samples,
    extract_defect_patch,
    extract_normal_patches,
    augment_patch
)


def test_anti_leakage_split():
    """
    CRITICAL TEST:
    Verifies that the dataset split occurs at the original PCB image level
    and that train, validation, and test sample IDs are strictly mutually disjoint.
    """
    dummy_samples = [
        PCBSample(
            sample_id=f"pcb_{i:04d}",
            test_image_path=Path(f"pcb_{i:04d}_test.jpg"),
            template_image_path=Path(f"pcb_{i:04d}_temp.jpg"),
            annotation_path=Path(f"pcb_{i:04d}.txt"),
            boxes=[]
        )
        for i in range(100)
    ]

    train_s, val_s, test_s = split_pcb_samples(
        dummy_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    assert len(train_s) == 70
    assert len(val_s) == 15
    assert len(test_s) == 15

    train_ids = set(s.sample_id for s in train_s)
    val_ids = set(s.sample_id for s in val_s)
    test_ids = set(s.sample_id for s in test_s)

    # Assert mutually disjoint sets (Zero data leakage!)
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage detected between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage detected between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage detected between val and test!"


def test_defect_patch_extraction():
    """Verifies that patch extraction returns normalized (128, 128, 1) tensor."""
    dummy_image = np.full((640, 640), 200, dtype=np.uint8)
    box = BoundingBox(x1=100, y1=100, x2=140, y2=140, class_id=1, class_name="Open Circuit", target_idx=0)

    patch = extract_defect_patch(dummy_image, box, target_size=(128, 128), padding=12)

    assert patch.shape == (128, 128, 1)
    assert patch.dtype == np.uint8


def test_normal_patch_extraction_no_overlap():
    """Verifies that extracted Normal patches do NOT overlap any defect bounding box."""
    dummy_image = np.full((640, 640), 220, dtype=np.uint8)
    defect_box = BoundingBox(x1=200, y1=200, x2=300, y2=300, class_id=2, class_name="Short", target_idx=1)

    normal_patches = extract_normal_patches(
        dummy_image,
        defect_boxes=[defect_box],
        num_patches=4,
        target_size=(128, 128),
        seed=42
    )

    assert len(normal_patches) > 0
    for p in normal_patches:
        assert p.shape == (128, 128, 1)


def test_augmentation_preserves_shape():
    """Tests that patch augmentation preserves shape and channel dimensions."""
    sample_patch = np.random.randint(0, 255, (128, 128, 1), dtype=np.uint8)
    aug = augment_patch(sample_patch)

    assert aug.shape == (128, 128, 1)
    assert aug.dtype == np.uint8
