"""Leakage-safe patch preparation and one preprocessing contract for training/inference."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import uuid
import cv2
import numpy as np
from PIL import Image
from config.config import CLASS_NAMES, CONFIG, Settings
from src.dataset_loader import DatasetError, PCBRecord, inspect_dataset, load_dataset, split_groups, validate_box

LOGGER = logging.getLogger(__name__)

def read_image(path: str | Path, max_pixels: int = CONFIG.max_image_pixels) -> np.ndarray:
    """Decode grayscale with Unicode-safe paths and reject corrupt/oversized images."""
    path = Path(path)
    try:
        with Image.open(path) as image:
            if image.width * image.height > max_pixels:
                raise ValueError(f"Image exceeds the {max_pixels:,}-pixel limit: {path}")
            image.verify()
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read image {path}: {error}") from error
    if image is None or image.size == 0:
        raise ValueError(f"OpenCV could not decode {path}.")
    return image

def preprocess_image(image: np.ndarray, image_size: int = CONFIG.image_size) -> np.ndarray:
    """Return finite float32 [0,1] grayscale HWC data; RGB arrays use RGB ordering."""
    image = np.asarray(image)
    was_integer = np.issubdtype(image.dtype, np.integer)
    if image.size == 0 or image.ndim not in (2, 3) or not np.issubdtype(image.dtype, np.number):
        raise ValueError("Expected a nonempty numeric grayscale or RGB image array.")
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[..., 0]
        elif image.shape[2] in (3, 4):
            image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY if image.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError("An image must have one, three, or four channels.")
    image = image.astype(np.float32)
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError("Pixels must be finite and in [0,255] or already normalized to [0,1].")
    if image.max() > 1 or was_integer:
        image = image / 255.0
    # Integer images consisting only of 0/1 also represent 8-bit intensity values.
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA if min(image.shape) > image_size else cv2.INTER_LINEAR)
    return np.clip(resized, 0, 1).astype(np.float32)[..., np.newaxis]

def intersection_area(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> int:
    return max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(0, min(first[3], second[3]) - max(first[1], second[1]))

def padded_box(box: tuple[int, int, int, int], width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    validate_box(box, width, height)
    if padding < 0:
        raise ValueError("Padding cannot be negative.")
    return (max(0, box[0] - padding), max(0, box[1] - padding), min(width, box[2] + padding), min(height, box[3] + padding))

def extract_defect_patch(image: np.ndarray, box: tuple[int, int, int, int], padding: int = CONFIG.padding,
                         image_size: int = CONFIG.image_size) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = padded_box(box, width, height, padding)
    return preprocess_image(image[y1:y2, x1:x2], image_size)

def generate_normal_boxes(image_shape: tuple[int, ...], boxes: list[tuple[int, int, int, int]],
                          rng: np.random.Generator, window_size: int = CONFIG.normal_window_size,
                          count: int = CONFIG.normal_patches_per_image, clearance: int = CONFIG.normal_clearance,
                          max_attempts: int = CONFIG.normal_max_attempts) -> list[tuple[int, int, int, int]]:
    """Select distinct windows outside all expanded defect boxes; never duplicate to fill quotas."""
    height, width = image_shape[:2]
    if min(window_size, max_attempts) < 1 or min(count, clearance) < 0:
        raise ValueError("Normal sampling settings are invalid.")
    if window_size > min(height, width):
        return []
    forbidden = [padded_box(box, width, height, clearance) for box in boxes]
    samples = []
    for _ in range(max_attempts):
        if len(samples) >= count:
            break
        x = int(rng.integers(0, width - window_size + 1))
        y = int(rng.integers(0, height - window_size + 1))
        candidate = (x, y, x + window_size, y + window_size)
        if candidate in samples or any(intersection_area(candidate, box) for box in forbidden):
            continue
        # Avoid nearly identical overlapping normal samples, without insisting on impossible quotas.
        area = window_size**2
        if any(intersection_area(candidate, box) / area > 0.5 for box in samples):
            continue
        samples.append(candidate)
    return samples

def _save_gray(path: Path, gray: np.ndarray, image_size: int) -> None:
    image = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA if min(gray.shape) > image_size else cv2.INTER_LINEAR)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"Failed to encode {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())

def prepare_dataset(settings: Settings = CONFIG, overwrite: bool = False, confirmed_schema: bool = False) -> Path:
    """Inspect, split groups, validate both images, crop, and atomically publish a manifest.

    Confirmation is required because upstream files could not be inspected in the
    restricted development environment. It records the selected label/origin contract.
    Rebuilds use new versioned directories; existing data is never recursively deleted.
    """
    settings.validate()
    destination = settings.processed_dir
    manifest_path = destination / "manifest.csv"
    if manifest_path.is_file() and not overwrite:
        load_manifest(settings)
        LOGGER.info("Existing validated manifest preserved: %s", manifest_path)
        return manifest_path
    if not confirmed_schema:
        raise DatasetError("Run src.dataset_loader, inspect its raw lines/README/class map, then pass --confirm-schema after verifying them.")
    inspection = inspect_dataset(settings.dataset_dir, settings.annotation_origin, settings.group_by_board)
    if inspection["missing_defect_ids"]:
        raise DatasetError(f"Dataset is missing defect IDs {inspection['missing_defect_ids']}; provide the full dataset.")
    records = load_dataset(settings.dataset_dir, settings.annotation_origin, settings.group_by_board)
    splits = split_groups(records, settings)
    destination.mkdir(parents=True, exist_ok=True)
    build = destination / f"dataset_{settings.seed}_{uuid.uuid4().hex[:8]}"
    build.mkdir()
    rows, skipped = [], []
    digest = hashlib.sha256()
    for split_name, split_records in splits.items():
        for record in split_records:
            try:
                test, template = read_image(record.test_path, settings.max_image_pixels), read_image(record.template_path, settings.max_image_pixels)
                if test.shape != template.shape:
                    raise DatasetError(f"Aligned pair dimensions differ: {record.image_id}.")
                height, width = test.shape
                for annotation in record.annotations:
                    validate_box(annotation.box, width, height)
                for path in (record.test_path, record.template_path, record.annotation_path):
                    digest.update(path.name.encode())
                    digest.update(path.read_bytes())
                seed_material = hashlib.sha256(f"{settings.seed}:{record.image_id}".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(seed_material[:8], "little"))
                sample_id = hashlib.sha256(record.image_id.encode()).hexdigest()[:16]
                crops = []
                for index, annotation in enumerate(record.annotations):
                    crop = padded_box(annotation.box, width, height, settings.padding)
                    if any(other.class_id != annotation.class_id and intersection_area(crop, other.box) > 0 for other in record.annotations):
                        LOGGER.warning("Skipping ambiguous multi-class crop %s annotation %s", record.image_id, index)
                        continue
                    crops.append((f"defect_{index}", annotation.class_id, crop, False))
                normals = generate_normal_boxes(test.shape, [annotation.box for annotation in record.annotations], rng,
                                                settings.normal_window_size, settings.normal_patches_per_image,
                                                settings.normal_clearance, settings.normal_max_attempts)
                if len(normals) < settings.normal_patches_per_image:
                    LOGGER.warning("Only %s/%s normal windows available in %s; no samples duplicated.", len(normals), settings.normal_patches_per_image, record.image_id)
                crops.extend((f"normal_{index}", 6, box, True) for index, box in enumerate(normals))
                for crop_name, class_id, box, is_normal in crops:
                    x1, y1, x2, y2 = box
                    patch_path = build / split_name / str(class_id) / f"{sample_id}_{crop_name}.png"
                    _save_gray(patch_path, test[y1:y2, x1:x2], settings.image_size)
                    rows.append({"patch_path": patch_path.relative_to(destination).as_posix(), "split": split_name,
                                 "class_id": class_id, "class_name": CLASS_NAMES[class_id], "image_id": record.image_id,
                                 "group_id": record.group_id, "source_image": str(record.test_path.resolve()),
                                 "template_image": str(record.template_path.resolve()), "annotation_path": str(record.annotation_path.resolve()),
                                 "x1": x1, "y1": y1, "x2": x2, "y2": y2, "is_normal": is_normal})
            except (OSError, ValueError, DatasetError) as error:
                skipped.append({"image_id": record.image_id, "reason": str(error)})
                LOGGER.warning("Skipping invalid pair %s: %s", record.image_id, error)
    if not rows:
        raise DatasetError("No patches extracted. Review the input images, origin, and annotations.")
    counts = {name: Counter(int(row["class_id"]) for row in rows if row["split"] == name) for name in splits}
    if missing := [CLASS_NAMES[index] for index in range(7) if not counts["train"][index]]:
        raise DatasetError(f"Training split is missing classes {missing}. Use more source groups or another documented seed; do not duplicate validation/test samples.")
    for name in ("validation", "test"):
        if not sum(counts[name].values()):
            raise DatasetError(f"No usable patches in {name} split.")
        if missing := [CLASS_NAMES[index] for index in range(7) if not counts[name][index]]:
            LOGGER.warning("%s split has no examples of %s; reports will show zero support.", name, missing)
    temporary = destination / "manifest.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"settings": settings.to_dict(), "schema_confirmed": True, "inspection": inspection,
                "dataset_fingerprint": digest.hexdigest(), "manifest_sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
                "groups_by_split": {name: sorted({record.group_id for record in records}) for name, records in splits.items()},
                "counts_by_split": {name: {CLASS_NAMES[index]: counts[name][index] for index in range(7)} for name in splits},
                "patch_count": len(rows), "skipped_pairs": skipped, "build_directory": build.name,
                "normal_definition": "test-image window outside every annotated box plus clearance; assumes annotation completeness"}
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    load_manifest(settings)
    LOGGER.info("Prepared %s patches; manifest: %s", len(rows), manifest_path)
    return manifest_path

def load_manifest(settings: Settings = CONFIG):
    """Read a prepared manifest, verify its hash, paths, label range, and split isolation."""
    import pandas as pd
    path = settings.processed_dir / "manifest.csv"
    metadata_path = settings.processed_dir / "metadata.json"
    if not path.is_file() or not metadata_path.is_file():
        raise DatasetError("Prepared data is missing. Run python -m src.preprocessing first.")
    metadata = json.loads(metadata_path.read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["manifest_sha256"]:
        raise DatasetError("Manifest checksum differs from preparation metadata; rerun preprocessing.")
    if metadata["settings"]["image_size"] != settings.image_size or tuple(metadata["settings"]["class_names"]) != settings.class_names:
        raise DatasetError("Prepared dimensions/class mapping differ from current configuration.")
    frame = pd.read_csv(path)
    required = {"patch_path", "split", "class_id", "image_id", "group_id", "source_image", "template_image"}
    if not required.issubset(frame.columns) or frame.empty:
        raise DatasetError("Manifest columns are missing or manifest is empty.")
    if not frame["class_id"].isin(range(7)).all() or not frame["split"].isin(("train", "validation", "test")).all():
        raise DatasetError("Manifest has invalid labels or split names.")
    for column in ("group_id", "image_id", "source_image", "template_image"):
        if (frame.groupby(column)["split"].nunique() > 1).any():
            raise DatasetError(f"Leakage detected for manifest column {column}.")
    if frame["patch_path"].duplicated().any():
        raise DatasetError("Duplicate patch paths detected.")
    for relative in frame["patch_path"]:
        resolved = (settings.processed_dir / relative).resolve()
        if not resolved.is_relative_to(settings.processed_dir.resolve()) or not resolved.is_file():
            raise DatasetError(f"Missing or unsafe prepared patch path: {relative}")
    return frame, metadata

def make_tf_dataset(frame, settings: Settings = CONFIG, training: bool = False):
    """Stream prepared PNGs; only training is shuffled. Model augmentation is training-only."""
    import tensorflow as tf
    paths = [str((settings.processed_dir / path).resolve()) for path in frame["patch_path"]]
    labels = frame["class_id"].to_numpy(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    def decode(path, label):
        image = tf.io.decode_png(tf.io.read_file(path), channels=1)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.ensure_shape(image, (settings.image_size, settings.image_size, 1))
        return image, label
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options).map(decode, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if training:
        dataset = dataset.shuffle(len(frame), seed=settings.seed, reshuffle_each_iteration=True)
    return dataset.batch(settings.batch_size).prefetch(tf.data.AUTOTUNE)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=CONFIG.dataset_dir)
    parser.add_argument("--origin", choices=("zero", "one"), default=CONFIG.annotation_origin)
    parser.add_argument("--confirm-schema", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Publish a new versioned build, preserving existing files.")
    parser.add_argument("--group-by-pair", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = replace(CONFIG.with_dataset(args.dataset), annotation_origin=args.origin, group_by_board=not args.group_by_pair)
    try:
        prepare_dataset(settings, args.overwrite, args.confirm_schema)
    except (DatasetError, OSError, ValueError) as error:
        parser.exit(1, f"Preprocessing failed: {error}\n")

if __name__ == "__main__":
    main()
