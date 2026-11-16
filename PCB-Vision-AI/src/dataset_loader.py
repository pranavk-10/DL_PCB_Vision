"""Inspect DeepPCB on disk, validate its text annotations, and split source groups.

Repository access was unavailable during development. This loader recognizes the
published five-integer xyxy/type schema, inspects actual files at runtime, and
fails on a different schema. The label map and coordinate origin remain explicit.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
from typing import Iterable
from config.config import CLASS_NAMES, CONFIG, DEEPPCB_TO_CLASS, Settings

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pgm"}

class DatasetError(ValueError):
    """A dataset cannot safely be interpreted or split."""

@dataclass(frozen=True)
class Annotation:
    box: tuple[int, int, int, int]
    class_id: int
    source_label: int

@dataclass(frozen=True)
class PCBRecord:
    image_id: str
    group_id: str
    test_path: Path
    template_path: Path
    annotation_path: Path
    annotations: tuple[Annotation, ...]

def validate_box(box: tuple[int, int, int, int], width: int, height: int) -> None:
    """Validate internal zero-based, half-open coordinates without silently clipping."""
    x1, y1, x2, y2 = box
    if width <= 0 or height <= 0 or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise DatasetError(f"Invalid bounding box {box} for image {width} x {height}.")

def parse_annotations(path: str | Path, origin: str = "zero") -> tuple[Annotation, ...]:
    """Read exactly x_min y_min x_max y_max defect_id integer rows.

    For one-based inclusive input, subtract one from minima only; maxima then
    become exclusive indices. Zero input is interpreted as zero-based xyxy.
    Empty files are valid defect-free annotations; malformed rows are errors.
    """
    path = Path(path)
    if origin not in {"zero", "one"}:
        raise DatasetError("Coordinate origin must be explicitly zero or one.")
    annotations = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) != 5:
            raise DatasetError(f"{path}:{line_number}: expected five integers, received {line!r}.")
        try:
            x1, y1, x2, y2, source_label = map(int, tokens)
        except ValueError as error:
            raise DatasetError(f"{path}:{line_number}: noninteger annotation {line!r}.") from error
        if source_label not in DEEPPCB_TO_CLASS:
            raise DatasetError(f"{path}:{line_number}: unknown DeepPCB class {source_label}; expected 1..6.")
        if origin == "one":
            x1, y1 = x1 - 1, y1 - 1
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise DatasetError(f"{path}:{line_number}: invalid xyxy box {(x1, y1, x2, y2)}.")
        annotations.append(Annotation((x1, y1, x2, y2), DEEPPCB_TO_CLASS[source_label], source_label))
    return tuple(annotations)

def resolve_dataset_root(directory: str | Path) -> Path:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise DatasetError(f"Dataset directory does not exist: {directory}. See docs/DATASET_DOCUMENTATION.md.")
    if (directory / "PCBData").is_dir():
        return directory / "PCBData"
    candidates = sorted(directory.glob("DeepPCB*/PCBData"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise DatasetError("Multiple extracted datasets found; supply the exact PCBData path.")
    return directory

def pair_identifier(path: Path) -> str:
    if not path.stem.lower().endswith("_test"):
        raise DatasetError(f"Test image does not use the _test suffix: {path}")
    return path.stem[:-5]

def board_group(path: Path, root: Path, by_board: bool = True) -> str:
    relative = path.relative_to(root)
    if by_board:
        for part in relative.parts[:-1]:
            if part.lower().startswith("group"):
                return part
    return str(relative.with_name(pair_identifier(path))).replace("\\", "/")

def load_dataset(directory: str | Path, origin: str = "zero", group_by_board: bool = True,
                 strict: bool = False) -> list[PCBRecord]:
    """Discover test/temp pairs and annotation files by actual basenames and proximity.

    Missing pairs or invalid annotations are logged and skipped unless strict.
    Corrupt image decoding and image-bound validation occur during preprocessing.
    Ambiguous annotations are never guessed.
    """
    root = resolve_dataset_root(directory)
    text_index = defaultdict(list)
    for path in root.rglob("*.txt"):
        text_index[path.stem].append(path)
    records = []
    errors = []
    tests = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem.lower().endswith("_test"))
    for test in tests:
        try:
            stem = pair_identifier(test)
            templates = [path for path in test.parent.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem == stem + "_temp"]
            if len(templates) != 1:
                raise DatasetError(f"Expected exactly one aligned _temp image for {test}; found {len(templates)}.")
            annotations = text_index.get(stem, []) + text_index.get(stem + "_test", [])
            if len(annotations) > 1:
                local = [path for path in annotations if path.parent.parent == test.parent.parent]
                annotations = local if len(local) == 1 else annotations
            if len(annotations) != 1:
                raise DatasetError(f"Expected one annotation for {test}; found {len(annotations)}.")
            annotation_path = annotations[0]
            image_id = str(test.relative_to(root).with_suffix("")).replace("\\", "/")
            records.append(PCBRecord(image_id, board_group(test, root, group_by_board), test, templates[0], annotation_path, parse_annotations(annotation_path, origin)))
        except (DatasetError, OSError) as error:
            errors.append(str(error))
            LOGGER.warning("Skipping dataset record: %s", error)
            if strict:
                raise DatasetError(str(error)) from error
    if not records:
        detail = " ".join(errors[:3])
        raise DatasetError(f"No valid _test/_temp annotated pairs found under {root}. {detail}")
    return records

def split_groups(records: Iterable[PCBRecord], settings: Settings = CONFIG) -> dict[str, list[PCBRecord]]:
    """Shuffle indivisible PCB groups before any cropping; reserve nonempty splits."""
    settings.validate()
    records = list(records)
    groups = sorted({record.group_id for record in records})
    if len(groups) < 3:
        raise DatasetError(f"At least three independent groups are required; found {len(groups)}.")
    random.Random(settings.seed).shuffle(groups)
    train_count = min(len(groups) - 2, max(1, round(len(groups) * settings.train_fraction)))
    validation_count = min(len(groups) - train_count - 1, max(1, round(len(groups) * settings.validation_fraction)))
    assignment = {group: "train" for group in groups[:train_count]}
    assignment.update({group: "validation" for group in groups[train_count:train_count + validation_count]})
    assignment.update({group: "test" for group in groups[train_count + validation_count:]})
    splits = {name: [record for record in records if assignment[record.group_id] == name] for name in ("train", "validation", "test")}
    assert_no_leakage(splits)
    return splits

def assert_no_leakage(splits: dict[str, list[PCBRecord]]) -> None:
    seen_groups, seen_images, seen_paths = {}, {}, {}
    for split, records in splits.items():
        for record in records:
            for value, seen, kind in [(record.group_id, seen_groups, "group"), (record.image_id, seen_images, "image"),
                                     (str(record.test_path.resolve()), seen_paths, "test path"), (str(record.template_path.resolve()), seen_paths, "template path")]:
                if value in seen and seen[value] != split:
                    raise DatasetError(f"Leakage: {kind} {value!r} appears in {seen[value]} and {split}.")
                seen[value] = split

def inspect_dataset(directory: str | Path, origin: str = "zero", group_by_board: bool = True) -> dict:
    """Return observed paths, raw annotation examples, label counts, and README evidence."""
    root = resolve_dataset_root(directory)
    records = load_dataset(root, origin, group_by_board)
    counts = Counter(annotation.source_label for record in records for annotation in record.annotations)
    readmes = sorted({*root.glob("README*"), *root.parent.glob("README*")})
    summary = {
        "dataset_root": str(root), "image_pairs": len(records), "images": 2 * len(records),
        "groups": len({record.group_id for record in records}), "annotation_schema": "five integer tokens: xmin ymin xmax ymax type",
        "coordinate_origin": origin, "counts_by_source_label": {str(key): counts.get(key, 0) for key in DEEPPCB_TO_CLASS},
        "counts_by_class": {CLASS_NAMES[value]: counts.get(key, 0) for key, value in DEEPPCB_TO_CLASS.items()},
        "missing_defect_ids": [key for key in DEEPPCB_TO_CLASS if not counts[key]],
        "mapping": {str(key): CLASS_NAMES[value] for key, value in DEEPPCB_TO_CLASS.items()},
        "samples": [{"test": str(record.test_path.relative_to(root)), "template": str(record.template_path.relative_to(root)),
                     "annotation": str(record.annotation_path.relative_to(root)), "group": record.group_id,
                     "raw_annotation_lines": record.annotation_path.read_text().splitlines()[:5]} for record in records[:6]],
        "readme_evidence": [{"path": str(path), "excerpt": path.read_text(encoding="utf-8", errors="replace")[:20000]} for path in readmes if path.is_file()],
        "validation_note": "Observed text schema and labels validated. Image decoding and all bounding boxes are validated by preprocessing. Confirm mapping/origin against upstream README before training."
    }
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=CONFIG.dataset_dir)
    parser.add_argument("--origin", choices=("zero", "one"), default=CONFIG.annotation_origin)
    parser.add_argument("--group-by-pair", action="store_true", help="Default groups by board group directory, a stronger leakage guard.")
    parser.add_argument("--output", type=Path, default=CONFIG.metrics_dir / "dataset_inspection.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        summary = inspect_dataset(args.dataset, args.origin, not args.group_by_pair)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.info("Inspected %s pairs and %s groups. Class counts: %s", summary["image_pairs"], summary["groups"], summary["counts_by_class"])
        LOGGER.info("Review raw paths, labels, and README evidence in %s before preprocessing.", args.output)
        if summary["missing_defect_ids"]:
            raise DatasetError(f"Not every defect category was found: missing {summary['missing_defect_ids']}.")
    except (DatasetError, OSError) as error:
        parser.exit(1, f"Dataset inspection failed: {error}\n")

if __name__ == "__main__":
    main()
