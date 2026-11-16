"""Download and safely extract the official DeepPCB source archive (no credentials)."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://codeload.github.com/tangsanli5201/DeepPCB/zip/refs/heads/master"

def download_dataset(destination: Path) -> Path:
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted = destination / "DeepPCB-master" / "PCBData"
    if extracted.is_dir() and any(extracted.rglob("*_test.*")):
        logging.info("Existing extracted dataset preserved: %s", extracted)
        return extracted
    archive = destination / "DeepPCB-source.zip"
    partial = destination / "DeepPCB-source.zip.part"
    if not archive.is_file():
        try:
            with urllib.request.urlopen(URL, timeout=30) as response, partial.open("wb") as output:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 5 * 1024**3:
                        raise ValueError("Download exceeded 5 GiB; check the upstream archive.")
                    output.write(chunk)
            partial.replace(archive)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlinks are unsupported: {member.filename}")
        package.extractall(destination)
    if not extracted.is_dir():
        raise ValueError("Archive extracted but PCBData is absent. Inspect upstream layout and supply --dataset manually.")
    (destination / "download_provenance.json").write_text(json.dumps({"source": URL, "repository": "https://github.com/tangsanli5201/DeepPCB", "archive": archive.name}, indent=2), encoding="utf-8")
    return extracted

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        logging.info("Dataset available at %s", download_dataset(args.destination))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        logging.error("Download failed: %s: %s", type(error).__name__, error)
        logging.error("Manual fallback: open https://github.com/tangsanli5201/DeepPCB, Code > Download ZIP; extract under data/raw; inspect the actual PCBData directory with python -m src.dataset_loader --dataset PATH.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
