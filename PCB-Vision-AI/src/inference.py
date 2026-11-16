"""Load provenance-checked trained models and perform consistent patch inference."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from config.config import CLASS_NAMES, CONFIG, Settings
from src.model import load_keras_model
from src.preprocessing import preprocess_image

@dataclass(frozen=True)
class PatchPrediction:
    class_id: int
    class_name: str
    confidence: float
    probabilities: tuple[float, ...]

def validate_probabilities(probabilities: np.ndarray, expected_samples: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (expected_samples, 7):
        raise ValueError(f"Expected ({expected_samples}, 7) probabilities; received {probabilities.shape}.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Model returned nonfinite or out-of-range probabilities.")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Model probabilities do not sum to one.")
    return probabilities

def decode_uploaded_image(data: bytes, max_pixels: int = CONFIG.max_image_pixels) -> np.ndarray:
    if not data:
        raise ValueError("The image is empty.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}:
                raise ValueError("Supported formats: JPEG, PNG, BMP, TIFF, WebP.")
            if image.width * image.height > max_pixels:
                raise ValueError(f"Image exceeds the {max_pixels:,}-pixel limit.")
            image.load()
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(f"Invalid or corrupted image: {error}") from error

class Predictor:
    """A local completed training artifact; never substitutes random weights."""
    def __init__(self, model_path: str | Path, settings: Settings = CONFIG):
        self.path = Path(model_path).resolve()
        self.settings = settings
        metadata_path = self.path.parent / "metadata.json"
        if not self.path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Trained model and metadata required at {self.path.parent}. Run python -m src.train.")
        self.metadata = json.loads(metadata_path.read_text())
        if self.metadata.get("status") != "trained":
            raise ValueError("Model metadata does not describe a completed training run.")
        if tuple(self.metadata.get("class_names", [])) != settings.class_names:
            raise ValueError("Model label mapping differs from project configuration.")
        if self.metadata.get("model_sha256") != hashlib.sha256(self.path.read_bytes()).hexdigest():
            raise ValueError("Model checksum differs from its training metadata.")
        trained_settings = self.metadata.get("settings", {})
        if trained_settings.get("image_size") != settings.image_size or trained_settings.get("channels") != 1:
            raise ValueError("Model preprocessing dimensions differ from current configuration.")
        self.model = load_keras_model(self.path)
        if tuple(self.model.input_shape[1:]) != (128, 128, 1) or self.model.output_shape[-1] != 7:
            raise ValueError("Model must accept 128x128x1 and produce seven probabilities.")

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        patches = np.asarray(patches, dtype=np.float32)
        if patches.ndim != 4 or patches.shape[1:] != (128, 128, 1) or len(patches) == 0:
            raise ValueError("Batch must have nonempty shape (N,128,128,1).")
        if not np.isfinite(patches).all() or patches.min() < 0 or patches.max() > 1:
            raise ValueError("Batch pixels must be finite and normalized to [0,1].")
        return validate_probabilities(self.model(patches, training=False).numpy(), len(patches))

    def predict_patch(self, image: np.ndarray) -> PatchPrediction:
        patch = preprocess_image(image, self.settings.image_size)
        probabilities = self.predict_batch(patch[np.newaxis])[0]
        class_id = int(probabilities.argmax())
        return PatchPrediction(class_id, CLASS_NAMES[class_id], float(probabilities[class_id]), tuple(float(value) for value in probabilities))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=CONFIG.model_path)
    args = parser.parse_args()
    try:
        result = Predictor(args.model).predict_patch(decode_uploaded_image(args.image.read_bytes()))
        print(json.dumps({"class_id": result.class_id, "class": result.class_name, "confidence": result.confidence,
                          "probabilities": dict(zip(CLASS_NAMES, result.probabilities)), "scope": "patch classification; full boards use localization"}, indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Inference failed: {error}\n")
if __name__ == "__main__":
    main()
