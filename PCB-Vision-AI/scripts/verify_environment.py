"""Verify interpreter isolation, required imports, and basic package interoperability."""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
import site
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
REQUIRED = {
    "tensorflow": "tensorflow",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "streamlit": "streamlit",
    "jupyter": "jupyter",
    "notebook": "notebook",
    "ipykernel": "ipykernel",
    "pillow": "PIL",
}


def main() -> int:
    failures = []

    def report(name: str, success: bool, detail: str) -> None:
        print(f"{'PASS' if success else 'FAIL'} {name}: {detail}", flush=True)
        if not success:
            failures.append(name)

    report(".venv exists", VENV.is_dir(), str(VENV))
    report("Python version", sys.version_info[:2] == (3, 11), sys.version.split()[0])
    report("64-bit interpreter", sys.maxsize > 2**32, str(sys.maxsize > 2**32))
    report("interpreter", Path(sys.executable).resolve().is_relative_to(VENV.resolve()), sys.executable)
    report("virtual environment prefix", Path(sys.prefix).resolve() == VENV.resolve() and sys.prefix != sys.base_prefix, sys.prefix)
    report("user site disabled", site.ENABLE_USER_SITE is False, str(site.ENABLE_USER_SITE))
    config = VENV / "pyvenv.cfg"
    isolated = config.is_file() and "include-system-site-packages = false" in config.read_text().lower()
    report("system packages disabled", isolated, str(config))
    if failures:
        print("Run this script with .\\.venv\\Scripts\\python.exe; do not install globally.")
        return 1

    try:
        import pip
    except ImportError as error:
        report("pip location", False, f"{type(error).__name__}: {error}")
        print("Repair pip with .\\.venv\\Scripts\\python.exe -m ensurepip --upgrade")
        return 1
    report("pip location", Path(pip.__file__).resolve().is_relative_to(VENV.resolve()), pip.__file__)
    subprocess.run([sys.executable, "-m", "pip", "--version"], check=True)
    modules = {}
    expected_versions = dict(line.strip().split("==", 1) for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip() and not line.lstrip().startswith("#"))
    for package, module_name in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
            version = importlib.metadata.version(package)
            paths = [Path(p).resolve() for p in getattr(module, "__path__", [])]
            if getattr(module, "__file__", None):
                paths.append(Path(module.__file__).resolve())
            local = bool(paths) and all(p.is_relative_to(VENV.resolve()) for p in paths)
            report(f"import {module_name}", local, f"{package}=={version}; locations={paths}")
            report(f"version {package}", version == expected_versions[package], f"installed={version}; required={expected_versions[package]}")
            modules[module_name] = module
        except Exception as error:
            report(f"import {module_name}", False, f"{type(error).__name__}: {error}")

    result = subprocess.run([sys.executable, "-m", "pip", "check"], check=False)
    report("pip check", result.returncode == 0, f"exit code {result.returncode}")

    if not failures:
        try:
            np, tf, cv2 = modules["numpy"], modules["tensorflow"], modules["cv2"]
            pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
            gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
            assert gray.shape == (4, 4)
            tensor = tf.convert_to_tensor(gray, dtype=tf.float32)
            assert float(tf.reduce_sum(tensor).numpy()) == float(gray.sum())
            frame = modules["pandas"].DataFrame({"intensity": gray.flatten()})
            preprocessing = importlib.import_module("sklearn.preprocessing")
            scaled = preprocessing.StandardScaler().fit_transform(frame)
            assert np.isfinite(scaled).all()
            modules["matplotlib"].use("Agg")
            pyplot = importlib.import_module("matplotlib.pyplot")
            figure, axes = pyplot.subplots()
            axes.imshow(gray, cmap="gray")
            figure.canvas.draw()
            pyplot.close(figure)
            report("scientific package interoperability", True, "TensorFlow, NumPy, OpenCV, Pandas, Scikit-learn, and Matplotlib smoke test")
        except Exception as error:
            report("scientific package interoperability", False, f"{type(error).__name__}: {error}")
    else:
        print("SKIP interoperability: required imports or environment checks failed.")

    print(f"Verification {'FAILED' if failures else 'PASSED'}: {len(failures)} failed checks.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
