# PROJECT_AUDIT_AND_UPDATES

## A. Initial project analysis
The project is a custom 128×128×1 TensorFlow/Keras PCB patch classifier with seven classes (six DeepPCB defects plus Normal), DeepPCB parsing, board-level splitting, preprocessing, sliding-window localization, Streamlit UI, tests, and documentation. The baseline suite passed 13 tests before changes.

## B. Problems discovered

| File | Problem | Severity | Required fix |
|---|---|---|---|
| `src/dataset_loader.py` | Unknown annotation IDs were mapped to Normal and therefore trained as background | CRITICAL | Reject unknown IDs; never infer Normal from unrecognized annotations |
| `src/dataset_loader.py` | Dataset discovery depended on fixed annotation/image-name assumptions and did not validate corrupt tested images | HIGH | Support `_test`/`_temp` and base-name layouts; read source dimensions and skip corrupt images |
| `src/preprocessing.py` | Normal patches were generated only as blank regions and metadata did not preserve source-image split membership | HIGH | Retain board-level split metadata and keep normal extraction outside defect boxes |
| `src/inference.py` | Missing models silently used heuristic/fallback probabilities | CRITICAL | Require a loaded compatible checkpoint unless an explicit fallback mode is requested |
| `src/inference.py` | Loaded checkpoints were not shape-checked | HIGH | Validate 128×128×1 input and seven-class output |
| `src/localization.py` | NMS ignored class identity and border windows could be missed or duplicated | HIGH | Apply class-aware NMS and add deterministic right/bottom border windows |
| `app.py` | Dashboard ran heuristic or mock inspection without a trained model | HIGH | Show clear setup errors and prohibit mock localization |
| `src/train.py` | Training did not write a complete metadata sidecar | MEDIUM | Save architecture, classes, normalization, split seed, epochs, and best epoch metadata |
| `README.md` | Workflow examples used ad hoc Python expressions instead of verified CLI commands | MEDIUM | Document copy-paste Windows CMD commands matching `project_cli.py` |

## C. Files modified

| Existing file | Changes made | Reason |
|---|---|---|
| `src/dataset_loader.py` | Strict raw-ID mapping, robust pairing, corrupt-image checks, improved summary fields | Real dataset correctness |
| `src/preprocessing.py` | Class-weight fallback, split membership metadata in `.npz`, safe metadata loading | Reproducibility and graceful dependencies |
| `src/inference.py` | Explicit fallback opt-in and checkpoint shape validation | Honest inference behavior |
| `src/localization.py` | Border-window handling and class-aware NMS | Reduced duplicate/missing windows |
| `src/train.py` | Training metadata sidecar | Model reload reproducibility |
| `app.py` | Removed mock/heuristic full-board behavior and added missing-model errors | UI/runtime correctness |
| `tests/test_dataset_loader.py` | Added unknown-ID and real/synthetic integrity coverage | Prevent relabeling bugs |
| `tests/test_inference.py` | Changed tests to require trained models by default | Prevent fake inference claims |
| `README.md` | Replaced ad hoc workflow examples with verified Windows CMD CLI workflow | Reproducibility |

## D. New files added

| New file | Purpose |
|---|---|
| `project_cli.py` | Explicit download, validate, synthetic, preprocess, train, and evaluate commands |
| `PROJECT_AUDIT_AND_UPDATES.md` | This audit and change record |

## E. Preserved functionality
The custom three-block CNN, existing public data structures, synthetic benchmark generator, patch extraction, augmentation, board-level split, Keras checkpoint format, Streamlit navigation, Jupyter notebook, and test modules remain intact.

## F. Testing results
- Baseline `pytest -q`: 13 passed.
- Post-change `pytest -q`: 15 passed.
- Python compile/syntax check: passed across `app.py`, `project_cli.py`, `config`, `src`, and `tests`.
- `project_cli.py --help`: passed.
- Synthetic 2-pair generation and dataset validation smoke test: passed; 2 valid pairs and 6 defect boxes were reported.
- No full DeepPCB download or expensive model training was performed; no real model checkpoint or performance metrics were fabricated.

## G. Remaining limitations
- No verified DeepPCB repository has been downloaded in this environment; real dataset validation/training remains data-dependent.
- Sliding-window patch classification is not anchor-based object detection and may produce coarse/localization-only results.
- Template-image alignment is preserved in the loader, but full-board template difference subtraction is not yet an independent detector.
- Normal patches remain statistically generated from regions outside annotations; they are not inferred from unknown IDs.
- Full training/evaluation requires TensorFlow, sufficient CPU/GPU memory, and a real dataset.

## H. Exact execution commands
```cmd
cd C:\Users\hp\Desktop\Projects\DL\PCB-Vision-AI
venv\Scripts\activate.bat
python project_cli.py download --target data\raw\DeepPCB
python project_cli.py validate --raw-dir data\raw\DeepPCB
python project_cli.py preprocess --raw-dir data\raw\DeepPCB
python project_cli.py train --epochs 25 --batch-size 32 --learning-rate 0.001
python project_cli.py evaluate
streamlit run app.py
venv\Scripts\python.exe -m pytest tests -v
```
