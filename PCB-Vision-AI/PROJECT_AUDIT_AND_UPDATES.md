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
| Local training environment | TensorFlow 2.17.1 reports no usable GPU; only Intel UHD graphics is present | HIGH | Provide a GPU-only Colab workflow and refuse local CPU training |
| `src/preprocessing.py` | Initial split did not honor the benchmark's official trainval/test split files | CRITICAL | Preserve official test split and carve validation only from trainval |

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
| `project_cli.py` | Require TensorFlow-visible GPU before training; persist training source and split details | Avoid CPU fallback and improve provenance |
| `src/preprocessing.py` | Read official split lists, retain the official test set, save class weights | Ensure correct final evaluation |
| `src/dataset_loader.py` | Index nested images once for faster discovery; ignore split/README text files | Make real dataset scans efficient |
| `src/inference.py` | Discover an existing checkpoint and validate paired metadata/class ordering | Automatic compatible model loading |
| `app.py` | Decode uploads with PIL, display immediately, and infer automatically; show class confidence ahead of optional window coordinates; plot training history from checkpoint metadata | Fix preview, simplify inspection, and show recorded training curves |
| `tests/test_preprocessing.py` | Assert official split counts and separation | Verify test isolation |
| `README.md` | Describe GPU Colab training and ordinary dashboard use | Avoid documenting local CPU training |

## D. New files added

| New file | Purpose |
|---|---|
| `project_cli.py` | Explicit download, validate, synthetic, preprocess, train, and evaluate commands |
| `PROJECT_AUDIT_AND_UPDATES.md` | This audit and change record |
| `notebooks/PCB_Vision_AI_DeepPCB_GPU_Training.ipynb` | GPU-only Colab training, official held-out evaluation, real-image localization smoke test, Drive export |

## E. Preserved functionality
The custom three-block CNN, existing public data structures, synthetic benchmark generator, patch extraction, augmentation, board-level split, Keras checkpoint format, Streamlit navigation, Jupyter notebook, and test modules remain intact.

## F. Testing results
- Baseline `pytest -q`: 13 passed.
- Post-change `pytest -q`: 16 passed.
- Python compile/syntax check: passed across `app.py`, `project_cli.py`, `config`, `src`, and `tests`.
- `project_cli.py --help`: passed.
- Synthetic 2-pair generation and dataset validation smoke test: passed; 2 valid pairs and 6 defect boxes were reported.
- Real DeepPCB validation: 1,500 pairs, 10,013 defect annotations, no missing templates or empty annotations.
- Official split verification: 850 train, 150 validation, 500 test image pairs; IDs are disjoint.
- Colab notebook: JSON and Python cell syntax checks passed.
- TensorFlow 2.17.1 on this PC sees no GPU (`tf.config.list_physical_devices('GPU') == []`); no local CPU training was performed.
- A local `models/pcb_cnn_best.keras` checkpoint is present (51,517,470 bytes), alongside its metadata sidecar; the app log confirms successful checkpoint loading.
- The metadata reports DeepPCB training on `/physical_device:GPU:0`, 21 completed epochs, best epoch 14, and best validation accuracy 98.72%. These are recorded checkpoint metadata, not an independent training-log audit.
- Held-out test metrics have not been computed in this environment and are not reported as verified.
- The current image-preview failure was traced to Streamlit media responses returning HTTP 500; the dashboard now renders images inline rather than through that endpoint.
- The Model Performance learning-curves tab now reads the checkpoint's recorded history directly; the live page displays 21 epochs and 98.72% best validation accuracy at epoch 14.
- The missing-model test now uses an isolated empty model path, so it remains valid when a real checkpoint is installed locally.

## G. Remaining limitations
- The real DeepPCB repository is present locally but is untracked and intentionally not committed.
- The trained `.keras` checkpoint and metadata are local and intentionally not committed as model artifacts.
- Final held-out test accuracy and macro F1 remain unverified until evaluation is run against the official test split.
- Sliding-window patch classification is not anchor-based object detection and may produce coarse/localization-only results.
- Template-image alignment is preserved in the loader, but full-board template difference subtraction is not yet an independent detector.
- Normal patches remain statistically generated from regions outside annotations; they are not inferred from unknown IDs.
- Full training/evaluation requires a Colab-accessible CUDA GPU, sufficient GPU memory, and a real dataset.

## H. Exact execution commands
```cmd
cd C:\Users\hp\Desktop\Projects\DL\PCB-Vision-AI
venv\Scripts\activate.bat
python project_cli.py download --target data\raw\DeepPCB
python project_cli.py validate --raw-dir data\raw\DeepPCB
python project_cli.py preprocess --raw-dir data\raw\DeepPCB
```
Train and export using `notebooks\PCB_Vision_AI_DeepPCB_GPU_Training.ipynb` in a Colab GPU runtime. Copy the `.keras` checkpoint and matching `.metadata.json` to `models\` once.
```cmd
streamlit run app.py
venv\Scripts\python.exe -m pytest tests -v
```
