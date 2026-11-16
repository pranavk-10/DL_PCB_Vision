# PCB Vision AI — PowerShell Command Cheat Sheet & Execution Reference

This document provides exact, copy-paste **Windows PowerShell** commands for every stage of the **PCB Vision AI** system lifecycle.

---

## 📋 Table of Contents
1. [Virtual Environment Setup & Activation](#1-virtual-environment-setup--activation)
2. [Dependency Installation & Environment Verification](#2-dependency-installation--environment-verification)
3. [Dataset Ingestion & Benchmark Generation](#3-dataset-ingestion--benchmark-generation)
4. [Data Preprocessing & Anti-Leakage Splitting](#4-data-preprocessing--anti-leakage-splitting)
5. [Model Training Commands](#5-model-training-commands)
6. [Empirical Research Experiments (ANN vs CNN, Adam vs SGD, Augmentation)](#6-empirical-research-experiments)
7. [Model Evaluation & Diagnostic Plot Generation](#7-model-evaluation--diagnostic-plot-generation)
8. [Sliding-Window Visual Localization](#8-sliding-window-visual-localization)
9. [Launching the Streamlit Web Application](#9-launching-the-streamlit-web-application)
10. [Running the Test Suites](#10-running-the-test-suites)
11. [Launching the Jupyter Notebook](#11-launching-the-jupyter-notebook)
12. [Maintenance, Cleanup & Deactivation](#12-maintenance-cleanup--deactivation)

---

## 1. Virtual Environment Setup & Activation

### Navigate to Project Directory
```powershell
cd c:\Users\hp\Desktop\Projects\DL\PCB-Vision-AI
```

### Create Virtual Environment (Using Python 3.11)
```powershell
py -3.11 -m venv venv
```

### Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(You will see `(venv)` prepended to your prompt).*

### Troubleshooting Execution Policy Error
If PowerShell states that `running scripts is disabled on this system`, execute this once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

---

## 2. Dependency Installation & Environment Verification

### Upgrade Pip
```powershell
python -m pip install --upgrade pip
```

### Install Project Dependencies
```powershell
pip install -r requirements.txt
```

### Verify Environment & Installed Package Versions
```powershell
python check_env.py
```

### Check Hardware Acceleration (GPU / CUDA Availability)
```powershell
python -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))"
```

---

## 3. Dataset Ingestion & Benchmark Generation

### Option A: Generate Built-in Offline Benchmark Dataset (Fastest)
Generates 60 physically consistent, $640 \times 640$ PCB image pairs with authentic traces and stamped defects across all 6 DeepPCB defect classes:
```powershell
python -c "from src.dataset_loader import create_synthetic_pcb_benchmark; create_synthetic_pcb_benchmark('data/raw', num_pairs=60)"
```

### Option B: Download Official DeepPCB Dataset via Git
```powershell
git clone https://github.com/tangsanli5201/DeepPCB.git temp_deeppcb
Copy-Item -Path "temp_deeppcb\PCBData\*" -Destination "data\raw" -Recurse -Force
Remove-Item -Path "temp_deeppcb" -Recurse -Force
```

---

## 4. Data Preprocessing & Anti-Leakage Splitting

### Run End-to-End Preprocessing Pipeline
Partitions images at the board level (70% Train, 15% Val, 15% Test), extracts padded $128 \times 128$ defect patches, synthesizes non-overlapping Normal patches, applies training-only augmentations, and saves arrays to `data/processed/pcb_dataset_processed.npz`:
```powershell
python -c "from src.preprocessing import PCBDataPipeline; pipeline = PCBDataPipeline(); pipeline.run_full_pipeline()"
```

### Run Exploratory Data Analysis (EDA) & Save Figures
Generates class distribution graphs and bounding box dimension plots in `results/figures/`:
```powershell
python -c "from src.dataset_loader import DeepPCBDatasetLoader; from src.data_analysis import run_full_eda; loader = DeepPCBDatasetLoader(); samples = loader.scan_dataset(); run_full_eda(samples)"
```

---

## 5. Model Training Commands

### Train Primary Custom CNN (Adam Optimizer)
Trains the 3-block CNN with `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau`:
```powershell
python -c "from src.preprocessing import PCBDataPipeline; from src.model import build_custom_cnn; from src.train import train_model; from config.config import DATA_PROCESSED_DIR; data = PCBDataPipeline.load_processed_dataset(DATA_PROCESSED_DIR / 'pcb_dataset_processed.npz'); model = build_custom_cnn(); train_model(model, data['X_train'], data['y_train'], data['X_val'], data['y_val'], epochs=15)"
```

### Train Baseline ANN (MLP) Architecture
```powershell
python -c "from src.preprocessing import PCBDataPipeline; from src.model import build_ann_baseline; from src.train import train_model; from config.config import DATA_PROCESSED_DIR, MODELS_DIR; data = PCBDataPipeline.load_processed_dataset(DATA_PROCESSED_DIR / 'pcb_dataset_processed.npz'); ann = build_ann_baseline(); train_model(ann, data['X_train'], data['y_train'], data['X_val'], data['y_val'], save_path=MODELS_DIR / 'pcb_ann_baseline.keras', epochs=15)"
```

---

## 6. Empirical Research Experiments

### Run Experiment 1: ANN vs. Custom CNN (Spatial Inductive Bias)
Trains both models under identical splits and exports results to `results/metrics/experiment_1_ann_vs_cnn.json`:
```powershell
python -c "from src.preprocessing import PCBDataPipeline; from src.train import run_experiment_1_ann_vs_cnn; from config.config import DATA_PROCESSED_DIR; data = PCBDataPipeline.load_processed_dataset(DATA_PROCESSED_DIR / 'pcb_dataset_processed.npz'); run_experiment_1_ann_vs_cnn(data, epochs=12)"
```

### Run Experiment 2: Adam Optimizer vs. SGD with Momentum
Compares adaptive per-parameter moments vs. momentum gradient descent, saving to `results/metrics/experiment_2_adam_vs_sgd.json`:
```powershell
python -c "from src.preprocessing import PCBDataPipeline; from src.train import run_experiment_2_adam_vs_sgd; from config.config import DATA_PROCESSED_DIR; data = PCBDataPipeline.load_processed_dataset(DATA_PROCESSED_DIR / 'pcb_dataset_processed.npz'); run_experiment_2_adam_vs_sgd(data, epochs=12)"
```

---

## 7. Model Evaluation & Diagnostic Plot Generation

### Run Full Test Set Evaluation & Generate Heatmaps
Evaluates model on held-out test data, producing `confusion_matrix.png`, `confusion_matrix_normalized.png`, `per_class_metrics.png`, and `final_evaluation_report.json`:
```powershell
python -c "from src.preprocessing import PCBDataPipeline; from src.inference import PCBDefectClassifier; from src.evaluate import run_full_evaluation; from config.config import DATA_PROCESSED_DIR; data = PCBDataPipeline.load_processed_dataset(DATA_PROCESSED_DIR / 'pcb_dataset_processed.npz'); clf = PCBDefectClassifier(); probs = clf.model.predict(data['X_test']) if clf.model else [clf.predict_patch(p)['raw_probabilities'] for p in data['X_test']]; run_full_evaluation(data['y_test'], probs)"
```

---

## 8. Sliding-Window Visual Localization

### Run Sliding-Window Inspection Scan on Full Board
Executes sliding window with Non-Maximum Suppression (NMS) on a target $640 \times 640$ PCB board:
```powershell
python -c "import cv2; from src.inference import PCBDefectClassifier; from src.localization import SlidingWindowInspector; clf = PCBDefectClassifier(); inspector = SlidingWindowInspector(clf.model if clf.model else clf); res = inspector.inspect_image('data/raw/synthetic_0000_test.jpg', stride=48, conf_threshold=0.70); cv2.imwrite('results/figures/inspection_scan_output.png', res['annotated_image']); print(f'Detected {len(res[\"detections\"])} defects. Saved overlay to results/figures/inspection_scan_output.png')"
```

---

## 9. Launching the Streamlit Web Application

### Standard Launch (Localhost on Port 8501)
```powershell
streamlit run app.py
```
*Your default browser will automatically open at `http://localhost:8501`.*

### Launch on a Specific Custom Port
```powershell
streamlit run app.py --server.port 8505
```

### Launch in Headless Server Mode
```powershell
streamlit run app.py --server.headless true
```

---

## 10. Running the Test Suites

### Run Built-in Comprehensive Unit Test Runner (Pure Python)
Executes all parsing, coordinate clamping, 7-class contract, and anti-leakage tests:
```powershell
python run_tests.py
```

### Run Tests via Pytest (Detailed Verbose Mode)
```powershell
pytest tests/ -v
```

### Run Only Dataset Loader & Parser Tests
```powershell
pytest tests/test_dataset_loader.py -v
```

### Run Only Anti-Leakage Preprocessing Tests
```powershell
pytest tests/test_preprocessing.py -v
```

---

## 11. Launching the Jupyter Notebook

### Launch Jupyter Lab
```powershell
jupyter lab notebooks/PCB_Vision_AI_Complete.ipynb
```

### Launch Classic Jupyter Notebook
```powershell
jupyter notebook notebooks/PCB_Vision_AI_Complete.ipynb
```

### Validate Notebook JSON Integrity Without Launching
```powershell
python -c "import json; nb = json.load(open('notebooks/PCB_Vision_AI_Complete.ipynb', 'r', encoding='utf-8')); print(f'Notebook valid! Contains {len(nb[\"cells\"])} cells covering all 40 sections.')"
```

---

## 12. Maintenance, Cleanup & Deactivation

### Clean Python Cache Files (`__pycache__`)
```powershell
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force
```

### Remove Generated Preprocessed Arrays (`data/processed/`)
```powershell
Remove-Item -Path "data\processed\*.npz" -Force -ErrorAction SilentlyContinue
```

### Deactivate Virtual Environment
```powershell
deactivate
```

---

## ⚡ Quick One-Liner Full Pipeline (From Scratch)
If you want to run the entire project setup in a single PowerShell sequence:
```powershell
.\venv\Scripts\Activate.ps1 ; python check_env.py ; python run_tests.py ; streamlit run app.py
```
