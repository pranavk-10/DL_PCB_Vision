# PCB Vision AI: Deep Learning-Based Automated PCB Defect Classification and Visual Inspection System Using CNN

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Academic Project](https://img.shields.io/badge/Discipline-Electronics%20%26%20Telecommunication-purple.svg)]()

> An end-to-end Automated Optical Inspection (AOI) Deep Learning system utilizing a custom Convolutional Neural Network (CNN) built from scratch to detect, classify, and visually localize manufacturing defects on Printed Circuit Boards (PCBs).

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Key Engineering Features](#-key-engineering-features)
4. [Dataset Information](#-dataset-information)
5. [Deep Learning Architecture](#-deep-learning-architecture)
6. [Project Directory Structure](#-project-directory-structure)
7. [Installation & Windows PowerShell Setup](#-installation--windows-powershell-setup)
8. [Workflow Execution Guide](#-workflow-execution-guide)
   - [Step 1: Dataset Preparation](#step-1-dataset-preparation)
   - [Step 2: Preprocessing & Anti-Leakage Splitting](#step-2-preprocessing--anti-leakage-splitting)
   - [Step 3: Model Training](#step-3-model-training)
   - [Step 4: Model Evaluation](#step-4-model-evaluation)
   - [Step 5: Launching Streamlit Web Dashboard](#step-5-launching-streamlit-web-dashboard)
   - [Step 6: Executing Unit Tests](#step-6-executing-unit-tests)
   - [Step 7: Running Jupyter Notebook](#step-7-running-jupyter-notebook)
9. [Empirical Research Experiments](#-empirical-research-experiments)
10. [Visual Defect Localization (Sliding Window & NMS)](#-visual-defect-localization)
11. [User Interface Screenshots](#-user-interface-screenshots)
12. [System Limitations](#-system-limitations)
13. [Future Scope](#-future-scope)
14. [Academic Citation & License](#-academic-citation--license)

---

## 🔭 Project Overview
Printed Circuit Boards (PCBs) are the core foundation of modern electronic devices. During photolithographic etching, chemical electroplating, and drilling, physical flaws emerge that break electrical connectivity or cause hazardous short circuits.

**PCB Vision AI** replaces error-prone manual microscopic inspection and fragile template subtraction with a robust **custom Convolutional Neural Network (CNN)**. It classifies $128 \times 128 \times 1$ image patches across 7 classes:
1. **Open Circuit** (electrical trace break)
2. **Short Circuit** (unintended solder/copper bridge)
3. **Mouse Bite** (edge erosion notch)
4. **Spur** (sharp conductor protrusion)
5. **Pinhole** (void inside copper track or pad)
6. **Spurious Copper** (unwanted isolated copper residue)
7. **Normal / Defect-Free** (pristine conductor track)

---

## ⚠️ Problem Statement
Rule-based optical inspection systems suffer from:
1. **Extreme Alignment Sensitivity**: Even $0.5^\circ$ board rotation or minor camera vibration causes false alarms during template subtraction.
2. **Illumination Vulnerability**: Ambient light changes cause pixel-difference artifacts.
3. **Binary Failure**: Traditional AOI indicates whether a difference exists, but cannot identify *what defect category* is present.

Our CNN solves these problems by learning **spatially invariant hierarchical features** directly from optical scans.

---

## ✨ Key Engineering Features
- **Trained From Scratch**: No pre-trained weights or black-box APIs used. Demonstrates fundamental understanding of backpropagation, non-linear activations, and loss optimization.
- **Anti-Leakage Data Partitioning**: Original PCB boards are partitioned *before* extracting patches, guaranteeing zero data leakage between train, validation, and test splits.
- **Controlled Data Augmentation**: Training-only spatial and photometric perturbations (rotations, flips, translation, brightness adjustments) that regularize the model without altering defect semantics.
- **Sliding-Window Visual Inspection**: Full-board scanning with Non-Maximum Suppression (NMS) to localize defects.
- **Interactive Streamlit Web Dashboard**: Real-time classification, probability distribution graphs, inspection overlays, and annotated image downloads.
- **Complete Self-Contained Jupyter Notebook**: 40-section comprehensive learning notebook containing complete underlying implementations and mathematical derivations.

---

## 📊 Dataset Information
The system natively supports the public **DeepPCB** benchmark dataset:
- **Repository**: [https://github.com/tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB)
- **Image Pairs**: ~1,500 aligned tested boards (`*_test.jpg`) and golden reference templates (`*_temp.jpg`).
- **Resolution**: $640 \times 640$ pixels, optical grayscale.
- **Annotations**: Bounding box coordinates with class integer identifiers: `x1 y1 x2 y2 type`.

*Offline Synthetic Benchmark:* For immediate offline testing and grading, `src/dataset_loader.py` includes a high-fidelity synthetic benchmark generator (`create_synthetic_pcb_benchmark`) that generates authentic circuit traces, SMD pads, via holes, and realistic stamped defects matching DeepPCB specifications.

---

## 🧠 Deep Learning Architecture

```
Input: 128x128x1 Grayscale Image Patch
  │
  ├── Block 1: Conv2D(32, 3x3, same) + ReLU ──> MaxPool2D(2x2, stride 2)  [Output: 64x64x32]
  │
  ├── Block 2: Conv2D(64, 3x3, same) + ReLU ──> MaxPool2D(2x2, stride 2)  [Output: 32x32x64]
  │
  ├── Block 3: Conv2D(128, 3x3, same) + ReLU ──> MaxPool2D(2x2, stride 2) [Output: 16x16x128]
  │
  └── Classification Head:
        ├── Flatten                                                        [Output: 32,768]
        ├── Dense(128, He Normal) + ReLU                                   [Output: 128]
        ├── Dropout(rate = 0.30)                                           [Output: 128]
        └── Dense(7, Softmax)                                              [Output: 7 Probabilities]
```

- **Total Trainable Parameters**: `4,288,007`
- **Loss Function**: Sparse Categorical Cross-Entropy
- **Optimizer**: Adam ($\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$)
- **Callbacks**: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`

---

## 📁 Project Directory Structure

```text
PCB-Vision-AI/
│
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Python package dependencies
├── README.md                   # Comprehensive project documentation
├── LICENSE                     # MIT Open Source License
├── .gitignore                  # Git tracking exclusion rules
│
├── config/
│   ├── __init__.py
│   └── config.py               # Central configuration & hyperparameters
│
├── data/
│   ├── raw/                    # Raw DeepPCB images & annotations
│   ├── processed/              # Preprocessed 128x128 .npz patch datasets
│   └── README.md
│
├── models/
│   ├── pcb_cnn_best.keras      # Best trained CNN model weights
│   └── README.md
│
├── notebooks/
│   └── PCB_Vision_AI_Complete.ipynb  # Self-contained 40-section Jupyter Notebook
│
├── src/
│   ├── __init__.py
│   ├── dataset_loader.py       # DeepPCB parser & benchmark generator
│   ├── preprocessing.py        # Patch extraction & anti-leakage splitting
│   ├── data_analysis.py        # EDA & statistical visualization
│   ├── model.py                # Custom CNN & Baseline ANN architectures
│   ├── train.py                # Training pipeline & experiment runners
│   ├── evaluate.py             # Evaluation metrics, confusion matrix, curves
│   ├── localization.py         # Sliding-window inspector & NMS
│   └── inference.py            # Real-time patch prediction engine
│
├── results/
│   ├── figures/                # Saved charts (loss, acc, confusion matrix)
│   ├── metrics/                # Evaluation and experiment JSON summaries
│   └── README.md
│
├── tests/
│   ├── test_dataset_loader.py  # Unit tests for parsing & boundary clamping
│   ├── test_preprocessing.py   # Unit tests for anti-leakage splitting
│   ├── test_model.py           # Unit tests for model forward pass & shapes
│   └── test_inference.py       # Unit tests for prediction outputs
│
└── docs/
    ├── PROJECT_REPORT.md       # 24-section formal academic project report
    ├── DEEP_LEARNING_THEORY.md # In-depth mathematical theory & derivations
    ├── MODEL_ARCHITECTURE.md   # Layer specifications, dimensions, equations
    ├── DATASET_DOCUMENTATION.md# DeepPCB structure, class IDs, anti-leakage
    ├── EXPERIMENTAL_RESULTS.md # Empirical analysis of 3 research experiments
    └── VIVA_QUESTIONS.md       # 45 viva exam questions & detailed answers
```

---

## 💻 Installation & Windows PowerShell Setup

Follow these exact commands in **Windows PowerShell**:

### 1. Open PowerShell and navigate to the project root
```powershell
cd c:\Users\hp\Desktop\Projects\DL\PCB-Vision-AI
```

### 2. Create a dedicated Python Virtual Environment
```powershell
py -3.11 -m venv venv
```

### 3. Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` and retry).*

### 4. Upgrade pip and install all dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Workflow Execution Guide (Windows CMD)

### Setup
```cmd
cd C:\Users\hp\Desktop\Projects\DL\PCB-Vision-AI
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 1: Download Real DeepPCB (explicit)
```cmd
python project_cli.py download --target data\raw\DeepPCB
```
For testing only, generate synthetic data with a separate explicit command:
```cmd
python project_cli.py synthetic --target data\raw\synthetic --num-pairs 120
```
Synthetic data must not be used as real DeepPCB training evidence.

### Step 2: Validate the Dataset
```cmd
python project_cli.py validate --raw-dir data\raw\DeepPCB
```

### Step 3: Preprocess with Leakage-Free Splits
```cmd
python project_cli.py preprocess --raw-dir data\raw\DeepPCB
```

### Step 4: Train the Custom CNN
```cmd
python project_cli.py train --epochs 25 --batch-size 32 --learning-rate 0.001
```

### Step 5: Evaluate the Untouched Test Split
```cmd
python project_cli.py evaluate
```

### Step 6: Launch the Streamlit Dashboard
```cmd
streamlit run app.py
```

### Step 7: Run Tests
```cmd
venv\Scripts\python.exe -m pytest tests -v
```

---

## 🔬 Empirical Research Experiments

| Experiment | Configuration | Core Research Finding |
| :--- | :--- | :--- |
| **Exp 1: ANN vs. CNN** | Baseline MLP (4.22M params) vs. Custom CNN (4.28M params) | 2D convolutions provide spatial translation equivariance; CNN outperforms ANN test accuracy by over 20%. |
| **Exp 2: Adam vs. SGD** | Adam ($\eta=0.001$) vs. SGD with Nesterov Momentum ($\eta=0.01$, $\gamma=0.9$) | Adam converges within 4-5 epochs due to per-parameter adaptive second moments. SGD yields smooth, monotonic descent. |
| **Exp 3: Augmentation** | Clean training patches vs. Photometric & spatial augmentation | Training augmentation narrows the train-validation generalization gap by over 60%, preventing coordinate memorization. |

---

## 📐 Visual Defect Localization
The sliding-window inspector extends patch-level classification to full $640 \times 640$ PCB images:
1. Divides the board into overlapping $128 \times 128$ windows with configurable stride (e.g., 48 px).
2. Runs batched CNN forward passes.
3. Filters predictions where predicted class $\ne$ Normal and confidence $\ge 0.70$.
4. Applies **Non-Maximum Suppression (NMS)** with an IoU threshold of 0.30 to merge duplicate boxes.
5. Renders color-coded bounding boxes, class names, and confidence tags.

---

## 🖼️ User Interface Screenshots
*(Visual placeholders — live application runs interactively via `streamlit run app.py`)*

- `[Screenshot Pending: Dashboard KPI & Architecture Overview]`
- `[Screenshot Pending: Single Patch Classification Result]`
- `[Screenshot Pending: Sliding-Window Localization Overlays on Full PCB]`
- `[Screenshot Pending: Confusion Matrix & Per-Class Performance Breakdown]`

---

## ⚠️ System Limitations
1. **Coarse Bounding Boxes**: Sliding window provides coarse localized bounding boxes determined by window stride, unlike single-shot anchor regressors (YOLO).
2. **Closed-World Taxonomy**: Can only classify the 7 predefined classes. Unseen defects are mapped to the nearest visual category.
3. **Severe Perspective Warping**: Significant optical rotations ($>20^\circ$) require pre-alignment against a reference golden template.

---

## 🔮 Future Scope
1. **YOLOv8 / Faster R-CNN Integration**: Transitioning to single-pass anchor-based object detection for real-time bounding box regression.
2. **Sub-Pixel Image Registration**: Incorporating SIFT/ORB homography alignment against CAD Gerber reference files.
3. **Edge AI Hardware Deployment**: Quantizing to INT8 TensorRT for execution on NVIDIA Jetson Orin Nano or Raspberry Pi 5.

---

## 📜 Academic Citation & License
This project is developed for academic demonstration and research in **Electronics and Telecommunication Engineering**. Distributed under the [MIT License](LICENSE).
