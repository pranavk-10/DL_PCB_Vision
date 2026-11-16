# Dataset Documentation: DeepPCB & Benchmark Generation

## 1. DeepPCB Dataset Overview
The primary benchmark utilized in this project is the **DeepPCB** dataset published by Tang et al. (Institute of Automation, Chinese Academy of Sciences):
- **Official Repository**: [https://github.com/tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB)
- **Source**: High-resolution optical scans of linear circuit boards tested on automated optical inspection (AOI) lines.
- **Dataset Composition**: Approximately 1,500 aligned PCB image pairs.
  - `{id}_test.jpg`: Tested PCB containing authentic or stamped defects.
  - `{id}_temp.jpg`: Defect-free reference template image aligned to the test board.
  - `{id}.txt`: Defect ground-truth annotation file.
- **Original Image Resolution**: $640 \times 640$ pixels, single-channel / optical grayscale intensity.

---

## 2. Defect Taxonomy and Class Mapping

The dataset annotates six primary manufacturing defect types. For our classification task, we add a seventh category representing **Normal (Defect-Free)** regions:

| Raw DeepPCB ID | Defect Category | Physical Manufacturing Root Cause | Target Index |
| :---: | :--- | :--- | :---: |
| **1** | **Open Circuit** | Physical break/scratch in copper trace due to foreign debris during exposure or over-etching. | **0** |
| **2** | **Short Circuit** | Unintended solder bridge or copper residue between parallel conductors. | **1** |
| **3** | **Mouse Bite** | Edge erosion notches caused by chemical over-etching or shearing stress along board edges. | **2** |
| **4** | **Spur** | Sharp copper protrusion sticking out from track due to improper mask alignment or burrs. | **3** |
| **5** | **Spurious Copper** | Residual copper island left on FR4 substrate due to incomplete etching bath dissolution. | **5** |
| **6** | **Pinhole** | Circular void within solid copper pads caused by gas bubbles trapped during electroplating. | **4** |
| **-** | **Normal / Defect-Free** | Non-defective, pristine conductor traces conforming to design rules. | **6** |

---

## 3. Ground-Truth Annotation File Format

Each `{id}.txt` file contains one line per defect box:
```text
x1 y1 x2 y2 type
```
or comma-separated:
```text
x1, y1, x2, y2, type
```

- `(x1, y1)`: Top-left bounding box pixel coordinate ($0 \le x_1 < x_2 \le 640$).
- `(y1, y2)`: Bottom-right bounding box pixel coordinate ($0 \le y_1 < y_2 \le 640$).
- `type`: Integer identifier (1 through 6).

### Coordinate Validation & Clamping
Our `DeepPCBParser` enforces:
1. Coordinate ordering: if $x_1 > x_2$, swap to ensure proper box geometry.
2. Boundary clamping: $x \in [0, 640]$, $y \in [0, 640]$.
3. Degenerate box filtering: filters out zero-area or single-pixel lines ($\Delta x \le 2$ or $\Delta y \le 2$).

---

## 4. Anti-Leakage Image-Level Dataset Partitioning

### The Data Leakage Risk in Patch-Based Inspection
If a user extracts all defect patches from 1,500 images first and then randomly shuffles them into 70% train and 30% test sets:
- Patches from the *same* physical PCB board will appear simultaneously in both training and test sets.
- The model memorizes board-specific substrate texture, illumination non-uniformity, and trace layout quirks.
- This creates **artificially inflated test accuracy** that fails catastrophically when deployed on new boards.

### Our Solution
We split the **original PCB images** first:
- **Training Set (70%)**: Images $\{S_{\text{train}}\}$
- **Validation Set (15%)**: Images $\{S_{\text{val}}\}$
- **Testing Set (15%)**: Images $\{S_{\text{test}}\}$

$$\{S_{\text{train}}\} \cap \{S_{\text{val}}\} = \emptyset, \quad \{S_{\text{train}}\} \cap \{S_{\text{test}}\} = \emptyset, \quad \{S_{\text{val}}\} \cap \{S_{\text{test}}\} = \emptyset$$

All defect and normal patches extracted from board $i$ remain strictly confined to that board's assigned split.

---

## 5. Offline Synthetic Benchmark Generation
To ensure the project is immediately runnable and testable offline without waiting for the external 1.5 GB GitHub repository clone:
- The system includes `create_synthetic_pcb_benchmark` in `src/dataset_loader.py`.
- It generates mathematically and physically consistent $640 \times 640$ PCB image pairs containing horizontal trace buses, IC pads, via holes, and realistic stamped defect instances across all 6 defect classes, complete with exact matching TXT annotations.
