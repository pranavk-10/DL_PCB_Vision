# PCB Vision AI - Dataset Storage Guide

## Directory Structure
- `data/raw/`: Store raw DeepPCB image pairs (`*_test.jpg`, `*_temp.jpg`) and annotations (`*.txt`).
- `data/processed/`: Stores extracted and normalized $128 \times 128$ `.npz` arrays (`pcb_dataset_processed.npz`).

## How to Acquire the DeepPCB Dataset
1. Clone or download from GitHub:
   ```bash
   git clone https://github.com/tangsanli5201/DeepPCB.git
   ```
2. Copy the `PCBData` subfolder contents directly into `data/raw/`.
3. If internet access is not available, run our built-in benchmark generator:
   ```bash
   python -c "from src.dataset_loader import create_synthetic_pcb_benchmark; create_synthetic_pcb_benchmark('data/raw', num_pairs=60)"
   ```
