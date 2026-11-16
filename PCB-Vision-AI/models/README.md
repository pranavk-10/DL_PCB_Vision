# PCB Vision AI - Trained Model Weights

This directory holds serialized TensorFlow/Keras model checkpoints (`.keras` format):
- `pcb_cnn_best.keras`: Checkpointed best model weights monitored on minimum validation loss (`val_loss`).
- `pcb_cnn_final.keras`: Final weights after full epoch completion.
- `pcb_ann_baseline.keras`: Baseline Artificial Neural Network (MLP) for Experiment 1.
- `pcb_cnn_sgd.keras`: CNN trained with SGD + Momentum for Experiment 2.
- `pcb_cnn_no_aug.keras`: CNN trained without data augmentation for Experiment 3.

All models can be loaded via:
```python
from tensorflow.keras.models import load_model
model = load_model("models/pcb_cnn_best.keras")
```
