# Model Architecture Specifications: PCB Vision AI

## 1. Architectural Philosophy and Design Principles
The primary neural network of **PCB Vision AI** is a custom, three-block Convolutional Neural Network (CNN) engineered specifically for Automated Optical Inspection (AOI) of Printed Circuit Board (PCB) image patches.

Rather than leveraging bulky, generic pretrained networks (such as ImageNet-trained ResNet or VGG) which carry excessive parameters and lack inductive bias for synthetic binary/monochrome PCB traces, this architecture is designed **from scratch** to be:
1. **Computationally Lightweight**: Suitable for real-time edge deployment on automated manufacturing lines.
2. **Spatially Invariant**: Invariant to small shifts and rotational variations of PCB defects.
3. **Academically Transparent**: Clearly exposes the mathematical role of convolution, pooling, non-linearity, and regularization.

---

## 2. Layer-by-Layer Architectural Breakdown

The network receives a single-channel grayscale patch of dimensions $128 \times 128 \times 1$.

```
[Input: 128x128x1]
        │
┌───────┴───────┐
│    BLOCK 1    │  Conv2D (32 filters, 3x3, same) + ReLU ──> MaxPool2D (2x2, stride 2)
└───────┬───────┘  Output: 64x64x32  | Receptive Field: 4x4
        │
┌───────┴───────┐
│    BLOCK 2    │  Conv2D (64 filters, 3x3, same) + ReLU ──> MaxPool2D (2x2, stride 2)
└───────┬───────┘  Output: 32x32x64  | Receptive Field: 10x10
        │
┌───────┴───────┐
│    BLOCK 3    │  Conv2D (128 filters, 3x3, same) + ReLU ──> MaxPool2D (2x2, stride 2)
└───────┬───────┘  Output: 16x16x128 | Receptive Field: 24x24
        │
┌───────┴───────┐
│ CLASSIFIER    │  Flatten (32,768) ──> Dense (128) + ReLU ──> Dropout (0.3) ──> Dense (7, Softmax)
└───────────────┘  Output: 7 Class Posterior Probabilities
```

### Detailed Layer Specification Table

| Stage | Layer Name | Layer Type | Kernel / Units | Padding / Stride | Activation | Output Shape | Trainable Params |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | `pcb_patch_input` | InputLayer | - | - | - | $(128, 128, 1)$ | $0$ |
| **Block 1** | `block1_conv` | Conv2D | $32 \times (3 \times 3)$ | same, $s=1$ | Linear | $(128, 128, 32)$ | $320$ |
| | `block1_relu` | Activation | - | - | ReLU | $(128, 128, 32)$ | $0$ |
| | `block1_pool` | MaxPooling2D | $2 \times 2$ | valid, $s=2$ | - | $(64, 64, 32)$ | $0$ |
| **Block 2** | `block2_conv` | Conv2D | $64 \times (3 \times 3)$ | same, $s=1$ | Linear | $(64, 64, 64)$ | $18,496$ |
| | `block2_relu` | Activation | - | - | ReLU | $(64, 64, 64)$ | $0$ |
| | `block2_pool` | MaxPooling2D | $2 \times 2$ | valid, $s=2$ | - | $(32, 32, 64)$ | $0$ |
| **Block 3** | `block3_conv` | Conv2D | $128 \times (3 \times 3)$| same, $s=1$ | Linear | $(32, 32, 128)$| $73,856$ |
| | `block3_relu` | Activation | - | - | ReLU | $(32, 32, 128)$| $0$ |
| | `block3_pool` | MaxPooling2D | $2 \times 2$ | valid, $s=2$ | - | $(16, 16, 128)$| $0$ |
| **Head** | `flatten_layer` | Flatten | - | - | - | $(32,768)$ | $0$ |
| | `dense_features` | Dense | $128$ | - | Linear | $(128)$ | $4,194,432$ |
| | `dense_relu` | Activation | - | - | ReLU | $(128)$ | $0$ |
| | `dropout_reg` | Dropout ($p=0.3$)| - | - | - | $(128)$ | $0$ |
| | `output_dense` | Dense | $7$ | - | Softmax | $(7)$ | $903$ |
| **Total** | | | | | | | **4,288,007** |

---

## 3. Mathematical Parameter Count Derivations

### Block 1 Convolution
$$\text{Params} = (\text{kernel\_h} \times \text{kernel\_w} \times C_{\text{in}} + 1) \times C_{\text{out}} = (3 \times 3 \times 1 + 1) \times 32 = 10 \times 32 = 320$$

### Block 2 Convolution
$$\text{Params} = (3 \times 3 \times 32 + 1) \times 64 = (288 + 1) \times 64 = 289 \times 64 = 18,496$$

### Block 3 Convolution
$$\text{Params} = (3 \times 3 \times 64 + 1) \times 128 = (576 + 1) \times 128 = 577 \times 128 = 73,856$$

### Fully Connected Feature Layer (`dense_features`)
$$\text{Params} = (N_{\text{in}} + 1) \times N_{\text{out}} = (32,768 + 1) \times 128 = 32,769 \times 128 = 4,194,432$$

### Output Classification Layer (`output_dense`)
$$\text{Params} = (128 + 1) \times 7 = 129 \times 7 = 903$$

$$\text{Total Trainable Parameters} = 320 + 18,496 + 73,856 + 4,194,432 + 903 = \mathbf{4,288,007}$$

---

## 4. Receptive Field Progression
The effective receptive field (RF) describes the local window in the input image that influences a neuron at layer $l$:
$$RF_l = RF_{l-1} + (k_l - 1) \cdot S_{l-1}$$

1. **Input**: $RF_0 = 1 \times 1$, Stride $S_0 = 1$
2. **Conv1 ($3 \times 3$, stride 1)**: $RF_1 = 1 + (3 - 1) \times 1 = 3 \times 3$
3. **Pool1 ($2 \times 2$, stride 2)**: $RF_{p1} = 3 + (2 - 1) \times 1 = 4 \times 4$, Stride $S_1 = 2$
4. **Conv2 ($3 \times 3$, stride 1)**: $RF_2 = 4 + (3 - 1) \times 2 = 8 \times 8$
5. **Pool2 ($2 \times 2$, stride 2)**: $RF_{p2} = 8 + (2 - 1) \times 2 = 10 \times 10$, Stride $S_2 = 4$
6. **Conv3 ($3 \times 3$, stride 1)**: $RF_3 = 10 + (3 - 1) \times 4 = 18 \times 18$
7. **Pool3 ($2 \times 2$, stride 2)**: $RF_{p3} = 18 + (2 - 1) \times 4 = 22 \times 22$ (spanning up to $24 \times 24$ with padding).

This receptive field size aligns cleanly with the bounding box scale of typical PCB defects (mean defect width $\approx 28 \text{ px}$ to $36 \text{ px}$), allowing the network to capture both the local defect discontinuity and the surrounding conductive trace context.

---

## 5. Baseline ANN (MLP) Specification
For **Experiment 1 (ANN vs. CNN)**, the baseline Artificial Neural Network accepts the flattened 16,384-dimensional vector:
- **Input**: $128 \times 128 \times 1 \to 16,384$ pixels
- **Dense Layer 1**: $256$ Neurons + ReLU (Params: $(16,384 + 1) \times 256 = 4,194,560$)
- **Dropout**: $p = 0.30$
- **Dense Layer 2**: $128$ Neurons + ReLU (Params: $(256 + 1) \times 128 = 32,896$)
- **Dropout**: $p = 0.30$
- **Output Layer**: $7$ Neurons + Softmax (Params: $(128 + 1) \times 7 = 903$)
- **Total ANN Parameters**: $\mathbf{4,228,359}$

Notice that while the ANN and CNN have comparable parameter counts ($\sim 4.2\text{M}$), the CNN devotes the vast majority of its early capacity to spatially localized weight sharing, whereas the ANN's parameters are entirely in dense global matrix multiplications that destroy 2D spatial adjacency.
