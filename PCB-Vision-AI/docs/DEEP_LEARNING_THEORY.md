# Deep Learning Theoretical Foundations: PCB Vision AI

## 1. Artificial Neural Networks (ANN) Fundamentals

### 1.1 The Biological to Artificial Neuron Analogy
In electronics and biological systems, signals travel along pathways and are modulated at junctions:
- **Dendrites** receive inputs $x_1, x_2, \dots, x_n$.
- **Synaptic weights** $w_1, w_2, \dots, w_n$ represent connection strengths.
- **Soma (Cell Body)** computes the accumulated potential (weighted sum + bias).
- **Axon & Hillock** apply a threshold activation function to decide whether to fire a signal to subsequent neurons.

### 1.2 Mathematical Formulation of an Artificial Neuron
For an input vector $\mathbf{x} \in \mathbb{R}^n$, weight vector $\mathbf{w} \in \mathbb{R}^n$, and scalar bias $b \in \mathbb{R}$, the pre-activation potential $z$ is:
$$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

The final scalar activation output $a$ is produced by applying a non-linear activation function $\sigma(\cdot)$:
$$a = \sigma(z) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

### 1.3 Dense (Fully-Connected) Layer Representation
In a layer with $m$ neurons receiving $n$ inputs:
$$\mathbf{z} = \mathbf{W} \mathbf{x} + \mathbf{b}, \quad \text{where } \mathbf{W} \in \mathbb{R}^{m \times n}, \mathbf{b} \in \mathbb{R}^m$$
$$\mathbf{a} = \sigma(\mathbf{z})$$

---

## 2. Convolutional Neural Networks (CNN)

### 2.1 Why Flat Dense Networks Fail for Image Data
1. **Curse of Dimensionality**: A $128 \times 128 \times 1$ image flattened yields $16,384$ values. Connecting this to 256 neurons requires over $4.19 \times 10^6$ parameters in just one layer, inviting severe overfitting.
2. **Loss of Spatial Locality**: Flattening converts adjacent pixels into disconnected array indices, destroying spatial relationships.
3. **Lack of Translation Invariance**: If a defect (e.g., an Open Circuit break) shifts by 5 pixels, a dense network sees entirely new input weights and cannot generalize.

### 2.2 The 2D Convolution Operation
A convolution applies a small parameter matrix (kernel) $K \in \mathbb{R}^{k_h \times k_w}$ across the input feature map $I$:
$$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i + m, j + n) K(m, n) + b$$

- **Weight Sharing**: The same kernel weights $K$ are reused across every spatial location.
- **Sparse Connectivity**: Each output neuron connects only to a small local receptive field.

### 2.3 Padding and Stride
- **Stride ($s$)**: The step size of kernel traversal. Stride $s=1$ processes every pixel; stride $s=2$ downsamples the output grid by a factor of 2.
- **Padding ($p$)**: Border extension to preserve boundary information.
  - *Valid Padding*: $p=0$, output dimension shrinks:
    $$O = \left\lfloor \frac{W - K}{s} \right\rfloor + 1$$
  - *Same Padding*: Borders zero-padded so output dimension matches input dimension (when $s=1$):
    $$p = \frac{K - 1}{2}$$

### 2.4 Max-Pooling
Max-pooling applies a non-linear reduction over a local window (typically $2 \times 2$, stride 2):
$$P(i, j) = \max_{0 \le m, n < 2} X(2i + m, 2j + n)$$
- **Benefits**:
  1. Halves spatial height and width, cutting memory and computation by 75%.
  2. Imparts local translation invariance: small shifts in defect features do not alter the pooled maximum.
  3. Expands the receptive field of downstream convolutional layers.

---

## 3. Activation Functions

### 3.1 Rectified Linear Unit (ReLU)
$$\text{ReLU}(z) = \max(0, z)$$
$$\frac{d}{dz}\text{ReLU}(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$
- **Advantages**:
  1. **Constant Gradient (1.0)** for positive inputs: avoids the vanishing gradient problem of Sigmoid/Tanh.
  2. **Computational Speed**: Single comparison instruction without costly exponent calculations.
  3. **Sparse Representation**: Deactivates neurons with negative potentials, yielding efficient representations.

### 3.2 Softmax Function
For multi-class classification over $C$ mutually exclusive categories ($C=7$ in our PCB system):
$$P(y = k \mid \mathbf{z}) = \hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{C} e^{z_j}}, \quad \forall k \in \{0, \dots, C-1\}$$
- **Properties**:
  1. Output is strictly bounded: $0 < \hat{y}_k < 1$.
  2. Sums to unity: $\sum_{k=1}^C \hat{y}_k = 1.0$.
  3. Acts as a smooth, differentiable approximation of the argmax function.

---

## 4. Backpropagation and Loss Derivations

### 4.1 Categorical Cross-Entropy Loss
For one-hot ground-truth label vector $\mathbf{y} \in \{0, 1\}^C$ and model probability vector $\hat{\mathbf{y}}$:
$$\mathcal{L} = -\sum_{k=1}^{C} y_k \log(\hat{y}_k)$$

### 4.2 Gradient Derivation (Softmax + Cross-Entropy)
Using the chain rule:
$$\frac{\partial \mathcal{L}}{\partial z_i} = \sum_{k=1}^{C} \frac{\partial \mathcal{L}}{\partial \hat{y}_k} \frac{\partial \hat{y}_k}{\partial z_i}$$

1. $\frac{\partial \mathcal{L}}{\partial \hat{y}_k} = -\frac{y_k}{\hat{y}_k}$
2. Derivative of Softmax:
   - When $k = i$:
     $$\frac{\partial \hat{y}_i}{\partial z_i} = \hat{y}_i (1 - \hat{y}_i)$$
   - When $k \ne i$:
     $$\frac{\partial \hat{y}_k}{\partial z_i} = -\hat{y}_k \hat{y}_i$$

Combining into the summation:
$$\frac{\partial \mathcal{L}}{\partial z_i} = -\frac{y_i}{\hat{y}_i} \hat{y}_i (1 - \hat{y}_i) - \sum_{k \ne i} \frac{y_k}{\hat{y}_k} (-\hat{y}_k \hat{y}_i)$$
$$= -y_i (1 - \hat{y}_i) + \hat{y}_i \sum_{k \ne i} y_k = -y_i + y_i \hat{y}_i + \hat{y}_i (1 - y_i) = \hat{y}_i - y_i$$

In vector form, the error signal entering the classification head is simply:
$$\mathbf{\delta}_{\text{output}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}$$

This elegant cancellation ensures stable, non-saturating gradients during backpropagation.

---

## 5. Optimization Algorithms

### 5.1 Stochastic Gradient Descent (SGD) with Momentum
Standard SGD updates parameters $\theta$ along the instantaneous mini-batch gradient $\mathbf{g}_t = \nabla_\theta \mathcal{L}_t$:
$$\mathbf{v}_t = \gamma \mathbf{v}_{t-1} + \eta \mathbf{g}_t$$
$$\theta_{t+1} = \theta_t - \mathbf{v}_t$$
- Momentum coefficient $\gamma \in [0.9, 0.99]$ dampens oscillations along steep ravine directions while accelerating traversal along flat plateaus.

### 5.2 Adaptive Moment Estimation (Adam)
Adam combines momentum (first moment of gradients) with RMSProp (second raw moment of gradients):
1. **First Moment (Mean Gradient)**:
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
2. **Second Moment (Uncentered Variance)**:
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
3. **Bias Correction** (accounts for initialization at zero):
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
4. **Parameter Update**:
   $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
   Standard defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$, $\eta = 0.001$.

---

## 6. Regularization Strategies

### 6.1 Dropout
During each training forward pass, each neuron is independently zeroed out with probability $p = 0.30$:
$$r_j \sim \text{Bernoulli}(1 - p), \quad \tilde{\mathbf{a}} = \frac{1}{1 - p} (\mathbf{a} \odot \mathbf{r})$$
- The scaling factor $\frac{1}{1 - p}$ (Inverted Dropout) maintains expected activation magnitude during training, allowing inference to run unmodified without weight rescaling.
- Prevents co-adaptation of neurons by forcing each feature detector to be independently useful.

### 6.2 Data Augmentation as Implicit Regularization
Augmentations (rotations, flips, slight shifts) mathematically act as an expanded integration over the domain of transformation group invariants:
$$\mathbb{E}_{T \sim \mathcal{T}} [\mathcal{L}(f(T(\mathbf{x})), y)]$$
This forces the learned decision surface to be smooth with respect to label-preserving optical variations.
