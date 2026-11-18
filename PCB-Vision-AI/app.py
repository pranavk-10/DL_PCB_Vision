"""
PCB Vision AI - Streamlit Web Application
An interactive, industrial-grade Visual Inspection and Defect Classification Dashboard.
Includes:
- Page 1: Home / Dashboard (Architecture, Defect Taxonomy, System Status)
- Page 2: PCB Image Inspection (Patch Classification & Sliding-Window Localization with Overlays)
- Page 3: Model Performance (Loss/Accuracy Curves, Confusion Matrix, Per-Class Reports, Experiments)
- Page 4: Model Information (Layer Explanations, Hyperparameters, Inductive Bias)
- Page 5: About (Academic Scope, Technical Stack, Future Directions)
"""

import sys
import io
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# Configure page layout and visual identity
st.set_page_config(
    page_title="PCB Vision AI | Industrial Inspection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add project root to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config.config import (
    DATASET_CONFIG,
    MODEL_CONFIG,
    LOCALIZATION_CONFIG,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
)
from src.inference import PCBDefectClassifier, preprocess_input_patch
from src.localization import SlidingWindowInspector
from src.model import TF_AVAILABLE

# Professional Industrial Inspection Theme CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0e2a47;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2b6cb0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-ok { background-color: #c6f6d5; color: #22543d; }
    .badge-warn { background-color: #feebc8; color: #7b341e; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_classifier():
    """Initializes and caches the PCB defect inference engine."""
    model_path = MODEL_CONFIG.cnn_model_path
    if not model_path.exists():
        # Check fallback best models
        candidates = list(MODELS_DIR.glob("*.keras"))
        if candidates:
            model_path = candidates[0]
    return PCBDefectClassifier(model_path=model_path)


classifier = get_classifier()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/circuit.png", width=70)
st.sidebar.title("PCB Vision AI")
st.sidebar.markdown("**Deep Learning Visual Inspection**")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home / Dashboard",
        "🔬 PCB Image Inspection",
        "📊 Model Performance",
        "🧠 Model Architecture & Theory",
        "ℹ️ About & System Specifications",
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Runtime Status")
if classifier.is_loaded():
    st.sidebar.markdown('<span class="status-badge badge-ok">● Trained CNN Loaded</span>', unsafe_allow_html=True)
    st.sidebar.caption(f"Weights: `{classifier.model_path.name}`")
else:
    st.sidebar.markdown('<span class="status-badge badge-warn">▲ No Compatible Model</span>', unsafe_allow_html=True)
    st.sidebar.caption("Run training or load a valid `.keras` checkpoint.")

st.sidebar.markdown(f"**TensorFlow Available:** `{'Yes' if TF_AVAILABLE else 'No'}`")
st.sidebar.markdown(f"**Target Classes:** `{len(DATASET_CONFIG.class_labels)}`")


# ==============================================================================
# PAGE 1: HOME / DASHBOARD
# ==============================================================================
if nav_choice == "🏠 Home / Dashboard":
    st.markdown('<div class="main-header">Automated PCB Defect Classification & Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An intelligent Computer Vision & Deep Learning system utilizing Custom Convolutional Neural Networks for Automated Optical Inspection (AOI) of Printed Circuit Boards.</div>', unsafe_allow_html=True)

    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Input Resolution</div>
            <div class="metric-value">128 × 128</div>
            <div class="metric-label">Single-Channel (Grayscale)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Defect Classes</div>
            <div class="metric-value">7 Classes</div>
            <div class="metric-label">6 Defects + 1 Normal</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Target Dataset</div>
            <div class="metric-value">DeepPCB</div>
            <div class="metric-label">1,500 Aligned Image Pairs</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Architecture</div>
            <div class="metric-value">Custom CNN</div>
            <div class="metric-label">Trained From Scratch</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Overview & Defect Taxonomy
    col_left, col_right = st.columns([1.1, 0.9])
    with col_left:
        st.subheader("📋 Industrial Defect Taxonomy")
        st.write(
            "During PCB manufacturing processes (photolithography, chemical etching, electroplating, and drilling), "
            "various electrical and geometric defects emerge. Our system classifies defects into 7 distinct categories:"
        )

        tax_data = [
            ("1. Open Circuit", "A complete disconnection or break along a conductive copper trace, preventing signal propagation."),
            ("2. Short Circuit", "An unintended conductive bridge or solder splash connecting two adjacent isolated traces."),
            ("3. Mouse Bite", "Localized surface erosion or notches along the trace edge, reducing cross-sectional area and causing impedance mismatch."),
            ("4. Spur", "A sharp copper protrusion radiating outward from a trace, risking insulation breakdown under high voltage."),
            ("5. Pinhole", "Small circular voids or missing copper within solid traces or SMD solder pads."),
            ("6. Spurious Copper", "Isolated, unwanted islands of copper substrate left behind due to incomplete chemical etching."),
            ("7. Normal / Defect-Free", "Flawless PCB conductor traces and ground planes conforming to industrial manufacturing tolerances.")
        ]
        for name, desc in tax_data:
            st.markdown(f"**{name}**: {desc}")

    with col_right:
        st.subheader("🔬 Inspection Methodology")
        st.info("""
        **Pipeline Overview:**
        1. **Optical Image Acquisition**: 640×640 industrial optical capture of tested circuit boards.
        2. **Patch Extraction & Anti-Leakage Partition**: Bounding box extraction with margin padding, partitioned at board level.
        3. **Hierarchical CNN Feature Extraction**:
           - **Block 1**: 32 filters (3×3) capture edges, lines, and trace contours.
           - **Block 2**: 64 filters (3×3) detect corners, notches, and bridges.
           - **Block 3**: 128 filters (3×3) build abstract defect geometry representations.
        4. **Softmax Classification**: Computes probability distribution over 7 classes.
        5. **Visual Localization**: Optional sliding-window scanning with Non-Maximum Suppression (NMS).
        """)

        st.subheader("🚀 Quick Start")
        st.markdown("""
        - Navigate to **🔬 PCB Image Inspection** to test a PCB patch or full board.
        - View loss curves and confusion matrices under **📊 Model Performance**.
        - Read backpropagation and mathematical theory under **🧠 Model Architecture & Theory**.
        """)


# ==============================================================================
# PAGE 2: PCB IMAGE INSPECTION
# ==============================================================================
elif nav_choice == "🔬 PCB Image Inspection":
    st.markdown('<div class="main-header">PCB Visual Defect Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a PCB patch or full board image for real-time defect classification and visual defect localization.</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Select Inspection Mode",
        ["🔍 Single Patch Classification (128×128)", "📐 Full Board Sliding-Window Localization (640×640)"],
        horizontal=True
    )

    st.markdown("---")

    if mode == "🔍 Single Patch Classification (128×128)":
        col_input, col_result = st.columns([1, 1.2])

        with col_input:
            st.subheader("1. Input Image Patch")
            uploaded_file = st.file_uploader("Upload Patch Image (PNG, JPG, BMP)", type=["jpg", "jpeg", "png", "bmp"])
            
            # Preset sample generator if no upload
            use_sample = st.checkbox("Or use built-in synthetic defect sample", value=(uploaded_file is None))

            img_to_process = None
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.image(img_to_process, caption="Uploaded Image Patch", use_container_width=True)
            elif use_sample:
                # Generate synthetic test patch with an Open Circuit
                sample_patch = np.full((128, 128), 240, dtype=np.uint8)
                cv2.line(sample_patch, (10, 64), (50, 64), 30, 10)  # Left trace
                cv2.line(sample_patch, (78, 64), (118, 64), 30, 10) # Right trace (Break in middle)
                # Add solder pad
                cv2.circle(sample_patch, (30, 30), 12, 30, -1)
                img_to_process = sample_patch
                st.image(img_to_process, caption="Synthetic Demo Patch (Open Circuit Break)", use_container_width=True)

        with col_result:
            st.subheader("2. Model Classification Result")
            if img_to_process is not None:
                if not classifier.is_loaded():
                    st.error("No compatible trained .keras model is loaded. Run `python project_cli.py train` before patch inference.")
                else:
                    with st.spinner("Executing CNN Forward Pass..."):
                        pred = classifier.predict_patch(img_to_process)

                    pred_class = pred["predicted_class"]
                    conf = pred["confidence"]
                    probs = pred["probabilities"]

                    is_normal = (pred_class == "Normal")
                    if is_normal:
                        st.success(f"**Predicted Status:** {pred_class} (Defect-Free) — Confidence: {conf * 100:.2f}%")
                    else:
                        st.error(f"**Defect Detected:** {pred_class} — Confidence: {conf * 100:.2f}%")

                    st.markdown("#### Class Probability Distribution")
                    for cls_name, prob in probs.items():
                        c_col1, c_col2 = st.columns([1.5, 3])
                        with c_col1:
                            st.caption(f"**{cls_name}**")
                        with c_col2:
                            st.progress(float(prob), text=f"{prob * 100:.1f}%")

                    with st.expander("Detailed Prediction Diagnostics"):
                        st.json({
                            "predicted_label": pred_class,
                            "class_index": pred["class_index"],
                            "confidence_score": float(conf),
                            "model_loaded": classifier.is_loaded()
                        })
            else:
                st.info("Upload an image patch to view CNN predictions.")

    else:
        # Full Board Sliding-Window Localization
        st.subheader("📐 Full Board Scanning & Defect Localization")
        st.write(
            "The sliding-window inspector tiles the full PCB image into overlapping 128×128 patches, "
            "passes them in parallel batches through the CNN, filters non-normal predictions, and consolidates "
            "overlapping detections using **Non-Maximum Suppression (NMS)**."
        )

        col_cfg, col_view = st.columns([1, 2])
        with col_cfg:
            st.markdown("#### Inspection Settings")
            stride = st.slider("Window Stride (pixels)", min_value=16, max_value=64, value=48, step=8,
                               help="Smaller stride increases localization resolution at higher computational cost.")
            conf_thresh = st.slider("Confidence Threshold", min_value=0.50, max_value=0.95, value=0.70, step=0.05,
                                    help="Minimum probability required to flag a suspicious patch.")
            iou_thresh = st.slider("NMS IoU Overlap Threshold", min_value=0.10, max_value=0.60, value=0.30, step=0.05,
                                   help="Suppresses candidate boxes overlapping more than this threshold.")

            full_file = st.file_uploader("Upload Full PCB Image (640×640)", type=["jpg", "png"], key="full_pcb")
            use_synthetic_board = st.checkbox("Use Demo Benchmark Board", value=(full_file is None))

        with col_view:
            target_board = None
            if full_file is not None:
                f_bytes = np.asarray(bytearray(full_file.read()), dtype=np.uint8)
                target_board = cv2.imdecode(f_bytes, cv2.IMREAD_COLOR)
            elif use_synthetic_board:
                # Generate sample 640x640 board
                board = np.full((640, 640), 240, dtype=np.uint8)
                # Traces
                cv2.line(board, (30, 150), (610, 150), 30, 10)
                # Defect 1: Open circuit cut
                cv2.rectangle(board, (280, 142), (310, 158), 240, -1)
                # Defect 2: Spur
                cv2.line(board, (30, 300), (610, 300), 30, 10)
                cv2.line(board, (420, 300), (435, 275), 30, 6)
                # Defect 3: Spurious copper
                cv2.circle(board, (180, 450), 12, 30, -1)
                target_board = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)

            if target_board is not None:
                st.image(target_board, caption="Input PCB Inspection Target", use_container_width=True)

                if st.button("🚀 Run Full Board Inspection Scan", type="primary"):
                    if not classifier.is_loaded():
                        st.error("No compatible trained .keras model is loaded. Run `python project_cli.py train` before full-board inspection.")
                    else:
                        inspector = SlidingWindowInspector(classifier.model)
                        inspector.config.iou_nms_threshold = iou_thresh

                        with st.spinner("Scanning PCB board with sliding window and NMS..."):
                            results = inspector.inspect_image(
                                target_board, stride=stride, conf_threshold=conf_thresh
                            )

                        st.success(f"Scan complete: {results['total_windows_scanned']} patches evaluated. {len(results['detections'])} defects localized.")
                        st.image(results["annotated_image"], caption="Annotated Inspection Result", use_container_width=True)

                        if results["detections"]:
                            st.markdown("#### Detected Defects Summary")
                            det_table = [
                                {
                                    "Defect Category": d["class_name"],
                                    "Confidence": f"{d['confidence'] * 100:.1f}%",
                                    "Bounding Box (x1, y1, x2, y2)": str(d["box"])
                                }
                                for d in results["detections"]
                            ]
                            st.table(det_table)

                        is_success, buffer = cv2.imencode(".png", results["annotated_image"])
                        if is_success:
                            st.download_button(
                                label="📥 Download Annotated Inspection Image",
                                data=buffer.tobytes(),
                                file_name="pcb_inspection_annotated.png",
                                mime="image/png"
                            )


# ==============================================================================
# PAGE 3: MODEL PERFORMANCE
# ==============================================================================
elif nav_choice == "📊 Model Performance":
    st.markdown('<div class="main-header">Model Performance & Empirical Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous evaluation metrics on the held-out test set, including confusion matrices, per-class classification reports, and experimental comparisons.</div>', unsafe_allow_html=True)

    # Check for existing evaluation metrics JSON
    metrics_path = METRICS_DIR / "final_evaluation_report.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    else:
        eval_data = None

    # Top Metric KPI row
    if eval_data and "overall" in eval_data:
        ov = eval_data["overall"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Accuracy", f"{ov.get('accuracy', 0.0) * 100:.2f}%")
        m2.metric("Macro Precision", f"{ov.get('macro_precision', 0.0) * 100:.2f}%")
        m3.metric("Macro Recall", f"{ov.get('macro_recall', 0.0) * 100:.2f}%")
        m4.metric("Macro F1-Score", f"{ov.get('macro_f1', 0.0) * 100:.2f}%")
    else:
        st.info("ℹ️ Evaluation metrics will update automatically after running model evaluation (`python -m src.evaluate`).")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Learning Curves",
        "🎯 Confusion Matrix",
        "📋 Per-Class Performance",
        "🧪 Experimental Studies"
    ])

    with tab1:
        st.subheader("Training and Validation Dynamics")
        col_l, col_r = st.columns(2)
        loss_img = FIGURES_DIR / "loss_curves.png"
        acc_img = FIGURES_DIR / "accuracy_curves.png"

        with col_l:
            if loss_img.exists():
                st.image(str(loss_img), caption="Cross-Entropy Loss vs. Epochs", use_container_width=True)
            else:
                st.warning("Loss curves plot will be generated during model training.")
        with col_r:
            if acc_img.exists():
                st.image(str(acc_img), caption="Accuracy (%) vs. Epochs", use_container_width=True)
            else:
                st.warning("Accuracy curves plot will be generated during model training.")

    with tab2:
        st.subheader("Confusion Matrix")
        cm_img = FIGURES_DIR / "confusion_matrix.png"
        cm_norm_img = FIGURES_DIR / "confusion_matrix_normalized.png"
        
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            if cm_img.exists():
                st.image(str(cm_img), caption="Raw Sample Counts Confusion Matrix", use_container_width=True)
            else:
                st.warning("Confusion matrix plot pending evaluation execution.")
        with c_m2:
            if cm_norm_img.exists():
                st.image(str(cm_norm_img), caption="Normalized Confusion Matrix (Recall)", use_container_width=True)
            else:
                st.warning("Normalized confusion matrix pending evaluation execution.")

    with tab3:
        st.subheader("Per-Class Precision, Recall, and F1-Score")
        per_class_img = FIGURES_DIR / "per_class_metrics.png"
        if per_class_img.exists():
            st.image(str(per_class_img), caption="Per-Class Performance Breakdown", use_container_width=True)

        if eval_data and "per_class" in eval_data:
            st.table(eval_data["per_class"])

    with tab4:
        st.subheader("Empirical Research Experiments")
        st.markdown("""
        The project evaluates three critical Deep Learning research questions:
        1. **Experiment 1 (ANN vs CNN)**: Demonstrating the translation invariance and spatial locality of 2D convolutions compared to flattened dense MLPs.
        2. **Experiment 2 (Adam vs SGD)**: Contrasting adaptive moment estimation with momentum-based stochastic gradient descent.
        3. **Experiment 3 (Data Augmentation)**: Quantifying the regularization benefit of defect-preserving photometric and spatial augmentations.
        """)

        exp1_file = METRICS_DIR / "experiment_1_ann_vs_cnn.json"
        exp2_file = METRICS_DIR / "experiment_2_adam_vs_sgd.json"
        exp3_file = METRICS_DIR / "experiment_3_augmentation.json"

        if exp1_file.exists():
            with open(exp1_file, "r", encoding="utf-8") as f:
                exp1 = json.load(f)
            st.markdown("#### Experiment 1: ANN vs Custom CNN")
            st.json(exp1)

        if exp2_file.exists():
            with open(exp2_file, "r", encoding="utf-8") as f:
                exp2 = json.load(f)
            st.markdown("#### Experiment 2: Adam vs SGD Optimizer")
            st.json(exp2)

        if exp3_file.exists():
            with open(exp3_file, "r", encoding="utf-8") as f:
                exp3 = json.load(f)
            st.markdown("#### Experiment 3: Data Augmentation Generalization")
            st.json(exp3)


# ==============================================================================
# PAGE 4: MODEL INFORMATION & THEORY
# ==============================================================================
elif nav_choice == "🧠 Model Architecture & Theory":
    st.markdown('<div class="main-header">Deep Learning Architecture & Foundations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive architectural specifications and academic mathematical foundations of the PCB Vision AI system.</div>', unsafe_allow_html=True)

    tab_arch, tab_math, tab_bias = st.tabs([
        "🏗️ Layer-by-Layer Architecture",
        "📐 Mathematical Foundations",
        "⚖️ Inductive Bias & Limitations"
    ])

    with tab_arch:
        st.subheader("Custom Convolutional Neural Network (CNN)")
        st.write(
            "The model is intentionally engineered from scratch without pretrained weights to demonstrate genuine mastery of "
            "convolutional feature representation and backpropagation."
        )

        st.markdown("""
        | Stage | Layer Name | Configuration | Output Dimensions | Receptive Field | Purpose |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        | **Input** | `pcb_patch_input` | Grayscale image patch | $128 \\times 128 \\times 1$ | $1 \\times 1$ | Raw optical reflectance intensity |
        | **Block 1** | `block1_conv` | 32 filters, $3 \\times 3$, padding='same' | $128 \\times 128 \\times 32$ | $3 \\times 3$ | Detects low-level edges, copper track borders |
        | | `block1_relu` | $\\text{ReLU}(x) = \\max(0, x)$ | $128 \\times 128 \\times 32$ | $3 \\times 3$ | Introduces essential non-linearity |
        | | `block1_pool` | MaxPool $2 \\times 2$, stride 2 | $64 \\times 64 \\times 32$ | $4 \\times 4$ | Downsamples spatial size, translation tolerance |
        | **Block 2** | `block2_conv` | 64 filters, $3 \\times 3$, padding='same' | $64 \\times 64 \\times 64$ | $8 \\times 8$ | Detects corners, trace bends, spur protrusions |
        | | `block2_relu` | $\\text{ReLU}(x) = \\max(0, x)$ | $64 \\times 64 \\times 64$ | $8 \\times 8$ | Non-linearity |
        | | `block2_pool` | MaxPool $2 \\times 2$, stride 2 | $32 \\times 32 \\times 64$ | $10 \\times 10$ | Downsamples spatial resolution |
        | **Block 3** | `block3_conv` | 128 filters, $3 \\times 3$, padding='same' | $32 \\times 32 \\times 128$ | $20 \\times 20$ | Complex defect shapes (mousebites, breaks, pinholes) |
        | | `block3_relu` | $\\text{ReLU}(x) = \\max(0, x)$ | $32 \\times 32 \\times 128$ | $20 \\times 20$ | Non-linearity |
        | | `block3_pool` | MaxPool $2 \\times 2$, stride 2 | $16 \\times 16 \\times 128$ | $24 \\times 24$ | Downsamples to $16 \\times 16$ |
        | **Head** | `flatten_layer` | Vector unrolling | $32,768$ | Global | Flattens spatial feature volume into vector |
        | | `dense_features`| 128 Neurons | $128$ | Global | High-level non-linear feature synthesis |
        | | `dense_relu` | $\\text{ReLU}(x)$ | $128$ | Global | Non-linearity |
        | | `dropout_reg` | Dropout (rate = 0.30) | $128$ | Global | Prevents co-adaptation of neurons, reduces overfitting |
        | | `output_dense` | 7 Neurons + Softmax | $7$ | Global | Outputs class posterior probabilities $P(y=k \\mid X)$ |
        """)

    with tab_math:
        st.subheader("Mathematical Formulations")

        st.markdown("#### 1. 2D Discrete Convolution")
        st.latex(r"S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i - m, j - n) \cdot K(m, n) + b")
        st.write(
            "In CNNs, kernel weights $K(m, n)$ are shared across the spatial dimensions of input $I$, drastically reducing "
            "parameter count compared to fully-connected layers while achieving translation equivariance."
        )

        st.markdown("#### 2. Rectified Linear Unit (ReLU) Activation")
        st.latex(r"f(z) = \max(0, z), \quad \frac{\partial f}{\partial z} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}")
        st.write(
            "ReLU eliminates the vanishing gradient problem inherent in Sigmoid and Tanh activations for positive activations, "
            "and induces sparse representations."
        )

        st.markdown("#### 3. Softmax Output Function")
        st.latex(r"P(y = k \mid \mathbf{z}) = \frac{e^{z_k}}{\sum_{j=1}^{C} e^{z_j}}, \quad \text{where } \sum_{k=1}^C P(y=k \mid \mathbf{z}) = 1")

        st.markdown("#### 4. Categorical Cross-Entropy Loss")
        st.latex(r"\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{C} y_{i, k} \log(\hat{y}_{i, k})")

        st.markdown("#### 5. Adam Optimization (Adaptive Moment Estimation)")
        st.latex(r"m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2")
        st.latex(r"\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t")

    with tab_bias:
        st.subheader("Inductive Bias and Engineering Limitations")
        st.markdown("""
        **Why CNN over ANN for Vision:**
        - **Spatial Locality**: Adjacent pixels in PCB images share strong semantic correlation (conductive traces are continuous).
        - **Weight Sharing**: A filter detecting an open break or spur operates identically whether the defect is in the top-left or bottom-right corner.
        - **Parameter Efficiency**: A dense layer connecting $128 \\times 128$ inputs to 128 neurons requires $16,384 \\times 128 \\approx 2,097,152$ parameters, whereas Conv2D requires only $3 \\times 3 \\times 32 = 288$ parameters.

        **System Limitations:**
        - **Coarse Localization**: Patch-based sliding window is not a native single-shot anchor-based bounding box regressor (e.g., YOLO or Faster R-CNN). Bounding box precision is constrained by the window stride.
        - **Template Alignment**: In real industrial AOI, subtle mechanical vibration or perspective shift requires sub-pixel homography alignment before subtraction.
        """)


# ==============================================================================
# PAGE 5: ABOUT & SYSTEM SPECIFICATIONS
# ==============================================================================
elif nav_choice == "ℹ️ About & System Specifications":
    st.markdown('<div class="main-header">About PCB Vision AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Academic Context, Engineering Specifications, and Future Directions.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        st.subheader("🎓 Academic Context")
        st.markdown("""
        **Project Name**: PCB Vision AI — Deep Learning-Based Automated PCB Defect Classification and Visual Inspection System Using Convolutional Neural Networks (CNN).  
        **Discipline**: Electronics and Telecommunication Engineering (ETE) / Computer Vision & Deep Learning.  
        **Target Audience**: Academic presentation, technical viva examination, and industrial manufacturing engineering review.

        **Core Engineering Contributions:**
        1. Fully custom 3-block CNN designed, implemented, and trained from scratch without relying on pretrained transfer weights or generic black-box APIs.
        2. Strict image-level data partitioning protocol preventing data leakage between training, validation, and testing splits.
        3. Comprehensive empirical studies (ANN vs. CNN, Adam vs. SGD, Augmentation study).
        4. Functional sliding-window visual localization engine with Non-Maximum Suppression (NMS).
        5. Interactive Streamlit industrial inspection GUI.
        """)

        st.subheader("🛠️ Technology Stack")
        st.markdown("""
        - **Deep Learning Framework**: TensorFlow 2.x / Keras
        - **Computer Vision**: OpenCV (cv2)
        - **Scientific Computing**: NumPy, Scikit-learn
        - **Visualization**: Matplotlib, Seaborn
        - **Web Dashboard**: Streamlit
        - **Dataset**: Tang et al., *DeepPCB: A Dataset for PCB Defect Detection*
        """)

    with col2:
        st.subheader("🔮 Future Improvements")
        st.markdown("""
        1. **End-to-End Object Detection**: Integrate anchor-based detection (e.g. YOLOv8 / Faster R-CNN) directly on the full 640×640 board images for joint localization and classification in one pass.
        2. **Sub-Pixel Image Registration**: Incorporate ORB / SIFT feature matching with RANSAC homography estimation for real-time board alignment against reference CAD gerber files.
        3. **Edge AI Hardware Deployment**: Quantize the trained model via INT8 / FP16 TensorRT or ONNX Runtime for deployment on embedded edge devices (NVIDIA Jetson Orin Nano, Raspberry Pi 5).
        4. **Multi-Spectral Illumination**: Combine optical inspection with infrared thermography to identify latent electrical shorts before solder flow.
        """)

st.markdown("---")
st.caption("PCB Vision AI — Deep Learning Automated Visual Inspection System | Electronics and Telecommunication Engineering")
