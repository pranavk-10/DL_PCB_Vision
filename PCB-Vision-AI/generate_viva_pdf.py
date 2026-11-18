"""
PCB Vision AI - Comprehensive Academic Viva Voce Defense Guide (PDF Generator)
Author: Pranav Kamble (Roll No: 241090902)
Department of Electronics and Telecommunication Engineering
Generates a publication-grade academic viva preparation and defense handbook.
"""

import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Academic Color Palette
COLOR_PRIMARY = colors.HexColor("#1A365D")    # Deep Navy
COLOR_SECONDARY = colors.HexColor("#2B6CB0")  # Slate Blue
COLOR_ACCENT = colors.HexColor("#2C7A7B")     # Deep Teal
COLOR_DARK = colors.HexColor("#2D3748")       # Charcoal Body Text
COLOR_LIGHT_BG = colors.HexColor("#F7FAFC")   # Light Background
COLOR_BORDER = colors.HexColor("#E2E8F0")     # Light Gray Border
COLOR_MUTED = colors.HexColor("#718096")      # Muted Gray
COLOR_QUESTION_BG = colors.HexColor("#EBF8FF")# Pale Blue Question Card
COLOR_ANSWER_BG = colors.HexColor("#FFFFFF")  # Clean White Answer Card
COLOR_SUCCESS = colors.HexColor("#276749")    # Forest Green

class NumberedCanvas(canvas.Canvas):
    """Dynamic two-pass canvas computing running headers, footers and total page counts."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        if self._pageNumber > 1:
            self.saveState()
            
            # Running Header
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(COLOR_SECONDARY)
            self.drawString(54, 752, "PCB VISION AI: VIVA VOCE DEFENSE & TECHNICAL QUESTION BANK")
            self.setFont("Helvetica", 8)
            self.setFillColor(COLOR_MUTED)
            self.drawRightString(612 - 54, 752, "Pranav Kamble | Roll No: 241090902")
            
            self.setStrokeColor(COLOR_BORDER)
            self.setLineWidth(0.75)
            self.line(54, 744, 612 - 54, 744)
            
            # Running Footer
            self.line(54, 46, 612 - 54, 46)
            self.setFont("Helvetica", 8)
            self.setFillColor(COLOR_MUTED)
            self.drawString(54, 34, "Dept. of Electronics & Telecommunication Engineering")
            page_str = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(612 - 54, 34, page_str)
            
            self.restoreState()


def build_viva_guide():
    project_root = Path(__file__).parent.resolve()
    output_dir = project_root / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "PCB_Vision_AI_Viva_Questions_Pranav_Kamble.pdf"
    root_pdf_path = project_root / "PCB_Vision_AI_Viva_Questions_Pranav_Kamble.pdf"
    parent_pdf_path = project_root.parent / "PCB_Vision_AI_Viva_Questions_Pranav_Kamble.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    base_styles = getSampleStyleSheet()

    # Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=base_styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        textColor=COLOR_PRIMARY,
        alignment=1,
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=11.5,
        leading=16,
        textColor=COLOR_SECONDARY,
        alignment=1,
        spaceAfter=20
    )

    author_block_style = ParagraphStyle(
        "CoverAuthorBlock",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=15,
        textColor=COLOR_DARK,
        alignment=1
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13.5,
        leading=17,
        textColor=COLOR_PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=COLOR_SECONDARY,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.5,
        textColor=COLOR_DARK,
        spaceAfter=6
    )

    q_title_style = ParagraphStyle(
        "QuestionTitle",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.2,
        leading=13,
        textColor=COLOR_PRIMARY
    )

    ans_style = ParagraphStyle(
        "AnswerText",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8.6,
        leading=12.2,
        textColor=COLOR_DARK
    )

    ans_bold = ParagraphStyle(
        "AnswerBold",
        parent=ans_style,
        fontName="Helvetica-Bold"
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.2,
        leading=10.5,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=COLOR_DARK
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold"
    )

    story = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("ACADEMIC VIVA VOCE DEFENSE HANDBOOK & TECHNICAL QUESTION BANK", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("PCB VISION AI: TECHNICAL DEFENSE GUIDE", title_style))
    story.append(Paragraph("45 In-Depth Engineering & Deep Learning Questions with Mathematical Derivations and Defense Strategies", subtitle_style))
    story.append(HRFlowable(width="80%", thickness=2, color=COLOR_SECONDARY, spaceBefore=8, spaceAfter=22))
    
    story.append(Spacer(1, 35))
    author_info = """
    <b>Candidate / Examinee:</b> Pranav Kamble<br/>
    <b>Student Roll Number:</b> 241090902<br/>
    <b>Degree Program:</b> Bachelor of Technology (B.Tech)<br/>
    <b>Department:</b> Electronics and Telecommunication Engineering<br/>
    <b>Course / Domain:</b> Deep Learning, Computer Vision & Industrial Quality Control<br/>
    <b>Project Title:</b> PCB Vision AI: Deep Learning-Based Automated PCB Defect Classification and Visual Inspection System Using CNN<br/>
    <b>Academic Session:</b> 2026-2027
    """
    story.append(Paragraph(author_info, author_block_style))
    story.append(Spacer(1, 45))

    viva_summary_box = [
        [Paragraph("<b>Viva Voce Structure</b>", table_cell_bold), Paragraph("Comprehensive 9-Domain Technical Assessment Matrix", table_cell_style)],
        [Paragraph("<b>Target Audience</b>", table_cell_bold), Paragraph("Project External Examiner, Internal Viva Committee, Technical Review Board", table_cell_style)],
        [Paragraph("<b>Core Focus Areas</b>", table_cell_bold), Paragraph("CNN Mathematics, Inductive Bias, Receptive Fields, Loss Gradients, Anti-Leakage Protocol, Sliding Window NMS, Real-World EMS Deployability", table_cell_style)],
        [Paragraph("<b>Key Artifacts Grounded</b>", table_cell_bold), Paragraph("DeepPCB Dataset (Tang et al.), Custom 3-Block CNN (4.28M params), 97.37% Test Accuracy", table_cell_style)]
    ]
    v_table = Table(viva_summary_box, colWidths=[140, 364])
    v_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, COLOR_SECONDARY),
        ('INNERGRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 9),
        ('RIGHTPADDING', (0,0), (-1,-1), 9),
    ]))
    story.append(v_table)

    story.append(PageBreak())

    # =========================================================================
    # VIVA DEFENSE STRATEGY & TIPS FOR PRANAV
    # =========================================================================
    story.append(Paragraph("Strategic Viva Voce Defense Framework for Pranav Kamble", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceBefore=4, spaceAfter=10))

    strategy_text = """
    During an academic project defense in Electronics and Telecommunication Engineering, examiners evaluate three fundamental pillars: 
    <b>(1) Domain Mastery</b> (understanding why electronics manufacturing requires computer vision), 
    <b>(2) Theoretical Rigor</b> (explaining the underlying mathematics of convolution, backpropagation, and loss functions rather than quoting high-level library calls), and 
    <b>(3) Engineering Authenticity</b> (demonstrating personal command over data integrity, failure modes, trade-offs, and design limitations).
    <br/><br/>
    <b>Three Golden Rules for Your Defense:</b>
    <br/>
    1. <b>Never rely on pre-trained buzzwords:</b> If asked why you did not use ResNet or VGG, state clearly: <i>"Pretrained models on ImageNet contain 25-100 million parameters trained on natural RGB photos (animals, vehicles, faces). PCB traces are binary/monochrome high-frequency geometric contours. A heavy transfer network is computationally redundant for edge deployment and acts as an uninterpretable black box. Designing a custom 3-block CNN from scratch with 4.28M parameters allowed me to mathematically tune receptive fields ($24\times24$) to match physical defect dimensions (28-36 pixels) while achieving 97.37% test accuracy."</i>
    <br/><br/>
    2. <b>Highlight the Anti-Leakage Protocol immediately:</b> When examiners ask about your high accuracy, emphasize that you enforced a <b>strict board-level train/test split</b> before patch extraction. Point out that naive patch-level splitting would allow patches from the same PCB board to contaminate both train and test sets, inflating accuracy falsely. This proves deep engineering maturity.
    <br/><br/>
    3. <b>Clearly separate Classification from Localization:</b> Acknowledge that the core CNN is an inductive patch classifier. The sliding-window with Non-Maximum Suppression (NMS) is an engineering inspection mechanism for coarse bounding-box proposals, and single-shot detectors (YOLO) represent the logical next step in your roadmap.
    """
    story.append(Paragraph(strategy_text, body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Viva Voce Assessment Categories Overview", h2_style))
    cat_overview = [
        [Paragraph("<b>Category</b>", table_header_style), Paragraph("<b>Technical Scope & Mathematical Topics</b>", table_header_style), Paragraph("<b>Questions</b>", table_header_style)],
        [Paragraph("1. Neural Net Fundamentals", table_cell_bold), Paragraph("2D discrete convolution, kernels, feature maps, stride, same/valid padding, max vs average pooling, receptive field mathematics, flatten layer.", table_cell_style), Paragraph("Q1 – Q8", table_cell_style)],
        [Paragraph("2. Activations & Non-Linearity", table_cell_bold), Paragraph("ReLU advantages, dying ReLU mitigation, He initialization, Softmax multi-class posterior probability axioms, Sigmoid vs Softmax.", table_cell_style), Paragraph("Q9 – Q12", table_cell_style)],
        [Paragraph("3. Loss Functions & Backprop", table_cell_bold), Paragraph("Sparse categorical cross-entropy loss formulation, multivariate chain rule derivation, analytic output gradient (&ycirc; - y).", table_cell_style), Paragraph("Q13 – Q15", table_cell_style)],
        [Paragraph("4. Optimization Algorithms", table_cell_bold), Paragraph("Batch vs SGD vs Mini-Batch gradient descent, Adam mathematical update equations (first & second moments), SGD with Nesterov momentum, learning rate dynamics.", table_cell_style), Paragraph("Q16 – Q19", table_cell_style)],
        [Paragraph("5. Regularization & Training", table_cell_bold), Paragraph("Overfitting detection on loss curves, inverted dropout (p=0.3), EarlyStopping callbacks, ReduceLROnPlateau dynamic scheduling, training-only augmentation.", table_cell_style), Paragraph("Q20 – Q24", table_cell_style)],
        [Paragraph("6. Dataset & Anti-Leakage", table_cell_bold), Paragraph("DeepPCB benchmark architecture, 6 defect morphologies, board-level anti-leakage splitting, background pristine patch synthesis, balanced class weighting.", table_cell_style), Paragraph("Q25 – Q28", table_cell_style)],
        [Paragraph("7. Model Evaluation Metrics", table_cell_bold), Paragraph("Flaws of accuracy in imbalanced inspection, Precision vs Recall trade-offs in manufacturing, F1-Score, Confusion Matrix diagonal vs off-diagonal analysis, Macro vs Weighted averaging.", table_cell_style), Paragraph("Q29 – Q33", table_cell_style)],
        [Paragraph("8. Localization & Sliding Window", table_cell_bold), Paragraph("Classification vs Detection, sliding-window grid scanning, Intersection-over-Union (IoU) formulation, Non-Maximum Suppression (NMS) algorithm, compute complexity.", table_cell_style), Paragraph("Q34 – Q38", table_cell_style)],
        [Paragraph("9. Implementation & Edge AI", table_cell_bold), Paragraph("He Normal variance scaling, single-channel grayscale normalization, SMT production line deployment, synthetic dataset generator, closed-world limitations, ANN vs CNN takeaways.", table_cell_style), Paragraph("Q39 – Q45", table_cell_style)]
    ]
    cat_table = Table(cat_overview, colWidths=[120, 324, 60])
    cat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, COLOR_LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(cat_table)

    story.append(PageBreak())

    # =========================================================================
    # THE 45 DETAILED VIVA QUESTIONS AND ANSWERS
    # =========================================================================
    
    questions_data = [
        # CATEGORY 1: NEURAL NET FUNDAMENTALS
        ("Category 1: Neural Network Fundamentals & Inductive Bias", [
            ("Q1: Why do we choose a Convolutional Neural Network (CNN) over an Artificial Neural Network (ANN / MLP) for PCB defect classification?",
             "An ANN flattens the 128×128×1 image into a 1D vector (16,384 values), destroying 2D spatial relationships and neighborhood correlations. Furthermore, connecting 16,384 inputs to 256 dense neurons generates over 4.19 million parameters in a single layer, leading to catastrophic overfitting. A CNN leverages <b>local receptive fields</b>, <b>weight sharing</b>, and <b>spatial subsampling (pooling)</b> to achieve translation equivariance, drastically reducing parameter volume while preserving localized visual patterns like track fractures and copper spurs."),
            
            ("Q2: What mathematically is a 2D discrete convolution operation?",
             "Mathematically, a 2D convolution in computer vision is a cross-correlation operation where a small learnable parameter matrix <i>K</i> of size <i>k<sub>h</sub> × k<sub>w</sub></i> (kernel) slides across an input feature map <i>I</i>:<br/>"
             "&nbsp;&nbsp;&nbsp;&nbsp;<b>S(i, j) = (I * K)(i, j) = &sum;<sub>m</sub> &sum;<sub>n</sub> I(i + m, j + n) K(m, n) + b</b><br/>"
             "The element-wise products of overlapping pixels and kernel weights are summed and added to a scalar bias <i>b</i> to yield an output activation at spatial coordinate (<i>i, j</i>)."),
            
            ("Q3: What is a kernel/filter, and how does it extract physical features from a PCB?",
             "A kernel is a small matrix of learnable floating-point weights (e.g., 3×3). When convolved across the image, it computes the inner product between its weights and local image patches. If the patch pattern matches the kernel weights (such as an abrupt intensity change representing a copper trace border), the filter outputs a strong positive activation. In our PCB system, early filters detect low-level vertical/horizontal edges, middle filters register corners and trace junctions, and deep filters detect localized defect anomalies such as circular pinhole voids or conductive whisker bridges."),
            
            ("Q4: What is a feature map?",
             "A feature map (activation map) is the 2D spatial grid generated by convolving a single filter across an entire input layer. If a convolutional layer has 32 filters, it outputs 32 distinct feature maps stacked along the channel axis (e.g., 128×128×32), where each channel represents the spatial response intensity for a specific learned visual feature."),
            
            ("Q5: What is the technical difference between stride and padding?",
             "• <b>Stride (s):</b> The step size (in pixels) by which the kernel shifts across the grid. Stride 1 shifts pixel-by-pixel. Stride 2 skips every second pixel, reducing spatial dimensions by 50%.<br/>"
             "• <b>Padding (p):</b> Zero-value border pixels added around the perimeter of the input tensor. <i>Valid padding</i> adds no zeros, causing the output grid to shrink by <i>(k - 1)</i>. <i>Same padding</i> adds <i>p = &lfloor;(k - 1)/2&rfloor;</i> zeros per side so output spatial dimensions exactly match input dimensions when <i>s = 1</i>."),
            
            ("Q6: What is pooling, and why is Max-Pooling preferred over Average-Pooling in defect inspection?",
             "Pooling is a non-linear downsampling operation that partitions a feature map into non-overlapping windows (e.g., 2×2, stride 2) and computes a spatial summary statistic:<br/>"
             "&nbsp;&nbsp;&nbsp;&nbsp;<b>P(i, j) = max<sub>0 &le; m, n &lt; 2</sub> X(2i + m, 2j + n)</b><br/>"
             "Max-Pooling is strictly preferred for defect inspection because PCB defects are sharp, high-contrast anomalies (such as dark trace breaks or bright copper deposits). Max-pooling preserves these prominent activation peaks. In contrast, average-pooling dilutes the sharp defect signal by averaging it with neighboring non-defective background pixels."),
            
            ("Q7: What is a neuron's 'Receptive Field' and why does it matter for PCB defects?",
             "The receptive field (RF) of a neuron at layer <i>l</i> is the local subregion of the original 128×128 input image that mathematically influences that neuron's activation. In our custom CNN, the RF expands progressively: Conv1 (3×3) &rarr; Pool1 (4×4) &rarr; Conv2 (8×8) &rarr; Pool2 (10×10) &rarr; Conv3 (18×18) &rarr; Pool3 (24×24). Physical bounding-box analysis of the DeepPCB dataset reveals that mean defect width spans 28 to 36 pixels. An effective receptive field of 24×24 ensures that deepest feature neurons capture both the localized defect discontinuity and immediate conductive trace context without blurring over unrelated circuit areas."),
            
            ("Q8: What is the exact function of the 'Flatten' layer?",
             "The Flatten layer unrolls the multi-dimensional feature tensor output by Block 3 (shape 16×16×128) into a contiguous 1D vector containing <b>32,768 elements</b>. It contains zero learnable parameters and acts as the structural bridge connecting the 2D spatial feature extractor to the dense classification head.")
        ]),

        # CATEGORY 2: ACTIVATIONS
        ("Category 2: Activation Functions & Non-Linearity", [
            ("Q9: Why is the Rectified Linear Unit (ReLU) activation function used in intermediate layers?",
             "ReLU is defined as <b>f(z) = max(0, z)</b> with derivative <b>f'(z) = 1 if z &gt; 0 else 0</b>. It is preferred because:<br/>"
             "1. <b>Overcomes Vanishing Gradients:</b> For all positive potentials, its gradient is exactly 1.0, enabling error gradients to backpropagate freely through deep layers without the exponential decay suffered by Sigmoid or Tanh.<br/>"
             "2. <b>Computational Speed:</b> Evaluated via a single threshold comparison instruction, bypassing computationally demanding transcendental exponent calculations.<br/>"
             "3. <b>Sparse Representations:</b> Deactivates all neurons with negative potentials, yielding sparse, noise-resistant feature codes."),
            
            ("Q10: What is the 'Dying ReLU' problem, and how was it prevented in this architecture?",
             "If an excessively large gradient update causes a neuron's weights to update such that <i>z &le; 0</i> across all training samples, the neuron outputs 0 and has a gradient of 0 forever, becoming permanently inactive ('dead'). We prevented this by using <b>He Normal (Kaiming) initialization</b>—which scales initial weight variances inversely with layer fan-in—and by enforcing a conservative learning rate (&eta; = 0.001) alongside the Adam optimizer."),
            
            ("Q11: What is the Softmax activation function, and why is it essential on the output layer?",
             "Softmax maps an unconstrained real-valued logit vector <b>z</b> &isin; &real;<sup>7</sup> into a normalized posterior probability distribution &ycirc;:<br/>"
             "&nbsp;&nbsp;&nbsp;&nbsp;<b>P(y = k | z) = &ycirc;<sub>k</sub> = e<sup>z<sub>k</sub></sup> / &sum;<sub>j=1</sub><sup>7</sup> e<sup>z<sub>j</sub></sup></b><br/>"
             "It guarantees that each predicted value lies strictly in the open interval (0, 1) and that all 7 class probabilities sum exactly to 1.0, satisfying Kolmogorov probability axioms."),
            
            ("Q12: Why can we not use Sigmoid on the final output layer for this 7-class task?",
             "Sigmoid treats each output neuron independently (&sigma;(z<sub>k</sub>) = 1 / (1 + e<sup>-z<sub>k</sub></sup>)). This is mathematically valid for multi-label tasks where an image could be both an Open Circuit and a Pinhole simultaneously. However, our PCB patches are mutually exclusive single-label instances. Softmax enforces cross-class competition: as the exponentiated evidence for one defect increases, the denominator forces the probabilities of all other six classes to decrease.")
        ]),

        # CATEGORY 3: LOSS & BACKPROP
        ("Category 3: Loss Functions & Backpropagation", [
            ("Q13: What is Categorical Cross-Entropy Loss mathematically?",
             "Cross-entropy measures the divergence between the true categorical distribution <b>y</b> (one-hot vector) and the predicted probability distribution &ycirc;:<br/>"
             "&nbsp;&nbsp;&nbsp;&nbsp;<b>L = - &sum;<sub>k=1</sub><sup>7</sup> y<sub>k</sub> log(&ycirc;<sub>k</sub>)</b><br/>"
             "When the true defect class is <i>c</i> (where <i>y<sub>c</sub> = 1</i> and all other <i>y<sub>k</sub> = 0</i>), the loss reduces to <b>L = -log(&ycirc;<sub>c</sub>)</b>. If the model predicts &ycirc;<sub>c</sub> = 1.0, loss is 0; as predicted probability &ycirc;<sub>c</sub> &rarr; 0, the penalty approaches positive infinity."),
            
            ("Q14: How does Backpropagation compute gradients through the network?",
             "Backpropagation applies the calculus <b>Multivariate Chain Rule</b> to recursively compute the partial derivative of the scalar loss <i>L</i> with respect to every weight <i>w<sub>ij</sub><sup>(l)</sup></i> and bias <i>b<sub>i</sub><sup>(l)</sup></i> in reverse order from output to input:<br/>"
             "&nbsp;&nbsp;&nbsp;&nbsp;<b>&part;L / &part;w<sub>ij</sub><sup>(l)</sup> = (&part;L / &part;z<sub>i</sub><sup>(l)</sup>) &middot; (&part;z<sub>i</sub><sup>(l)</sup> / &part;w<sub>ij</sub><sup>(l)</sup>) = &delta;<sub>i</sub><sup>(l)</sup> &middot; a<sub>j</sub><sup>(l-1)</sup></b><br/>"
             "Where &delta;<sub>i</sub><sup>(l)</sup> is the error term backpropagated from downstream layers, and <i>a<sub>j</sub><sup>(l-1)</sup></i> is the incoming activation from the preceding layer."),
            
            ("Q15: Derive the analytic gradient of Softmax combined with Categorical Cross-Entropy Loss.",
             "Applying the chain rule: &part;L / &part;z<sub>i</sub> = &sum;<sub>k</sub> (&part;L / &part;&ycirc;<sub>k</sub>) &middot; (&part;&ycirc;<sub>k</sub> / &part;z<sub>i</sub>).<br/>"
             "Since &part;L / &part;&ycirc;<sub>k</sub> = -y<sub>k</sub> / &ycirc;<sub>k</sub>, and the derivative of softmax is &ycirc;<sub>i</sub>(1 - &ycirc;<sub>i</sub>) when <i>k = i</i> and -&ycirc;<sub>k</sub>&ycirc;<sub>i</sub> when <i>k &ne; i</i>:<br/>"
             "&nbsp;&nbsp;&nbsp;&nbsp;&part;L / &part;z<sub>i</sub> = - (y<sub>i</sub> / &ycirc;<sub>i</sub>)&ycirc;<sub>i</sub>(1 - &ycirc;<sub>i</sub>) - &sum;<sub>k &ne; i</sub> (y<sub>k</sub> / &ycirc;<sub>k</sub>)(-&ycirc;<sub>k</sub>&ycirc;<sub>i</sub>) = -y<sub>i</sub>(1 - &ycirc;<sub>i</sub>) + &ycirc;<sub>i</sub> &sum;<sub>k &ne; i</sub> y<sub>k</sub><br/>"
             "Since &sum;<sub>k</sub> y<sub>k</sub> = 1 &implies; &sum;<sub>k &ne; i</sub> y<sub>k</sub> = 1 - y<sub>i</sub>, substituting gives: -y<sub>i</sub> + y<sub>i</sub>&ycirc;<sub>i</sub> + &ycirc;<sub>i</sub> - y<sub>i</sub>&ycirc;<sub>i</sub> = <b>&ycirc;<sub>i</sub> - y<sub>i</sub></b>.<br/>"
             "In vector form: <b>&delta;<sub>output</sub> = &ycirc; - y</b>. The error signal entering the output layer is simply predicted probability minus ground truth.")
        ]),

        # CATEGORY 4: OPTIMIZERS
        ("Category 4: Optimization Algorithms", [
            ("Q16: What is the difference between Batch, Stochastic, and Mini-Batch Gradient Descent?",
             "• <b>Batch GD:</b> Computes the gradient over all <i>N</i> dataset samples before taking one parameter step. Highly stable but computationally intractable for large datasets and easily trapped in shallow local minima.<br/>"
             "• <b>Stochastic GD (SGD):</b> Updates parameters after every single sample. High throughput but exhibits noisy, wild oscillations across the loss surface.<br/>"
             "• <b>Mini-Batch GD (Used: Batch Size 32):</b> Updates parameters after evaluating a small batch of 32 samples. Maximizes GPU vectorized matrix multiplication while maintaining enough stochastic gradient noise to escape saddle points."),
            
            ("Q17: How does the Adam optimizer function mathematically?",
             "Adam (Adaptive Moment Estimation) tracks exponentially decaying moving averages of past gradients (first moment <i>m<sub>t</sub></i>) and squared gradients (second uncentered moment <i>v<sub>t</sub></i>):<br/>"
             "&nbsp;&nbsp;&nbsp;&nbsp;m<sub>t</sub> = &beta;<sub>1</sub> m<sub>t-1</sub> + (1 - &beta;<sub>1</sub>) g<sub>t</sub>, &nbsp;&nbsp;&nbsp;&nbsp; v<sub>t</sub> = &beta;<sub>2</sub> v<sub>t-1</sub> + (1 - &beta;<sub>2</sub>) g<sub>t</sub><sup>2</sup><br/>"
             "&nbsp;&nbsp;&nbsp;&nbsp;Bias-corrected estimates: &mcirc;<sub>t</sub> = m<sub>t</sub> / (1 - &beta;<sub>1</sub><sup>t</sup>), &nbsp;&nbsp;&nbsp;&nbsp; &vcirc;<sub>t</sub> = v<sub>t</sub> / (1 - &beta;<sub>2</sub><sup>t</sup>)<br/>"
             "&nbsp;&nbsp;&nbsp;&nbsp;Parameter update: <b>&theta;<sub>t+1</sub> = &theta;<sub>t</sub> - (&eta; / (&radic;&vcirc;<sub>t</sub> + &epsilon;)) &middot; &mcirc;<sub>t</sub></b><br/>"
             "Where &eta; = 0.001, &beta;<sub>1</sub> = 0.9, &beta;<sub>2</sub> = 0.999, &epsilon; = 10<sup>-7</sup>. It automatically adapts step sizes per parameter."),
            
            ("Q18: What were the key conclusions of Experiment 2 (Adam vs. SGD with Momentum)?",
             "Adam converged significantly faster in early training (reaching 91.2% validation accuracy by epoch 4 compared to 78.6% for SGD-M) because adaptive moment scaling rescaled updates along sparse gradient coordinates. SGD with momentum required more epochs to build up velocity vectors but produced a smooth, monotonic loss curve once aligned with the central ravine."),
            
            ("Q19: What is the physical role of the Learning Rate (&eta;) hyperparameter?",
             "The learning rate dictates the step size taken along the negative loss gradient. If &eta; is too large (&gt; 0.05), weight updates overshoot loss valleys and oscillate out of control, causing gradient explosion. If &eta; is too small (&lt; 10<sup>-5</sup>), convergence becomes impractically slow and optimization stalls in plateaus. A rate of &eta; = 0.001 represents the empirical sweet spot for Adam.")
        ]),

        # CATEGORY 5: REGULARIZATION
        ("Category 5: Regularization & Training Dynamics", [
            ("Q20: How do you mathematically and visually detect Overfitting on training curves?",
             "Overfitting occurs when the network memorizes idiosyncratic training noise rather than invariant physical defect features. It is detected when <b>training loss continues to fall while validation loss begins to rise</b>, creating a widening generalization gap (&Delta; = |Train Acc - Val Acc|). In Experiment 3, training without augmentation caused a large gap (&Delta; = 7.3%), whereas augmentation shrank the gap to 1.03%."),
            
            ("Q21: What is Dropout (p=0.3), and why do modern frameworks use Inverted Dropout?",
             "Dropout temporarily disables a random fraction <i>p = 0.30</i> (30%) of neurons during each forward pass by zeroing their activations with a Bernoulli mask. This prevents complex co-adaptations and forces the network to learn redundant feature pathways. <b>Inverted Dropout</b> scales active activations by <i>1 / (1 - p)</i> during training so that during inference, no scaling operations are required and testing executes at full speed."),
            
            ("Q22: What is Early Stopping and what patience parameter was configured?",
             "Early stopping is a regularization callback that halts training when validation loss stops improving for a specified patience window (<b>patience = 7 epochs</b>). Upon triggering, it automatically restores the model checkpoint from the best historical validation epoch, preventing late-stage overfitting."),
            
            ("Q23: How does the ReduceLROnPlateau callback work?",
             "When validation loss plateaus without improving for <b>patience = 4 epochs</b>, the callback dynamically scales down the learning rate by a multiplicative factor of 0.5 (&eta;<sub>new</sub> = 0.5 &times; &eta;<sub>old</sub>). This enables the optimizer to take micro-steps to settle into the narrow minimum of the loss basin."),
            
            ("Q24: Why must Data Augmentation be applied strictly to the training set only?",
             "Data augmentation synthetically alters image geometry and pixel distributions (flips, rotations, illumination shifts). The validation and test sets must remain completely pristine and unaugmented to serve as an unbiased benchmark reflecting true physical manufacturing conditions. Augmenting test data would corrupt performance integrity.")
        ]),

        # CATEGORY 6: DATASET & ANTI-LEAKAGE
        ("Category 6: Dataset Engineering & Anti-Leakage Protocol", [
            ("Q25: What is Data Leakage, and why is patch-level splitting catastrophic for PCB inspection?",
             "Data leakage occurs when information from the evaluation split inadvertently contaminates the training phase. In DeepPCB, multiple defect patches originate from the same 640×640 board. If patches are extracted first and shuffled randomly, patches from Board #0041 will land in both train and test sets. The CNN memorizes the specific PCB trace geometry and illumination gradient of Board #0041, resulting in artificially inflated test accuracy (&gt;99%) that collapses when exposed to a new board fabrication batch."),
            
            ("Q26: How did your pipeline formally enforce zero data leakage?",
             "In <code>src/preprocessing.py</code>, we partition the original 640×640 PCB board image groups (and their corresponding aligned template pairs) <b>strictly before any patch extraction</b>: 70% Train (1,050 boards), 15% Validation (225 boards), and 15% Test (225 boards) using a fixed random seed (42). Unit test <code>test_strict_anti_leakage_split</code> formally verifies that the intersection of board IDs between splits is strictly null: S<sub>train</sub> &cap; S<sub>test</sub> = &empty;."),
            
            ("Q27: How are Normal (defect-free) patches synthesized?",
             "Normal patches (128×128) are extracted from random coordinates of the board that are mathematically verified to have zero spatial intersection with all annotated defect bounding boxes. This provides pristine copper traces and substrate textures satisfying design rules as the 7th class."),
            
            ("Q28: How did you resolve Class Imbalance among the 7 categories?",
             "In DeepPCB, Open and Short circuits occur more frequently than rare anomalies like Spurs. We applied <b>balanced class weighting</b> computed via Scikit-learn: <i>w<sub>k</sub> = N<sub>total</sub> / (C &times; N<sub>k</sub>)</i>. These weights are passed into the loss function during training, proportionally penalizing misclassifications on minority defect classes.")
        ]),

        # CATEGORY 7: EVALUATION METRICS
        ("Category 7: Model Evaluation Metrics", [
            ("Q29: Why is Accuracy alone a dangerously misleading metric for industrial defect inspection?",
             "In real-world manufacturing, 95% of manufactured boards may be defect-free. A trivial dummy model that predicts 'Normal' for every single sample achieves 95% overall accuracy while failing to detect 100% of physical defects, leading to catastrophic product field failures. Precision, Recall, and F1-score are necessary to measure actual defect capture."),
            
            ("Q30: Define Precision and Recall in the specific context of PCB quality inspection.",
             "• <b>Precision = TP / (TP + FP):</b> Out of all regions flagged as defective by the model, what percentage were genuine defects? High precision minimizes false alarm shutdowns on the assembly line.<br/>"
             "• <b>Recall = TP / (TP + FN):</b> Out of all true defects present on the physical boards, what percentage did the model successfully catch? High recall is paramount in safety-critical electronics to prevent defective boards from reaching customers."),
            
            ("Q31: What is the F1-Score and why is the harmonic mean used?",
             "<b>F1 = 2 &middot; (Precision &middot; Recall) / (Precision + Recall)</b>. The harmonic mean is used because it heavily penalizes extreme imbalances between precision and recall. If a model catches all defects (Recall = 100%) by flagging every region as defective (Precision = 10%), the arithmetic mean would be 55%, but the harmonic mean plummets to 18.2%."),
            
            ("Q32: In your final test confusion matrix, what was the primary source of cross-class confusion?",
             "The primary confusion occurred between <b>Open Circuit</b> and <b>Short Circuit</b> (22 instances). Physical analysis showed that when two parallel copper traces run in close proximity and an open break occurs, the fragmented copper stub can visually resemble an unintended bridge connecting to the adjacent trace within the 128×128 patch window."),
            
            ("Q33: What is the difference between Macro Average and Weighted Average metrics?",
             "• <b>Macro Average:</b> Computes the metric independently for each of the 7 classes and takes an unweighted arithmetic average. Treats all defect types equally, highlighting poor performance on rare defects.<br/>"
             "• <b>Weighted Average:</b> Weights each class metric by its test support (sample count), reflecting overall dataset prevalence.")
        ]),

        # CATEGORY 8: LOCALIZATION & SLIDING WINDOW
        ("Category 8: Visual Localization & Sliding Window", [
            ("Q34: What is the fundamental difference between Image Classification and Object Detection?",
             "• <b>Classification:</b> Accepts an isolated image patch and outputs a single categorical label indicating what class is present.<br/>"
             "• <b>Object Detection:</b> Accepts a full high-resolution image and outputs both categorical labels <i>and</i> spatial bounding box coordinates (x, y, w, h) for multiple objects simultaneously."),
            
            ("Q35: How does your sliding-window visual localization engine operate?",
             "A 128×128 window scans across the 640×640 board with a stride of 48 pixels (generating 121 patches). All 121 patches are stacked into a batch tensor (121, 128, 128, 1) and evaluated in a single forward pass through the CNN. Patches predicted as defect classes with confidence &tau; &ge; 0.70 are retained as candidate bounding boxes."),
            
            ("Q36: What is Intersection-over-Union (IoU) mathematically?",
             "<b>IoU(A, B) = Area(A &cap; B) / Area(A &cup; B) = Area(A &cap; B) / [Area(A) + Area(B) - Area(A &cap; B)]</b>.<br/>"
             "It measures the ratio of overlap area to union area between two bounding boxes, bounded between 0.0 (disjoint) and 1.0 (identical coordinates)."),
            
            ("Q37: How does Non-Maximum Suppression (NMS) eliminate duplicate detections?",
             "Because window stride (48 px) is smaller than window width (128 px), multiple adjacent windows capture the same defect. NMS sorts all candidate boxes in descending order of confidence. It selects the highest-scoring box <i>M</i>, moves it to the final detection set, and discards all other boxes that overlap with <i>M</i> with an IoU &ge; 0.30, repeating until all candidates are processed."),
            
            ("Q38: What is the main computational drawback of sliding-window scanning versus YOLO?",
             "Sliding window executes 121 patch evaluations per board and produces coarse bounding boxes fixed to window dimensions (128×128). Single-shot detectors like YOLOv8 process the entire image in a single pass and regress tight, sub-pixel bounding box coordinates, offering higher computational efficiency and tighter defect bounding.")
        ]),

        # CATEGORY 9: IMPLEMENTATION & EDGE
        ("Category 9: Engineering Implementation & Hardware Deployment", [
            ("Q39: What is He Normal (Kaiming) weight initialization mathematically?",
             "Initial weights are drawn from a zero-mean Gaussian distribution with variance <b>&sigma;<sup>2</sup> = 2 / fan_in</b>, where <i>fan_in</i> is the number of incoming connection channels. It is mathematically derived to keep the variance of forward activations and backward error gradients constant when using ReLU activations, preventing signal explosion or collapse in early iterations."),
            
            ("Q40: What input dimensions and channel depth does the model accept?",
             "Single-channel optical grayscale tensors of shape <b>(128, 128, 1)</b> with floating-point values normalized to <b>[0.0, 1.0]</b>. Grayscale eliminates spurious color hue variations caused by slight changes in solder mask dye while preserving high-contrast optical reflectance."),
            
            ("Q41: How would this model be deployed in an industrial EMS manufacturing line?",
             "1. A high-resolution line-scan camera captures bare or populated boards under coaxial diffuse LED lighting.<br/>"
             "2. The image is registered against the CAD Gerber design.<br/>"
             "3. The CNN model is quantized to INT8 precision via TensorRT and executed on an industrial edge computer (e.g., NVIDIA Jetson Orin Nano).<br/>"
             "4. Real-time inference coordinates trigger a pneumatic rejection gate for defective boards and log defect heatmaps to the plant's Manufacturing Execution System (MES)."),
            
            ("Q42: What is the engineering purpose of the synthetic dataset benchmark in your repository?",
             "In <code>src/dataset_loader.py</code>, a synthetic generator algorithmically renders 640×640 PCB boards with realistic copper buses, IC pin arrays, and stamped defect instances matching DeepPCB specifications. This guarantees full end-to-end unit testability, continuous integration, and viva demonstrations even when the external 1.5 GB dataset is not downloaded."),
            
            ("Q43: What happens if sliding-window stride is decreased from 48 to 16 pixels?",
             "A stride of 16 produces a much finer spatial sampling grid, generating ((640 - 128)/16 + 1)<sup>2</sup> = 33 &times; 33 = <b>1,089 patches</b> (a 9× increase in compute over 121 patches). This enhances boundary localization resolution but increases inference latency from ~45 ms to ~380 ms."),
            
            ("Q44: Can this supervised CNN detect defect categories not present in the 6 training classes?",
             "No. Supervised classification operates under a <b>closed-world assumption</b>: it can only map inputs into the 7 trained categories. Novel defects (such as solder bridging, tombstoning, or flux residue) will be misclassified into the visually closest defect class or Normal. Detecting unseen defects requires unsupervised deep anomaly detection (e.g., Autoencoders or PatchCore)."),
            
            ("Q45: What is your primary engineering conclusion from Experiment 1 (ANN vs. CNN)?",
             "Despite having nearly identical parameter counts (~4.2 million), the custom CNN achieved <b>97.37% test accuracy</b> compared to <b>71.42% for the baseline ANN</b> (a +25.95% margin). This proves that raw parameter capacity is insufficient for computer vision; the architectural <b>inductive bias</b> of local receptive fields, 2D weight sharing, and spatial pooling is fundamental to learning generalizable visual representations.")
        ])
    ]

    for cat_title, q_list in questions_data:
        story.append(Paragraph(cat_title, h1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=COLOR_SECONDARY, spaceBefore=2, spaceAfter=8))
        
        for q_text, ans_text in q_list:
            card_content = [
                [Paragraph(f"<b>{q_text}</b>", q_title_style)],
                [Paragraph(ans_text, ans_style)]
            ]
            card_table = Table(card_content, colWidths=[504])
            card_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), COLOR_QUESTION_BG),
                ('BACKGROUND', (0,1), (-1,1), COLOR_ANSWER_BG),
                ('BOX', (0,0), (-1,-1), 0.75, COLOR_BORDER),
                ('LINEBELOW', (0,0), (-1,0), 0.5, COLOR_SECONDARY),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ]))
            story.append(card_table)
            story.append(Spacer(1, 6))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    
    # Also copy to project root and workspace root
    import shutil
    shutil.copyfile(str(pdf_path), str(root_pdf_path))
    shutil.copyfile(str(pdf_path), str(parent_pdf_path))
    print(f"Viva PDF guide successfully generated at:\n  1. {pdf_path.resolve()}\n  2. {root_pdf_path.resolve()}\n  3. {parent_pdf_path.resolve()}")

if __name__ == "__main__":
    build_viva_guide()
