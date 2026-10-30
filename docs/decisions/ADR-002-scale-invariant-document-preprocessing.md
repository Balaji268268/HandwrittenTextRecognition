# ADR-002: Scale-Invariant Document Preprocessing, Deskewing, and Doodle Filtering

## Status
Accepted

## Date
2026-10-30

## Context
Real-world handwritten text recognition involves diverse document modalities:
1. **Uncalibrated Document Scales**: Images captured by mobile phones or varying scanners range from $150\text{ DPI}$ to $1200\text{ DPI}$, with character stroke heights ranging from $15\text{ px}$ to over $300\text{ px}$. Directly resizing a full document or line to fixed $64\text{ px}$ height distorts stroke aspect ratios and morphological fidelity.
2. **Angular Skew and Lighting Gradients**: Smartphone captures frequently exhibit tilt ($\pm 30^\circ$) and severe non-uniform shadows (e.g., desk lamps, camera angle occlusions).
3. **Ruled and Lined Paper**: Notebooks and prescription slips feature horizontal printed lines intersecting ascenders and descenders.
4. **Dense Paragraph Slicing vs. Non-Text Doodles**: Naive horizontal dilation merges dense paragraph lines into a single unreadable mass (e.g., 11-line letter squashed into a single line yielding `◖n◗`). Conversely, isolated non-text drawings (e.g., smiling suns, flowers, stars) get passed into the text recognition model and hallucinated as erroneous words with pseudo-high confidence (e.g., sun doodle hallucinated as `"drior,"`).

## Decision
We implement a unified, production-grade **Scale-Invariant Document Engine** in `sarah/data/utils.py` comprising:
1. **Dynamic Character Scale Estimation ($\text{med\_h}$)**:
   Extract connected components of ink pixels; filter out punctuation ($H < 8\text{ px}$) and full-page artifacts ($H > 0.8 \cdot H_{\text{page}}$); calculate the median height $\text{med\_h} = \text{median}(\{h_i\})$.
2. **Illumination Normalization via Background Closure Division**:
   Estimate non-uniform background illumination $\mathbf{B}$ using morphological grayscale closing with structuring element $S_{\text{light}}$ of radius $2 \cdot \text{med\_h}$:
   $$\mathbf{I}_{\text{norm}} = \frac{\mathbf{I}}{\mathbf{B} + \epsilon} \times 255$$
3. **Automated Document Deskewing**:
   Locate ink contours, compute minimum-area bounding box angle $\theta \in [-30^\circ, +30^\circ]$, and apply affine rotation matrix:
   $$\mathbf{I}_{\text{deskewed}} = \text{warpAffine}\left(\mathbf{I}, \mathbf{R}(\theta), (W, H), \text{borderValue}=255\right)$$
4. **Ruled-Line Suppression**:
   Detect and eliminate ruled notebook lines using horizontal morphological opening with structuring element $K_{\text{line}} = \text{getStructuringElement}(\text{MORPH\_RECT}, (2.5 \cdot \text{med\_h}, 1))$:
   $$\mathbf{I}_{\text{suppressed}} = \mathbf{I}_{\text{bin}} \setminus (\mathbf{I}_{\text{bin}} \circ K_{\text{line}})$$
5. **Adaptive Valley-Splitting Line Segmentation**:
   For dense multi-line text (ink height $> 2.2 \cdot \text{med\_h}$), compute the horizontal projection profile histogram $\mathcal{P}(y) = \sum_{x} \mathbf{I}_{\text{bin}}(x, y)$. Identify local inter-line minima (valleys) using peak prominence analysis, slicing multi-line paragraphs into distinct, clean line strips.
6. **True Un-Dilated Aspect Ratio Doodle Discrimination**:
   Dilation artificially widens square doodles (causing a circular sun with aspect ratio $1.15$ to stretch past $1.9$). We evaluate geometric features directly on the **un-dilated raw ink**:
   $$AR_{\text{true}} = \frac{W_{\text{ink}}}{H_{\text{ink}}} < 1.8 \quad \text{AND} \quad H_{\text{ink}} > 0.9 \cdot \text{med\_h} \implies \text{Classified as Non-Text Doodle (Rejected)}$$

## Alternatives Considered

### 1. Naive Direct Resizing (Direct Squash)
- **Pros**: Zero segmentation latency; single-pass feedforward.
- **Cons**: Catastrophic distortion on multi-line documents. Squeezing a $153 \times 148$ greeting card to $64 \times 1024$ compresses character strokes into unreadable vertical needles, yielding garbled predictions (e.g. `"fiisil"` at $5.1\%$ confidence).
- **Rejected**: Fails fundamentally on multi-line documents and greeting cards.

### 2. A* Path Planning / Seam Carving Line Segmentation
- **Pros**: Capable of tracing overlapping ascenders and descenders in unconstrained handwriting.
- **Cons**: Computational complexity scales as $\mathcal{O}(H \times W^2)$, requiring $> 2.5\text{ s}$ per full-page document, which violates real-time production requirements.
- **Rejected**: Unsuitable for responsive web and mobile deployment pipelines.

### 3. Pretrained Object Detection Networks (YOLOv8 / Faster-RCNN for Line Detection)
- **Pros**: High bounding box accuracy.
- **Cons**: Introduces external pretrained weights; high parameter footprint ($> 25\text{M}$ params); requires tens of thousands of manually annotated bounding-box datasets.
- **Rejected**: Violates the "zero pretrained model" architectural commitment.

## Consequences

### Positive
- Fully automated scale calibration regardless of scanner resolution or smartphone distance.
- Robust handling of mobile phone illumination gradients, eliminating shadow-induced thresholding artifacts.
- Dense multi-line paragraphs (e.g. 11-line historical letter) are cleanly separated along natural inter-line valleys into 11 distinct lines.
- Non-text doodles (smiling suns, decorative hearts) are reliably filtered out before feeding the recognition model, preventing hallucinated token sequences.
- Completely training-free and model-agnostic; runs in $< 45\text{ ms}$ on standard CPU.

### Negative / Trade-offs
- Heavily overlapping ascenders and descenders (e.g. loops of 'y' penetrating the loop of 'd' below) can occasionally be clipped if the valley is perfectly flat; handled gracefully through vertical safety margins.

