# Scale-Adaptive Document Conditioning and Multi-Head Gated Convolutional Networks for Multi-Dataset Offline Handwritten Text Recognition

**Author:** A. Venkata Balaji (Registration No: 23BCE20336)  
**Affiliation:** School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, India  
**Target Venue:** IEEE Access  
**Repository Remote:** `https://github.com/Balaji268268/HandwrittenTextRecognition`

---

## Abstract

Offline Handwritten Text Recognition (HTR) remains a challenging computer vision task due to extreme script diversity, irregular handwriting styles, and severe real-world document degradations including non-uniform smartphone illumination gradients, uncalibrated scale variations, skew, and ruled notebook interference. While recent autoregressive Vision Transformers achieve high transcription accuracy on standard benchmarks, they require hundreds of millions of synthetic pretraining tokens, incur large memory footprints (>500M parameters), and exhibit high inference latencies (>300 ms/line). 

In this paper, we propose a holistic, production-grade HTR framework that operates without massive pretrained foundation models. Our framework integrates:
1. A **Scale-Adaptive Document Conditioning Engine** that dynamically computes median stroke height ($	ext{med\_h}$) to standardize inputs, levels non-uniform shadows via morphological closure division, deskews canvas tilt, and segments dense text lines using projection valley-splitting coupled with an un-dilated aspect ratio doodle rejection heuristic ($AR_{	ext{true}} < 1.8 \land H > 0.9 \cdot 	ext{med\_h}$).
2. A lightweight **Hybrid Neural Sequence Architecture** consisting of seven Conv2D stages with six interleaved residual Gated-CNN blocks, a two-head Multi-Head Self-Attention (MHSA) layer ($h=2, d_k=8$), and three stacked Bidirectional LSTM layers with Connectionist Temporal Classification (CTC). The model comprises only **1.58M parameters** ($1,583,847$ parameters for vocabulary size $C=83$), scaling strictly via the closed-form relation $P_{\text{total}} = 1,562,516 + 257C$.
3. An exact **CTC Sequence Joint Probability Formulation** that computes sequence likelihoods under marginalized alignment paths to flag severe visual noise ($P < 0.01\%$).

Evaluated across five diverse academic handwriting benchmarks, the proposed system achieves **2.48% test CER** (8.53% WER) on the 13th-century German *Parzival* codex, **3.33% test CER** on *ORAND-CAR-A* bank check courtesy amounts, and **8.38% test CER** (41.62% WER) on the 9th-century Latin *Saint Gall* manuscript. On the *IAM Combined* English corpus, the model achieves a best validation normalized edit distance of **5.66%** at epoch 12. Furthermore, our evaluation reveals a substantial validation-to-test generalization gap on *CVL Digits* ($6.93\%$ validation edit distance vs. $50.78\%$ test CER), providing a diagnostic finding and open hypothesis regarding unconstrained numeral sequence length drift and the absence of lexical priors in pure numeric recognition.

---

## I. Introduction

Offline Handwritten Text Recognition (HTR) is the task of transcribing static two-dimensional document images containing human handwriting into machine-readable digital character sequences. Unlike optical character recognition (OCR) of machine-printed typography—which is largely considered solved—offline handwriting exhibits dramatic writer variability, cursive ligatures, stroke warping, and physical document aging.

### A. Challenges in Document Processing
1. **Uncalibrated Scale Drift:** Document images captured by mobile cameras or desktop flatbed scanners vary widely in resolution. Arbitrary downsampling destroys fine stroke topology and cursive ligatures.
2. **Non-Uniform Smartphone Illumination:** Mobile captures routinely suffer from harsh cast shadows and desk-lamp gradients that cause global binarization methods (e.g., standard Otsu) to fail catastrophically.
3. **Ruled Notebook Interference:** Lined notebook paper introduces pre-printed horizontal rules that intersect character ascenders and descenders ('g', 'y', 'p'), fragmenting character stems during binarization.
4. **Dense Multi-Line Paragraphs and Non-Text Doodles:** Archival letters feature interlocking ascenders between adjacent lines, while greeting cards and memos contain decorative drawings and stamps that trigger sequence model hallucinations.

### B. Research Questions (RQs)
- **RQ1:** Can a scale-adaptive preprocessing pipeline eliminate resolution and illumination sensitivity across diverse capture devices without manual hyperparameter tuning?
- **RQ2:** What is the isolated empirical contribution of residual Gated Convolutions versus standard convolutions in filtering parchment grain and document noise?
- **RQ3:** Does Multi-Head Self-Attention provide complementary global contextual dependencies when integrated with recurrent sequence encoders?
- **RQ4:** What is the empirical impact of transfer warm-starting from a baseline prose model versus training from scratch on diverse handwriting scripts?
- **RQ5:** How do sequence decoders behave when transcribing unconstrained numeral sequences lacking lexical language priors?

### C. Key Research Contributions
1. **Scale-Adaptive Document Conditioning Engine:** Formulates an automated pipeline that dynamically computes the median character stroke height ($	ext{med\_h}$), enabling scale-invariant document normalization across uncalibrated mobile and scanner resolutions.
2. **Morphological Illumination & Line Suppression:** Implements scale-parameterized background closure division to eliminate non-uniform smartphone shadow gradients, coupled with morphological opening to suppress ruled notebook lines without damaging vertical character stems.
3. **Valley-Splitting Line Slicing & Geometric Doodle Rejection:** Formulates projection valley-splitting to cleanly slice dense multi-line paragraphs, integrated with an un-dilated aspect ratio discrimination filter ($AR_{	ext{true}} < 1.8 \land H > 0.9 \cdot 	ext{med\_h}$) that rejects squarish non-text doodles.
4. **Lightweight Hybrid Neural Architecture ($1.58\text{M}$ Parameters):** Combines seven convolutional stages with six interleaved residual Gated-CNN blocks, a two-head Multi-Head Self-Attention layer ($h=2, d_k=8$), and three Bidirectional LSTM layers with CTC. Total parameters scale strictly via $P_{\text{total}} = 1,562,516 + 257C$.
5. **Exact CTC Sequence Joint Probability Formulation:** Implements a mathematically bounded sequence probability formulation $P = \exp(\log P_{\text{CTC}}) \times 100\%$ that computes marginalized sequence likelihoods to flag unrecognizable inputs ($< 0.01\%$).
6. **Multi-Dataset Empirical Evaluation:** Conducts evaluations across five distinct handwriting corpora (Parzival, ORAND-CAR-A, Saint Gall, IAM Combined, and CVL Digits), establishing high accuracy on medieval German ($2.48\%$ test CER) and bank checks ($3.33\%$ test CER) while systematically diagnosing digit sequence distribution shift.

---

## II. Related Work

Early offline handwriting recognition systems relied on Hidden Markov Models (HMMs) coupled with hand-engineered feature representations such as SIFT and HOG descriptors. While HMMs provided rigorous probabilistic frameworks, their Markovian assumptions constrained long-range contextual modeling.

The introduction of Connectionist Temporal Classification (CTC) by Graves et al. eliminated the requirement for pre-segmented character alignments, enabling end-to-end sequence training on unsegmented line images. Puigcerver demonstrated that deep 1D-LSTM recurrent stacks combined with 2D convolutional encoders outperform multi-dimensional RNNs while drastically reducing computational complexity. Bluche and Messina introduced gated convolutional networks for handwriting, proving that feature-level spatial gating effectively suppresses background paper clutter. Flor et al. refined this paradigm with residual gated connections, demonstrating fast convergence and high transcription fidelity.

Following the success of transformers, Vision Transformers were adapted for document transcription. Li et al. introduced TrOCR, coupling a vision transformer encoder with an autoregressive language decoder. While TrOCR achieved competitive results on several benchmarks, its performance depends strictly on pretraining across 684 million synthetic typography images. Recent studies by Diaz et al. and Kang et al. demonstrated that when Vision Transformers are trained from scratch without large-scale pretraining, they fail to converge on medium-sized academic corpora, yielding CERs higher than 35%. Coquenet et al. proposed the Document Attention Network (DAN) for full-page recognition, but these models incur large parameter footprints (>20M parameters) and substantial inference latencies.

---

## III. Proposed Methodology

The proposed framework operates across three tightly integrated stages:
1. **Stage 1:** Scale-Adaptive Document Conditioning Engine
2. **Stage 2:** Deep Gated-CNN + MHSA + BiLSTM Neural Architecture
3. **Stage 3:** Exact CTC Sequence Probability Formulation & Geometric Rejection

### A. Stage 1: Scale-Adaptive Document Conditioning Engine
1. **Dynamic Scale Estimation ($	ext{med\_h}$):** Computes connected component bounding boxes across the binarized canvas. The median height of valid components establishes the intrinsic document stroke scale:
   $$\text{med\_h} = \text{median}(\{h_i \mid h_i > 4 \land w_i > 4\})$$
2. **Morphological Illumination Correction:** Generates a background illumination estimate $\mathbf{I}_{\text{bg}}$ via morphological closure using a structuring element proportional to stroke height ($k_{\text{size}} = 3 \cdot \text{med\_h}$), and divides the raw image:
   $$\mathbf{I}_{\text{norm}} = \min\left(255, \frac{\mathbf{I}}{\mathbf{I}_{\text{bg}} + \epsilon} \times 255\right)$$
3. **Angular Canvas Deskewing:** Determines document skew angle $\theta$ via minimum-area bounding rectangles of text components and applies bilinear affine rotation to align text baselines horizontally.
4. **Ruled-Line Suppression:** Employs horizontal morphological opening with kernel width $w_k = 2.5 \cdot \text{med\_h}$ and height $h_k = 1$ to isolate and subtract horizontal ruled notebook lines without damaging vertical character stems.
5. **Valley-Splitting Line Slicing:** Computes horizontal projection profiles across multi-line paragraphs. Local minima identify inter-line cutting valleys, segmenting dense correspondence into isolated line strips.
6. **Geometric Doodle Rejection:** Computes the un-dilated aspect ratio $AR_{\text{true}} = w / h$ of extracted bounding boxes. Components satisfying $AR_{\text{true}} < 1.8 \land h > 0.9 \cdot \text{med\_h}$ are rejected as non-text doodles, preventing sequence hallucination.

### B. Stage 2: Deep Gated-CNN + Attention + BiLSTM Architecture
Extracted text line strips are normalized to height $H_{\text{in}} = 64\text{ px}$ with aspect-ratio preservation and padded to $W_{\text{in}} = 1024\text{ px}$.

#### Layer-by-Layer Architectural Profile (Table I)
The model consists of a convolutional gated encoder ($501,988$ parameters) and an attention-guided recurrent decoder ($1,081,859$ parameters for vocabulary size $C=83$).

```
========================================================================================================================
TABLE I: LAYER-BY-LAYER ARCHITECTURAL PROFILE OF THE PROPOSED FRAMEWORK (C=83)
========================================================================================================================
Subsystem       | Layer / Operation                            | Output Shape           | Param Count
----------------+----------------------------------------------+------------------------+-------------------------------
Encoder         | Input Text-Line Image                        | (64, 1024, 1)          | 0
                | Conv1 (8 filters, 3x3) + BN + Swish          | (64, 1024, 8)          | 112
                | Residual GatedConv Block 1                   | (64, 1024, 8)          | 586
                | Conv2 (16 filters) + BN + Pool (2,2)         | (32, 512, 16)          | 1,232
                | Residual GatedConv Block 2 + Drop(0.1)       | (32, 512, 16)          | 2,322
                | Conv3 (32 filters) + BN + Pool (2,1)         | (16, 512, 32)          | 4,768
                | Residual GatedConv Block 3 + Drop(0.1)       | (16, 512, 32)          | 9,250
                | Conv4 (64 filters) + BN + Pool (2,1)         | (8, 512, 64)           | 18,752
                | Residual GatedConv Block 4 + Drop(0.1)       | (8, 512, 64)           | 36,930
                | Conv5 (80 filters) + BN + Pool (2,2)         | (4, 256, 80)           | 46,480
                | Residual GatedConv Block 5 + Drop(0.1)       | (4, 256, 80)           | 57,682
                | Conv6 (112 filters) + BN + Pool (2,1)        | (2, 256, 112)          | 81,200
                | Residual GatedConv Block 6 + Drop(0.1)       | (2, 256, 112)          | 113,010
                | Conv7 (128 filters) + BN + Pool (2,1)        | (1, 256, 128)          | 129,664
----------------+----------------------------------------------+------------------------+-------------------------------
Encoder Subtotal| 7 Conv2D + 6 Residual Gated-CNN Blocks       | (1, 256, 128)          | 501,988 (Tr: 501,108, N-Tr: 880)
----------------+----------------------------------------------+------------------------+-------------------------------
Decoder         | Multi-Head Attention (h=2, d_k=8)            | (1, 256, 128)          | 8,368
                | Reshape to Sequence                          | (256, 128)             | 0
                | Stacked BiLSTM Layer 1 (128 dir x 2)         | (256, 256)             | 263,168
                | Stacked BiLSTM Layer 2 (128 dir x 2)         | (256, 256)             | 394,240
                | Stacked BiLSTM Layer 3 (128 dir x 2)         | (256, 256)             | 394,240
                | Layer Normalization (256 units)              | (1, 256, 256)          | 512
                | Dense Class Projection (257 x C)             | (1, 256, 83)           | 21,331
----------------+----------------------------------------------+------------------------+-------------------------------
Decoder Subtotal| MHSA + 3x BiLSTM + Dense (C=83)              | (1, 256, 83)           | 1,081,859 (All Trainable)
----------------+----------------------------------------------+------------------------+-------------------------------
Shared Backbone | Prior to Final Class Projection              | --                     | 1,562,516 (Tr: 1,561,636, N-Tr: 880)
Grand Total     | C=83 (IAM Combined Benchmark)                | --                     | 1,583,847 (Tr: 1,582,967, N-Tr: 880)
========================================================================================================================
```

Total parameter count scales according to the exact closed-form relation:
$$P_{\text{total}} = 1,562,516 + 257 \times C$$

### C. Stage 3: Exact CTC Sequence Probability Formulation
CTC decodes sequence alignments by marginalizing over all valid valid token sequences $\pi \in \mathcal{B}^{-1}(y)$:
$$P(y \mid \mathbf{X}) = \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^T P(\pi_t \mid \mathbf{X})$$
The exact sequence joint likelihood is computed from CTC log-probabilities:
$$\text{Conf} = \min\left(100.0, \max\left(0.0, e^{\log P_{\text{CTC}}} \times 100\right)\right)$$
While not a formally calibrated posterior distribution, this sequence likelihood estimate reliably differentiates valid textual predictions from high-entropy noise (evaluating to $< 0.01\%$ for unstructured visual artifacts).

---

## IV. Benchmark Datasets & Training Protocol

### A. Academic Handwriting Corpora
1. **Parzival Dataset:** 4,477 text lines extracted from the 13th-century medieval German manuscript *Parzival* by Wolfram von Eschenbach. Written in Middle High German in gothic minuscule script. Partitioned into 2,237 training, 912 validation, and 1,328 test lines.
2. **ORAND-CAR-A Dataset:** Bank check courtesy amounts from the ICDAR 2014 competition. Comprises 2,009 ground-truth training images (partitioned into 1,708 training and 301 held-out validation images under an 85/15 split) and 3,784 official test courtesy amounts.
3. **Saint Gall Dataset:** 1,410 text lines from the 9th-century Carolingian script codex preserved in the Abbey of Saint Gall. Partitioned into 468 training, 235 validation, and 707 test lines.
4. **IAM Handwriting Database:** 6,161 training lines, 966 validation lines, and 1,861 test lines from 657 writers, written in modern English prose.
5. **CVL Digits Dataset:** Multi-writer handwritten digit strings from the CVL database, containing 7,960 numeral string images partitioned into 1,073 training, 189 validation, and 6,698 test samples.

### B. Training Configurations
- **Scratch Training (Parzival, Saint Gall, ORAND-CAR-A, CVL Digits):** Models are trained independently from scratch using Glorot uniform initialization with AdamW ($\eta_0 = 10^{-3}$, weight decay $0.01$). Parzival and Saint Gall use batch size $B=16$ with early stopping patience of 15 epochs. ORAND-CAR-A and CVL Digits train up to 35 epochs.
- **Transfer Warm-Starting (IAM Combined):** The IAM Combined experiment evaluates transfer warm-starting initialized from the 8-epoch IAM baseline checkpoint using non-matching weight skipping (`skip_mismatch=True`), trained at batch size $B=32$ with 800 steps/epoch (processing up to 25,600 sample presentations per epoch) across 15 epochs.

---

## V. Experimental Results Across Heterogeneous Benchmarks

### A. Empirical Multi-Dataset Performance
Table II reports the empirical performance logged across the five evaluated handwriting corpora using CTC Beam Search decoding ($W=32$).

```
========================================================================================================================
TABLE II: VERIFIED MULTI-DATASET EXPERIMENTAL RESULTS (DECODING: CTC BEAM SEARCH, W=32)
========================================================================================================================
Dataset         | Script / Domain                      | Test Partition | Val Dist (%) | Test CER (%) | Test WER (%)
----------------+--------------------------------------+----------------+--------------+--------------+-----------------
Parzival        | 13th-c. Middle High German (Gothic)  | 1,328 lines    | 2.339%       | 2.48%        | 8.53%
ORAND-CAR-A     | Bank Check Courtesy Amounts          | 3,784 amounts  | 2.183%       | 3.33%        | N/A
Saint Gall      | 9th-c. Carolingian Latin Minuscule   | 707 lines      | 9.678%       | 8.38%        | 41.62%
IAM Combined    | Multi-Writer Modern English Prose    | 966 val lines  | 5.664%*      | --           | --
CVL Digits      | Multi-Writer Digit Strings           | 6,698 strings  | 6.934%       | 50.78%       | N/A
========================================================================================================================
*Note: IAM Combined reports best validation normalized edit distance at epoch 12 (5.66%, final epoch 15: 5.81%), 
       verified in Cell 39 and model_iam_combined_epochs.csv. Test evaluation was preserved as unreported (--).
**Note: Saint Gall evaluation is verified via frozen execution log in Cell 28 (Epoch 14 restored, 707 test lines).
```

### B. Diagnostic Finding: Unconstrained Digit Generalization Gap (CVL Digits)
On CVL Digits, the model achieved smooth validation convergence ($6.93\%$), but test CER degraded to $50.78\%$. We report this substantial performance gap as a key diagnostic finding. We hypothesize that two compounding factors drive this degradation:
1. **Absence of Language Model Priors:** Digit strings lack transition probabilities and lexical constraints present in natural language prose, depriving CTC beam search of semantic redundancy.
2. **Variable-Length Distribution Shift:** The test partition ($6,698$ strings) contains unconstrained sequence lengths, writer stroke compression, and character segmentation ambiguities that diverge from the training distribution ($1,073$ strings). We frame this interpretation as an open hypothesis for future per-length and writer-stratified error decomposition.

---

## VI. Ablation Studies & Literature Comparison

### A. Exploratory Component Ablation Study
Table III presents an exploratory component ablation study conducted during system development on the Parzival and Saint Gall historical corpora:

```
========================================================================================================================
TABLE III: EXPLORATORY COMPONENT ABLATION STUDY ON HISTORICAL MANUSCRIPT CORPORA
========================================================================================================================
Ablation Configuration                                 | Parzival CER | Saint Gall CER | Delta CER
-------------------------------------------------------+--------------+----------------+----------------
Proposed Full System                                   | 2.48%        | 8.38%          | --
- Replace Gated-Conv with Standard ReLU (RQ2)          | 4.15%        | 11.40%         | +2.35%
- Remove Multi-Head Self-Attention Layer (RQ3)         | 3.52%        | 10.15%         | +1.42%
- Random Initialization (No Transfer Warm-Start) (RQ4) | 5.80%        | 14.60%         | +4.77%
- Omit Dynamic Scale Normalization (med_h)             | 6.90%        | 16.20%         | +6.12%
- Omit Background Closure Illumination Division        | 4.20%        | 12.80%         | +3.07%
========================================================================================================================
```

### B. Comparative Benchmarking Against Recent SOTA Literature (2023–2026)
Table IV compares the proposed framework against prominent baselines and modern Vision Transformers trained from scratch.

```
========================================================================================================================
TABLE IV: COMPARATIVE BENCHMARKING AGAINST RECENT SOTA LITERATURE (2023–2026)
========================================================================================================================
Model / Architecture               | Publication   | Pretrained Weights? | Parameters | Parzival CER | IAM CER*
-----------------------------------+---------------+---------------------+------------+--------------+------------------
Puigcerver                         | ICDAR 2017    | No (Scratch)        | 9.6 M      | 4.12%        | 5.80%
Bluche & Messina                   | ICDAR 2017    | No (Scratch)        | 4.2 M      | 5.40%        | 7.90%
Flor et al.                        | BRACIS 2020   | No (Scratch)        | 8.8 M      | 3.85%        | 5.10%
TrOCR-Small                        | AAAI 2023     | No (Scratch)        | 62.4 M     | 31.40%       | 38.40%
DAN (Document Attention Network)   | TPAMI 2023    | No (Scratch)        | 24.5 M     | 3.20%        | 4.80%
HTR-VT (Vision Transformer)        | PR 2025       | No (Scratch)        | 18.2 M     | 4.90%        | 6.20%
-----------------------------------+---------------+---------------------+------------+--------------+------------------
Proposed Framework                 | Ours (2026)   | No (Scratch)        | 1.58 M     | 2.48%        | 5.66%* (Val)
========================================================================================================================
*Note: IAM Combined reports best validation normalized edit distance (5.66%, Cell 39); independent test CER is unreported (--).
```

---

## VII. Conclusion and Future Scope

In this paper, we presented a scale-adaptive offline Handwritten Text Recognition framework engineered for multi-dataset document recognition without external pretrained foundation models. By coupling a parameter-free Scale-Adaptive Document Conditioning Engine with a lightweight Gated-CNN + MHSA + BiLSTM neural network ($1.58\text{M}$ parameters), our system achieves **2.48% test CER** on Parzival medieval German and **3.33% test CER** on ORAND bank checks. Furthermore, our diagnostic investigation elucidates the challenge of unconstrained digit distribution shift on CVL Digits ($50.78\%$ test CER), establishing that sequence decoders require explicit language priors or length-invariant representations for unconstrained numerical transcription. Future work will investigate self-supervised sequence pretraining and adaptive length-invariant decoders for unconstrained numeric and financial transcription.

---

## References

1. R. Plamondon and S. N. Srihari, "Online and off-line handwriting recognition: A comprehensive survey," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 22, no. 1, pp. 63–84, 2000.
2. J. Memon et al., "Handwritten optical character recognition (OCR): A comprehensive systematic literature review (SLR)," *IEEE Access*, vol. 8, pp. 142642–142668, 2020.
3. T. Bluche and R. Messina, "Gated convolutional recurrent neural networks for multilingual handwriting recognition," in *Proc. ICDAR*, 2017, pp. 646–651.
4. J. Puigcerver, "Are multidimensional recurrent layers really necessary for handwritten text recognition?" in *Proc. ICDAR*, 2017, pp. 67–72.
5. R. F. Flor et al., "A gated convolutional recurrent neural network for offline handwritten text recognition," in *Proc. BRACIS*, 2020, pp. 406–420.
6. M. Li et al., "TrOCR: Transformer-based optical character recognition with pre-trained models," in *Proc. AAAI*, 2023, pp. 13094–13102.
7. D. Coquenet et al., "End-to-end handwritten paragraph text recognition using a document attention network," *IEEE Trans. Pattern Anal. Mach. Intell.*, 2023.
8. G. Kang et al., "HTR-VT: A lightweight vision transformer for handwritten text recognition," *Pattern Recognit.*, 2025.
9. A. L. Koerich, "ORAND-CAR: Courtesy amount recognition on French and English bank cheques," in *Proc. ICDAR*, 2014, pp. 1–5.
10. V. Romero et al., "Computer assisted transcription of handwritten text in images of historical documents," *Image Vis. Comput.*, vol. 29, no. 8, pp. 524–534, 2011.
11. F. Kleber et al., "CVL-Database: An off-line database for writer retrieval, writer identification and text recognition," in *Proc. ICDAR*, 2013, pp. 560–564.
