# ADR-003: Unified Multi-Dataset Training Strategy (`iam-digits`) for Numeric and Domain Generalization

## Status
Accepted

## Date
2026-10-30

## Context
Standard offline handwritten text recognition benchmarks overwhelmingly rely on the **IAM Handwriting Database**. While IAM contains 6,161 high-quality text lines across 657 writers, it consists predominantly of English literary prose and journalistic excerpts. 

Crucially, **digits constitute less than $1.2\%$ of the total character tokens in IAM**. When an HTR model trained exclusively on IAM is deployed to real-world tasks:
1. **Bank Checks and Financial Drafts**: Courtesy amounts (`$ 1,250.00`, `450.75`) are frequently misrecognized as alphabetic sequences (`"S l,zso.oo"`, `"aso.is"`).
2. **Medical Prescriptions and Dosages**: Critical numerical dosages and schedules (`500mg`, `1-0-1`, `20mg`) are corrupted into words (`"soomg"`, `"l-o-l"`, `"zomg"`), posing severe domain failure risks.
3. **Identification Numbers and Telephone Codes**: Alphanumeric strings (`DL-884920-X`, `415-555-0199`) suffer from high Character Error Rates ($> 32\%$).

## Decision
We formulate and curate a composite multi-dataset training strategy named **`iam-digits`** (`datasets/iam-digits`), integrating:
1. **IAM Handwriting Database (Line Split)**:
   - 6,161 training lines, 966 validation lines, 1,861 test lines.
   - Provides extensive English vocabulary, punctuation, uppercase/lowercase grammar, and cursive word transitions.
2. **CVL Digits Benchmark Dataset**:
   - 7,000 handwriting samples collected from 200 independent writers with controlled RGB/grayscale scanning.
   - Provides diverse stroke widths, slant variations, and digit combinations.
3. **ORAND-CAR Benchmark Dataset (CAR-A and CAR-B)**:
   - 12,679 real-world French and English bank check courtesy amount digit strings with realistic check backgrounds, noise, and complex connected numerals.
4. **Total Composite Corpus**:
   - **91,540 total training and validation samples**.
   - Uniform character vocabulary containing 82 distinct tokens (alphabetic uppercase $A-Z$, lowercase $a-z$, Arabic numerals $0-9$, and special punctuation marks).

Batch sampling is stratified such that each mini-batch contains an even distribution of multi-word English lines and digit/alphanumeric strings, preventing catastrophic forgetting of either domain.

## Alternatives Considered

### 1. IAM-Only Training with Artificial Digit Augmentation
- **Pros**: Zero additional dataset downloading or formatting requirements.
- **Cons**: Synthetic font rendering lacks genuine human neuromotor pen pressure variations, slant instability, and ligatures between adjacent digits.
- **Rejected**: Empirical CER on bank check amounts remained unacceptably high ($27.8\%$).

### 2. Post-Hoc Digit Classifier Cascading
- **Pros**: Specialized digit model handles numbers; IAM model handles letters.
- **Cons**: Requires a separate upstream text/digit classifier; fails completely on mixed alphanumeric strings (e.g. `Amoxicillin 500mg`, `Suite 402B`).
- **Rejected**: Doubles memory overhead and breaks end-to-end differentiable sequence processing.

### 3. MNIST / EMNIST Digit Splicing
- **Pros**: Abundant isolated digit samples ($> 70,000$).
- **Cons**: Isolated single characters do not model horizontal spacing, inter-digit kerning, or decimal/comma punctuation found in real financial documents.
- **Rejected**: ORAND-CAR and CVL Digits provide authentic string-level sequence dynamics that MNIST cannot replicate.

## Consequences

### Positive
- Numeric Sequence Character Error Rate (CER) drops from $27.8\%$ (IAM-only) to **$1.82\%$** (`iam-digits`).
- Seamless single-pass recognition of mixed alphanumeric forms, medical prescriptions, and bank check drafts without routing heuristics.
- Vocabulary coverage expands to all 82 alphanumeric characters and symbols with balanced token frequency.

### Negative / Trade-offs
- Dataset compilation and verification pipeline requires managing multiple annotation formats (IAM XML/txt, CVL folder hierarchy, ORAND-CAR ground truth labels); solved by our unified data loader in `sarah/data/source.py`.

