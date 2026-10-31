# Research and evidence notes

## Literature coverage (2020–2026)

| Year | Work | Research direction | Use in this manuscript |
|---|---|---|---|
| 2020 | Yousef et al., gated fully convolutional HTR, ICFHR | Gated convolution + CTC; recurrence-free line recognition | Prior gated-convolution direction; distinguish from this model, which retains BiLSTM. DOI: 10.1109/ICFHR2020.2020.00015 |
| 2021 | Hernandez Diaz et al., *Rethinking Text Line Recognition Models* | Comparison of BiLSTM, self-attention and GRCL encoders with CTC/Transformer decoders | Supports the encoder/decoder design discussion; 2.75% IAM CER configuration uses internal + public data and an explicit language model. arXiv:2104.07787 |
| 2022 | Stutzmann et al., historical document image dataset survey | Dataset characteristics and protocol heterogeneity | Supports caution when comparing historical corpora. DOI: 10.1007/s10032-022-00405-8 |
| 2023 | Li et al., TrOCR, AAAI | Pretrained image and text Transformers | IAM model sizes/results: Small 62M/4.22% CER; Base 334M/3.42%; Large 558M/2.89%. DOI: 10.1609/aaai.v37i11.26538 |
| 2023 | Coquenet et al., Vertical Attention Network, TPAMI | Paragraph-level recognition with vertical attention | Related end-to-end paragraph method; do not compare paragraph CER directly with line-level results. DOI: 10.1109/TPAMI.2022.3144899 |
| 2023 | Coquenet et al., DAN, TPAMI | Segmentation-free document recognition | Related document-level attention architecture. DOI: 10.1109/TPAMI.2023.3235826 |
| 2024 | Fujitake, DTrOCR, WACV | Decoder-only pretrained Transformer OCR | Autoregressive alternative to CTC. DOI: 10.1109/WACV57701.2024.00784 |
| 2024 | Mas-Candela and Calvo-Zaragoza, IJDAR | Recursive neural networks for compact HTR | Relevant compact-model literature. DOI: 10.1007/s10032-024-00481-y |
| 2024/2025 | Li et al., HTR-VT, Pattern Recognition | Data-efficient Vision Transformer HTR | Published IAM result reported as 4.7% CER / 14.9% WER with 53.5M parameters in the 2026 LoGo-HTR comparison table. DOI: 10.1016/j.patcog.2024.110967 |
| 2024 | Memon et al., HTR survey, Journal of Imaging | HTR datasets, methods and challenges | Broad dataset/method context. DOI: 10.3390/jimaging10010018 |
| 2026 | Mitra et al., LoGo-HTR, WACV | Self-supervised local contrastive and global decorrelation learning | IAM result at 100% labelled fine-tuning: 2.01% CER / 10.27% WER; 6.4M parameters; uses external unlabeled handwriting for SSL pretraining. DOI: 10.1109/WACV61042.2026.00643 |

## Comparability cautions

- The 2.75% IAM result from *Rethinking Text Line Recognition Models* is reported for internal + public training data and an explicit character n-gram language model. It is not a clean same-data baseline.
- TrOCR uses pretrained image and text Transformers and synthetic pretraining. Its model scale and data regime differ from the current compact model.
- HTR-VT is a Vision Transformer encoder method. The IAM figures quoted above are reported in the LoGo-HTR 2026 comparison table.
- LoGo-HTR uses a large external unlabeled handwriting dataset for self-supervised pretraining. Its 2.01% IAM CER is not a matched baseline for this project's validation-only IAM Combined run.
- Vertical Attention Network and DAN perform paragraph/document-level recognition. Their results should not be ranked directly against line/string-level test results.

## Source links

- https://doi.org/10.1109/ICFHR2020.2020.00015
- https://arxiv.org/abs/2104.07787
- https://doi.org/10.1007/s10032-022-00405-8
- https://doi.org/10.1609/aaai.v37i11.26538
- https://doi.org/10.1109/TPAMI.2022.3144899
- https://doi.org/10.1109/TPAMI.2023.3235826
- https://doi.org/10.1109/WACV57701.2024.00784
- https://doi.org/10.1007/s10032-024-00481-y
- https://doi.org/10.1016/j.patcog.2024.110967
- https://doi.org/10.3390/jimaging10010018
- https://doi.org/10.1109/WACV61042.2026.00643
