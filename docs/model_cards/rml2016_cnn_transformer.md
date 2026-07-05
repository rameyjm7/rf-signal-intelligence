# Model Card: RML2016 CNN-Transformer

## Summary

| Field | Value |
|---|---|
| Model id | `cnn_transformer` |
| Dataset | RML2016.10A |
| Architecture | CNN-transformer experiment |
| Task | 11-way modulation classification |
| Primary artifact | `outputs/rml2016/rml2016_cnn_transformer_comparison.json` when regenerated locally |
| Status | Comparison artifact, not the primary maintained CLI checkpoint |

## Intended Use

Evaluate modulation-classification performance on RML2016.10A and compare full-distribution performance against higher-SNR slices. This model is useful for showing the effect of SNR on RFML classification.

## Input Shape

RML2016 examples are `(128, 2)` I/Q sequences. Some repo models append SNR as a third feature channel, producing `(128, 3)`. The CNN-transformer comparison should document the exact input transform used when the local artifact is regenerated.

## Preprocessing

- Load `RML2016.10a_dict.pkl`.
- Convert dictionary keys `(modulation, snr)` into examples and labels.
- Normalize or reshape I/Q according to the notebook/training cell.
- Preserve SNR metadata for all-SNR and high-SNR slice reporting.

## Classes

`8PSK`, `AM-DSB`, `AM-SSB`, `BPSK`, `CPFSK`, `GFSK`, `PAM4`, `QAM16`, `QAM64`, `QPSK`, `WBFM`.

## Training Split

The comparison row uses an RML2016 train/test evaluation artifact from the notebook workflow. The standard repo evaluation protocol uses stratified train/test splitting where applicable and reports both all-SNR and high-SNR slices.

## Evaluation Protocol

Two protocols are reported in the README comparison table:

- Full-distribution all-SNR test.
- High-SNR test slice with `SNR > -2 dB`.

## Headline Metrics

| Split | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---:|---:|---:|---:|
| All SNR levels | 0.6645 | 0.6592 | 0.6592 | 44,000 |
| SNR > -2 dB | 0.8969 | 0.8900 | 0.8913 | Not recorded in comparison CSV |

These rows are not directly comparable to NoisyDroneRFv2 or DeepRadar2022 because the label space, sample length, and dataset generation process differ.

## Known Limitations

- The CNN-transformer artifact is a comparison result, not the main maintained RML2016 model registry entry.
- Full-distribution accuracy is strongly affected by very low SNR examples.
- High-SNR metrics should be labeled as filtered-slice metrics, not whole-dataset performance.
- RML2016 synthetic channel conditions do not represent all real receiver impairments.

## Latency Target

No hard latency target is established for this artifact. The model is intended for offline comparison unless exported and benchmarked separately.

## Export / Deployment Status

No committed export artifact is documented for this specific CNN-transformer comparison row. The repo contains more maintained Keras checkpoints for RML2016 LSTM-style models under `models/rml2016/`.
