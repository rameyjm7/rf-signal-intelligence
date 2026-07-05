# Model Card: DeepRadar2022 CNN-Transformer

## Summary

| Field | Value |
|---|---|
| Model id | `cnn_transformer_deepradar2022_continued_stage2` |
| Dataset | DeepRadar2022 |
| Architecture | CNN-transformer experiment |
| Task | Radar waveform classification |
| Primary artifact | `outputs/deepradar2022/deepradar2022_cnn_transformer_stage2_result.json` when regenerated locally |
| Status | Comparison artifact |

## Intended Use

Evaluate radar waveform classification performance on DeepRadar2022 and provide a cross-domain comparison point next to modulation-recognition datasets.

## Input Shape

DeepRadar2022 samples are represented as `(1024, 2)` I/Q sequences in the dataset. Some repo workflows derive a third envelope channel, producing `(1024, 3)`.

## Preprocessing

- Load `X_test.mat`, `Y_test.mat`, and `lbl_test.mat`.
- Convert stored arrays to `(examples, 1024, 2)`.
- Use class labels from `Y_*` one-hot targets.
- Use `lbl_*` metadata for SNR-aware analysis.
- For maintained CNN-BiLSTM evaluation, append an I/Q envelope channel as the third feature.

## Classes

23 radar waveform classes represented by numeric labels in the dataset files. The repo reads labels directly from the MATLAB artifacts rather than maintaining a committed text class list.

## Training Split

DeepRadar2022 provides train/validation/test files in local dataset paths. The comparison artifact is evaluated against the test split.

## Evaluation Protocol

Current comparison protocol:

- Load the continued stage-2 CNN-transformer result artifact.
- Report all-test accuracy, macro F1, and weighted F1 in the cross-dataset comparison table.

## Headline Metrics

| Split | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---:|---:|---:|---:|
| All test | 0.4461 | 0.3973 | 0.3973 | Not recorded |

The integration check for the maintained DeepRadar2022 CNN-BiLSTM checkpoint also tests highest-SNR behavior, but that is a separate model/protocol from this comparison card.

## Known Limitations

- This card describes the CNN-transformer comparison artifact, while the model registry also contains a CNN-BiLSTM checkpoint.
- Class names are numeric unless mapped externally.
- The dataset is radar waveform data, so metrics are not directly comparable to drone/controller RF or modulation-recognition tasks.
- The headline metric is from a saved comparison artifact; regenerate before using in formal claims.

## Latency Target

No live latency target is defined. The model is currently documented as an offline comparison artifact.

## Export / Deployment Status

No committed ONNX/TensorRT export is documented for this comparison artifact.
