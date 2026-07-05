# Dataset Card: DeepRadar2022

## Source

DeepRadar2022 radar waveform dataset, loaded from local MATLAB/HDF5 files:

- `X_train.mat`, `Y_train.mat`, `lbl_train.mat`
- `X_val.mat`, `Y_val.mat`, `lbl_val.mat`
- `X_test.mat`, `Y_test.mat`, `lbl_test.mat`

The evaluation config uses the test files under `configs/deepradar2022_eval.yaml`.

## Class List

23 radar waveform classes represented by the dataset label matrices. The repo treats these as numeric classes unless an external label-name mapping is provided.

## SNR Range

SNR metadata is read from the `lbl_*` matrices. The exact range should be reported from the local dataset artifact when regenerating metrics.

## Sample Count

The committed comparison table records DeepRadar2022 all-test metrics but does not record the sample count. Regenerate the evaluation artifacts before making publication-grade sample-count claims.

## Preprocessing Assumptions

- Stored arrays are converted to `(examples, 1024, 2)` I/Q.
- Some workflows append a derived envelope channel, producing `(1024, 3)`.
- Labels are read from one-hot `Y_*` files.
- SNR-aware plots use the metadata in `lbl_*`.

## License / Usage Limitations

Use according to the original DeepRadar2022 dataset terms. This repository does not grant additional dataset redistribution rights.

## Leakage Risks

- Keep provided train/validation/test files separate.
- Do not mix derived windows from the same source example across splits.
- Clearly distinguish radar waveform metrics from modulation or drone-RF metrics.

## Recommended Evaluation Split

Use the provided test split for headline metrics and report:

- model checkpoint
- feature representation, especially whether envelope channel is used
- SNR range
- test sample count
- per-class and macro metrics
