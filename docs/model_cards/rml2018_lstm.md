# Model Card: RML2018 LSTM

## Summary

| Field | Value |
|---|---|
| Model id | `rml2018_lstm_rnn` |
| Checkpoint | `models/rml2018/rml2018_lstm_rnn.keras` |
| Dataset | RML2018.01A |
| Architecture | LSTM-RNN |
| Task | 24-way modulation classification |
| Status | Maintained evaluation checkpoint |

## Intended Use

Classify RML2018.01A modulation examples and provide a larger-scale modulation-recognition benchmark than RML2016. It is intended for offline evaluation and comparison, not live SDR drone detection.

## Input Shape

`(1024, 3)` float32 tensor.

The first two channels are I/Q. The third channel is SNR repeated across the time dimension.

## Preprocessing

- Load `GOLD_XYZ_OSC.0001_1024.hdf5`.
- Read I/Q from `X`, labels from `Y`, and SNR from `Z`.
- Append SNR as a third per-timestep feature channel.
- Resolve class-order ambiguity using `classes-fixed.txt` when available.
- Optionally filter SNR according to `configs/rml2018_eval.yaml`.

## Classes

24 modulation classes from the local `classes.txt` / `classes-fixed.txt` files. The class-order file is part of the evaluation protocol because class-order ambiguity can materially change metrics.

## Training Split

The current config uses a filtered evaluation setup:

- `snr_min_db=-6`
- `snr_max_db=30`
- `max_samples_per_class=3000`
- `test_size=0.20`
- random seed `42`

## Evaluation Protocol

Current repo protocol:

- Evaluate the pinned Keras checkpoint on the filtered RML2018 split.
- Calibrate/verify label order against the fixed class mapping.
- Report checkpoint accuracy in the cross-dataset comparison table.

## Headline Metrics

| Split | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---:|---:|---:|---:|
| Current filtered all-test protocol | 0.8295 | Not recorded | Not recorded | Not recorded |

Older README text also retains a legacy full-distribution baseline around 0.72 accuracy over 72,000 evaluation samples.

## Known Limitations

- RML2018 class-order handling is easy to get wrong; use `classes-fixed.txt` when reproducing reported metrics.
- Metrics are not directly comparable to RML2016 because sample length, class set, and dataset scale differ.
- SNR as an input feature helps benchmark performance but may not be available or calibrated in every deployment.
- The current card records the available repo metrics; full per-class metrics should be regenerated for publication.

## Latency Target

No live latency target is defined. The model is suitable for batch evaluation and could be benchmarked for streaming use after export.

## Export / Deployment Status

Keras checkpoint exists in the model registry. ONNX/TensorRT export is not currently documented for this checkpoint.
