Notebook execution order and naming convention:

- `10_*`: data acquisition and dataset setup
- `20_*`: exploratory data analysis
- `30_*`: model training
- `40_*`: dataset-specific evaluation
- `50_*`: cross-model comparison

Current notebooks:

- `10_download_data.ipynb`
- `11_download_additional_datasets.ipynb`
- `20_signal_eda.ipynb`
- `30_lstm_rml2016.ipynb`
- `31_lstm_rml2018.ipynb`
- `32_lstm_deepradar2022.ipynb`
- `33_vgg_spectrogram_noisy_drone_rf_v2.ipynb`
- `40_evaluation_rml2016.ipynb`
- `41_evaluation_rml2018.ipynb`
- `42_evaluation_deepradar2022.ipynb`
- `43_evaluation_cross_dataset_ensemble.ipynb`
- `44_evaluation_noisy_drone_rf_v2.ipynb`
- `50_evaluation_comparison.ipynb`

Notes for Noisy Drone RF v2:

- `33_vgg_spectrogram_noisy_drone_rf_v2.ipynb` trains/evaluates the VGG full-complex spectrogram model.
- `44_evaluation_noisy_drone_rf_v2.ipynb` is eval-only for the canonical VGG model.
- `50_evaluation_comparison.ipynb` has a dedicated Noisy Drone RF v2 eval-only cell, then a final comparison cell.
- Saved result artifacts live in `outputs/noisy_drone_rf_v2_eval/` and `outputs/50_evaluation_comparison/`; notebooks should stay cleared before commit.

Notes for `31_lstm_rml2018.ipynb`:

- Cell 1: base RML2018 training (auto-selects GPU-safe batch size).
- Cell 2: short fine-tuning + checkpoint selection (auto-selects GPU-safe batch size).
- Cell 3: standalone 500-epoch continuation cell that can run from a fresh kernel.
- Cell 3 saves consolidated metrics for later plotting:
  - `outputs/rml2018/rml2018_checkpoint_metrics.json`
  - `outputs/rml2018/rml2018_checkpoint_metrics.csv`
