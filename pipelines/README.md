Pipeline execution order and naming convention:

- `10_*`: data acquisition and dataset setup
- `20_*`: exploratory data analysis
- `30_*`: model training
- `40_*`: dataset-specific evaluation
- `50_*`: cross-model comparison

Pipeline convention:

- Keep pipelines as thin orchestration and visualization layers.
- Put reusable dataset loading, feature extraction, evaluation, and artifact-writing code under `src/rf_signal_intelligence/`.
- Prefer the `rfsi` CLI for reproducible runs that should not require opening a pipeline script.

Current pipelines:

- `10_download_data.py`
- `20_signal_eda.py`
- `30_lstm_rml2016.py`
- `31_lstm_rml2018.py`
- `32_lstm_deepradar2022.py`
- `33_vgg_spectrogram_noisy_drone_rf_v2.py`
- `34b_fast_high_snr_rfuav.py`
- `40_evaluation_rml2016.py`
- `41_evaluation_rml2018.py`
- `42_evaluation_deepradar2022.py`
- `43_evaluation_cross_dataset_ensemble.py`
- `44_evaluation_noisy_drone_rf_v2.py`
- `45_evaluation_rfuav.py`
- `50_evaluation_comparison.py`
- `63b_gan_drone_iq_generator.py`

Notes for Noisy Drone RF v2:

- `33_vgg_spectrogram_noisy_drone_rf_v2.py` documents the reusable VGG full-complex spectrogram workflow and CLI commands.
- `44_evaluation_noisy_drone_rf_v2.py` calls the reusable eval workflow for the canonical VGG model.
- `50_evaluation_comparison.py` calls the reusable cross-dataset comparison workflow.
- Saved result artifacts live in `outputs/noisy_drone_rf_v2_eval/`, `outputs/50_evaluation_comparison/`, and `outputs/pipeline_figures/`.

Notes for `31_lstm_rml2018.py`:

- Cell 1: base RML2018 training (auto-selects GPU-safe batch size).
- Cell 2: short fine-tuning + checkpoint selection (auto-selects GPU-safe batch size).
- Cell 3: standalone 500-epoch continuation cell that can run from a fresh kernel.
- Cell 3 saves consolidated metrics for later plotting:
  - `outputs/rml2018/rml2018_checkpoint_metrics.json`
  - `outputs/rml2018/rml2018_checkpoint_metrics.csv`
