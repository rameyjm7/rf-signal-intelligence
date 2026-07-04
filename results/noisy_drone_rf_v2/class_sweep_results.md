# Live OTA Noisy Drone RF Classification Results

Generated: `2026-07-04T14:49:53+00:00`

This test replays labeled NoisyDroneRF IQ samples over the air from one SDR and classifies the received live RF capture from another SDR. The result below is an end-to-end TX/RX hardware classification check, not just offline inference on dataset files.

## Summary

- Trials: `18`
- Exact final prediction matches: `18/18`
- Accuracy: `1.000`
- Classes: `DJI, FutabaT14, FutabaT7, Graupner, Taranis, Turnigy`
- CSV: `../../outputs/class_sweep.csv`
- RX IQ windows: `../../outputs/class_sweep_iq`
- Waterfall snapshots: `../../outputs/class_sweep_plots`

## OTA SDR Setup

| Setting | Value |
|---|---:|
| Model | `models/noisy_drone_rf_v2/noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras` |
| TX SDR | `driver=bladerf,serial=7faa712b1fab42f4b84e494171b91721` |
| TX frontend | `bladeRF TX1` |
| TX antenna | `TX` |
| RX SDR | `driver=hackrf` |
| RX frontend | `RX channel 0` |
| RX antenna | `` |
| Frequency | `2399000000 Hz` |
| Sample rate | `20000000 S/s` |
| Bandwidth | `20000000 Hz` |
| RX gain | `60.0` |
| TX gain | `60.0` |
| TX amplitude | `0.2` |
| TX min SNR | `20` |
| Window samples | `1048576` |
| Capture samples | `4194304` |
| Window score mode | `auto` |
| Decision mode | `hybrid` |
| Non-noise threshold | `0.55` |

## Confusion Matrix

Rows are transmitted dataset labels. Columns are final live OTA predictions.

| TX \ RX | DJI | FutabaT14 | FutabaT7 | Graupner | Taranis | Turnigy |
|---|---:|---:|---:|---:|---:|---:|
| DJI | 3 | 0 | 0 | 0 | 0 | 0 |
| FutabaT14 | 0 | 3 | 0 | 0 | 0 | 0 |
| FutabaT7 | 0 | 0 | 3 | 0 | 0 | 0 |
| Graupner | 0 | 0 | 0 | 3 | 0 | 0 |
| Taranis | 0 | 0 | 0 | 0 | 3 | 0 |
| Turnigy | 0 | 0 | 0 | 0 | 0 | 3 |

## Waterfall Snapshots

Each image is rendered from the selected live RX IQ window used for classification. The overlay shows the transmitted class, final prediction, confidence, and capture power.

### Trial 1: DJI -> DJI

![Waterfall trial 1](../../outputs/class_sweep_plots/001_DJI_waterfall.png)

### Trial 2: DJI -> DJI

![Waterfall trial 2](../../outputs/class_sweep_plots/002_DJI_waterfall.png)

### Trial 3: DJI -> DJI

![Waterfall trial 3](../../outputs/class_sweep_plots/003_DJI_waterfall.png)

### Trial 4: FutabaT14 -> FutabaT14

![Waterfall trial 4](../../outputs/class_sweep_plots/004_FutabaT14_waterfall.png)

### Trial 5: FutabaT14 -> FutabaT14

![Waterfall trial 5](../../outputs/class_sweep_plots/005_FutabaT14_waterfall.png)

### Trial 6: FutabaT14 -> FutabaT14

![Waterfall trial 6](../../outputs/class_sweep_plots/006_FutabaT14_waterfall.png)

### Trial 7: FutabaT7 -> FutabaT7

![Waterfall trial 7](../../outputs/class_sweep_plots/007_FutabaT7_waterfall.png)

### Trial 8: FutabaT7 -> FutabaT7

![Waterfall trial 8](../../outputs/class_sweep_plots/008_FutabaT7_waterfall.png)

### Trial 9: FutabaT7 -> FutabaT7

![Waterfall trial 9](../../outputs/class_sweep_plots/009_FutabaT7_waterfall.png)

### Trial 10: Graupner -> Graupner

![Waterfall trial 10](../../outputs/class_sweep_plots/010_Graupner_waterfall.png)

### Trial 11: Graupner -> Graupner

![Waterfall trial 11](../../outputs/class_sweep_plots/011_Graupner_waterfall.png)

### Trial 12: Graupner -> Graupner

![Waterfall trial 12](../../outputs/class_sweep_plots/012_Graupner_waterfall.png)

### Trial 13: Taranis -> Taranis

![Waterfall trial 13](../../outputs/class_sweep_plots/013_Taranis_waterfall.png)

### Trial 14: Taranis -> Taranis

![Waterfall trial 14](../../outputs/class_sweep_plots/014_Taranis_waterfall.png)

### Trial 15: Taranis -> Taranis

![Waterfall trial 15](../../outputs/class_sweep_plots/015_Taranis_waterfall.png)

### Trial 16: Turnigy -> Turnigy

![Waterfall trial 16](../../outputs/class_sweep_plots/016_Turnigy_waterfall.png)

### Trial 17: Turnigy -> Turnigy

![Waterfall trial 17](../../outputs/class_sweep_plots/017_Turnigy_waterfall.png)

### Trial 18: Turnigy -> Turnigy

![Waterfall trial 18](../../outputs/class_sweep_plots/018_Turnigy_waterfall.png)


## Command

```bash
/home/jake/workspace/SDR/RF_Sentinel/.venv/bin/python3 /home/jake/workspace/SDR/rf-signal-intelligence/scripts/live_noisy_drone_rf_classifier.py --tx-test-all-classes --tx-test-classes DJI,FutabaT14,FutabaT7,Graupner,Taranis,Turnigy --tx-test-count 3
```

## Per-Class Summary

| Class | Pass/Total | Accuracy | Min Target Confidence | Mean Target Confidence | Mean Capture Power dB | Max Full-Scale % |
|---|---:|---:|---:|---:|---:|---:|
| DJI | 3/3 | 1.000 | 0.998 | 0.999 | -25.4 | 0.000 |
| FutabaT14 | 3/3 | 1.000 | 1.000 | 1.000 | -17.3 | 0.830 |
| FutabaT7 | 3/3 | 1.000 | 0.999 | 1.000 | -17.0 | 0.910 |
| Graupner | 3/3 | 1.000 | 0.991 | 0.994 | -21.8 | 0.130 |
| Taranis | 3/3 | 1.000 | 0.996 | 0.998 | -19.9 | 0.170 |
| Turnigy | 3/3 | 1.000 | 0.969 | 0.984 | -17.3 | 0.910 |

## Per-Trial Results

| Trial | Target | Prediction | Confidence | Best Non-Noise | Target Confidence | Capture Power dB | Full-Scale % | TX Sample | RX IQ | Waterfall |
|---:|---|---|---:|---|---:|---:|---:|---|---|---|
| 1 | DJI | DJI | 1.000 | DJI | 1.000 | -28.5 | 0.00 | IQdata_sample709_target0_snr30.pt true=DJI target=0 snr=30dB | `../../outputs/class_sweep_iq/001_DJI.npy` | `../../outputs/class_sweep_plots/001_DJI_waterfall.png` |
| 2 | DJI | DJI | 1.000 | DJI | 1.000 | -25.4 | 0.00 | IQdata_sample1122_target0_snr30.pt true=DJI target=0 snr=30dB | `../../outputs/class_sweep_iq/002_DJI.npy` | `../../outputs/class_sweep_plots/002_DJI_waterfall.png` |
| 3 | DJI | DJI | 0.982 | DJI | 0.998 | -22.3 | 0.00 | IQdata_sample868_target0_snr24.pt true=DJI target=0 snr=24dB | `../../outputs/class_sweep_iq/003_DJI.npy` | `../../outputs/class_sweep_plots/003_DJI_waterfall.png` |
| 4 | FutabaT14 | FutabaT14 | 1.000 | FutabaT14 | 1.000 | -18.2 | 0.75 | IQdata_sample3367_target1_snr22.pt true=FutabaT14 target=1 snr=22dB | `../../outputs/class_sweep_iq/004_FutabaT14.npy` | `../../outputs/class_sweep_plots/004_FutabaT14_waterfall.png` |
| 5 | FutabaT14 | FutabaT14 | 1.000 | FutabaT14 | 1.000 | -16.5 | 0.83 | IQdata_sample3944_target1_snr20.pt true=FutabaT14 target=1 snr=20dB | `../../outputs/class_sweep_iq/005_FutabaT14.npy` | `../../outputs/class_sweep_plots/005_FutabaT14_waterfall.png` |
| 6 | FutabaT14 | FutabaT14 | 1.000 | FutabaT14 | 1.000 | -17.3 | 0.82 | IQdata_sample4618_target1_snr20.pt true=FutabaT14 target=1 snr=20dB | `../../outputs/class_sweep_iq/006_FutabaT14.npy` | `../../outputs/class_sweep_plots/006_FutabaT14_waterfall.png` |
| 7 | FutabaT7 | FutabaT7 | 1.000 | FutabaT7 | 1.000 | -16.6 | 0.75 | IQdata_sample5426_target2_snr20.pt true=FutabaT7 target=2 snr=20dB | `../../outputs/class_sweep_iq/007_FutabaT7.npy` | `../../outputs/class_sweep_plots/007_FutabaT7_waterfall.png` |
| 8 | FutabaT7 | FutabaT7 | 1.000 | FutabaT7 | 1.000 | -16.6 | 0.91 | IQdata_sample5085_target2_snr20.pt true=FutabaT7 target=2 snr=20dB | `../../outputs/class_sweep_iq/008_FutabaT7.npy` | `../../outputs/class_sweep_plots/008_FutabaT7_waterfall.png` |
| 9 | FutabaT7 | FutabaT7 | 0.999 | FutabaT7 | 0.999 | -17.8 | 0.75 | IQdata_sample4963_target2_snr22.pt true=FutabaT7 target=2 snr=22dB | `../../outputs/class_sweep_iq/009_FutabaT7.npy` | `../../outputs/class_sweep_plots/009_FutabaT7_waterfall.png` |
| 10 | Graupner | Graupner | 0.981 | Graupner | 0.992 | -22.2 | 0.00 | IQdata_sample6065_target3_snr30.pt true=Graupner target=3 snr=30dB | `../../outputs/class_sweep_iq/010_Graupner.npy` | `../../outputs/class_sweep_plots/010_Graupner_waterfall.png` |
| 11 | Graupner | Graupner | 0.999 | Graupner | 0.999 | -19.9 | 0.13 | IQdata_sample6086_target3_snr30.pt true=Graupner target=3 snr=30dB | `../../outputs/class_sweep_iq/011_Graupner.npy` | `../../outputs/class_sweep_plots/011_Graupner_waterfall.png` |
| 12 | Graupner | Graupner | 0.961 | Graupner | 0.991 | -23.3 | 0.00 | IQdata_sample6308_target3_snr22.pt true=Graupner target=3 snr=22dB | `../../outputs/class_sweep_iq/012_Graupner.npy` | `../../outputs/class_sweep_plots/012_Graupner_waterfall.png` |
| 13 | Taranis | Taranis | 0.996 | Taranis | 0.996 | -20.1 | 0.17 | IQdata_sample6929_target5_snr28.pt true=Taranis target=5 snr=28dB | `../../outputs/class_sweep_iq/013_Taranis.npy` | `../../outputs/class_sweep_plots/013_Taranis_waterfall.png` |
| 14 | Taranis | Taranis | 0.999 | Taranis | 0.999 | -20.8 | 0.00 | IQdata_sample7626_target5_snr20.pt true=Taranis target=5 snr=20dB | `../../outputs/class_sweep_iq/014_Taranis.npy` | `../../outputs/class_sweep_plots/014_Taranis_waterfall.png` |
| 15 | Taranis | Taranis | 0.999 | Taranis | 0.999 | -18.9 | 0.17 | IQdata_sample7763_target5_snr22.pt true=Taranis target=5 snr=22dB | `../../outputs/class_sweep_iq/015_Taranis.npy` | `../../outputs/class_sweep_plots/015_Taranis_waterfall.png` |
| 16 | Turnigy | Turnigy | 0.983 | Turnigy | 0.983 | -16.0 | 0.90 | IQdata_sample8189_target6_snr30.pt true=Turnigy target=6 snr=30dB | `../../outputs/class_sweep_iq/016_Turnigy.npy` | `../../outputs/class_sweep_plots/016_Turnigy_waterfall.png` |
| 17 | Turnigy | Turnigy | 1.000 | Turnigy | 1.000 | -19.3 | 0.02 | IQdata_sample8390_target6_snr26.pt true=Turnigy target=6 snr=26dB | `../../outputs/class_sweep_iq/017_Turnigy.npy` | `../../outputs/class_sweep_plots/017_Turnigy_waterfall.png` |
| 18 | Turnigy | Turnigy | 0.969 | Turnigy | 0.969 | -16.6 | 0.91 | IQdata_sample8860_target6_snr30.pt true=Turnigy target=6 snr=30dB | `../../outputs/class_sweep_iq/018_Turnigy.npy` | `../../outputs/class_sweep_plots/018_Turnigy_waterfall.png` |

## Notes

- `prediction` is the final script decision after the configured decision policy.
- `best_non_noise` and `target confidence` are conditional on the non-noise class mass.
- Full-scale percentages above zero indicate some clipping or saturation in the saved RX window.
