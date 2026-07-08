# Documentation

This directory contains the formal project reports and supporting research documents for the work:

Wireless Signal Classification via Deep Learning  
Jacob M. Ramey, Paras Goda (Equal Contributors)

Primary Report: Wireless Signal Classification via Deep Learning (Generative Adversarial, Open Set Incremental and Dual Head)

## Overview

The documents in this directory detail the full academic analysis, experimental results, and architectural designs for the wireless-signal classification research project. The work includes:

- Cross-dataset evaluation (RML2016 → RML2018a)
- Open-set incremental learning (OSIL)
- GAN-assisted sample balancing
- Dual-head time–frequency neural architecture
- Transformer-based preliminary experiments

## File List

1. Wireless Signal Classification via Deep Learning – Final Jacob M. Ramey.pdf  
   Formal 7‑page technical report describing the OSIL engine, GAN augmentation, dual‑head CNN+BiLSTM, and cross-corpus experiments.

2. Wireless Signal Classification via Machine Learning – Final.pdf  
   Earlier version of the analysis focusing on baseline LSTM and CNN comparisons.

## Summary of Included Research

The main report presents:

### Cross‑Corpus Generalization
A bidirectional LSTM trained on RML2016 is ported unchanged to the 24‑class RML2018a dataset, achieving 74% accuracy after 200 epochs.

### Open‑Set Incremental Learning (OSIL)
A lightweight incremental-learning framework combining entropy-based gating, Mahalanobis rejection, replay buffers, and classifier expansion.

### GAN‑Assisted Augmentation
A class‑conditioned AC‑GAN generates high‑SNR synthetic waveforms to counter dataset imbalance.

### Dual‑Head Time–Frequency Network
A fusion model combining:
- Frequency‑domain CNN features from FFT magnitude patches.
- Time‑domain BiLSTM features from I/Q sequences.

### Transformer Exploration
Initial experiments with a Spectral Transformer using I/Q patches as tokens.

## Usage

These documents serve as the reference material for the experiments and architectures found in:

- pipelines/
- src/rf_signal_intelligence/
- models/
- docker/

They describe the theoretical and experimental motivations underlying the models and pipelines implemented in this repository.

## Model And Dataset Cards

Model cards:

- [NoisyDroneRFv2 VGG full-complex spectrogram](model_cards/noisy_drone_rf_v2_vgg.md)
- [RML2016 CNN-transformer](model_cards/rml2016_cnn_transformer.md)
- [RML2018 LSTM](model_cards/rml2018_lstm.md)
- [DeepRadar2022 CNN-transformer](model_cards/deepradar2022_cnn_transformer.md)

Dataset cards:

- [Noisy Drone RF Signal Classification v2](dataset_cards/noisy_drone_rf_v2.md)
- [RML2016.10A](dataset_cards/rml2016.md)
- [RML2018.01A](dataset_cards/rml2018.md)
- [DeepRadar2022](dataset_cards/deepradar2022.md)

## Result References

- [Offline model result archive](results/offline_model_results.md)
- [NoisyDroneRFv2 result card](results/noisy_drone_rf_v2/README.md)
- [Live OTA result reports](../results/README.md)
- [Evaluation protocols](evaluation_protocols.md)
