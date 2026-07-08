#!/usr/bin/env python3
"""Pipeline converted from the legacy 20_signal_eda workflow."""

from __future__ import annotations

from pathlib import Path


# Pipeline artifact helpers: converted from notebook workflows, so plot displays are saved.
def _pipeline_artifact_dir() -> Path:
    path = Path("outputs/pipeline_figures") / Path(__file__).stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_current_figure(filename: str) -> None:
    import matplotlib.pyplot as plt

    path = _pipeline_artifact_dir() / filename
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"saved figure: {path}")

# %% Cell 1
# Cell 1 : 20 Signal EDA
# 20 Signal EDA
# 
# Basic exploratory analysis for RF I/Q signals using the RML2016.10a dataset.
# 
# This notebook covers:
# - dataset loading and summary
# - SNR and class distribution checks
# - time-domain I/Q plots
# - constellation plot
# - simple frequency-domain magnitude view

# %% Cell 2
# Cell 2 : Import libraries and configure dependencies
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

# %% Cell 3
# Cell 3 : Set up file paths and runtime configuration
from pathlib import Path
import yaml

notebook_dir = Path().resolve()
project_root = notebook_dir.parent if notebook_dir.name == 'pipelines' else notebook_dir

cfg_path = project_root / 'configs' / 'local_data_paths.yaml'
if cfg_path.exists():
    local_cfg = yaml.safe_load(cfg_path.read_text())
    data_path = Path(local_cfg.get('datasets', {}).get('rml2016', {}).get('pkl', '/scratch/rameyjm7/datasets/RML2016/RML2016.10a_dict.pkl'))
else:
    data_path = Path('/scratch/rameyjm7/datasets/RML2016/RML2016.10a_dict.pkl')

if not data_path.exists():
    raise FileNotFoundError(f'Dataset not found: {data_path}')

with data_path.open('rb') as f:
    data = pickle.load(f, encoding='latin1')

mods = sorted({m for (m, _) in data.keys()})
snrs = sorted({s for (_, s) in data.keys()})

print(f'Dataset: {data_path}')
print(f'Number of modulation classes: {len(mods)}')
print(f'Number of SNR levels: {len(snrs)}')
print(f'Modulations: {mods}')
print(f'SNR range: {snrs[0]} to {snrs[-1]} dB')

# %% Cell 4
# Cell 4 : Plot sample counts by SNR
samples_per_key = {k: len(v) for k, v in data.items()}
print('Min samples per (mod, snr):', min(samples_per_key.values()))
print('Max samples per (mod, snr):', max(samples_per_key.values()))

# SNR distribution (counting samples across all classes)
snr_counts = {snr: 0 for snr in snrs}
for (mod, snr), signals in data.items():
    snr_counts[snr] += len(signals)

plt.figure(figsize=(10, 4))
plt.plot(list(snr_counts.keys()), list(snr_counts.values()), marker='o')
plt.title('Sample Count by SNR (RML2016.10a)')
plt.xlabel('SNR (dB)')
plt.ylabel('Number of samples')
plt.tight_layout()
_save_current_figure("cell_04_figure_01.png")

# %% Cell 5
# Cell 5 : Select a high-SNR example signal
# Pick an example signal to visualize
example_mod = 'QPSK' if 'QPSK' in mods else mods[0]
example_snr = max(snrs)
example_idx = 0

signal = data[(example_mod, example_snr)][example_idx]
i = np.asarray(signal[0])
q = np.asarray(signal[1])
t = np.arange(len(i))

print(f'Example: modulation={example_mod}, SNR={example_snr} dB, len={len(i)}')

# %% Cell 6
# Cell 6 : Plot example I and Q time series
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(t, i, color='tab:blue', linewidth=1.2)
axes[0].set_title(f'I Component ({example_mod}, {example_snr} dB)')
axes[0].set_ylabel('Amplitude')

axes[1].plot(t, q, color='tab:orange', linewidth=1.2)
axes[1].set_title(f'Q Component ({example_mod}, {example_snr} dB)')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Amplitude')

plt.tight_layout()
_save_current_figure("cell_06_figure_02.png")

# %% Cell 7
# Cell 7 : Plot example constellation
plt.figure(figsize=(6, 6))
plt.scatter(i, q, s=18, alpha=0.7, color='tab:green')
plt.title(f'Constellation Plot ({example_mod}, {example_snr} dB)')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
_save_current_figure("cell_07_figure_03.png")

# %% Cell 8
# Cell 8 : Plot example FFT magnitude
# Simple spectrum view from complex baseband signal
x = i + 1j * q
X = np.fft.fftshift(np.fft.fft(x))
f = np.fft.fftshift(np.fft.fftfreq(len(x), d=1.0))
mag_db = 20 * np.log10(np.abs(X) + 1e-8)

plt.figure(figsize=(10, 4))
plt.plot(f, mag_db, color='tab:purple')
plt.title(f'FFT Magnitude (dB) - {example_mod}, {example_snr} dB')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude (dB)')
plt.tight_layout()
_save_current_figure("cell_08_figure_04.png")

# %% Cell 9
# Cell 9 : Plot multi-class constellation snapshots
# Quick multi-class snapshot at highest SNR
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()
picked_mods = mods[:6]

for ax, mod in zip(axes, picked_mods):
    sig = data[(mod, max(snrs))][0]
    ii = np.asarray(sig[0])
    qq = np.asarray(sig[1])
    ax.scatter(ii, qq, s=10, alpha=0.6)
    ax.set_title(mod)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')

plt.suptitle(f'Constellation Snapshots at SNR={max(snrs)} dB', y=1.02)
plt.tight_layout()
_save_current_figure("cell_09_figure_05.png")
