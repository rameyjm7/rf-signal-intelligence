#!/usr/bin/env python3
"""Pipeline converted from the legacy 45_evaluation_rfuav workflow."""

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
# Cell 1 : Configure RFUAV VGG full-complex spectrogram evaluation
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

notebook_dir = Path().resolve()
if notebook_dir.name == 'pipelines':
    project_root = notebook_dir.parent if notebook_dir.name == 'pipelines' else notebook_dir
elif (notebook_dir / 'pipelines').exists() and (notebook_dir / 'src').exists():
    project_root = notebook_dir
elif (notebook_dir / 'ML-wireless-signal-classification').exists():
    project_root = notebook_dir / 'ML-wireless-signal-classification'
else:
    project_root = notebook_dir

cfg_path = project_root / 'configs' / 'local_data_paths.yaml'
if cfg_path.exists():
    local_cfg = yaml.safe_load(cfg_path.read_text()) or {}
    dcfg = local_cfg.get('datasets', {}).get('rfuav', {}) or {}
    data_dir = Path(dcfg.get('data_dir', Path(local_cfg.get('dataset_root', '/scratch/rameyjm7/datasets')) / 'RFUAV'))
    archive_dir = Path(dcfg.get('archive_dir', data_dir / 'archives'))
    extract_root = Path(dcfg.get('extract_dir', data_dir / 'extracted'))
else:
    data_dir = Path('/scratch/rameyjm7/datasets/RFUAV')
    archive_dir = data_dir / 'archives'
    extract_root = data_dir / 'extracted'

SCRATCH_ROOT = Path(os.getenv('RFUAV_SCRATCH_ROOT', '/scratch/rameyjm7/ML-wireless-signal-classification'))
outputs_dir = project_root / 'outputs' / 'rfuav_eval'
model_dir = project_root / 'models' / 'rfuav'
archive_dir = Path(os.getenv('RFUAV_ARCHIVE_DIR', str(archive_dir)))
extract_root = Path(os.getenv('RFUAV_EXTRACT_ROOT', str(extract_root)))
cache_dir = Path(os.getenv('RFUAV_SPEC_CACHE_DIR', str(SCRATCH_ROOT / 'cache' / 'rfuav' / '34b_high_snr_spectrogram_cnn_cache')))
manifest_path = Path(os.getenv('RFUAV_MANIFEST_PATH', str(SCRATCH_ROOT / 'manifests' / 'rfuav_manifest.csv')))

outputs_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

MAX_IQ_SAMPLES = int(os.getenv('RFUAV34B_MAX_IQ_SAMPLES', os.getenv('RFUAV_MAX_IQ_SAMPLES', '32768')))
DATA_FRACTION = float(os.getenv('RFUAV_DATA_FRACTION', '1.0'))
BATCH_SIZE = int(os.getenv('RFUAV_BATCH_SIZE', '8'))
SPEC_NFFT = int(os.getenv('RFUAV34B_SPEC_NFFT', os.getenv('RFUAV_SPEC_NFFT', '256')))
SPEC_HOP = int(os.getenv('RFUAV34B_SPEC_HOP', os.getenv('RFUAV_SPEC_HOP', '128')))
SPEC_TIME_BINS = int(os.getenv('RFUAV34B_SPEC_TIME_BINS', os.getenv('RFUAV_SPEC_TIME_BINS', '64')))
EVAL_LIMIT = int(os.getenv('RFUAV_EVAL_LIMIT', '0'))
BALANCED_EVAL = os.getenv('RFUAV_BALANCED_EVAL', '1').lower() not in {'0', 'false', 'no'}
BALANCED_SAMPLES_PER_CLASS = int(os.getenv('RFUAV_BALANCED_SAMPLES_PER_CLASS', '0'))
RFUAV_RAW_DTYPE = np.float32
RANDOM_STATE = 3407
input_shape = (SPEC_NFFT, SPEC_TIME_BINS, 2)

print('Project root:', project_root)
print('RFUAV dataset:', data_dir)
print('Archive dir:', archive_dir)
print('Extract root:', extract_root)
print('Manifest:', manifest_path)
print('Cache:', cache_dir)
print('Model dir:', model_dir)
print('Input shape:', input_shape)
assert extract_root.exists(), f'Missing extracted RFUAV data: {extract_root}. Run notebook 10 Cell 6 first.'

# %% Cell 2
# Cell 2 : Load or rebuild the RFUAV evaluation manifest and deterministic test split
IQ_SUFFIXES = {'.npy', '.npz', '.mat', '.csv', '.bin', '.dat', '.iq', '.complex', '.txt'}
META_SUFFIXES = {'.md', '.pdf', '.png', '.jpg', '.jpeg', '.json', '.yaml', '.yml', '.extract_done', '.extract_complete'}


def discover_iq_files(root: Path) -> list[Path]:
    files = []
    for file_path in root.rglob('*'):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix in META_SUFFIXES:
            continue
        if suffix in IQ_SUFFIXES or file_path.stat().st_size > 1024:
            files.append(file_path)
    return sorted(files)


def build_manifest_from_extract_root(root: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label = class_dir.name
        for file_path in discover_iq_files(class_dir):
            rows.append({
                'filepath': str(file_path),
                'label': label,
                'archive': f'{label}.rar',
                'relative_path': str(file_path.relative_to(class_dir)),
                'size_bytes': int(file_path.stat().st_size),
            })
    if not rows:
        raise RuntimeError(f'No RFUAV candidate IQ files found under {root}')
    return pd.DataFrame(rows)

if manifest_path.exists() and os.getenv('RFUAV_REBUILD_MANIFEST', '0').lower() not in {'1', 'true', 'yes'}:
    manifest_df = pd.read_csv(manifest_path)
    print('Loaded manifest:', manifest_path, 'rows:', len(manifest_df))
else:
    manifest_df = build_manifest_from_extract_root(extract_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)
    print('Rebuilt manifest:', manifest_path, 'rows:', len(manifest_df))

if DATA_FRACTION < 1.0:
    manifest_df = (
        manifest_df.groupby('label', group_keys=False)
        .sample(frac=DATA_FRACTION, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

label_names = sorted(manifest_df['label'].unique().tolist())
label_to_idx = {name: idx for idx, name in enumerate(label_names)}
manifest_df['label_idx'] = manifest_df['label'].map(label_to_idx).astype(np.int64)
num_classes = len(label_names)

train_df, test_df = train_test_split(
    manifest_df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=manifest_df['label_idx'],
)
train_df, val_df = train_test_split(
    train_df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=train_df['label_idx'],
)
test_df = test_df.reset_index(drop=True)

if BALANCED_EVAL:
    counts = test_df['label_idx'].value_counts().sort_index()
    n_per_class = int(BALANCED_SAMPLES_PER_CLASS) if BALANCED_SAMPLES_PER_CLASS > 0 else int(counts.min())
    balanced_test_df = (
        test_df.groupby('label_idx', group_keys=False)
        .sample(n=n_per_class, replace=False, random_state=RANDOM_STATE)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
else:
    balanced_test_df = None

print('Classes:', num_classes)
print('Class counts:', manifest_df['label'].value_counts().sort_index().to_dict())
print('Train/val/test:', len(train_df), len(val_df), len(test_df))
if balanced_test_df is not None:
    print('Balanced test counts:', balanced_test_df['label'].value_counts().sort_index().to_dict())

# %% Cell 3
# Cell 3 : Evaluate RFUAV model and save natural/balanced confusion matrices
def _read_numeric_text(path: Path) -> np.ndarray:
    try:
        return np.asarray(np.loadtxt(path, delimiter=','))
    except Exception:
        return np.asarray(np.loadtxt(path))


def load_iq_file(path_like) -> np.ndarray:
    path = Path(path_like)
    suffix = path.suffix.lower()
    if suffix == '.npy':
        arr = np.load(path, allow_pickle=True)
    elif suffix == '.npz':
        obj = np.load(path, allow_pickle=True)
        key = 'x' if 'x' in obj else ('iq' if 'iq' in obj else obj.files[0])
        arr = obj[key]
    elif suffix == '.mat':
        import scipy.io as sio
        try:
            mat = sio.loadmat(path)
            keys = [k for k in mat.keys() if not k.startswith('__')]
            if not keys:
                raise ValueError(f'No arrays in mat file: {path}')
            arr = mat[keys[0]]
        except NotImplementedError:
            import h5py
            with h5py.File(path, 'r') as h5f:
                key = next(iter(h5f.keys()))
                arr = np.array(h5f[key])
    elif suffix in {'.csv', '.txt'}:
        arr = _read_numeric_text(path)
    else:
        raw = np.fromfile(path, dtype=RFUAV_RAW_DTYPE)
        if raw.size < 4 or not np.isfinite(raw).all():
            raw = np.fromfile(path, dtype=np.int16).astype(np.float32)
        arr = raw

    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if np.iscomplexobj(arr):
        iq = np.stack([arr.real, arr.imag], axis=-1)
    elif arr.ndim == 1:
        if arr.size % 2:
            arr = arr[:-1]
        iq = arr.reshape(-1, 2)
    elif arr.ndim >= 2 and 2 in arr.shape:
        if arr.shape[-1] == 2:
            iq = arr.reshape(-1, 2)
        else:
            iq = np.moveaxis(arr, list(arr.shape).index(2), -1).reshape(-1, 2)
    else:
        flat = arr.reshape(-1)
        if flat.size % 2:
            flat = flat[:-1]
        iq = flat.reshape(-1, 2)
    iq = np.asarray(iq, dtype=np.float32)
    iq = iq[np.isfinite(iq).all(axis=1)]
    if iq.shape[0] < SPEC_NFFT:
        iq = np.pad(iq, ((0, SPEC_NFFT - iq.shape[0]), (0, 0)), mode='constant')
    return iq


def select_active_window(iq: np.ndarray, max_samples=MAX_IQ_SAMPLES) -> np.ndarray:
    if iq.shape[0] <= max_samples:
        out = iq
    else:
        power = np.sum(iq.astype(np.float32) ** 2, axis=1)
        kernel = np.ones(min(8192, max_samples), dtype=np.float32)
        smooth = np.convolve(power, kernel / kernel.size, mode='same')
        center = int(np.argmax(smooth))
        start = max(0, min(center - max_samples // 2, iq.shape[0] - max_samples))
        out = iq[start:start + max_samples]
    scale = np.sqrt(np.mean(out.astype(np.float32) ** 2) + 1e-8)
    return (out / scale).astype(np.float32)


def cache_key(filepath: str) -> str:
    path = Path(filepath)
    stat = path.stat()
    token = f'{path}|{stat.st_size}|{stat.st_mtime_ns}|{MAX_IQ_SAMPLES}|{SPEC_NFFT}|{SPEC_HOP}|{SPEC_TIME_BINS}'
    return hashlib.sha1(token.encode('utf-8')).hexdigest()


def safe_load_cache(path: Path):
    try:
        with np.load(path) as data:
            arr = np.asarray(data['x'], dtype=np.float32)
        if arr.shape == input_shape and np.isfinite(arr).all():
            return arr
    except Exception:
        path.unlink(missing_ok=True)
    return None


def write_cache_atomic(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp_{os.getpid()}')
    try:
        np.savez_compressed(tmp, x=arr.astype(np.float32))
        generated = tmp if tmp.exists() else Path(str(tmp) + '.npz')
        generated.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
        Path(str(tmp) + '.npz').unlink(missing_ok=True)


def iq_to_complex_spectrogram(iq: np.ndarray) -> np.ndarray:
    iq = select_active_window(iq)
    complex_iq = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
    starts = np.arange(0, len(complex_iq) - SPEC_NFFT + 1, SPEC_HOP)
    if starts.size == 0:
        starts = np.array([0])
    window = np.hanning(SPEC_NFFT).astype(np.float32)
    frames = np.stack([complex_iq[start:start + SPEC_NFFT] * window for start in starts], axis=0)
    fft_complex = np.fft.fftshift(np.fft.fft(frames, n=SPEC_NFFT, axis=1), axes=1).T / float(SPEC_NFFT)
    real_part = fft_complex.real.astype(np.float32)
    imag_part = fft_complex.imag.astype(np.float32)
    if real_part.shape[1] > SPEC_TIME_BINS:
        idx = np.linspace(0, real_part.shape[1] - 1, SPEC_TIME_BINS).round().astype(int)
        real_part = real_part[:, idx]
        imag_part = imag_part[:, idx]
    elif real_part.shape[1] < SPEC_TIME_BINS:
        pad = SPEC_TIME_BINS - real_part.shape[1]
        real_part = np.pad(real_part, ((0, 0), (0, pad)), mode='edge')
        imag_part = np.pad(imag_part, ((0, 0), (0, pad)), mode='edge')
    spec = np.stack([real_part, imag_part], axis=-1).astype(np.float32)
    spec = spec / (np.std(spec) + 1e-6)
    return np.clip(spec, -6.0, 6.0).astype(np.float32)


def prepare_spectrogram(filepath: str) -> np.ndarray:
    cache_path = cache_dir / f'{cache_key(filepath)}.npz'
    cached = safe_load_cache(cache_path) if cache_path.exists() else None
    if cached is not None:
        return cached
    spec = iq_to_complex_spectrogram(load_iq_file(filepath))
    write_cache_atomic(cache_path, spec)
    return spec

best_path = model_dir / 'rfuav_34b_high_snr_spectrogram_cnn_best.keras'
final_path = model_dir / 'rfuav_34b_high_snr_spectrogram_cnn_final.keras'
model_path = best_path if best_path.exists() else final_path
if not model_path.exists():
    raise FileNotFoundError(f'Missing RFUAV model: {model_path}. Run notebook 34 training first.')
model = load_model(model_path, compile=False)
print('Loaded RFUAV 34b model:', model_path)


def predict_frame(frame: pd.DataFrame):
    if EVAL_LIMIT > 0:
        frame = frame.head(EVAL_LIMIT).copy()
    probs = []
    for row in frame.itertuples(index=False):
        x = prepare_spectrogram(row.filepath)[None, ...]
        probs.append(model.predict(x, batch_size=BATCH_SIZE, verbose=0)[0])
    return frame, np.asarray(probs, dtype=np.float32)


def evaluate_and_plot(frame: pd.DataFrame, split_name: str, title_suffix: str):
    eval_df, probs = predict_frame(frame)
    y_true = eval_df['label_idx'].to_numpy(dtype=np.int64)
    y_pred = probs.argmax(axis=1)
    print(f'RFUAV {split_name} report')
    print('accuracy:', accuracy_score(y_true, y_pred))
    print('macro_f1:', f1_score(y_true, y_pred, average='macro', zero_division=0))
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    pred_path = outputs_dir / f'45_rfuav_34b_high_snr_spectrogram_cnn_{split_name}_predictions.csv'
    pred_df = eval_df[['filepath', 'label', 'label_idx']].copy()
    pred_df['pred_idx'] = y_pred
    pred_df['pred_label'] = [label_names[idx] for idx in y_pred]
    pred_df['correct'] = (y_true == y_pred).astype(int)
    pred_df.to_csv(pred_path, index=False)
    print('Saved predictions:', pred_path)

    fig, ax = plt.subplots(figsize=(max(12, num_classes * 0.45), max(10, num_classes * 0.35)))
    sns.heatmap(confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)), cmap='Blues', xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title(f'RFUAV 34b High-SNR Spectrogram CNN - {title_suffix}')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    plt.tight_layout()
    cm_path = outputs_dir / f'45_rfuav_34b_high_snr_spectrogram_cnn_{split_name}_confusion_matrix.png'
    plt.savefig(cm_path, dpi=180)
    print('Saved:', cm_path)
    _save_current_figure("cell_03_figure_01.png")

    return {
        f'{split_name}_accuracy': float(accuracy_score(y_true, y_pred)),
        f'{split_name}_macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        f'{split_name}_weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        f'{split_name}_samples': int(len(eval_df)),
        f'{split_name}_predictions': str(pred_path),
        f'{split_name}_confusion_matrix': str(cm_path),
    }

metrics = {
    'model': 'rfuav_34b_high_snr_spectrogram_cnn',
    'model_path': str(model_path),
    'num_classes': int(num_classes),
    'label_names': label_names,
}
metrics.update(evaluate_and_plot(test_df, 'natural_test', 'Natural Test Distribution'))
if balanced_test_df is not None:
    metrics.update(evaluate_and_plot(balanced_test_df, 'balanced_test', 'Balanced Test Distribution'))

metrics_path = outputs_dir / '45_rfuav_34b_high_snr_spectrogram_cnn_metrics.json'
metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print('Saved metrics:', metrics_path)

# %% Cell 4
# Cell 4 : Plot RFUAV cumulative training curves if history is present
history_csv_path = model_dir / 'rfuav_34b_high_snr_spectrogram_cnn_training_history.csv'
if not history_csv_path.exists():
    print('No RFUAV training history found yet:', history_csv_path)
else:
    history_df = pd.read_csv(history_csv_path)
    print('History rows:', len(history_df))
    print(history_df.tail())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history_df['global_epoch'], history_df.get('accuracy', []), marker='o', label='train_accuracy')
    if 'val_accuracy' in history_df:
        axes[0].plot(history_df['global_epoch'], history_df['val_accuracy'], marker='o', label='val_accuracy')
    if 'val_macro_f1' in history_df:
        axes[0].plot(history_df['global_epoch'], history_df['val_macro_f1'], marker='o', label='val_macro_f1')
    axes[0].set_title('RFUAV 34b High-SNR Spectrogram CNN Accuracy')
    axes[0].set_xlabel('Global epoch')
    axes[0].set_ylabel('Score')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history_df['global_epoch'], history_df.get('loss', []), marker='o', label='train_loss')
    if 'val_loss' in history_df:
        axes[1].plot(history_df['global_epoch'], history_df['val_loss'], marker='o', label='val_loss')
    axes[1].set_title('RFUAV 34b High-SNR Spectrogram CNN Loss')
    axes[1].set_xlabel('Global epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plot_path = outputs_dir / '45_rfuav_34b_high_snr_spectrogram_cnn_training_curves.png'
    plt.savefig(plot_path, dpi=180)
    print('Saved:', plot_path)
    _save_current_figure("cell_04_figure_02.png")
