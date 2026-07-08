#!/usr/bin/env python3
"""Pipeline converted from the legacy 34b_fast_high_snr_rfuav workflow."""

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
# Cell 1 : Configure fast high-SNR RFUAV lightweight spectrogram CNN experiment
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, MaxPooling2D
from tensorflow.keras.models import Model, load_model

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
RFUAV34B_CACHE_VERSION = os.getenv('RFUAV34B_CACHE_VERSION', 'v6_labelsafe_burst')

for path in (outputs_dir, model_dir, extract_root, cache_dir, manifest_path.parent):
    path.mkdir(parents=True, exist_ok=True)

MAX_IQ_SAMPLES = int(os.getenv('RFUAV34B_MAX_IQ_SAMPLES', os.getenv('RFUAV_MAX_IQ_SAMPLES', '32768')))
DATA_FRACTION = float(os.getenv('RFUAV_DATA_FRACTION', '1.0'))
BATCH_SIZE = int(os.getenv('RFUAV34B_BATCH_SIZE', os.getenv('RFUAV_BATCH_SIZE', '8')))
SHUFFLE_BUFFER = int(os.getenv('RFUAV_SHUFFLE_BUFFER', '256'))
SPEC_NFFT = int(os.getenv('RFUAV34B_SPEC_NFFT', os.getenv('RFUAV_SPEC_NFFT', '256')))
SPEC_HOP = int(os.getenv('RFUAV34B_SPEC_HOP', os.getenv('RFUAV_SPEC_HOP', '128')))
SPEC_TIME_BINS = int(os.getenv('RFUAV34B_SPEC_TIME_BINS', os.getenv('RFUAV_SPEC_TIME_BINS', '64')))
SPEC_EVAL_WINDOWS = int(os.getenv('RFUAV_SPEC_EVAL_WINDOWS', '1'))
CNN_WEIGHT_DECAY = float(os.getenv('RFUAV34B_CNN_WEIGHT_DECAY', os.getenv('RFUAV_CNN_WEIGHT_DECAY', '1e-5')))
EVAL_LIMIT = int(os.getenv('RFUAV_EVAL_LIMIT', '0'))
RFUAV_RAW_DTYPE = np.float32  # Author tooling reads raw datapacks as interleaved float32 I/Q.
RFUAV_SAMPLE_RATE_HZ = float(os.getenv('RFUAV_SAMPLE_RATE_HZ', '100e6'))
RANDOM_STATE = 3407

print('Project root:', project_root)
print('RFUAV dataset:', data_dir)
print('Archive dir:', archive_dir)
print('Archive extraction root:', extract_root)
print('Spectrogram cache:', cache_dir)
print('Manifest:', manifest_path)
print('34b cache version:', RFUAV34B_CACHE_VERSION)
print('Light spectrogram bins:', SPEC_NFFT, 'x', SPEC_TIME_BINS)
print('Raw dtype:', RFUAV_RAW_DTYPE, 'Sample rate:', RFUAV_SAMPLE_RATE_HZ)
print('Data fraction:', DATA_FRACTION)
print('34b high-SNR samples/class:', os.getenv('RFUAV34B_SAMPLES_PER_CLASS', '40'))
assert data_dir.exists(), f'Missing RFUAV dataset directory: {data_dir}'
assert archive_dir.exists(), f'Missing RFUAV archive directory: {archive_dir}. Run notebook 10 Cell 6 first.'

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('GPUs:', gpus)
except Exception as exc:
    print('GPU memory-growth setup skipped:', repr(exc))

archive_paths = sorted(archive_dir.glob('*.rar'))
print('RFUAV class archives found:', len(archive_paths))
for archive in archive_paths[:10]:
    print(' -', archive.name)

# %% Cell 2
# Cell 2 : Build/load RFUAV manifest, then keep highest-SNR balanced subset
IQ_SUFFIXES = set(os.getenv('RFUAV34B_IQ_SUFFIXES', '.iq').lower().split(','))
IQ_SUFFIXES = {suffix if suffix.startswith('.') else f'.{suffix}' for suffix in IQ_SUFFIXES if suffix}
print('34b accepted IQ suffixes:', sorted(IQ_SUFFIXES))
META_SUFFIXES = {'.md', '.pdf', '.png', '.jpg', '.jpeg', '.json', '.yaml', '.yml', '.xml'}


def safe_label_from_archive(path) -> str:
    path = Path(str(path))
    label = path.stem.strip() if path.suffix else path.name.strip()
    label = re.sub(r'\s+', '_', label)
    label = re.sub(r'[^A-Za-z0-9_\-]+', '', label)
    return label or 'unknown'


def extractor_command(archive_path: Path, extract_to: Path):
    if shutil.which('unrar'):
        return ['unrar', 'x', '-o+', str(archive_path), str(extract_to) + '/']
    if shutil.which('bsdtar'):
        return ['bsdtar', '-xf', str(archive_path), '-C', str(extract_to)]
    if shutil.which('7z'):
        return ['7z', 'x', '-y', f'-o{extract_to}', str(archive_path)]
    raise RuntimeError('No RAR extractor found. Install one of: unrar, bsdtar/libarchive, or p7zip/7z.')


def extract_archive(path: Path, target: Path):
    done = target / '.extract_done'
    done_from_notebook10 = target / '.extract_complete'
    # Marker files alone are not enough: previous partial extraction runs left
    # empty class directories with .extract_complete, which broke manifest rebuilds.
    if (done.exists() or done_from_notebook10.exists()) and discover_iq_files(target):
        return
    target.mkdir(parents=True, exist_ok=True)
    print('Extracting:', path.name, '->', target)
    subprocess.run(extractor_command(path, target), check=True)
    done.write_text(datetime.now(timezone.utc).isoformat(), encoding='utf-8')


def discover_iq_files(root: Path) -> list[Path]:
    files = []
    for file_path in root.rglob('*'):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix in META_SUFFIXES:
            continue
        if suffix in IQ_SUFFIXES:
            files.append(file_path)
    return sorted(files)


def rebuild_manifest_from_extracted() -> pd.DataFrame:
    rows = []
    if not extract_root.exists():
        extract_root.mkdir(parents=True, exist_ok=True)

    # If the normalized extraction folders exist but contain no IQ, extract from
    # the local RAR archives. This keeps notebook 34b self-healing after scratch
    # cleanup or partial notebook-10 extraction runs.
    for archive in archive_paths:
        label = safe_label_from_archive(archive)
        class_extract_dir = extract_root / label
        existing_iq = discover_iq_files(class_extract_dir) if class_extract_dir.exists() else []
        if not existing_iq:
            extract_archive(archive, class_extract_dir)

    archive_label_lookup = {safe_label_from_archive(path): path.name for path in archive_paths}
    seen = set()
    for class_dir in sorted([path for path in extract_root.iterdir() if path.is_dir()]):
        label = safe_label_from_archive(class_dir)
        archive_name = archive_label_lookup.get(label, f'{label}.rar')
        iq_files = discover_iq_files(class_dir)
        print(label, 'existing extracted files:', len(iq_files))
        for file_path in iq_files:
            if file_path in seen:
                continue
            seen.add(file_path)
            try:
                rel = file_path.relative_to(class_dir)
            except ValueError:
                rel = file_path.name
            rows.append({
                'filepath': str(file_path),
                'label': label,
                'archive': archive_name,
                'relative_path': str(rel),
                'size_bytes': int(file_path.stat().st_size),
            })

    # Last-resort whole-tree scan: useful if archives unpack with unexpected
    # nesting or original space-containing folder names.
    if not rows:
        print('No class-folder IQ files found; scanning entire RFUAV extracted tree.')
        archive_labels = sorted(archive_label_lookup, key=len, reverse=True)
        for file_path in discover_iq_files(extract_root):
            safe_parts = [safe_label_from_archive(part) for part in file_path.relative_to(extract_root).parts]
            label = next((candidate for candidate in archive_labels if candidate in safe_parts), safe_parts[0] if safe_parts else 'unknown')
            rows.append({
                'filepath': str(file_path),
                'label': label,
                'archive': archive_label_lookup.get(label, f'{label}.rar'),
                'relative_path': str(file_path.relative_to(extract_root)),
                'size_bytes': int(file_path.stat().st_size),
            })
    return pd.DataFrame(rows)

if manifest_path.exists() and os.getenv('RFUAV_REBUILD_MANIFEST', '0').lower() not in {'1', 'true', 'yes'}:
    manifest_df = pd.read_csv(manifest_path)
    print('Loaded manifest:', manifest_path, 'rows:', len(manifest_df))
else:
    rows = []
    if not archive_paths:
        raise FileNotFoundError(f'No .rar archives found in {data_dir}. Run notebook 10 Cell 6 first.')
    for archive in archive_paths:
        label = safe_label_from_archive(archive)
        class_extract_dir = extract_root / label
        extract_archive(archive, class_extract_dir)
        iq_files = discover_iq_files(class_extract_dir)
        print(label, 'files:', len(iq_files))
        for file_path in iq_files:
            rel = file_path.relative_to(class_extract_dir)
            rows.append({
                'filepath': str(file_path),
                'label': label,
                'archive': archive.name,
                'relative_path': str(rel),
                'size_bytes': int(file_path.stat().st_size),
            })
    manifest_df = pd.DataFrame(rows)
    if manifest_df.empty:
        raise RuntimeError('RFUAV archives extracted, but no candidate IQ files were discovered. Inspect extracted files under extract_root.')
    manifest_df.to_csv(manifest_path, index=False)
    print('Saved manifest:', manifest_path)


def path_is_label_safe(path: Path, label: str) -> bool:
    safe_label = safe_label_from_archive(Path(str(label)))
    safe_parts = {safe_label_from_archive(part) for part in path.relative_to(extract_root).parts} if extract_root in path.parents else {safe_label_from_archive(part) for part in path.parts}
    return safe_label in safe_parts


def repair_or_drop_missing_files(frame: pd.DataFrame) -> pd.DataFrame:
    # Cached manifests can outlive scratch cleanup or extraction layout changes.
    # DO NOT repair by bare filename globally: RFUAV archives reuse names like pack1_0-1s.iq.
    # A global filename repair can assign one DAUTEL capture to every class, which makes
    # training mathematically impossible. Only accept repairs under a path that contains
    # the row's normalized label.
    repaired = frame.copy()
    missing_rows = []
    unsafe_rows = []
    repaired_count = 0
    search_cache = {}
    for idx, row in repaired.iterrows():
        path = Path(row['filepath'])
        label = str(row.get('label', ''))
        if path.exists():
            if path_is_label_safe(path, label):
                continue
            unsafe_rows.append(idx)
            continue
        filename = path.name
        cache_key_name = (filename, safe_label_from_archive(Path(label)))
        matches = search_cache.get(cache_key_name)
        if matches is None:
            all_matches = sorted(extract_root.rglob(filename)) if extract_root.exists() else []
            matches = [candidate for candidate in all_matches if path_is_label_safe(candidate, label)]
            search_cache[cache_key_name] = matches
        if matches:
            repaired.at[idx, 'filepath'] = str(matches[0])
            repaired_count += 1
        else:
            missing_rows.append(idx)
    if unsafe_rows:
        print(f'Dropping {len(unsafe_rows)} manifest rows whose filepath does not match their label.')
        repaired = repaired.drop(index=unsafe_rows)
    if missing_rows:
        print(f'Dropping {len(missing_rows)} manifest rows with missing files and no label-safe repair.')
        repaired = repaired.drop(index=missing_rows)
    if repaired_count:
        print(f'Repaired {repaired_count} stale manifest paths by label-safe filename search.')
    return repaired.reset_index(drop=True)


def manifest_has_label_conflicts(frame: pd.DataFrame) -> bool:
    if frame.empty or 'filepath' not in frame or 'label' not in frame:
        return True
    conflicts = frame.groupby('filepath')['label'].nunique()
    conflict_count = int((conflicts > 1).sum())
    if conflict_count:
        print(f'RFUAV manifest invalid: {conflict_count} source files are assigned to multiple labels.')
        examples = conflicts[conflicts > 1].head(5).index.tolist()
        for example in examples:
            labels = sorted(frame.loc[frame['filepath'] == example, 'label'].astype(str).unique().tolist())
            print(' conflict:', example, 'labels=', labels[:8])
        return True
    return False


def manifest_too_small(frame: pd.DataFrame) -> bool:
    expected_min_classes = int(os.getenv('RFUAV34B_MIN_VALID_CLASSES', str(min(max(len(archive_paths), 1), 10))))
    actual_classes = int(frame['label'].nunique()) if not frame.empty and 'label' in frame else 0
    if actual_classes < expected_min_classes:
        print(f'RFUAV manifest has only {actual_classes} valid classes; expected at least {expected_min_classes}.')
        return True
    return False


manifest_df = repair_or_drop_missing_files(manifest_df)
if manifest_df.empty or manifest_has_label_conflicts(manifest_df) or manifest_too_small(manifest_df):
    print('Cached RFUAV manifest is invalid/incomplete; rebuilding from archives/extracted files under:', extract_root)
    manifest_df = rebuild_manifest_from_extracted()
    manifest_df = repair_or_drop_missing_files(manifest_df)
    if manifest_df.empty:
        raise FileNotFoundError(
            f'No RFUAV IQ files found under {extract_root}. Run notebook 10 Cell 6/7 or verify extraction.'
        )
    if manifest_has_label_conflicts(manifest_df):
        raise RuntimeError('Rebuilt RFUAV manifest still has source files assigned to multiple labels; inspect extraction layout.')
    manifest_df.to_csv(manifest_path, index=False)
    print('Rebuilt manifest:', manifest_path, 'rows:', len(manifest_df))

# Keep only actual IQ-like payloads. RFUAV also ships pack*.xml metadata next to .iq files;
# training on XML bytes as float32 IQ causes total collapse.
before_iq_filter = len(manifest_df)
manifest_df = manifest_df[manifest_df['filepath'].map(lambda value: Path(value).suffix.lower() in IQ_SUFFIXES)].reset_index(drop=True)
dropped_non_iq = before_iq_filter - len(manifest_df)
if dropped_non_iq:
    print(f'Dropped {dropped_non_iq} non-IQ manifest rows before training.')
if manifest_df.empty:
    raise RuntimeError('RFUAV manifest has no IQ payload rows after filtering. Run notebook 10 Cell 7 and rebuild manifest.')
if manifest_has_label_conflicts(manifest_df):
    raise RuntimeError('RFUAV manifest has conflicting labels after IQ filtering; refusing to train on poisoned labels.')
manifest_df.to_csv(manifest_path, index=False)
print('Saved label-safe RFUAV manifest:', manifest_path, 'rows:', len(manifest_df))

label_names = sorted(manifest_df['label'].unique().tolist())
label_to_idx = {name: idx for idx, name in enumerate(label_names)}
manifest_df['label_idx'] = manifest_df['label'].map(label_to_idx).astype(np.int64)
num_classes = len(label_names)
print('classes before 34b filtering:', num_classes)
print('raw class counts:')
print(manifest_df.groupby('label').size().sort_values(ascending=False).head(40))

# 34b fast lane: infer SNR from paths when present, keep highest-SNR rows, then class-balance a small subset.
def infer_snr_from_text(value) -> float:
    text = str(value)
    patterns = [
        r'(?i)snr[_=\- ]*(-?\d+(?:\.\d+)?)',
        r'(?i)(-?\d+(?:\.\d+)?)\s*dB',
        r'(?i)dB[_=\- ]*(-?\d+(?:\.\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return np.nan

snr_sources = ['filepath', 'relative_path', 'archive', 'label']
manifest_df['snr'] = np.nan
for column in snr_sources:
    if column in manifest_df.columns:
        inferred = manifest_df[column].map(infer_snr_from_text)
        manifest_df['snr'] = manifest_df['snr'].where(manifest_df['snr'].notna(), inferred)

snr_threshold_env = os.getenv('RFUAV34B_MIN_SNR', '').strip()
highest_snr_levels = int(os.getenv('RFUAV34B_TOP_SNR_LEVELS', '8'))
if manifest_df['snr'].notna().any():
    if snr_threshold_env:
        threshold = float(snr_threshold_env)
        manifest_df = manifest_df[manifest_df['snr'] >= threshold].reset_index(drop=True)
        print('34b SNR filter: threshold >=', threshold)
    else:
        unique_snrs = sorted(manifest_df['snr'].dropna().unique(), reverse=True)
        keep_snrs = set(unique_snrs[:highest_snr_levels])
        manifest_df = manifest_df[manifest_df['snr'].isin(keep_snrs)].reset_index(drop=True)
        print('34b SNR filter: top SNR levels kept:', sorted(keep_snrs))
else:
    print('34b warning: no SNR values inferred from RFUAV paths; using class-balanced subset across all rows.')

samples_per_class = int(os.getenv('RFUAV34B_SAMPLES_PER_CLASS', '40'))
if samples_per_class > 0:
    parts = []
    for label_idx, group in manifest_df.groupby('label_idx', sort=True):
        if group.empty:
            continue
        n = min(samples_per_class, len(group))
        parts.append(group.sample(n=n, replace=False, random_state=RANDOM_STATE + int(label_idx)))
    manifest_df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    print('34b balanced cap per class:', samples_per_class)

# Recompute label list after filtering in case a class has no high-SNR samples.
if manifest_df.empty:
    raise RuntimeError('RFUAV 34b filtering left no rows. Relax RFUAV34B_MIN_SNR or increase RFUAV34B_TOP_SNR_LEVELS.')

label_names = sorted(manifest_df['label'].unique().tolist())
label_to_idx = {name: idx for idx, name in enumerate(label_names)}
manifest_df['label_idx'] = manifest_df['label'].map(label_to_idx).astype(np.int64)
num_classes = len(label_names)
print('34b classes after filtering:', num_classes)
print('34b rows after filtering:', len(manifest_df))
print('34b class counts:')
print(manifest_df.groupby('label').size().sort_values(ascending=False).head(80))
min_rows_per_class = int(manifest_df.groupby('label').size().min())
if min_rows_per_class < 20:
    print(f'34b warning: least populated class has only {min_rows_per_class} rows; accuracy will be noisy. Consider RFUAV34B_TOP_SNR_LEVELS=12 or RFUAV34B_SAMPLES_PER_CLASS=300.')
if 'snr' in manifest_df.columns and manifest_df['snr'].notna().any():
    print('34b SNR counts:')
    print(manifest_df['snr'].value_counts().sort_index())

# %% Cell 3
# Cell 3 : Define RFUAV IQ loading, fast spectrogram caching, and datasets
def _read_numeric_text(path: Path) -> np.ndarray:
    try:
        arr = np.loadtxt(path, delimiter=',')
    except Exception:
        arr = np.loadtxt(path)
    return np.asarray(arr)


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
        # RFUAV author tooling uses np.fromfile(..., dtype=np.float32), then I + 1j*Q.
        raw = np.fromfile(path, dtype=RFUAV_RAW_DTYPE)
        if raw.size < 4 or not np.isfinite(raw).all():
            raw = np.fromfile(path, dtype=np.int16).astype(np.float32)
        arr = raw

    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if np.iscomplexobj(arr):
        iq = np.stack([arr.real, arr.imag], axis=-1)
    elif arr.ndim == 1:
        if arr.size % 2 != 0:
            arr = arr[:-1]
        iq = arr.reshape(-1, 2)
    elif arr.ndim >= 2 and 2 in arr.shape:
        if arr.shape[-1] == 2:
            iq = arr.reshape(-1, 2)
        else:
            iq = np.moveaxis(arr, list(arr.shape).index(2), -1).reshape(-1, 2)
    else:
        flat = arr.reshape(-1)
        if flat.size % 2 != 0:
            flat = flat[:-1]
        iq = flat.reshape(-1, 2)
    iq = np.asarray(iq, dtype=np.float32)
    iq = iq[np.isfinite(iq).all(axis=1)]
    if iq.shape[0] < SPEC_NFFT:
        iq = np.pad(iq, ((0, SPEC_NFFT - iq.shape[0]), (0, 0)), mode='constant')
    return iq


def normalize_iq_window(out: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(out.astype(np.float32) ** 2) + 1e-8)
    return (out / scale).astype(np.float32)


def select_active_window(iq: np.ndarray, max_samples=MAX_IQ_SAMPLES) -> np.ndarray:
    if iq.shape[0] <= max_samples:
        return normalize_iq_window(iq)
    power = np.sum(iq.astype(np.float32) ** 2, axis=1)
    kernel = np.ones(min(8192, max_samples), dtype=np.float32)
    smooth = np.convolve(power, kernel / kernel.size, mode='same')
    center = int(np.argmax(smooth))
    start = max(0, min(center - max_samples // 2, iq.shape[0] - max_samples))
    return normalize_iq_window(iq[start:start + max_samples])


def select_random_window(iq: np.ndarray, max_samples=MAX_IQ_SAMPLES) -> np.ndarray:
    if iq.shape[0] <= max_samples:
        return normalize_iq_window(iq)
    start = np.random.randint(0, iq.shape[0] - max_samples + 1)
    return normalize_iq_window(iq[start:start + max_samples])


def cache_key(filepath: str) -> str:
    path = Path(filepath)
    stat = path.stat()
    token = f'{RFUAV34B_CACHE_VERSION}|{path}|{stat.st_size}|{stat.st_mtime_ns}|{MAX_IQ_SAMPLES}|{SPEC_NFFT}|{SPEC_HOP}|{SPEC_TIME_BINS}'
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
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, x=arr.astype(np.float32))
    tmp.replace(path)


def complex_spectrogram(iq: np.ndarray) -> np.ndarray:
    # Lightweight 33c-style representation: real/imag STFT channels only.
    complex_iq = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
    if complex_iq.size < SPEC_NFFT:
        complex_iq = np.pad(complex_iq, (0, SPEC_NFFT - complex_iq.size), mode='constant')
    starts = np.arange(0, max(1, complex_iq.size - SPEC_NFFT + 1), SPEC_HOP, dtype=np.int64)
    if starts.size == 0:
        starts = np.array([0], dtype=np.int64)
    if starts.size >= SPEC_TIME_BINS:
        idx = np.linspace(0, starts.size - 1, SPEC_TIME_BINS).round().astype(np.int64)
        starts = starts[idx]
    else:
        starts = np.pad(starts, (0, SPEC_TIME_BINS - starts.size), mode='edge')
    window = np.hanning(SPEC_NFFT).astype(np.float32)
    frames = np.stack([complex_iq[start:start + SPEC_NFFT] * window for start in starts], axis=0)
    spec = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1) / float(SPEC_NFFT)
    out = np.stack([spec.real.astype(np.float32), spec.imag.astype(np.float32)], axis=-1)
    out = out / (np.std(out) + 1e-6)
    return np.clip(out, -6.0, 6.0).astype(np.float32)

input_shape = (SPEC_TIME_BINS, SPEC_NFFT, 2)


def prepare_spectrogram(filepath: str, training=False) -> np.ndarray:
    # Eval uses cached strongest-window spectrograms. Training can use random windows
    # to get more views per RFUAV file without expanding the manifest.
    if not training:
        cache_path = cache_dir / f'{cache_key(filepath)}.npz'
        if cache_path.exists():
            cached = safe_load_cache(cache_path)
            if cached is not None:
                return cached
        iq = select_active_window(load_iq_file(filepath))
        spec = complex_spectrogram(iq)
        write_cache_atomic(cache_path, spec)
        return spec
    iq = select_random_window(load_iq_file(filepath))
    return complex_spectrogram(iq)

if DATA_FRACTION < 1.0:
    manifest_df = (
        manifest_df.groupby('label_idx', group_keys=False)
        .apply(lambda frame: frame.sample(max(1, int(math.ceil(len(frame) * DATA_FRACTION))), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )
    print('Fractioned rows:', len(manifest_df))

def split_by_class(frame: pd.DataFrame, train_frac=0.70, val_frac=0.15, random_state=RANDOM_STATE):
    # sklearn's second stratified split fails when tmp has only one sample for a class.
    # This deterministic per-class split keeps rare classes usable and avoids dropping data.
    rng = np.random.default_rng(random_state)
    train_parts = []
    val_parts = []
    test_parts = []
    for label_idx, group in frame.groupby('label_idx', sort=True):
        group = group.sample(frac=1.0, random_state=random_state + int(label_idx)).reset_index(drop=True)
        n = len(group)
        if n < 3:
            # Too small for all splits; keep it in train and let balanced replay upsample it.
            train_parts.append(group)
            print(f'Class {label_idx} has only {n} sample(s); assigning all to train.')
            continue
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])
    train = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True) if val_parts else pd.DataFrame(columns=frame.columns)
    test = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True) if test_parts else pd.DataFrame(columns=frame.columns)
    if val.empty or test.empty:
        raise ValueError('RFUAV split produced an empty val/test split. Increase DATA_FRACTION or add more RFUAV samples.')
    return train, val, test

RFUAV34B_VIEW_SPLIT = os.getenv('RFUAV34B_VIEW_SPLIT', '1').lower() not in {'0', 'false', 'no'}
if RFUAV34B_VIEW_SPLIT:
    # RFUAV high-SNR fast mode may have only 2-3 raw files per class. A raw-file split
    # leaves validation/test with only a few classes, making the confusion matrix useless.
    # Use deterministic window splits instead: train/val/test share source files but use
    # different RF windows during precompute. Treat this as a pipeline/model sanity benchmark,
    # not an independent dataset estimate.
    train_df = manifest_df.copy().reset_index(drop=True)
    val_df = manifest_df.copy().reset_index(drop=True)
    test_df = manifest_df.copy().reset_index(drop=True)
    print('RFUAV34B_VIEW_SPLIT=1: using deterministic window splits over all classes.')
else:
    train_df, val_df, test_df = split_by_class(manifest_df)
print('train/val/test:', len(train_df), len(val_df), len(test_df))
print('val class counts:', val_df['label_idx'].value_counts().sort_index().to_dict())
print('test class counts:', test_df['label_idx'].value_counts().sort_index().to_dict())

class_counts = train_df['label_idx'].value_counts().sort_index()
max_count = int(class_counts.max())
balanced_train_df = pd.concat([
    train_df[train_df['label_idx'] == label_idx].sample(max_count, replace=True, random_state=RANDOM_STATE + int(label_idx))
    for label_idx in class_counts.index
], ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
print('balanced train rows:', len(balanced_train_df))


def make_dataset(frame: pd.DataFrame, training=False):
    paths = frame['filepath'].astype(str).to_numpy()
    labels = frame['label_idx'].to_numpy(dtype=np.int64)

    def gen():
        for filepath, label in zip(paths, labels):
            yield prepare_spectrogram(filepath), np.int64(label)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )
    if training:
        ds = ds.shuffle(min(SHUFFLE_BUFFER, len(frame)), reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(balanced_train_df, training=True)
val_ds = make_dataset(val_df, training=False)
test_ds = make_dataset(test_df, training=False)
train_steps = int(math.ceil(len(balanced_train_df) / BATCH_SIZE))
validation_steps = int(math.ceil(len(val_df) / BATCH_SIZE))
print('steps:', train_steps, validation_steps)

# %% Cell 4
# Cell 4 : Precompute compact RFUAV 34b spectrogram tensors for faster training
# First run does the raw IQ read + FFT work. Later training/eval cells load these compact tensors directly.
precompute_dir = Path(os.getenv('RFUAV34B_PRECOMPUTE_DIR', str(cache_dir / 'precomputed_tensors')))
precompute_dir.mkdir(parents=True, exist_ok=True)
RFUAV34B_PRECOMPUTE_MAX_RAW_PER_CLASS = int(os.getenv('RFUAV34B_PRECOMPUTE_MAX_RAW_PER_CLASS', '12'))
RFUAV34B_PRECOMPUTE_AUGMENTS = int(os.getenv('RFUAV34B_PRECOMPUTE_AUGMENTS', '3'))
RFUAV34B_FORCE_PRECOMPUTE = os.getenv('RFUAV34B_FORCE_PRECOMPUTE', '0').lower() in {'1', 'true', 'yes'}
RFUAV34B_COMPRESS_PRECOMPUTE = os.getenv('RFUAV34B_COMPRESS_PRECOMPUTE', '0').lower() in {'1', 'true', 'yes'}
RFUAV34B_BURST_CANDIDATES = int(os.getenv('RFUAV34B_BURST_CANDIDATES', '8'))
RFUAV34B_BURST_SMOOTH = int(os.getenv('RFUAV34B_BURST_SMOOTH', str(min(4096, MAX_IQ_SAMPLES))))
RFUAV34B_BURST_MIN_SPACING = int(os.getenv('RFUAV34B_BURST_MIN_SPACING', str(max(1, MAX_IQ_SAMPLES))))
print('34b precompute max raw/class:', RFUAV34B_PRECOMPUTE_MAX_RAW_PER_CLASS)
print('34b burst candidates/smooth/min-spacing:', RFUAV34B_BURST_CANDIDATES, RFUAV34B_BURST_SMOOTH, RFUAV34B_BURST_MIN_SPACING)


def precompute_key(filepath: str, split: str, view_idx: int) -> str:
    path = Path(filepath)
    stat = path.stat()
    token = '|'.join([
        str(path), str(stat.st_size), str(stat.st_mtime_ns), split, str(view_idx),
        RFUAV34B_CACHE_VERSION, str(MAX_IQ_SAMPLES), str(SPEC_NFFT), str(SPEC_HOP), str(SPEC_TIME_BINS), str(RFUAV34B_BURST_CANDIDATES), 'v8_top_energy_bursts',
    ])
    return hashlib.sha1(token.encode('utf-8')).hexdigest()


def burst_window_starts(iq: np.ndarray, filepath: str, candidate_count: int = RFUAV34B_BURST_CANDIDATES) -> list[int]:
    """Return deterministic high-energy, non-overlapping-ish window starts.

    RFUAV captures can contain short controller bursts plus long quiet regions. Random crops
    make training slow and collapse-prone, so precompute views from the strongest energy
    regions first and only fall back to deterministic random windows when needed.
    """
    if iq.shape[0] <= MAX_IQ_SAMPLES:
        return [0]
    power = np.sum(iq.astype(np.float32) ** 2, axis=1)
    smooth_len = max(1, min(int(RFUAV34B_BURST_SMOOTH), len(power), MAX_IQ_SAMPLES))
    kernel = np.ones(smooth_len, dtype=np.float32) / float(smooth_len)
    smooth = np.convolve(power, kernel, mode='same')
    order = np.argsort(smooth)[::-1]
    starts = []
    min_spacing = max(1, int(RFUAV34B_BURST_MIN_SPACING))
    for center in order:
        start = int(max(0, min(int(center) - MAX_IQ_SAMPLES // 2, iq.shape[0] - MAX_IQ_SAMPLES)))
        if all(abs(start - existing) >= min_spacing for existing in starts):
            starts.append(start)
            if len(starts) >= candidate_count:
                break
    # Deterministic random fallback fills any remaining requested views.
    if len(starts) < candidate_count:
        seed = int(hashlib.sha1(f'{filepath}|burst_fallback|{RANDOM_STATE}'.encode('utf-8')).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        attempts = 0
        while len(starts) < candidate_count and attempts < candidate_count * 20:
            start = int(rng.integers(0, iq.shape[0] - MAX_IQ_SAMPLES + 1))
            if all(abs(start - existing) >= min_spacing for existing in starts):
                starts.append(start)
            attempts += 1
    return starts or [0]


def seeded_window(iq: np.ndarray, filepath: str, view_idx: int, starts: list[int] | None = None) -> np.ndarray:
    if iq.shape[0] <= MAX_IQ_SAMPLES:
        return normalize_iq_window(iq)
    starts = starts if starts else burst_window_starts(iq, filepath)
    start = int(starts[int(view_idx) % len(starts)])
    return normalize_iq_window(iq[start:start + MAX_IQ_SAMPLES])


def save_precomputed(path: Path, spec: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as handle:
        if RFUAV34B_COMPRESS_PRECOMPUTE:
            np.savez_compressed(handle, x=spec.astype(np.float32))
        else:
            np.savez(handle, x=spec.astype(np.float32))
    tmp.replace(path)


def valid_precomputed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            arr = np.asarray(data['x'], dtype=np.float32)
        return arr.shape == input_shape and np.isfinite(arr).all()
    except Exception:
        path.unlink(missing_ok=True)
        return False


def load_precomputed_spectrogram(path_like) -> np.ndarray:
    with np.load(Path(path_like)) as data:
        arr = np.asarray(data['x'], dtype=np.float32)
    if arr.shape != input_shape:
        raise ValueError(f'Bad precomputed spectrogram shape {arr.shape}; expected {input_shape}: {path_like}')
    return arr


def cap_raw_frame_for_precompute(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if RFUAV34B_PRECOMPUTE_MAX_RAW_PER_CLASS <= 0:
        return frame.reset_index(drop=True)
    capped = (
        frame.groupby('label_idx', group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), RFUAV34B_PRECOMPUTE_MAX_RAW_PER_CLASS), random_state=RANDOM_STATE + abs(hash(split)) % 10000))
        .reset_index(drop=True)
    )
    print(f'Capped {split} raw files for precompute:', len(frame), '->', len(capped))
    return capped


def precompute_frame(frame: pd.DataFrame, split: str, augment_count: int, start_view_idx: int = 0) -> pd.DataFrame:
    frame = cap_raw_frame_for_precompute(frame, split)
    rows = []
    total = len(frame)
    for row_idx, row in enumerate(frame.itertuples(index=False), start=1):
        filepath = str(row.filepath)
        label_idx = int(row.label_idx)
        views = max(1, augment_count)
        iq = None
        burst_starts = None
        for local_view_idx in range(views):
            view_idx = start_view_idx + local_view_idx
            out_path = precompute_dir / split / f'{precompute_key(filepath, split, view_idx)}.npz'
            if RFUAV34B_FORCE_PRECOMPUTE or not valid_precomputed(out_path):
                if iq is None:
                    iq = load_iq_file(filepath)
                    burst_starts = burst_window_starts(iq, filepath, candidate_count=max(views, RFUAV34B_BURST_CANDIDATES))
                spec = complex_spectrogram(seeded_window(iq, filepath, view_idx, starts=burst_starts))
                save_precomputed(out_path, spec)
            rows.append({
                'spec_path': str(out_path),
                'source_filepath': filepath,
                'label_idx': label_idx,
                'label': getattr(row, 'label', label_names[label_idx]),
                'split': split,
                'view_idx': view_idx,
            })
        if row_idx <= 5 or row_idx % 25 == 0 or row_idx == total:
            print(f'Precomputed {split}: {row_idx}/{total} raw files -> {len(rows)} tensor rows', flush=True)
    result = pd.DataFrame(rows)
    manifest_out = precompute_dir / f'{split}_manifest.csv'
    result.to_csv(manifest_out, index=False)
    print(f'Saved {split} precompute manifest:', manifest_out, 'rows:', len(result))
    return result


train_source_df = balanced_train_df if os.getenv('RFUAV34B_TRAIN_ON_BALANCED_REPLAY', '1').lower() not in {'0', 'false', 'no'} else train_df
precomputed_train_df = precompute_frame(train_source_df, 'train', RFUAV34B_PRECOMPUTE_AUGMENTS, start_view_idx=0)
precomputed_val_df = precompute_frame(val_df, 'val', int(os.getenv('RFUAV34B_VAL_VIEWS', '1')), start_view_idx=100)
precomputed_test_df = precompute_frame(test_df, 'test', int(os.getenv('RFUAV34B_TEST_VIEWS', '1')), start_view_idx=200)

print('Precomputed train class counts:', precomputed_train_df['label_idx'].value_counts().sort_index().to_dict())
print('Precomputed val class counts:', precomputed_val_df['label_idx'].value_counts().sort_index().to_dict())
print('Precomputed test class counts:', precomputed_test_df['label_idx'].value_counts().sort_index().to_dict())

# %% Cell 5
# Cell 5 : Build or resume the RFUAV 34b high-SNR spectrogram CNN model
def conv_block(x, filters, dropout=0.0):
    # Two small kernels learn local burst texture better than one wider kernel,
    # while staying much lighter than the 34 VGG-style full model.
    for _ in range(2):
        x = Conv2D(
            filters,
            3,
            padding='same',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(CNN_WEIGHT_DECAY),
        )(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    return x


def build_cnn_model():
    inputs = Input(shape=input_shape)
    x = conv_block(inputs, 32, dropout=0.05)
    x = conv_block(x, 64, dropout=0.08)
    x = conv_block(x, 96, dropout=0.12)
    x = conv_block(x, 160, dropout=0.18)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(CNN_WEIGHT_DECAY))(x)
    x = Dropout(0.30)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='rfuav_34b_high_snr_spectrogram_cnn')

best_path = model_dir / 'rfuav_34b_high_snr_spectrogram_cnn_best.keras'
final_path = model_dir / 'rfuav_34b_high_snr_spectrogram_cnn_final.keras'
history_path = model_dir / 'rfuav_34b_high_snr_spectrogram_cnn_history.json'
history_csv_path = model_dir / 'rfuav_34b_high_snr_spectrogram_cnn_training_history.csv'

RFUAV34B_RESET_MODEL = os.getenv('RFUAV34B_RESET_MODEL', '1').lower() not in {'0', 'false', 'no'}
resume_path = None if RFUAV34B_RESET_MODEL else (final_path if final_path.exists() else (best_path if best_path.exists() else None))
if RFUAV34B_RESET_MODEL:
    print('RFUAV34B_RESET_MODEL=1; starting fresh instead of resuming the collapsed checkpoint.')
if resume_path is not None:
    print('Attempting resume from:', resume_path)
    try:
        model = load_model(resume_path, compile=False)
        if tuple(model.input_shape[1:]) != tuple(input_shape) or model.output_shape[-1] != num_classes:
            print('Checkpoint shape/classes mismatch; starting fresh.', model.input_shape, model.output_shape)
            model = build_cnn_model()
        else:
            print('Resuming compatible checkpoint.')
    except Exception as exc:
        print('Could not load checkpoint, starting fresh:', repr(exc))
        model = build_cnn_model()
else:
    model = build_cnn_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=float(os.getenv('RFUAV34B_CNN_LR', os.getenv('RFUAV_CNN_LR', '1e-4'))),
        clipnorm=1.0,
    ),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    jit_compile=os.getenv('RFUAV_JIT_COMPILE', '0').lower() in {'1', 'true', 'yes'},
)
model.summary()

# %% Cell 6
# Cell 6 : Train or continue RFUAV 34b high-SNR spectrogram CNN model and save cumulative history
# Fast-start knobs for RFUAV: cold spectrogram generation is expensive, so avoid filling a huge shuffle buffer before epoch 1.
RFUAV34B_FAST_START = os.getenv('RFUAV34B_FAST_START', '1').lower() not in {'0', 'false', 'no'}
RFUAV34B_CELL5_SHUFFLE_BUFFER = int(os.getenv('RFUAV34B_CELL5_SHUFFLE_BUFFER', '512'))
RFUAV34B_TRAIN_ON_BALANCED_REPLAY = os.getenv('RFUAV34B_TRAIN_ON_BALANCED_REPLAY', '1').lower() not in {'0', 'false', 'no'}
RFUAV34B_VAL_METRIC_LIMIT_DEFAULT = '0'
RFUAV34B_PROGRESS_EVERY = int(os.getenv('RFUAV34B_PROGRESS_EVERY', '0'))
RFUAV34B_PREFETCH = int(os.getenv('RFUAV34B_PREFETCH', '2'))

def sanitize_existing_paths(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = frame.copy().reset_index(drop=True)
    exists_mask = frame['filepath'].map(lambda value: Path(value).exists())
    missing_count = int((~exists_mask).sum())
    if missing_count:
        print(f'{name}: dropping {missing_count} missing files before dataset construction.')
    frame = frame[exists_mask].reset_index(drop=True)
    if frame.empty:
        raise FileNotFoundError(f'{name} has no existing RFUAV files after filtering. Re-run Cell 2 or notebook 10 extraction.')
    return frame

# Cell 5 may be rerun after scratch cleanup, so sanitize all active splits here too.
train_df = sanitize_existing_paths(train_df, 'train_df')
val_df = sanitize_existing_paths(val_df, 'val_df')
if 'test_df' in globals():
    test_df = sanitize_existing_paths(test_df, 'test_df')
if 'balanced_train_df' in globals():
    balanced_train_df = sanitize_existing_paths(balanced_train_df, 'balanced_train_df')

def load_precomputed_spectrogram(path_like) -> np.ndarray:
    with np.load(Path(path_like)) as data:
        arr = np.asarray(data['x'], dtype=np.float32)
    if arr.shape != input_shape:
        raise ValueError(f'Bad precomputed spectrogram shape {arr.shape}; expected {input_shape}: {path_like}')
    return arr


if 'precomputed_train_df' in globals() and 'precomputed_val_df' in globals():
    print('Using precomputed RFUAV34B spectrogram tensors for training.')
    train_frame_for_fit = precomputed_train_df.copy()
    val_frame_for_fit = precomputed_val_df.copy()
else:
    print('Precomputed tensors not found; falling back to raw IQ generator. Run Cell 4 for faster training.')
    train_frame_for_fit = balanced_train_df if RFUAV34B_TRAIN_ON_BALANCED_REPLAY else train_df
    val_frame_for_fit = val_df

print('RFUAV train class counts:', train_frame_for_fit['label_idx'].value_counts().sort_index().to_dict())
print('RFUAV val class counts:', val_frame_for_fit['label_idx'].value_counts().sort_index().to_dict())


def make_fit_dataset(frame: pd.DataFrame, training=False):
    use_precomputed = 'spec_path' in frame.columns
    paths = frame['spec_path' if use_precomputed else 'filepath'].astype(str).to_numpy()
    labels = frame['label_idx'].to_numpy(dtype=np.int64)

    def gen():
        total = len(paths)
        for idx, (filepath, label) in enumerate(zip(paths, labels), start=1):
            if RFUAV34B_PROGRESS_EVERY > 0 and (idx <= BATCH_SIZE or idx % RFUAV34B_PROGRESS_EVERY == 0):
                print(f'RFUAV34B generator {idx}/{total}: label={int(label)} file={Path(filepath).name}', flush=True)
            if use_precomputed:
                yield load_precomputed_spectrogram(filepath), np.int64(label)
            else:
                yield prepare_spectrogram(filepath, training=training), np.int64(label)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )
    if RFUAV34B_CELL5_SHUFFLE_BUFFER > 1:
        ds = ds.shuffle(
            min(RFUAV34B_CELL5_SHUFFLE_BUFFER, len(frame)),
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True,
        )
    # Keep the dataset finite. Keras re-iterates it each epoch, and avoiding repeat()
    # prevents misleading zero-loss/zero-accuracy history rows when generators exhaust.
    ds = ds.batch(BATCH_SIZE)
    if RFUAV34B_PREFETCH > 0:
        ds = ds.prefetch(RFUAV34B_PREFETCH)
    return ds

# Rebuild train/validation datasets here so Cell 5 can be made faster and stale-path safe without rerunning Cell 3.
train_ds = make_fit_dataset(train_frame_for_fit, training=True)
val_ds = make_fit_dataset(val_frame_for_fit, training=False)
train_steps = int(math.ceil(len(train_frame_for_fit) / BATCH_SIZE))
validation_steps = int(math.ceil(len(val_frame_for_fit) / BATCH_SIZE))
print('RFUAV 34b Cell 5 fast start:', RFUAV34B_FAST_START)
print('RFUAV 34b Cell 5 shuffle buffer:', RFUAV34B_CELL5_SHUFFLE_BUFFER)
print('RFUAV 34b Cell 5 generator progress every:', RFUAV34B_PROGRESS_EVERY)
print('RFUAV 34b Cell 5 prefetch:', RFUAV34B_PREFETCH)
print('RFUAV training rows used:', len(train_frame_for_fit), 'balanced replay:', RFUAV34B_TRAIN_ON_BALANCED_REPLAY)
print('RFUAV train steps:', train_steps)
print('RFUAV datasets repeat enabled: False')

# Recompile here so Cell 5 can recover from too-aggressive Cell 4 defaults without rerunning Cell 4.
# Early RFUAV batches can look collapsed with many classes; safer LR + no XLA makes debugging saner.
RFUAV34B_CELL5_LR = float(os.getenv('RFUAV34B_CELL5_LR', '3e-4'))
RFUAV_CELL5_CLIPNORM = float(os.getenv('RFUAV_CELL5_CLIPNORM', '1.0'))
try:
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        label_smoothing=float(os.getenv('RFUAV34B_LABEL_SMOOTHING', '0.0'))
    )
except TypeError:
    # Older Keras builds do not expose label_smoothing for sparse CE.
    loss_obj = 'sparse_categorical_crossentropy'

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=RFUAV34B_CELL5_LR, clipnorm=RFUAV_CELL5_CLIPNORM),
    loss=loss_obj,
    metrics=['accuracy'],
    jit_compile=os.getenv('RFUAV_JIT_COMPILE', '0').lower() in {'1', 'true', 'yes'},
)
print('RFUAV 34b Cell 5 learning rate:', RFUAV34B_CELL5_LR)
print('RFUAV 34b Cell 5 loss:', loss_obj)

# Quick sanity check: verify labels are diverse and predictions are finite before training.
peek_x, peek_y = next(iter(train_ds.take(1)))
peek_probs = model.predict(peek_x, verbose=0)
print('RFUAV peek labels:', peek_y.numpy().tolist())
print('RFUAV peek unique labels:', sorted(set(peek_y.numpy().tolist())))
print('RFUAV peek probs finite:', bool(np.isfinite(peek_probs).all()), 'min/max:', float(np.min(peek_probs)), float(np.max(peek_probs)))
print('RFUAV peek predicted classes:', np.argmax(peek_probs, axis=1).tolist())

class ValidationMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, validation_frame, max_samples=0):
        super().__init__()
        self.validation_frame = validation_frame.head(max_samples).copy() if max_samples > 0 else validation_frame.copy()
        print(f'ValidationMacroF1 samples: {len(self.validation_frame)} / {len(validation_frame)}')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true = self.validation_frame['label_idx'].to_numpy(dtype=np.int64)
        y_pred = []
        for row in self.validation_frame.itertuples(index=False):
            x = (load_precomputed_spectrogram(row.spec_path) if hasattr(row, 'spec_path') else prepare_spectrogram(row.filepath))[None, ...]
            probs = self.model.predict(x, batch_size=BATCH_SIZE, verbose=0)[0]
            y_pred.append(int(np.argmax(probs)))
        macro_f1 = float(f1_score(y_true, np.asarray(y_pred), average='macro', zero_division=0))
        logs['val_macro_f1'] = macro_f1
        print(f' - val_macro_f1: {macro_f1:.4f}')

callbacks = [
    ValidationMacroF1(val_frame_for_fit, max_samples=int(os.getenv('RFUAV34B_VAL_METRIC_LIMIT', RFUAV34B_VAL_METRIC_LIMIT_DEFAULT))),
    ModelCheckpoint(best_path, monitor='val_macro_f1', mode='max', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_macro_f1', mode='max', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
]


if os.getenv('RFUAV34B_CLEAR_HISTORY_ON_RESET', '1').lower() not in {'0', 'false', 'no'} and os.getenv('RFUAV34B_RESET_MODEL', '1').lower() not in {'0', 'false', 'no'}:
    if history_csv_path.exists():
        archive_history_path = history_csv_path.with_suffix('.archived_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.csv')
        history_csv_path.replace(archive_history_path)
        print('Archived prior collapsed cumulative history:', archive_history_path)
existing_history_df = pd.read_csv(history_csv_path) if history_csv_path.exists() else pd.DataFrame()
last_global_epoch = int(existing_history_df['global_epoch'].max()) if not existing_history_df.empty else 0
run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
run_epochs = int(os.getenv('RFUAV34B_CNN_EPOCHS', '40'))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=run_epochs,
    callbacks=callbacks,
    verbose=1,
)

if best_path.exists():
    best_model = load_model(best_path, compile=False)
    best_model.save(final_path)
else:
    model.save(final_path)

history_payload = {
    'run_id': run_id,
    'history': history.history,
    'label_names': label_names,
    'config': {
        'input_shape': list(input_shape),
        'num_classes': int(num_classes),
        'data_fraction': float(DATA_FRACTION),
        'spec_nfft': int(SPEC_NFFT),
        'spec_hop': int(SPEC_HOP),
        'spec_time_bins': int(SPEC_TIME_BINS),
        'max_iq_samples': int(MAX_IQ_SAMPLES),
    },
}
history_path.write_text(json.dumps(history_payload, indent=2), encoding='utf-8')
run_history_path = model_dir / f'rfuav_34b_high_snr_spectrogram_cnn_history_{run_id}.json'
run_history_path.write_text(json.dumps(history_payload, indent=2), encoding='utf-8')

rows = []
for epoch_idx in range(len(history.history.get('loss', []))):
    row = {'run_id': run_id, 'epoch': epoch_idx + 1, 'global_epoch': last_global_epoch + epoch_idx + 1}
    for key, values in history.history.items():
        row[key] = values[epoch_idx]
    rows.append(row)
new_history_df = pd.DataFrame(rows)
combined_history_df = pd.concat([existing_history_df, new_history_df], ignore_index=True)
combined_history_df.to_csv(history_csv_path, index=False)
print('Saved canonical final model:', final_path)
print('Saved history:', history_path)
print('Saved cumulative history:', history_csv_path)

# %% Cell 7
# Cell 7 : Evaluate RFUAV 34b high-SNR spectrogram CNN with multi-window TTA and optional prior calibration
model = load_model(final_path if final_path.exists() else best_path, compile=False)

RFUAV34B_EVAL_TTA_WINDOWS = int(os.getenv('RFUAV34B_EVAL_TTA_WINDOWS', '8'))
RFUAV34B_EVAL_PRIOR_CALIBRATION = os.getenv('RFUAV34B_EVAL_PRIOR_CALIBRATION', '1').lower() not in {'0', 'false', 'no'}
RFUAV34B_EVAL_PROGRESS_EVERY = int(os.getenv('RFUAV34B_EVAL_PROGRESS_EVERY', '10'))
RFUAV34B_EVAL_BALANCED_PER_CLASS = int(os.getenv('RFUAV34B_EVAL_BALANCED_PER_CLASS', '0'))
RFUAV34B_EVAL_USE_BEST = os.getenv('RFUAV34B_EVAL_USE_BEST', '1').lower() not in {'0', 'false', 'no'}
RFUAV34B_EVAL_PRIOR_ALPHAS = [float(x) for x in os.getenv('RFUAV34B_EVAL_PRIOR_ALPHAS', '0,0.1,0.2,0.35,0.5,0.75,1.0').split(',') if x.strip()]
RFUAV34B_EVAL_TEMPERATURES = [float(x) for x in os.getenv('RFUAV34B_EVAL_TEMPERATURES', '0.75,1.0,1.25,1.5').split(',') if x.strip()]

if RFUAV34B_EVAL_USE_BEST and best_path.exists():
    model = load_model(best_path, compile=False)
    model_used_path = best_path
else:
    model_used_path = final_path if final_path.exists() else best_path
print('Loaded RFUAV34B eval model:', model_used_path)
print('Eval TTA windows:', RFUAV34B_EVAL_TTA_WINDOWS)
print('Eval prior calibration:', RFUAV34B_EVAL_PRIOR_CALIBRATION)


def row_source_filepath(row):
    if hasattr(row, 'source_filepath') and pd.notna(row.source_filepath):
        return str(row.source_filepath)
    if hasattr(row, 'filepath') and pd.notna(row.filepath):
        return str(row.filepath)
    return None


def row_cached_spec(row):
    if hasattr(row, 'spec_path') and pd.notna(row.spec_path) and Path(row.spec_path).exists():
        return load_precomputed_spectrogram(row.spec_path)
    if hasattr(row, 'filepath') and pd.notna(row.filepath) and Path(row.filepath).exists():
        return prepare_spectrogram(row.filepath)
    if hasattr(row, 'source_filepath') and pd.notna(row.source_filepath) and Path(row.source_filepath).exists():
        return prepare_spectrogram(row.source_filepath)
    raise FileNotFoundError(f'No usable RFUAV path found in eval row: {row}')


def predict_row_tta(row):
    source = row_source_filepath(row)
    specs = []
    if source is not None and Path(source).exists() and RFUAV34B_EVAL_TTA_WINDOWS > 1:
        # Average strongest-window plus deterministic random windows from the same capture.
        iq = load_iq_file(source)
        for view_idx in range(RFUAV34B_EVAL_TTA_WINDOWS):
            if 'seeded_window' in globals():
                window = seeded_window(iq, source, view_idx)
            elif view_idx == 0:
                window = select_active_window(iq)
            else:
                window = select_random_window(iq)
            specs.append(complex_spectrogram(window))
    else:
        specs.append(row_cached_spec(row))
    batch = np.stack(specs, axis=0).astype(np.float32)
    window_probs = model.predict(batch, batch_size=BATCH_SIZE, verbose=0)
    return np.mean(window_probs, axis=0).astype(np.float32)


def predict_frame(frame: pd.DataFrame, name='eval'):
    if EVAL_LIMIT > 0:
        frame = frame.head(EVAL_LIMIT).copy()
    def has_any_usable_path(row):
        for attr in ('source_filepath', 'spec_path', 'filepath'):
            if hasattr(row, attr):
                value = getattr(row, attr)
                if pd.notna(value) and Path(value).exists():
                    return True
        return False

    exists_mask = [has_any_usable_path(row) for row in frame.itertuples(index=False)]
    if not all(exists_mask):
        print(f'{name}: dropping {int(len(exists_mask) - sum(exists_mask))} rows with no usable raw or cached path before prediction.')
        frame = frame.loc[exists_mask].reset_index(drop=True)
    probs = []
    total = len(frame)
    for idx, row in enumerate(frame.itertuples(index=False), start=1):
        probs.append(predict_row_tta(row))
        if RFUAV34B_EVAL_PROGRESS_EVERY > 0 and (idx <= 3 or idx % RFUAV34B_EVAL_PROGRESS_EVERY == 0 or idx == total):
            print(f'{name} prediction {idx}/{total}', flush=True)
    return frame.reset_index(drop=True), np.asarray(probs, dtype=np.float32)


def make_balanced_eval_frame(frame: pd.DataFrame, per_class: int) -> pd.DataFrame:
    if per_class <= 0:
        return frame.copy().reset_index(drop=True)
    pieces = []
    for label_idx, group in frame.groupby('label_idx', sort=True):
        if group.empty:
            continue
        n = min(per_class, len(group))
        pieces.append(group.sample(n=n, replace=False, random_state=RANDOM_STATE + int(label_idx) + 9900))
    if not pieces:
        raise RuntimeError('No rows available for balanced RFUAV34B evaluation frame.')
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE + 9901).reset_index(drop=True)


def apply_prior_calibration(raw_probs: np.ndarray, log_prior_shift: np.ndarray | None, alpha=1.0, temperature=1.0):
    logits = np.log(np.clip(raw_probs, 1e-8, 1.0))
    logits = logits / max(float(temperature), 1e-6)
    if log_prior_shift is not None and alpha > 0:
        logits = logits + float(alpha) * log_prior_shift[None, :]
    logits = logits - np.max(logits, axis=1, keepdims=True)
    calibrated = np.exp(logits)
    calibrated = calibrated / np.sum(calibrated, axis=1, keepdims=True)
    return calibrated.astype(np.float32)

# Build the eval frame. By default this is the held-out test split, but you can request
# more rows per class with RFUAV34B_EVAL_BALANCED_PER_CLASS for a less tiny matrix.
eval_source_df = precomputed_test_df if 'precomputed_test_df' in globals() else test_df
eval_source_df = make_balanced_eval_frame(eval_source_df, RFUAV34B_EVAL_BALANCED_PER_CLASS)

# Estimate a lightweight prior correction from validation predictions, then tune how much
# to apply. Full prior correction can over-steer, so choose alpha/temperature by validation macro-F1.
log_prior_shift = None
best_calibration = {'alpha': 0.0, 'temperature': 1.0, 'val_macro_f1': np.nan, 'val_accuracy': np.nan}
calibration_sweep_df = pd.DataFrame()
if RFUAV34B_EVAL_PRIOR_CALIBRATION:
    calib_source_df = precomputed_val_df if 'precomputed_val_df' in globals() else val_df
    calib_df, calib_probs = predict_frame(calib_source_df, name='calibration')
    calib_true = calib_df['label_idx'].to_numpy(dtype=np.int64)
    empirical_true_prior = np.bincount(calib_true, minlength=num_classes).astype(np.float64)
    empirical_true_prior = empirical_true_prior / max(empirical_true_prior.sum(), 1.0)
    predicted_prior = calib_probs.mean(axis=0).astype(np.float64)
    predicted_prior = predicted_prior / max(predicted_prior.sum(), 1e-12)
    log_prior_shift = np.log(empirical_true_prior + 1e-6) - np.log(predicted_prior + 1e-6)
    print('Prior calibration max abs shift:', float(np.max(np.abs(log_prior_shift))))

    sweep_rows = []
    for alpha in RFUAV34B_EVAL_PRIOR_ALPHAS:
        for temperature in RFUAV34B_EVAL_TEMPERATURES:
            tuned_probs = apply_prior_calibration(calib_probs, log_prior_shift, alpha=alpha, temperature=temperature)
            tuned_pred = tuned_probs.argmax(axis=1)
            sweep_rows.append({
                'alpha': float(alpha),
                'temperature': float(temperature),
                'val_accuracy': float(accuracy_score(calib_true, tuned_pred)),
                'val_macro_f1': float(f1_score(calib_true, tuned_pred, average='macro', zero_division=0)),
                'predicted_classes': int(len(np.unique(tuned_pred))),
            })
    calibration_sweep_df = pd.DataFrame(sweep_rows).sort_values(
        ['val_macro_f1', 'val_accuracy', 'predicted_classes'], ascending=False
    ).reset_index(drop=True)
    if not calibration_sweep_df.empty:
        best_calibration = calibration_sweep_df.iloc[0].to_dict()
    sweep_path = outputs_dir / '34_rfuav_34b_high_snr_spectrogram_cnn_calibration_sweep.csv'
    calibration_sweep_df.to_csv(sweep_path, index=False)
    print('Best calibration:', best_calibration)
    print('Saved calibration sweep:', sweep_path)
    print(calibration_sweep_df.head(10))

eval_df, raw_probs = predict_frame(eval_source_df, name='test')
probs = apply_prior_calibration(
    raw_probs,
    log_prior_shift,
    alpha=float(best_calibration.get('alpha', 0.0)),
    temperature=float(best_calibration.get('temperature', 1.0)),
)
y_true = eval_df['label_idx'].to_numpy(dtype=np.int64)
y_pred = probs.argmax(axis=1)

acc = float(accuracy_score(y_true, y_pred))
macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
raw_acc = float(accuracy_score(y_true, raw_probs.argmax(axis=1)))
raw_macro_f1 = float(f1_score(y_true, raw_probs.argmax(axis=1), average='macro', zero_division=0))
print('RFUAV 34b high-SNR spectrogram CNN full-complex spectrogram report')
print('raw_accuracy:', raw_acc, 'raw_macro_f1:', raw_macro_f1)
print('accuracy:', acc)
print('macro_f1:', macro_f1)
print('prediction counts:', dict(zip(*np.unique(y_pred, return_counts=True))))
print('truth counts:', dict(zip(*np.unique(y_true, return_counts=True))))
print(classification_report(y_true, y_pred, labels=np.arange(num_classes), target_names=label_names, zero_division=0))

metrics = {
    'model': 'rfuav_34b_high_snr_spectrogram_cnn',
    'accuracy': acc,
    'macro_f1': macro_f1,
    'weighted_f1': weighted_f1,
    'raw_accuracy_before_prior_calibration': raw_acc,
    'raw_macro_f1_before_prior_calibration': raw_macro_f1,
    'tta_windows': int(RFUAV34B_EVAL_TTA_WINDOWS),
    'prior_calibration': bool(RFUAV34B_EVAL_PRIOR_CALIBRATION),
    'best_calibration': best_calibration,
    'balanced_eval_per_class': int(RFUAV34B_EVAL_BALANCED_PER_CLASS),
    'model_path': str(model_used_path),
    'num_classes': int(num_classes),
    'test_samples': int(len(eval_df)),
    'label_names': label_names,
}
metrics_path = outputs_dir / '34_rfuav_34b_high_snr_spectrogram_cnn_metrics.json'
metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print('Saved metrics:', metrics_path)

predictions_path = outputs_dir / '34_rfuav_34b_high_snr_spectrogram_cnn_predictions.csv'
pred_df = eval_df.copy()
pred_df['y_true'] = y_true
pred_df['y_pred'] = y_pred
pred_df['true_label'] = [label_names[idx] for idx in y_true]
pred_df['pred_label'] = [label_names[idx] for idx in y_pred]
pred_df['confidence'] = np.max(probs, axis=1)
pred_df.to_csv(predictions_path, index=False)
print('Saved predictions:', predictions_path)

fig, ax = plt.subplots(figsize=(max(12, num_classes * 0.45), max(10, num_classes * 0.35)))
sns.heatmap(confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)), cmap='Blues', xticklabels=label_names, yticklabels=label_names, ax=ax)
ax.set_title(f"RFUAV 34b high-SNR spectrogram CNN Confusion Matrix (TTA={RFUAV34B_EVAL_TTA_WINDOWS}, alpha={float(best_calibration.get('alpha', 0.0)):.2f}, temp={float(best_calibration.get('temperature', 1.0)):.2f})")
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
plt.tight_layout()
cm_path = outputs_dir / '34_rfuav_34b_high_snr_spectrogram_cnn_confusion_matrix.png'
plt.savefig(cm_path, dpi=180)
print('Saved:', cm_path)
_save_current_figure("cell_07_figure_01.png")

# %% Cell 8
# Cell 8 : Plot cumulative RFUAV 34b high-SNR spectrogram CNN training and validation curves
if not history_csv_path.exists():
    raise FileNotFoundError(f'Missing cumulative history file: {history_csv_path}')

history_df = pd.read_csv(history_csv_path)
print('History rows:', len(history_df))
print('Runs:', history_df['run_id'].nunique())
archived_histories = sorted(history_csv_path.parent.glob(history_csv_path.stem + '.archived_*.csv'))
if archived_histories:
    print('Archived old histories:', [p.name for p in archived_histories[-3:]])
print(history_df.groupby('run_id').agg({'epoch': 'max', 'val_accuracy': 'max'}).tail(10))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(history_df['global_epoch'], history_df.get('accuracy', []), marker='o', label='train_accuracy')
if 'val_accuracy' in history_df:
    axes[0].plot(history_df['global_epoch'], history_df['val_accuracy'], marker='o', label='val_accuracy')
    best_idx = history_df['val_accuracy'].idxmax()
    axes[0].scatter([history_df.loc[best_idx, 'global_epoch']], [history_df.loc[best_idx, 'val_accuracy']], color='red', zorder=5, label='best_val_accuracy')
if 'val_macro_f1' in history_df:
    axes[0].plot(history_df['global_epoch'], history_df['val_macro_f1'], marker='o', label='val_macro_f1')
axes[0].set_title('RFUAV 34b high-SNR spectrogram CNN Full-Complex Spectrogram Accuracy')
axes[0].set_xlabel('Global epoch')
axes[0].set_ylabel('Score')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(history_df['global_epoch'], history_df.get('loss', []), marker='o', label='train_loss')
if 'val_loss' in history_df:
    axes[1].plot(history_df['global_epoch'], history_df['val_loss'], marker='o', label='val_loss')
axes[1].set_title('RFUAV 34b high-SNR spectrogram CNN Full-Complex Spectrogram Loss')
axes[1].set_xlabel('Global epoch')
axes[1].set_ylabel('Loss')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
plt.tight_layout()
plot_path = outputs_dir / '34_rfuav_34b_high_snr_spectrogram_cnn_training_curves.png'
plt.savefig(plot_path, dpi=180)
print('Saved:', plot_path)
_save_current_figure("cell_08_figure_02.png")

# %% Cell 9
# Cell 9 : Larger balanced RFUAV 34b evaluation over raw held-out frames
# This is a heavier version of Cell 7. It uses the raw `test_df` frame instead of the small
# precomputed test frame, so the confusion matrix has more samples/class.

RFUAV34B_LARGE_EVAL_BALANCED_PER_CLASS = int(os.getenv('RFUAV34B_LARGE_EVAL_BALANCED_PER_CLASS', '100'))
RFUAV34B_LARGE_EVAL_TTA_WINDOWS = int(os.getenv('RFUAV34B_LARGE_EVAL_TTA_WINDOWS', '4'))
RFUAV34B_LARGE_EVAL_PROGRESS_EVERY = int(os.getenv('RFUAV34B_LARGE_EVAL_PROGRESS_EVERY', '25'))
RFUAV34B_LARGE_EVAL_PRIOR_CALIBRATION = os.getenv('RFUAV34B_LARGE_EVAL_PRIOR_CALIBRATION', '1').lower() not in {'0', 'false', 'no'}
RFUAV34B_LARGE_EVAL_USE_BEST = os.getenv('RFUAV34B_LARGE_EVAL_USE_BEST', '1').lower() not in {'0', 'false', 'no'}
RFUAV34B_LARGE_EVAL_PRIOR_ALPHAS = [float(x) for x in os.getenv('RFUAV34B_LARGE_EVAL_PRIOR_ALPHAS', '0,0.1,0.2,0.35,0.5,0.75,1.0').split(',') if x.strip()]
RFUAV34B_LARGE_EVAL_TEMPERATURES = [float(x) for x in os.getenv('RFUAV34B_LARGE_EVAL_TEMPERATURES', '0.75,1.0,1.25,1.5').split(',') if x.strip()]

required_names = ['test_df', 'val_df', 'make_balanced_eval_frame', 'predict_row_tta', 'apply_prior_calibration']
missing = [name for name in required_names if name not in globals()]
if missing:
    raise RuntimeError(f'Run Cells 1-7 before this large eval cell. Missing: {missing}')

large_model_path = best_path if RFUAV34B_LARGE_EVAL_USE_BEST and best_path.exists() else final_path
large_model = load_model(large_model_path, compile=False)
model = large_model  # predict_row_tta from Cell 7 reads the global `model`.
model_used_path = large_model_path

# predict_row_tta reads these globals from Cell 7. Temporarily override them for large eval.
old_tta = globals().get('RFUAV34B_EVAL_TTA_WINDOWS', None)
old_progress = globals().get('RFUAV34B_EVAL_PROGRESS_EVERY', None)
RFUAV34B_EVAL_TTA_WINDOWS = RFUAV34B_LARGE_EVAL_TTA_WINDOWS
RFUAV34B_EVAL_PROGRESS_EVERY = RFUAV34B_LARGE_EVAL_PROGRESS_EVERY

print('Loaded RFUAV34B large-eval model:', model_used_path)
print('Large eval balanced rows/class:', RFUAV34B_LARGE_EVAL_BALANCED_PER_CLASS)
print('Large eval TTA windows:', RFUAV34B_LARGE_EVAL_TTA_WINDOWS)


def has_any_usable_path(row):
    for attr in ('source_filepath', 'spec_path', 'filepath'):
        if hasattr(row, attr):
            value = getattr(row, attr)
            if pd.notna(value) and Path(value).exists():
                return True
    return False


def usable_row_count(frame: pd.DataFrame) -> int:
    return int(sum(has_any_usable_path(row) for row in frame.itertuples(index=False)))


def select_large_eval_source(raw_frame: pd.DataFrame, precomputed_name: str, rows_per_class: int, name: str) -> pd.DataFrame:
    source = make_balanced_eval_frame(raw_frame, rows_per_class)
    usable = usable_row_count(source)
    if usable == 0 and precomputed_name in globals():
        fallback = globals()[precomputed_name]
        print(f'{name}: raw balanced frame has no usable paths; falling back to {precomputed_name}.')
        source = make_balanced_eval_frame(fallback, rows_per_class)
        usable = usable_row_count(source)
    print(f'{name}: selected rows={len(source)}, usable paths={usable}')
    return source


def predict_large_frame(frame: pd.DataFrame, name='large_eval'):
    exists_mask = [has_any_usable_path(row) for row in frame.itertuples(index=False)]
    if not all(exists_mask):
        print(f'{name}: dropping {int(len(exists_mask) - sum(exists_mask))} rows with no usable raw or cached path before prediction.')
        frame = frame.loc[exists_mask].reset_index(drop=True)
    probs = []
    total = len(frame)
    for idx, row in enumerate(frame.itertuples(index=False), start=1):
        probs.append(predict_row_tta(row))
        if RFUAV34B_LARGE_EVAL_PROGRESS_EVERY > 0 and (idx <= 3 or idx % RFUAV34B_LARGE_EVAL_PROGRESS_EVERY == 0 or idx == total):
            print(f'{name} prediction {idx}/{total}', flush=True)
    if not probs:
        return frame.reset_index(drop=True), np.empty((0, num_classes), dtype=np.float32)
    return frame.reset_index(drop=True), np.asarray(probs, dtype=np.float32)

large_eval_source_df = select_large_eval_source(
    test_df,
    precomputed_name='precomputed_test_df',
    rows_per_class=RFUAV34B_LARGE_EVAL_BALANCED_PER_CLASS,
    name='large_test',
)
print('Large eval rows:', len(large_eval_source_df))
print('Large eval class counts:', large_eval_source_df['label_idx'].value_counts().sort_index().to_dict())

log_prior_shift = None
best_large_calibration = {'alpha': 0.0, 'temperature': 1.0, 'val_macro_f1': np.nan, 'val_accuracy': np.nan}
large_calibration_sweep_df = pd.DataFrame()
if RFUAV34B_LARGE_EVAL_PRIOR_CALIBRATION:
    large_calib_df_source = select_large_eval_source(
        val_df,
        precomputed_name='precomputed_val_df',
        rows_per_class=min(RFUAV34B_LARGE_EVAL_BALANCED_PER_CLASS, int(os.getenv('RFUAV34B_LARGE_CALIBRATION_PER_CLASS', '24'))),
        name='large_calibration',
    )
    large_calib_df, large_calib_probs = predict_large_frame(large_calib_df_source, name='large_calibration')
    if large_calib_probs.ndim != 2 or len(large_calib_df) == 0:
        print('Large calibration produced no predictions; disabling prior calibration for this run.')
        RFUAV34B_LARGE_EVAL_PRIOR_CALIBRATION = False
    else:
        large_calib_true = large_calib_df['label_idx'].to_numpy(dtype=np.int64)
        empirical_true_prior = np.bincount(large_calib_true, minlength=num_classes).astype(np.float64)
        empirical_true_prior = empirical_true_prior / max(empirical_true_prior.sum(), 1.0)
        predicted_prior = large_calib_probs.mean(axis=0).astype(np.float64)
        predicted_prior = predicted_prior / max(predicted_prior.sum(), 1e-12)
        log_prior_shift = np.log(empirical_true_prior + 1e-6) - np.log(predicted_prior + 1e-6)

        sweep_rows = []
        for alpha in RFUAV34B_LARGE_EVAL_PRIOR_ALPHAS:
            for temperature in RFUAV34B_LARGE_EVAL_TEMPERATURES:
                tuned_probs = apply_prior_calibration(large_calib_probs, log_prior_shift, alpha=alpha, temperature=temperature)
                tuned_pred = tuned_probs.argmax(axis=1)
                sweep_rows.append({
                    'alpha': float(alpha),
                    'temperature': float(temperature),
                    'val_accuracy': float(accuracy_score(large_calib_true, tuned_pred)),
                    'val_macro_f1': float(f1_score(large_calib_true, tuned_pred, average='macro', zero_division=0)),
                    'predicted_classes': int(len(np.unique(tuned_pred))),
                })
        large_calibration_sweep_df = pd.DataFrame(sweep_rows).sort_values(
            ['val_macro_f1', 'val_accuracy', 'predicted_classes'], ascending=False
        ).reset_index(drop=True)
        if not large_calibration_sweep_df.empty:
            best_large_calibration = large_calibration_sweep_df.iloc[0].to_dict()
        large_sweep_path = outputs_dir / '34_rfuav_34b_high_snr_spectrogram_cnn_large_eval_calibration_sweep.csv'
        large_calibration_sweep_df.to_csv(large_sweep_path, index=False)
        print('Large eval best calibration:', best_large_calibration)
        print('Saved large calibration sweep:', large_sweep_path)
        print(large_calibration_sweep_df.head(10))

large_eval_df, large_raw_probs = predict_large_frame(large_eval_source_df, name='large_test')
if large_raw_probs.ndim != 2 or large_raw_probs.shape[0] == 0 or len(large_eval_df) == 0:
    raise RuntimeError(
        'Large RFUAV eval produced no predictions. Check that test_df rows have valid '
        'source_filepath/spec_path/filepath values and that cached tensors still exist.'
    )
large_probs = apply_prior_calibration(
    large_raw_probs,
    log_prior_shift,
    alpha=float(best_large_calibration.get('alpha', 0.0)),
    temperature=float(best_large_calibration.get('temperature', 1.0)),
)
large_y_true = large_eval_df['label_idx'].to_numpy(dtype=np.int64)
large_y_pred = large_probs.argmax(axis=1)
large_raw_pred = large_raw_probs.argmax(axis=1)

large_acc = float(accuracy_score(large_y_true, large_y_pred))
large_macro_f1 = float(f1_score(large_y_true, large_y_pred, average='macro', zero_division=0))
large_weighted_f1 = float(f1_score(large_y_true, large_y_pred, average='weighted', zero_division=0))
large_raw_acc = float(accuracy_score(large_y_true, large_raw_pred))
large_raw_macro_f1 = float(f1_score(large_y_true, large_raw_pred, average='macro', zero_division=0))

print('RFUAV 34b LARGE high-SNR spectrogram CNN report')
print('raw_accuracy:', large_raw_acc, 'raw_macro_f1:', large_raw_macro_f1)
print('accuracy:', large_acc, 'macro_f1:', large_macro_f1, 'weighted_f1:', large_weighted_f1)
print('prediction counts:', dict(zip(*np.unique(large_y_pred, return_counts=True))))
print('truth counts:', dict(zip(*np.unique(large_y_true, return_counts=True))))
print(classification_report(large_y_true, large_y_pred, labels=np.arange(num_classes), target_names=label_names, zero_division=0))

large_metrics = {
    'model': 'rfuav_34b_high_snr_spectrogram_cnn_large_eval',
    'accuracy': large_acc,
    'macro_f1': large_macro_f1,
    'weighted_f1': large_weighted_f1,
    'raw_accuracy_before_prior_calibration': large_raw_acc,
    'raw_macro_f1_before_prior_calibration': large_raw_macro_f1,
    'tta_windows': int(RFUAV34B_LARGE_EVAL_TTA_WINDOWS),
    'prior_calibration': bool(RFUAV34B_LARGE_EVAL_PRIOR_CALIBRATION),
    'best_calibration': best_large_calibration,
    'balanced_eval_per_class': int(RFUAV34B_LARGE_EVAL_BALANCED_PER_CLASS),
    'model_path': str(model_used_path),
    'num_classes': int(num_classes),
    'test_samples': int(len(large_eval_df)),
    'label_names': label_names,
}
large_metrics_path = outputs_dir / '34_rfuav_34b_high_snr_spectrogram_cnn_large_eval_metrics.json'
large_metrics_path.write_text(json.dumps(large_metrics, indent=2), encoding='utf-8')
print('Saved large eval metrics:', large_metrics_path)

large_predictions_path = outputs_dir / '34_rfuav_34b_high_snr_spectrogram_cnn_large_eval_predictions.csv'
large_pred_df = large_eval_df.copy()
large_pred_df['y_true'] = large_y_true
large_pred_df['y_pred'] = large_y_pred
large_pred_df['true_label'] = [label_names[idx] for idx in large_y_true]
large_pred_df['pred_label'] = [label_names[idx] for idx in large_y_pred]
large_pred_df['confidence'] = np.max(large_probs, axis=1)
large_pred_df.to_csv(large_predictions_path, index=False)
print('Saved large eval predictions:', large_predictions_path)

fig, ax = plt.subplots(figsize=(max(12, num_classes * 0.45), max(10, num_classes * 0.35)))
sns.heatmap(confusion_matrix(large_y_true, large_y_pred, labels=np.arange(num_classes)), cmap='Blues', xticklabels=label_names, yticklabels=label_names, ax=ax)
ax.set_title(f"RFUAV 34b LARGE high-SNR spectrogram CNN Confusion Matrix (n={len(large_eval_df)}, TTA={RFUAV34B_LARGE_EVAL_TTA_WINDOWS}, alpha={float(best_large_calibration.get('alpha', 0.0)):.2f}, temp={float(best_large_calibration.get('temperature', 1.0)):.2f})")
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
plt.tight_layout()
large_cm_path = outputs_dir / '34_rfuav_34b_high_snr_spectrogram_cnn_large_eval_confusion_matrix.png'
plt.savefig(large_cm_path, dpi=180)
print('Saved:', large_cm_path)
_save_current_figure("cell_09_figure_03.png")

# Restore quick-eval globals in case the previous Cell 7 is re-run later.
if old_tta is not None:
    RFUAV34B_EVAL_TTA_WINDOWS = old_tta
if old_progress is not None:
    RFUAV34B_EVAL_PROGRESS_EVERY = old_progress
