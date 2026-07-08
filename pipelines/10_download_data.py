#!/usr/bin/env python3
"""Pipeline converted from the legacy 10_download_data workflow."""

from __future__ import annotations

from pathlib import Path

# %% Cell 1
# Cell 1 : Configure dataset root and define reusable Kaggle download helpers
from pathlib import Path
import os
import shutil
import sys
import subprocess

DATASET_ROOT = Path('/scratch/rameyjm7/datasets').resolve()
DATASET_ROOT.mkdir(parents=True, exist_ok=True)

print(f'Dataset root: {DATASET_ROOT}')

try:
    import kagglehub
except ImportError:
    print('kagglehub not found; installing...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kagglehub'])
    import kagglehub


def download_and_copy(dataset_ref: str, target_dir: Path, allowed_suffixes: tuple[str, ...]):
    target_dir.mkdir(parents=True, exist_ok=True)
    download_path = Path(kagglehub.dataset_download(dataset_ref))
    print(f'Downloaded {dataset_ref} to cache: {download_path}')

    copied = 0
    for src in download_path.rglob('*'):
        if src.is_file() and src.suffix.lower() in allowed_suffixes:
            dst = target_dir / src.name
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                continue
            shutil.copy2(src, dst)
            copied += 1
            print(f'Copied: {src} -> {dst}')

    if copied == 0:
        print(f'No new files copied for {dataset_ref}.')
    else:
        print(f'Copied {copied} files for {dataset_ref} into {target_dir}.')

# %% Cell 2
# Cell 2 : Download RML2018 artifacts into /scratch/rameyjm7/datasets/RML2018
rml2018_dir = DATASET_ROOT / 'RML2018'
download_and_copy(
    dataset_ref='pinxau1000/radioml2018',
    target_dir=rml2018_dir,
    allowed_suffixes=('.h5', '.hdf5', '.txt', '.pkl'),
)

# %% Cell 3
# Cell 3 : Download RML2016 artifacts into /scratch/rameyjm7/datasets/RML2016
rml2016_dir = DATASET_ROOT / 'RML2016'
download_and_copy(
    dataset_ref='zaslee/rml2016-10a',
    target_dir=rml2016_dir,
    allowed_suffixes=('.pkl', '.bz2', '.tar', '.gz'),
)

# %% Cell 4
# Cell 4 : Download DeepRadar2022 artifacts into /scratch/rameyjm7/datasets/DeepRadar2022
deepradar_dir = DATASET_ROOT / 'DeepRadar2022'
download_and_copy(
    dataset_ref='khilian/deepradar',
    target_dir=deepradar_dir,
    allowed_suffixes=('.mat',),
)

# %% Cell 5
# Cell 5 : Download Noisy Drone RF Signal Classification v2 artifacts into /scratch/rameyjm7/datasets/NoisyDroneRFv2
noisy_drone_rf_dir = DATASET_ROOT / 'NoisyDroneRFv2'
download_and_copy(
    dataset_ref='sgluege/noisy-drone-rf-signal-classification-v2',
    target_dir=noisy_drone_rf_dir,
    allowed_suffixes=('.pt', '.csv'),
)

# %% Cell 6
# Cell 6 : Download RFUAV archives via wget into /scratch/rameyjm7/datasets/RFUAV/archives
# RFUAV is hosted on Hugging Face/Xet. The HF API can rate-limit many file calls, so use direct wget URLs.
from urllib.parse import quote

rfuav_dir = DATASET_ROOT / 'RFUAV'
rfuav_archive_dir = rfuav_dir / 'archives'
rfuav_extract_dir = rfuav_dir / 'extracted'
rfuav_archive_dir.mkdir(parents=True, exist_ok=True)
rfuav_extract_dir.mkdir(parents=True, exist_ok=True)

RFUAV_ARCHIVES = [
    'DAUTEL EVO NANO.rar',
    'DEVENTION DEVO.rar',
    'DJI AVATA2.rar',
    'DJI FPV COMBO.rar',
    'DJI MAVIC3 PRO.rar',
    'DJI MINI3.rar',
    'DJI MINI4 PRO.rar',
    'FLYSKY EL 18.rar',
    'FLYSKY FS I6X.rar',
    'FLYSKY NV 14.rar',
    'FRSKY X14.rar',
    'FRSKY X20R.rar',
    'FRSKY X9DP2019.rar',
    'FUTABA T10J.rar',
    'FUTABA T14SG.rar',
    'FUTABA T16IZ.rar',
    'FUTABA T18SZ.rar',
    'Herelink Hx4.rar',
    'JR PROPO XG14.rar',
    'JR PROPO XG7.rar',
    'JUMPER T14.rar',
    'JUMPER TProV2.rar',
    'RadioMaster BOXER.rar',
    'RadioMaster TX16S.rar',
    'Radiolink AT10 II.rar',
    'Radiolink AT9S Pro.rar',
    'SIYI FT24.rar',
    'SIYI MK15.rar',
    'SIYI MK32.rar',
    'SKYDROID H12.rar',
    'SKYDROID T10.rar',
    'WFLY ET10.rar',
    'WFLY ET16S.rar',
    'WFLY WFT09SII.rar',
    'YUNZHUO H12.rar',
    'YUNZHUO H16.rar',
    'YUNZHUO H30.rar',
]

# If an older run put archives directly in RFUAV root, move them into the layout used by 34/34b.
for archive in rfuav_dir.glob('*.rar'):
    target = rfuav_archive_dir / archive.name
    if not target.exists():
        archive.replace(target)
        print('Moved existing RFUAV archive into archives/:', target.name)

max_archives_env = os.getenv('RFUAV_MAX_ARCHIVES', '').strip()
max_archives = len(RFUAV_ARCHIVES) if not max_archives_env else int(max_archives_env)
archives_to_fetch = RFUAV_ARCHIVES[:max_archives]
print('RFUAV target:', rfuav_dir)
print('RFUAV archive dir:', rfuav_archive_dir)
print('RFUAV archives requested:', len(archives_to_fetch), '/', len(RFUAV_ARCHIVES))

if not shutil.which('wget'):
    raise RuntimeError('wget is required for RFUAV direct downloads on this notebook path.')

base_url = 'https://huggingface.co/datasets/kitofrank/RFUAV/resolve/main/'
for name in archives_to_fetch:
    dst = rfuav_archive_dir / name
    if dst.exists() and dst.stat().st_size > 1024 * 1024:
        print('RFUAV archive exists, skipping:', dst.name, f'({dst.stat().st_size / 1e9:.2f} GB)')
        continue
    url = base_url + quote(name) + '?download=true'
    print('Downloading RFUAV archive:', name)
    subprocess.run([
        'wget',
        '--continue',
        '--tries=5',
        '--timeout=60',
        '--read-timeout=60',
        '--output-document', str(dst),
        url,
    ], check=True)

archive_paths = sorted(rfuav_archive_dir.glob('*.rar'))
manifest_path = rfuav_dir / 'rfuav_archives_manifest.csv'
try:
    import pandas as pd
    pd.DataFrame({
        'archive_name': [path.name for path in archive_paths],
        'archive_path': [str(path) for path in archive_paths],
        'archive_size_bytes': [path.stat().st_size for path in archive_paths],
        'extract_path': [str(rfuav_extract_dir / path.stem.replace(' ', '_')) for path in archive_paths],
    }).to_csv(manifest_path, index=False)
    print('Saved RFUAV archive manifest:', manifest_path)
except Exception as exc:
    print('Could not write RFUAV manifest:', repr(exc))

print('RFUAV archives available:', len(archive_paths))
print('Next: run Cell 7 to extract, or run 34b and let it extract needed archives.')

# %% Cell 7
# Cell 7 : Optionally extract RFUAV archives for pipelines 34/34b
# 34b can self-extract, but this cell prepares the dataset ahead of time.
import re

rfuav_dir = DATASET_ROOT / 'RFUAV'
rfuav_archive_dir = rfuav_dir / 'archives'
rfuav_extract_dir = rfuav_dir / 'extracted'
rfuav_extract_dir.mkdir(parents=True, exist_ok=True)


def safe_label_from_archive_name(name: str) -> str:
    label = Path(name).stem.strip()
    label = re.sub(r'\s+', '_', label)
    label = re.sub(r'[^A-Za-z0-9_\-]+', '', label)
    return label or 'unknown'


def has_extracted_iq(path: Path) -> bool:
    return any(path.rglob('*.iq'))


def extractor_command(archive_path: Path, extract_to: Path):
    if shutil.which('unrar'):
        return ['unrar', 'x', '-o+', str(archive_path), str(extract_to) + '/']
    if shutil.which('bsdtar'):
        return ['bsdtar', '-xf', str(archive_path), '-C', str(extract_to)]
    if shutil.which('7z'):
        return ['7z', 'x', '-y', f'-o{extract_to}', str(archive_path)]
    raise RuntimeError('No RAR extractor found. Install one of: unrar, bsdtar/libarchive, or p7zip/7z.')

max_extract_env = os.getenv('RFUAV_MAX_EXTRACT_ARCHIVES', os.getenv('RFUAV_MAX_ARCHIVES', '')).strip()
archives = sorted(rfuav_archive_dir.glob('*.rar'))
if max_extract_env:
    archives = archives[:int(max_extract_env)]
print('RFUAV archives to inspect/extract:', len(archives))

for archive in archives:
    label = safe_label_from_archive_name(archive.name)
    target = rfuav_extract_dir / label
    marker = target / '.extract_complete'
    if marker.exists() and has_extracted_iq(target):
        print('RFUAV already extracted:', archive.name)
        continue
    if target.exists() and has_extracted_iq(target):
        marker.write_text('existing extraction with IQ files\n', encoding='utf-8')
        print('RFUAV extraction marker refreshed:', archive.name)
        continue
    target.mkdir(parents=True, exist_ok=True)
    print('Extracting RFUAV:', archive.name, '->', target)
    subprocess.run(extractor_command(archive, target), check=True)
    marker.write_text('extracted\n', encoding='utf-8')

print('RFUAV extracted IQ files:', len(list(rfuav_extract_dir.rglob('*.iq'))))
print('RFUAV extract dir:', rfuav_extract_dir)

# %% Cell 8
# Cell 8 : Report expected files and print symlink command for repo data/ if needed
expected = {
    'RML2016': ['RML2016.10a_dict.pkl'],
    'RML2018': ['GOLD_XYZ_OSC.0001_1024.hdf5', 'classes.txt', 'classes-fixed.txt'],
    'DeepRadar2022': ['X_test.mat', 'Y_test.mat', 'lbl_test.mat'],
    'NoisyDroneRFv2': ['class_stats.csv', 'SNR_stats.csv'],
    'RFUAV': ['rfuav_archives_manifest.csv'],
}

for ds_name, names in expected.items():
    ds_dir = DATASET_ROOT / ds_name
    print(f'{ds_name}: {ds_dir}')
    for name in names:
        p = ds_dir / name
        print(f'  {name}:', 'FOUND' if p.exists() else 'MISSING')
    if ds_name == 'NoisyDroneRFv2':
        pt_count = len(list(ds_dir.rglob('IQdata_sample*_target*_snr*.pt'))) if ds_dir.exists() else 0
        print(f'  IQdata_sample*_target*_snr*.pt files: {pt_count}')
    if ds_name == 'RFUAV':
        rar_count = len(list((ds_dir / 'archives').glob('*.rar'))) if (ds_dir / 'archives').exists() else 0
        iq_count = len(list((ds_dir / 'extracted').rglob('*.iq'))) if (ds_dir / 'extracted').exists() else 0
        print(f'  archives/*.rar files: {rar_count}')
        print(f'  extracted/**/*.iq files: {iq_count}')

print('If repo data symlink needs reset:')
print('cd /home/rameyjm7/workspace/rf-signal-intelligence')
print('rm -f data && ln -s /scratch/rameyjm7/datasets data')

# %% Cell 9
# Cell 9 : Write local data path config for pipelines 20/30s/40 to consume
import yaml

repo_root = Path.cwd().resolve().parent if Path.cwd().name == 'pipelines' else Path.cwd().resolve()
config_path = repo_root / 'configs' / 'local_data_paths.yaml'
config_path.parent.mkdir(parents=True, exist_ok=True)

local_paths = {
    'version': 1,
    'dataset_root': str(DATASET_ROOT),
    'datasets': {
        'rml2016': {
            'pkl': str(DATASET_ROOT / 'RML2016' / 'RML2016.10a_dict.pkl'),
        },
        'rml2018': {
            'hdf5': str(DATASET_ROOT / 'RML2018' / 'GOLD_XYZ_OSC.0001_1024.hdf5'),
            'classes': str(DATASET_ROOT / 'RML2018' / 'classes.txt'),
            'classes_fixed': str(DATASET_ROOT / 'RML2018' / 'classes-fixed.txt'),
        },
        'deepradar2022': {
            'x_test': str(DATASET_ROOT / 'DeepRadar2022' / 'X_test.mat'),
            'y_test': str(DATASET_ROOT / 'DeepRadar2022' / 'Y_test.mat'),
            'lbl_test': str(DATASET_ROOT / 'DeepRadar2022' / 'lbl_test.mat'),
            'x_train': str(DATASET_ROOT / 'DeepRadar2022' / 'X_train.mat'),
            'y_train': str(DATASET_ROOT / 'DeepRadar2022' / 'Y_train.mat'),
            'lbl_train': str(DATASET_ROOT / 'DeepRadar2022' / 'lbl_train.mat'),
            'x_val': str(DATASET_ROOT / 'DeepRadar2022' / 'X_val.mat'),
            'y_val': str(DATASET_ROOT / 'DeepRadar2022' / 'Y_val.mat'),
            'lbl_val': str(DATASET_ROOT / 'DeepRadar2022' / 'lbl_val.mat'),
        },
        'noisy_drone_rf_v2': {
            'data_dir': str(DATASET_ROOT / 'NoisyDroneRFv2'),
            'class_stats': str(DATASET_ROOT / 'NoisyDroneRFv2' / 'class_stats.csv'),
            'snr_stats': str(DATASET_ROOT / 'NoisyDroneRFv2' / 'SNR_stats.csv'),
        },
        'rfuav': {
            'data_dir': str(DATASET_ROOT / 'RFUAV'),
            'archive_dir': str(DATASET_ROOT / 'RFUAV' / 'archives'),
            'extract_dir': str(DATASET_ROOT / 'RFUAV' / 'extracted'),
            'manifest': str(DATASET_ROOT / 'RFUAV' / 'rfuav_archives_manifest.csv'),
        },
    },
}

with config_path.open('w', encoding='utf-8') as f:
    yaml.safe_dump(local_paths, f, sort_keys=False)

print(f'Wrote local data config: {config_path}')
print('You can now run pipelines 20/30s/40 without hardcoded repo/data paths.')
