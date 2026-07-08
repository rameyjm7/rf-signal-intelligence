#!/usr/bin/env python3
"""Pipeline converted from the legacy 43_evaluation_cross_dataset_ensemble workflow."""

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
# Cell 1 : Explain notebook goals and evaluation modes
# Cell 1: Explain notebook goals and evaluation modes
# 33: Cross-Dataset Ensemble Evaluation (Diagnostic + Strict Mode)
# 
# This notebook loads all three datasets and all three trained models, then:
# - performs dataset-level diagnostics,
# - resolves RML2018 class-order ambiguity,
# - and computes one combined confusion matrix.
# 
# Default behavior uses strict dataset-to-model mapping to avoid cross-dataset label leakage.

# %% Cell 2
# Cell 2 : Import dependencies and configure reproducibility
from pathlib import Path
import ast
import pickle
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from scipy.io import loadmat
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

np.random.seed(42)

# %% Cell 3
# Cell 3 : Resolve paths, load local config, and load base models
repo = Path.cwd().resolve().parent if Path.cwd().resolve().name == 'pipelines' else Path.cwd().resolve()

cfg_path = repo / 'configs' / 'local_data_paths.yaml'
if cfg_path.exists():
    local_cfg = yaml.safe_load(cfg_path.read_text())
    dataset_root = Path(local_cfg.get('dataset_root', '/scratch/rameyjm7/datasets'))
else:
    local_cfg = {}
    dataset_root = Path('/scratch/rameyjm7/datasets')

# Honor pinned best checkpoint when available.
best_ckpt_file = repo / 'models' / 'rml2018' / 'checkpoints' / 'best_checkpoint.txt'
pinned_rml2018 = None
if best_ckpt_file.exists():
    candidate = Path(best_ckpt_file.read_text().strip())
    if candidate.exists():
        pinned_rml2018 = candidate
        print('Pinned RML2018 checkpoint from best_checkpoint.txt:', pinned_rml2018)

# Load fixed models first; RML2018 candidate will be selected later.
model_specs = {
    'rml2016': repo / 'models' / 'rml2016' / 'rml2016_lstm_rnn_2024.keras',
    'deepradar2022': repo / 'models' / 'deepradar2022' / 'deepradar2022_cnn_bilstm_final.keras',
}

models = {}
for model_id, model_path in model_specs.items():
    if not model_path.exists():
        raise FileNotFoundError(f'Missing model file: {model_path}')
    models[model_id] = load_model(model_path, compile=False)
    print(model_id, 'input_shape=', models[model_id].input_shape, 'output_shape=', models[model_id].output_shape)

rml2018_model_candidates = []
rml2018_primary = repo / 'models' / 'rml2018' / 'rml2018_lstm_rnn.keras'
rml2018_balanced = repo / 'models' / 'rml2018' / 'rml2018_lstm_balanced.keras'
rml2018_ckpts = sorted((repo / 'models' / 'rml2018' / 'checkpoints').glob('*.keras'))

if pinned_rml2018 is not None:
    seed_candidates = [pinned_rml2018]
else:
    seed_candidates = [rml2018_primary, rml2018_balanced, *rml2018_ckpts]

for candidate in seed_candidates:
    if candidate.exists() and candidate not in rml2018_model_candidates:
        rml2018_model_candidates.append(candidate)

if not rml2018_model_candidates:
    raise FileNotFoundError('No RML2018 model candidates found in models/rml2018.')

print('RML2018 candidate models:')
for c in rml2018_model_candidates:
    print(' -', c)

print('dataset_root =', dataset_root)
if not dataset_root.exists():
    raise FileNotFoundError(f'Dataset root does not exist: {dataset_root}')


outputs_dir = repo / 'outputs' / 'cross_dataset_ensemble'
outputs_dir.mkdir(parents=True, exist_ok=True)
print('outputs_dir =', outputs_dir)

# %% Cell 4
# Cell 4 : Load and sample RML2016, RML2018, and DeepRadar2022 datasets
from sklearn.preprocessing import LabelEncoder
def parse_classes_file(path: Path):
    text = path.read_text()
    match = re.search(r'classes\s*=\s*(\[[\s\S]*?\])', text)
    if not match:
        raise ValueError(f'Could not parse classes list from {path}')
    return ast.literal_eval(match.group(1))


def load_rml2016(n_per_class=200):
    data_path = dataset_root / 'RML2016' / 'RML2016.10a_dict.pkl'
    with data_path.open('rb') as f:
        data = pickle.load(f, encoding='latin1')

    classes = sorted({mod for (mod, _snr) in data.keys()})
    class_to_idx = {c: i for i, c in enumerate(classes)}

    rows = []
    y_idx = []
    max_snr = max(snr for (_mod, snr) in data.keys())

    for mod in classes:
        signals = data[(mod, max_snr)]
        take = min(n_per_class, len(signals))
        picks = np.random.choice(len(signals), size=take, replace=False)
        for i in picks:
            sig = signals[i]
            iq = np.vstack([sig[0], sig[1]]).T.astype(np.float32)
            snr_ch = np.full((iq.shape[0], 1), max_snr, dtype=np.float32)
            rows.append(np.hstack([iq, snr_ch]))
            y_idx.append(class_to_idx[mod])

    return np.asarray(rows, dtype=np.float32), np.asarray(y_idx, dtype=np.int64), classes


def load_rml2018(n_per_class=200):
    data_file = dataset_root / 'RML2018' / 'GOLD_XYZ_OSC.0001_1024.hdf5'
    classes_orig = parse_classes_file(dataset_root / 'RML2018' / 'classes.txt')
    classes_fixed = parse_classes_file(dataset_root / 'RML2018' / 'classes-fixed.txt')
    remap_orig_to_fixed = np.array([classes_fixed.index(c) for c in classes_orig], dtype=np.int64)

    with h5py.File(data_file, 'r') as h5:
        x_ds = h5['X']
        y_ds = h5['Y']
        z_ds = h5['Z']

        snr = z_ds[:, 0]
        max_snr = int(np.max(snr))
        max_idx = np.where(snr == max_snr)[0]
        y_max = np.argmax(y_ds[max_idx], axis=1)

        picked = []
        for cls in np.unique(y_max):
            cls_idx = max_idx[y_max == cls]
            take = min(n_per_class, len(cls_idx))
            picked.extend(np.random.choice(cls_idx, size=take, replace=False).tolist())

        picked = np.array(sorted(picked), dtype=np.int64)
        x_iq = np.asarray(x_ds[picked], dtype=np.float32)
        y_onehot = np.asarray(y_ds[picked], dtype=np.int64)
        snr_vals = np.asarray(z_ds[picked, 0], dtype=np.float32)

    snr_ch = np.repeat(snr_vals[:, None, None], x_iq.shape[1], axis=1)
    x = np.concatenate([x_iq, snr_ch], axis=2).astype(np.float32)

    y_true_orig = np.argmax(y_onehot, axis=1).astype(np.int64)
    y_true_fixed = remap_orig_to_fixed[y_true_orig]

    # Match training-time encoding used in notebook 31 (alphabetical LabelEncoder order)
    y_names = np.array([classes_orig[i] for i in y_true_orig])
    le = LabelEncoder()
    y_true_le = le.fit_transform(y_names)
    classes_le = list(le.classes_)

    return x, y_true_orig, y_true_fixed, y_true_le, classes_orig, classes_fixed, classes_le


def load_deepradar2022(n_per_class=200):
    x_file = dataset_root / 'DeepRadar2022' / 'X_test.mat'
    y_file = dataset_root / 'DeepRadar2022' / 'Y_test.mat'
    lbl_file = dataset_root / 'DeepRadar2022' / 'lbl_test.mat'

    y_mat = loadmat(y_file)
    lbl_mat = loadmat(lbl_file)
    y_key = next(k for k in y_mat.keys() if not k.startswith('__'))
    lbl_key = next(k for k in lbl_mat.keys() if not k.startswith('__'))

    y_onehot = y_mat[y_key]
    lbl = lbl_mat[lbl_key]

    snr = lbl[:, 1]
    max_snr = float(np.max(snr))
    max_idx = np.where(snr == max_snr)[0]
    y_max = np.argmax(y_onehot[max_idx], axis=1)

    picked = []
    for cls in np.unique(y_max):
        cls_idx = max_idx[y_max == cls]
        take = min(n_per_class, len(cls_idx))
        picked.extend(np.random.choice(cls_idx, size=take, replace=False).tolist())

    picked = np.array(sorted(picked), dtype=np.int64)

    with h5py.File(x_file, 'r') as h5:
        x_raw = np.asarray(h5['X_test'][:, :, picked], dtype=np.float32)

    x_iq = np.transpose(x_raw, (2, 1, 0))
    envelope = np.sqrt(np.sum(np.square(x_iq), axis=2, keepdims=True))
    x = np.concatenate([x_iq, envelope], axis=2).astype(np.float32)

    y_idx = np.argmax(y_onehot[picked], axis=1).astype(np.int64)
    n_classes = int(y_onehot.shape[1])
    classes = [f'class_{i}' for i in range(n_classes)]
    return x, y_idx, classes

rml2016 = load_rml2016()
rml2018_raw = load_rml2018()
deepradar = load_deepradar2022()

print('rml2016:', rml2016[0].shape, 'classes=', len(rml2016[2]))
print('rml2018:', rml2018_raw[0].shape, 'classes_orig=', len(rml2018_raw[4]), 'classes_fixed=', len(rml2018_raw[5]), 'classes_labelencoder=', len(rml2018_raw[6]))
print('deepradar2022:', deepradar[0].shape, 'classes=', len(deepradar[2]))

# %% Cell 5
# Cell 5 : Auto-select best RML2018 model and class ordering using calibration accuracy
x18, y18_orig, y18_fixed, y18_le, classes18_orig, classes18_fixed, classes18_le = rml2018_raw

best = None
for candidate_path in rml2018_model_candidates:
    try:
        candidate_model = load_model(candidate_path, compile=False)
    except Exception as exc:
        print(f'Skip {candidate_path.name}: failed to load -> {exc}')
        continue

    pred18 = np.argmax(candidate_model.predict(x18, verbose=0), axis=1)

    acc_vs_orig = accuracy_score(y18_orig, pred18)
    acc_vs_fixed = accuracy_score(y18_fixed, pred18)
    acc_vs_le = accuracy_score(y18_le, pred18)

    candidates = [
        ('classes', y18_orig, classes18_orig, acc_vs_orig),
        ('classes-fixed', y18_fixed, classes18_fixed, acc_vs_fixed),
        ('labelencoder', y18_le, classes18_le, acc_vs_le),
    ]
    chosen_order, chosen_y, chosen_classes, chosen_acc = max(candidates, key=lambda t: t[3])

    binc = np.bincount(pred18, minlength=max(len(classes18_orig), len(classes18_fixed), len(classes18_le)))
    dominant_ratio = float(binc.max()) / float(len(pred18))

    print(
        f"{candidate_path.name}: acc_orig={acc_vs_orig:.4f}, "
        f"acc_fixed={acc_vs_fixed:.4f}, acc_le={acc_vs_le:.4f}, "
        f"chosen={chosen_order}, dominant_pred_ratio={dominant_ratio:.4f}"
    )

    score = chosen_acc
    if best is None or score > best['score']:
        if best is not None:
            del best['model']
        best = {
            'path': candidate_path,
            'model': candidate_model,
            'score': score,
            'order': chosen_order,
            'y': chosen_y,
            'classes': chosen_classes,
            'dominant_ratio': dominant_ratio,
        }
    else:
        del candidate_model

if best is None:
    raise RuntimeError('No valid RML2018 model candidate could be loaded.')

models['rml2018'] = best['model']
rml2018 = (x18, best['y'], best['classes'])
rml2018_order_used = best['order']

print('\nSelected RML2018 model:', best['path'])
print('Selected order:', rml2018_order_used)
print('Selected calibration accuracy:', f"{best['score']:.4f}")
print('Selected dominant prediction ratio:', f"{best['dominant_ratio']:.4f}")

# %% Cell 6
# Cell 6 : Build unified dataset dictionary and choose evaluation mode
datasets = {
    'rml2016': rml2016,
    'rml2018': rml2018,
    'deepradar2022': deepradar,
}

# Default safe mode: each dataset evaluated by its own trained model.
EVAL_MODE = 'strict_dataset_model'  # options: strict_dataset_model, compatible_ensemble

dataset_model_map = {
    'rml2016': 'rml2016',
    'rml2018': 'rml2018',
    'deepradar2022': 'deepradar2022',
}

for ds_id, (x, y_idx, classes) in datasets.items():
    print(ds_id, 'samples=', x.shape[0], 'shape=', x.shape[1:], 'classes=', len(classes))
print('EVAL_MODE =', EVAL_MODE)

# %% Cell 7
# Cell 7 : Build global label space and helper mappings
global_labels = []
for ds_id, (_x, _y_idx, classes) in datasets.items():
    global_labels.extend([f'{ds_id}:{c}' for c in classes])

global_labels = sorted(global_labels)
global_index = {label: i for i, label in enumerate(global_labels)}

model_to_global_labels = {}
for ds_id, (_x, _y_idx, classes) in datasets.items():
    model_to_global_labels[ds_id] = [f'{ds_id}:{c}' for c in classes]

print('Global label count =', len(global_labels))

# %% Cell 8
# Cell 8 : Run per-dataset diagnostics (confusion matrix + accuracy)
for ds_id, (x, y_idx, classes) in datasets.items():
    model = models[dataset_model_map[ds_id]]
    y_pred = np.argmax(model.predict(x, verbose=0), axis=1)
    acc = accuracy_score(y_idx, y_pred)
    print(f'\n{ds_id} strict-model accuracy: {acc:.4f}')

    cm_local = confusion_matrix(y_idx, y_pred, labels=np.arange(len(classes)))

    diag_report = classification_report(y_idx, y_pred, target_names=classes, zero_division=0)
    report_file = outputs_dir / f'43_{ds_id}_diagnostic_classification_report.txt'
    report_file.write_text(diag_report)
    print('Saved diagnostic report:', report_file)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_local, cmap='Blues')
    plt.title(f'{ds_id} Diagnostic Confusion Matrix (strict model)')
    plt.xlabel('Predicted class index')
    plt.ylabel('True class index')
    plt.tight_layout()
    _save_current_figure("cell_08_figure_01.png")

# %% Cell 9
# Cell 9 : Generate combined global predictions in strict or compatible-ensemble mode
def compatible_models_for_input(x):
    matched = []
    for model_id, model in models.items():
        if tuple(model.input_shape[1:]) == tuple(x.shape[1:]):
            matched.append((model_id, model))
    return matched


y_true_global = []
y_pred_global = []

for ds_id, (x, y_idx, classes) in datasets.items():
    local_to_global = {i: global_index[f'{ds_id}:{classes[i]}'] for i in range(len(classes))}

    if EVAL_MODE == 'strict_dataset_model':
        model_id = dataset_model_map[ds_id]
        probs = models[model_id].predict(x, verbose=0)
        pred_local = np.argmax(probs, axis=1)
    else:
        matched = compatible_models_for_input(x)
        if not matched:
            raise RuntimeError(f'No compatible models for {ds_id} input shape {x.shape[1:]}')

        stitched_scores = []
        for model_id, model in matched:
            probs = model.predict(x, verbose=0)
            global_scores = np.zeros((x.shape[0], len(global_labels)), dtype=np.float32)
            model_global_labels = model_to_global_labels[model_id]
            for i_local, g_label in enumerate(model_global_labels):
                if i_local < probs.shape[1]:
                    global_scores[:, global_index[g_label]] = probs[:, i_local]
            stitched_scores.append(global_scores)

        ensemble_scores = np.mean(stitched_scores, axis=0)
        pred_global = np.argmax(ensemble_scores, axis=1)

        true_global = [local_to_global[int(i)] for i in y_idx]
        y_true_global.extend(true_global)
        y_pred_global.extend(pred_global.tolist())
        continue

    true_global = [local_to_global[int(i)] for i in y_idx]
    pred_global = [local_to_global[int(i)] for i in pred_local]
    y_true_global.extend(true_global)
    y_pred_global.extend(pred_global)

y_true_global = np.asarray(y_true_global, dtype=np.int64)
y_pred_global = np.asarray(y_pred_global, dtype=np.int64)
print('Combined sample count =', len(y_true_global))

# %% Cell 10
# Cell 10 : Plot combined confusion matrix and print global classification report
cm = confusion_matrix(y_true_global, y_pred_global, labels=np.arange(len(global_labels)))

plt.figure(figsize=(24, 20))
sns.heatmap(cm, cmap='Blues', xticklabels=global_labels, yticklabels=global_labels)
plt.title(f'Combined Confusion Matrix Across RML2016 + RML2018 + DeepRadar2022 ({EVAL_MODE})')
plt.xlabel('Predicted Global Label')
plt.ylabel('True Global Label')
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
_save_current_figure("cell_10_figure_02.png")

print(classification_report(y_true_global, y_pred_global, target_names=global_labels, zero_division=0))

# %% Cell 11
# Cell 11 : Plot and save per-dataset SNR line charts (RML datasets)
import csv

for ds_id, (x, y_idx, classes) in datasets.items():
    # SNR is available if third channel is constant per sample (RML2016/RML2018)
    if x.shape[2] < 3:
        print(f'{ds_id}: no third channel, skipping SNR charts.')
        continue

    snr_vals = x[:, 0, 2]
    if not np.allclose(x[:, :, 2], snr_vals[:, None]):
        print(f'{ds_id}: third channel is not constant-per-sample, skipping SNR charts.')
        continue

    snr_vals = snr_vals.astype(int)
    model = models[dataset_model_map[ds_id]]
    y_pred_local = np.argmax(model.predict(x, verbose=0), axis=1)

    snr_unique = np.array(sorted(np.unique(snr_vals)), dtype=int)
    overall_acc = []
    per_class = np.full((len(classes), len(snr_unique)), np.nan, dtype=np.float32)

    for j, snr in enumerate(snr_unique):
        idx = np.where(snr_vals == snr)[0]
        overall_acc.append(float(np.mean(y_pred_local[idx] == y_idx[idx])) * 100.0)

        for c in range(len(classes)):
            m = idx[y_idx[idx] == c]
            if len(m) > 0:
                per_class[c, j] = float(np.mean(y_pred_local[m] == y_idx[m])) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    axes[0].plot(snr_unique, overall_acc, marker='o', color='blue')
    axes[0].set_title(f'Recognition Accuracy vs. SNR ({ds_id})')
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].grid(True, alpha=0.4)

    for c, cls_name in enumerate(classes):
        axes[1].plot(snr_unique, per_class[c], marker='o', linewidth=1.0, label=cls_name)
    axes[1].set_title(f'Accuracy vs. SNR per Class ({ds_id})')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, alpha=0.4)
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)

    plt.tight_layout()
    png = outputs_dir / f'43_{ds_id}_accuracy_vs_snr_line_plots.png'
    plt.savefig(png, dpi=180)
    print('Saved line charts:', png)
    _save_current_figure("cell_11_figure_03.png")

    csv_path = outputs_dir / f'43_{ds_id}_accuracy_vs_snr_line.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['snr_db', 'accuracy_percent'])
        for s, a in zip(snr_unique, overall_acc):
            writer.writerow([int(s), f"{a:.6f}"])
    print('Saved overall line data:', csv_path)
