#!/usr/bin/env python3
"""Pipeline converted from the legacy 63b_gan_drone_iq_generator workflow."""

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
# Cell 1 : 63b: Conditional GAN for Raw Drone I/Q Generation
# 63b: Conditional GAN for Raw Drone I/Q Generation

# This notebook trains a conditional GAN to generate raw I/Q windows for the
# Noisy Drone RF v2 classes.
#
# Unlike notebook `63`, this generator emits time-domain I/Q arrays shaped
# `(IQ_LEN, 2)`. The frozen notebook `33` VGG spectrogram classifier can still
# be used as a teacher by converting generated I/Q into spectrograms inside the
# generator loss.
#
# Default `IQ_LEN=4096` is intentionally short for stable experimentation.
# Increase it only after the generated I/Q looks sane and the teacher/class
# metrics improve.

# %% Cell 2
# Cell 2 : Configure raw-IQ GAN paths and training knobs
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import sys
if 'ipykernel' not in sys.modules and not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv1D,
    Conv1DTranspose,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import load_model

try:
    project_root = Path(__file__).resolve().parents[1]
except NameError:
    project_root = Path.cwd().resolve()
    if project_root.name == 'pipelines':
        project_root = project_root.parent

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
tf.get_logger().setLevel('ERROR')

CLASS_NAMES = ['DJI', 'FutabaT14', 'FutabaT7', 'Graupner', 'Noise', 'Taranis', 'Turnigy']
NUM_CLASSES = len(CLASS_NAMES)

model_dir = project_root / 'models' / 'noisy_drone_rf_v2'
outputs_dir = project_root / 'outputs' / 'noisy_drone_rf_v2_iq_gan'
outputs_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

manifest_path = project_root / 'outputs' / 'noisy_drone_rf_v2_eval' / '33_noisy_drone_rf_v2_vgg_full_complex_replay_manifest.csv'
if not manifest_path.exists():
    manifest_path = project_root / 'outputs' / 'noisy_drone_rf_v2_eval' / '33_noisy_drone_rf_v2_replay_manifest.csv'
teacher_model_candidates = [
    model_dir / 'noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras',
    model_dir / 'noisy_drone_rf_v2_vgg_complex_spectrogram_best.keras',
    model_dir / 'noisy_drone_rf_v2_vgg_spectrogram_best.keras',
]
teacher_model_path = next((path for path in teacher_model_candidates if path.exists()), teacher_model_candidates[0])

generator_path = model_dir / 'noisy_drone_rf_v2_conditional_iq_generator.keras'
teacher_aligned_generator_path = model_dir / 'noisy_drone_rf_v2_conditional_iq_generator_teacher_aligned.keras'
teacher_aligned_cell10_generator_path = model_dir / 'noisy_drone_rf_v2_conditional_iq_generator_teacher_aligned_cell10.keras'
discriminator_path = model_dir / 'noisy_drone_rf_v2_conditional_iq_discriminator.keras'
history_csv_path = outputs_dir / '63b_noisy_drone_rf_v2_iq_gan_training_history.csv'


def resolve_best_iq_generator_path() -> Path:
    """Prefer the repaired/teacher-aligned generator for eval/export."""
    repaired_path = model_dir / 'noisy_drone_rf_v2_conditional_iq_generator_teacher_aligned_repaired.keras'
    for candidate in (
        repaired_path,
        teacher_aligned_cell10_generator_path,
        teacher_aligned_generator_path,
        generator_path,
    ):
        if candidate.exists():
            return candidate
    return repaired_path

RANDOM_STATE = int(os.getenv('NOISY_DRONE_IQ_GAN_RANDOM_STATE', '42'))
DATA_FRACTION = float(os.getenv('NOISY_DRONE_IQ_GAN_DATA_FRACTION', '0.35'))
MIN_SNR_DB = float(os.getenv('NOISY_DRONE_IQ_GAN_MIN_SNR_DB', '-6'))
BATCH_SIZE = int(os.getenv('NOISY_DRONE_IQ_GAN_BATCH_SIZE', '32'))
EPOCHS = int(os.getenv('NOISY_DRONE_IQ_GAN_EPOCHS', '40'))
STEPS_PER_EPOCH = int(os.getenv('NOISY_DRONE_IQ_GAN_STEPS_PER_EPOCH', '300'))
LATENT_DIM = int(os.getenv('NOISY_DRONE_IQ_GAN_LATENT_DIM', '128'))
IQ_LEN = int(os.getenv('NOISY_DRONE_IQ_GAN_IQ_LEN', '4096'))
GEN_LR = float(os.getenv('NOISY_DRONE_IQ_GAN_GEN_LR', '1e-4'))
DISC_LR = float(os.getenv('NOISY_DRONE_IQ_GAN_DISC_LR', '8e-5'))
CLASS_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_CLASS_WEIGHT', '1.50'))
TEACHER_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_WEIGHT', '0.01'))
TEACHER_EVERY = int(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_EVERY', '5'))
DISC_CLASS_PRETRAIN_EPOCHS = int(os.getenv('NOISY_DRONE_IQ_GAN_DISC_CLASS_PRETRAIN_EPOCHS', '3'))
DISC_CLASS_PRETRAIN_STEPS = int(os.getenv('NOISY_DRONE_IQ_GAN_DISC_CLASS_PRETRAIN_STEPS', '200'))
IQ_MOMENT_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_IQ_MOMENT_WEIGHT', '0.25'))
FREEZE_DISCRIMINATOR = os.getenv('NOISY_DRONE_IQ_GAN_FREEZE_DISCRIMINATOR', '0') == '1'
SAVE_EVERY = int(os.getenv('NOISY_DRONE_IQ_GAN_SAVE_EVERY', '5'))

# Teacher spectrogram parameters mirror notebook 33 enough for class-consistency pressure.
TEACHER_SPEC_NFFT = int(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_SPEC_NFFT', '1024'))
TEACHER_SPEC_TIME_BINS = int(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_SPEC_TIME_BINS', '1024'))
TEACHER_FRAME_LEN = int(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_FRAME_LEN', '1024'))

print('Project root:', project_root)
print('Manifest:', manifest_path)
print('Teacher:', teacher_model_path)
print('I/Q output shape:', (IQ_LEN, 2))
print('Batch:', BATCH_SIZE, 'epochs:', EPOCHS, 'steps/epoch:', STEPS_PER_EPOCH)
print('Teacher weight:', TEACHER_WEIGHT, 'teacher every:', TEACHER_EVERY)
print('Class/moment/pretrain:', CLASS_WEIGHT, IQ_MOMENT_WEIGHT, DISC_CLASS_PRETRAIN_EPOCHS, DISC_CLASS_PRETRAIN_STEPS)
print('Freeze discriminator:', FREEZE_DISCRIMINATOR)

# %% Cell 3
# Cell 3 : Load the Noisy Drone RF v2 replay manifest and balance classes
if not manifest_path.exists():
    raise FileNotFoundError(f'Missing replay manifest: {manifest_path}. Run notebook 33 first.')

manifest_df = pd.read_csv(manifest_path)
for required in ['filepath', 'label_idx', 'snr']:
    if required not in manifest_df.columns:
        raise ValueError(f'Manifest missing required column: {required}')

manifest_df = manifest_df.copy()
manifest_df['filepath'] = manifest_df['filepath'].astype(str)
manifest_df['label_idx'] = manifest_df['label_idx'].astype(int)
manifest_df['snr'] = manifest_df['snr'].astype(float)
manifest_df = manifest_df[manifest_df['snr'] >= MIN_SNR_DB]
manifest_df = manifest_df[manifest_df['label_idx'].between(0, NUM_CLASSES - 1)]
manifest_df = manifest_df[manifest_df['filepath'].map(lambda p: Path(p).exists())]
manifest_df['label'] = manifest_df['label_idx'].map(lambda i: CLASS_NAMES[int(i)])

if manifest_df.empty:
    raise FileNotFoundError('No usable rows after filtering. Check /scratch data and manifest paths.')

if DATA_FRACTION < 1.0:
    manifest_df, _ = train_test_split(
        manifest_df,
        train_size=DATA_FRACTION,
        stratify=manifest_df['label_idx'],
        random_state=RANDOM_STATE,
    )

min_count = int(manifest_df['label_idx'].value_counts().min())
balanced_df = (
    manifest_df.groupby('label_idx', group_keys=False)
    .sample(n=min_count, random_state=RANDOM_STATE)
    .sample(frac=1.0, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)
train_df, val_df = train_test_split(
    balanced_df,
    test_size=0.15,
    stratify=balanced_df['label_idx'],
    random_state=RANDOM_STATE,
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

manifest_out = outputs_dir / '63b_noisy_drone_rf_v2_iq_gan_balanced_manifest.csv'
balanced_df.to_csv(manifest_out, index=False)
print('Filtered rows:', len(manifest_df))
print('Balanced rows:', len(balanced_df), 'train:', len(train_df), 'val:', len(val_df))
print(balanced_df['label'].value_counts().sort_index())
print('Saved:', manifest_out)

# %% Cell 4
# Cell 4 : Define robust .pt I/Q loading and normalized I/Q window extraction
def _unwrap_tensor_container(obj):
    seen = set()
    while True:
        if id(obj) in seen:
            break
        seen.add(id(obj))
        if isinstance(obj, dict):
            preferred = ['iq', 'x', 'data', 'samples', 'signal', 'arr', 'tensor']
            picked = next((key for key in preferred if key in obj), None)
            if picked is None:
                tensorish = [k for k, v in obj.items() if hasattr(v, 'detach') or isinstance(v, (np.ndarray, list, tuple))]
                picked = tensorish[0] if tensorish else list(obj.keys())[0]
            obj = obj[picked]
            continue
        if isinstance(obj, (list, tuple)) and len(obj) == 1:
            obj = obj[0]
            continue
        break
    return obj

def load_pt_iq(filepath):
    obj = torch.load(filepath, map_location='cpu', weights_only=False)
    obj = _unwrap_tensor_container(obj)
    arr = obj.detach().cpu().numpy() if hasattr(obj, 'detach') else np.asarray(obj)
    if np.iscomplexobj(arr):
        arr = np.stack([arr.real, arr.imag], axis=-1)
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            arr = arr[:-1]
        arr = arr.reshape(-1, 2)
    elif arr.ndim == 2:
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T
        elif arr.shape[1] > 2:
            arr = arr[:, :2]
    elif arr.ndim >= 3:
        arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[1] > 2:
            arr = arr[:, :2]
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f'Could not coerce {filepath} into (N, 2); got {arr.shape}')
    return arr.astype(np.float32)

def normalize_iq(iq):
    iq = np.asarray(iq, dtype=np.float32)
    iq = iq - np.mean(iq, axis=0, keepdims=True)
    power = np.sqrt(np.mean(np.sum(iq ** 2, axis=1))) + 1e-8
    iq = iq / power
    return np.clip(iq, -5.0, 5.0).astype(np.float32)

def extract_iq_window(filepath, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    iq = load_pt_iq(filepath)
    if iq.shape[0] < IQ_LEN:
        reps = int(np.ceil(IQ_LEN / max(1, iq.shape[0])))
        iq = np.tile(iq, (reps, 1))[:IQ_LEN]
    elif iq.shape[0] > IQ_LEN:
        start = int(rng.integers(0, iq.shape[0] - IQ_LEN + 1))
        iq = iq[start:start + IQ_LEN]
    return normalize_iq(iq)

preview = extract_iq_window(train_df.iloc[0]['filepath'], np.random.default_rng(RANDOM_STATE))
print('Preview I/Q:', preview.shape, preview.dtype, float(preview.min()), float(preview.max()))
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(preview[:1000, 0], label='I', linewidth=0.8)
ax.plot(preview[:1000, 1], label='Q', linewidth=0.8)
ax.set_title(f"Raw I/Q preview: {train_df.iloc[0]['label']}")
ax.legend()
ax.grid(True, alpha=0.25)
_save_current_figure("cell_04_figure_01.png")

# %% Cell 5
# Cell 5 : Build balanced I/Q tf.data streams
def make_balanced_generator(frame):
    frame = frame.copy().reset_index(drop=True)
    groups = {int(k): g.reset_index(drop=True) for k, g in frame.groupby('label_idx')}
    labels = sorted(groups)
    def gen():
        rng = np.random.default_rng()
        while True:
            label = int(rng.choice(labels))
            group = groups[label]
            row = group.iloc[int(rng.integers(0, len(group)))].to_dict()
            try:
                yield extract_iq_window(row['filepath'], rng), np.int64(label)
            except Exception as exc:
                print(f'Skipping corrupt row: {row.get("filepath")} ({type(exc).__name__}: {exc})')
                continue
    return gen

def make_iq_dataset(frame, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_generator(
        make_balanced_generator(frame),
        output_signature=(
            tf.TensorSpec(shape=(IQ_LEN, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = make_iq_dataset(train_df)
val_ds = make_iq_dataset(val_df)

x_batch, y_batch = next(iter(train_ds))
print('Batch:', x_batch.shape, y_batch.shape, 'labels:', y_batch.numpy()[:10].tolist())

# %% Cell 6
# Cell 6 : Build the raw-I/Q generator, discriminator, and optional frozen 33 teacher
def build_iq_generator():
    z_in = tf.keras.Input(shape=(LATENT_DIM,), name='noise')
    y_in = tf.keras.Input(shape=(), dtype='int32', name='label')
    y = Embedding(NUM_CLASSES, LATENT_DIM, name='class_embedding')(y_in)
    y = Flatten(name='class_embedding_flat')(y)
    x = Concatenate(name='noise_plus_class')([z_in, y])
    seed_len = IQ_LEN // 64
    if seed_len < 4:
        raise ValueError('IQ_LEN must be at least 256 for this generator')
    x = Dense(seed_len * 256, use_bias=False, name='dense_seed')(x)
    x = BatchNormalization(name='seed_bn')(x)
    x = LeakyReLU(0.2, name='seed_lrelu')(x)
    x = Reshape((seed_len, 256), name='seed_reshape')(x)
    for filters, name in [(192, 'up1'), (128, 'up2'), (96, 'up3'), (64, 'up4'), (48, 'up5'), (32, 'up6')]:
        x = Conv1DTranspose(filters, 7, strides=2, padding='same', use_bias=False, name=f'{name}_deconv')(x)
        x = BatchNormalization(name=f'{name}_bn')(x)
        x = LeakyReLU(0.2, name=f'{name}_lrelu')(x)
    out = Conv1D(2, 7, padding='same', activation='tanh', name='iq')(x)
    return Model([z_in, y_in], out, name='noisy_drone_iq_generator')

def build_iq_discriminator():
    x_in = Input(shape=(IQ_LEN, 2), name='iq_input')
    x = x_in
    for filters, name in [(48, 'down1'), (64, 'down2'), (96, 'down3'), (128, 'down4'), (192, 'down5')]:
        x = Conv1D(filters, 7, strides=2, padding='same', name=f'{name}_conv')(x)
        x = LeakyReLU(0.2, name=f'{name}_lrelu')(x)
        x = Dropout(0.15, name=f'{name}_dropout')(x)
    x = GlobalAveragePooling1D(name='gap')(x)
    x = Dense(192, name='shared_dense')(x)
    x = LeakyReLU(0.2, name='shared_lrelu')(x)
    real_fake = Dense(1, activation='sigmoid', name='real_fake')(x)
    class_logits = Dense(NUM_CLASSES, activation='softmax', name='class')(x)
    return Model(x_in, [real_fake, class_logits], name='noisy_drone_iq_discriminator')

generator = load_model(generator_path, compile=False) if generator_path.exists() else build_iq_generator()
discriminator = load_model(discriminator_path, compile=False) if discriminator_path.exists() else build_iq_discriminator()
discriminator.trainable = not FREEZE_DISCRIMINATOR

teacher_model = None
if TEACHER_WEIGHT > 0:
    if not teacher_model_path.exists():
        raise FileNotFoundError(f'Teacher requested but missing: {teacher_model_path}')
    teacher_model = load_model(teacher_model_path, compile=False)
    teacher_model.trainable = False
    print('Loaded frozen teacher:', teacher_model_path)
else:
    print('Teacher disabled')

generator.summary()
discriminator.summary()

# %% Cell 7
# Cell 7 : Train the conditional raw-I/Q GAN
def safe_binary_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-5, 1.0 - 1e-5)
    return tf.reduce_mean(tf.keras.backend.binary_crossentropy(y_true, y_pred))

def safe_sparse_ce(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    row_sum = tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    looks_like_probs = tf.reduce_all(y_pred >= 0.0) & tf.reduce_all(tf.abs(row_sum - 1.0) < 1e-2)
    probs = tf.cond(looks_like_probs, lambda: y_pred, lambda: tf.nn.softmax(y_pred, axis=-1))
    probs = tf.clip_by_value(probs, 1e-5, 1.0)
    probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
    y_onehot = tf.one_hot(tf.cast(y_true, tf.int32), NUM_CLASSES)
    ce = -tf.reduce_sum(y_onehot * tf.math.log(probs), axis=-1)
    return tf.reduce_mean(tf.clip_by_value(ce, 0.0, 8.0))

def iq_to_teacher_spectrogram_tf(iq):
    # iq: (B, IQ_LEN, 2). tf.signal.stft requires real-valued inputs, so compute
    # separate I/Q STFTs and use their log magnitudes as teacher channels.
    iq = tf.cast(iq, tf.float32)
    iq = iq - tf.reduce_mean(iq, axis=1, keepdims=True)
    power = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(iq), axis=-1), axis=1, keepdims=True)) + 1e-6
    iq = iq / power[:, :, None]

    frame_step = max(1, (IQ_LEN - TEACHER_FRAME_LEN) // max(1, TEACHER_SPEC_TIME_BINS - 1))
    specs = []
    for channel in [0, 1]:
        stft = tf.signal.stft(
            iq[:, :, channel],
            frame_length=TEACHER_FRAME_LEN,
            frame_step=frame_step,
            fft_length=TEACHER_SPEC_NFFT,
            window_fn=tf.signal.hann_window,
            pad_end=True,
        )
        stft = tf.signal.fftshift(stft, axes=-1)
        stft = tf.transpose(stft, [0, 2, 1])
        stft = stft[:, :, :TEACHER_SPEC_TIME_BINS]
        time_pad = TEACHER_SPEC_TIME_BINS - tf.shape(stft)[2]
        stft = tf.pad(stft, [[0, 0], [0, 0], [0, time_pad]])
        mag = tf.math.log1p(tf.abs(stft))
        scale = tf.reduce_mean(mag, axis=[1, 2], keepdims=True) + 1e-6
        mag = tf.clip_by_value(mag / scale, 0.0, 5.0) / 5.0
        specs.append(mag)
    teacher_x = tf.cast(tf.stack(specs, axis=-1), tf.float32)
    # Real-valued STFT returns one-sided bins (e.g. 513 for fft_length=1024).
    # The frozen notebook 33 VGG expects exactly 1024 x 1024 x 2, so resize here.
    teacher_x = tf.image.resize(teacher_x, [TEACHER_SPEC_NFFT, TEACHER_SPEC_TIME_BINS], method='bilinear')
    return tf.cast(teacher_x, tf.float32)

g_optimizer = tf.keras.optimizers.Adam(learning_rate=GEN_LR, beta_1=0.5, beta_2=0.999)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=DISC_LR, beta_1=0.5, beta_2=0.999)
pretrain_optimizer = tf.keras.optimizers.Adam(learning_rate=DISC_LR, beta_1=0.5, beta_2=0.999)
global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

@tf.function
def pretrain_discriminator_class_step(real_x, labels):
    real_targets = tf.ones((tf.shape(real_x)[0], 1)) * 0.9
    with tf.GradientTape() as tape:
        real_pred, real_cls = discriminator(real_x, training=True)
        real_adv = safe_binary_crossentropy(real_targets, real_pred)
        real_cls_loss = safe_sparse_ce(labels, real_cls)
        loss = 0.25 * real_adv + real_cls_loss
    grads = tape.gradient(loss, discriminator.trainable_variables)
    pretrain_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
    return {'pretrain_loss': loss, 'pretrain_cls': real_cls_loss, 'pretrain_adv': real_adv}

if DISC_CLASS_PRETRAIN_EPOCHS > 0 and not FREEZE_DISCRIMINATOR:
    print('Pretraining discriminator class head on real I/Q before GAN training...')
    pretrain_iterator = iter(train_ds)
    for pre_epoch in range(1, DISC_CLASS_PRETRAIN_EPOCHS + 1):
        pre_rows = []
        for pre_step in range(1, DISC_CLASS_PRETRAIN_STEPS + 1):
            real_x_pre, labels_pre = next(pretrain_iterator)
            metrics = pretrain_discriminator_class_step(real_x_pre, labels_pre)
            pre_rows.append({k: float(v.numpy()) for k, v in metrics.items()})
            if pre_step == 1 or pre_step % 50 == 0:
                row = pre_rows[-1]
                print(
                    f"pretrain epoch={pre_epoch:03d} step={pre_step:04d}/{DISC_CLASS_PRETRAIN_STEPS} "
                    f"loss={row['pretrain_loss']:.4f} cls={row['pretrain_cls']:.4f}"
                )
        summary = {key: float(np.mean([row[key] for row in pre_rows])) for key in pre_rows[0]}
        print('pretrain summary:', {'epoch': pre_epoch, **summary})
    discriminator.save(discriminator_path)
    print('Saved class-pretrained discriminator:', discriminator_path)

@tf.function
def train_step(real_x, labels):
    batch_size = tf.shape(real_x)[0]
    noise = tf.random.normal([batch_size, LATENT_DIM])
    real_targets = tf.ones((batch_size, 1)) * 0.9
    fake_targets = tf.zeros((batch_size, 1))

    if FREEZE_DISCRIMINATOR:
        fake_probe = generator([noise, labels], training=False)
        real_pred, real_cls = discriminator(real_x, training=False)
        fake_pred, fake_cls = discriminator(fake_probe, training=False)
        d_adv = safe_binary_crossentropy(real_targets, real_pred) + safe_binary_crossentropy(fake_targets, fake_pred)
        d_cls = safe_sparse_ce(labels, real_cls)
        d_loss = d_adv + CLASS_WEIGHT * d_cls
    else:
        with tf.GradientTape() as disc_tape:
            fake_probe = generator([noise, labels], training=True)
            real_pred, real_cls = discriminator(real_x, training=True)
            fake_pred, fake_cls = discriminator(fake_probe, training=True)
            d_adv = safe_binary_crossentropy(real_targets, real_pred) + safe_binary_crossentropy(fake_targets, fake_pred)
            d_cls = safe_sparse_ce(labels, real_cls)
            d_loss = d_adv + CLASS_WEIGHT * d_cls
        d_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    noise = tf.random.normal([batch_size, LATENT_DIM])
    with tf.GradientTape() as gen_tape:
        fake_x = generator([noise, labels], training=True)
        fake_pred, fake_cls = discriminator(fake_x, training=False)
        g_adv = safe_binary_crossentropy(tf.ones((batch_size, 1)) * 0.9, fake_pred)
        g_cls = safe_sparse_ce(labels, fake_cls)
        real_mean = tf.reduce_mean(real_x, axis=[1, 2])
        fake_mean = tf.reduce_mean(fake_x, axis=[1, 2])
        real_std = tf.math.reduce_std(real_x, axis=[1, 2])
        fake_std = tf.math.reduce_std(fake_x, axis=[1, 2])
        iq_moment_loss = tf.reduce_mean(tf.abs(real_mean - fake_mean) + tf.abs(real_std - fake_std))

        teacher_loss = tf.constant(0.0, dtype=tf.float32)
        use_teacher = teacher_model is not None and TEACHER_WEIGHT > 0 and tf.equal(global_step % TEACHER_EVERY, 0)
        if use_teacher:
            teacher_x = iq_to_teacher_spectrogram_tf(fake_x)
            teacher_raw = teacher_model(teacher_x, training=False)
            teacher_loss = safe_sparse_ce(labels, teacher_raw)
        g_loss = g_adv + CLASS_WEIGHT * g_cls + TEACHER_WEIGHT * teacher_loss + IQ_MOMENT_WEIGHT * iq_moment_loss
    g_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    global_step.assign_add(1)

    return {
        'd_loss': d_loss,
        'd_adv': d_adv,
        'd_cls': d_cls,
        'g_loss': g_loss,
        'g_adv': g_adv,
        'g_cls': g_cls,
        'teacher_loss': teacher_loss,
        'iq_moment_loss': iq_moment_loss,
    }

def save_iq_grid(epoch, n_per_class=3):
    labels = np.repeat(np.arange(NUM_CLASSES), n_per_class).astype(np.int32)
    noise = np.random.default_rng(12345).standard_normal((len(labels), LATENT_DIM)).astype(np.float32)
    fake = generator.predict([noise, labels], batch_size=BATCH_SIZE, verbose=0)
    fig, axes = plt.subplots(NUM_CLASSES, n_per_class, figsize=(n_per_class * 4, NUM_CLASSES * 2.1), sharex=True)
    for row in range(NUM_CLASSES):
        for col in range(n_per_class):
            idx = row * n_per_class + col
            ax = axes[row, col]
            ax.plot(fake[idx, : min(1024, IQ_LEN), 0], linewidth=0.6, label='I')
            ax.plot(fake[idx, : min(1024, IQ_LEN), 1], linewidth=0.6, label='Q')
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[row])
    axes[0, 0].legend(loc='upper right')
    fig.suptitle(f'Generated raw I/Q windows - epoch {epoch}')
    fig.tight_layout()
    path = outputs_dir / f'63b_generated_iq_grid_epoch_{epoch:03d}.png'
    fig.savefig(path, dpi=150)
    _save_current_figure("cell_07_figure_02.png")
    plt.close(fig)
    return path

run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
history_rows = []
iterator = iter(train_ds)
for epoch in range(1, EPOCHS + 1):
    epoch_metrics = []
    for step in range(1, STEPS_PER_EPOCH + 1):
        real_x, labels = next(iterator)
        metrics = train_step(real_x, labels)
        row = {k: float(v.numpy()) for k, v in metrics.items()}
        epoch_metrics.append(row)
        if step == 1 or step % 50 == 0:
            print(f"epoch={epoch:03d} step={step:04d}/{STEPS_PER_EPOCH} d_loss={row['d_loss']:.4f} g_loss={row['g_loss']:.4f} teacher={row['teacher_loss']:.4f}")
    summary = {'run_id': run_id, 'epoch': epoch}
    for key in epoch_metrics[0]:
        summary[key] = float(np.mean([m[key] for m in epoch_metrics]))
    history_rows.append(summary)
    print('epoch summary:', summary)
    if epoch == 1 or epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
        generator.save(generator_path)
        discriminator.save(discriminator_path)
        grid_path = save_iq_grid(epoch)
        print('Saved generator:', generator_path)
        print('Saved discriminator:', discriminator_path)
        print('Saved grid:', grid_path)

history_df = pd.DataFrame(history_rows)
if history_csv_path.exists():
    history_df = pd.concat([pd.read_csv(history_csv_path), history_df], ignore_index=True)
history_df.to_csv(history_csv_path, index=False)
print('Saved history:', history_csv_path)

# %% Cell 8
# Cell 8 : Evaluate generated I/Q with discriminator and optional frozen 33 teacher
eval_generator_path = resolve_best_iq_generator_path() if 'resolve_best_iq_generator_path' in globals() else generator_path
print('Loading eval generator:', eval_generator_path)
generator = load_model(eval_generator_path, compile=False)
discriminator = load_model(discriminator_path, compile=False)
teacher_model = load_model(teacher_model_path, compile=False) if teacher_model_path.exists() else None
if teacher_model is not None:
    teacher_model.trainable = False

samples_per_class = int(os.getenv('NOISY_DRONE_IQ_GAN_EVAL_SAMPLES_PER_CLASS', '64'))
labels = np.repeat(np.arange(NUM_CLASSES), samples_per_class).astype(np.int32)
noise = np.random.default_rng(RANDOM_STATE).standard_normal((len(labels), LATENT_DIM)).astype(np.float32)
fake = generator.predict([noise, labels], batch_size=BATCH_SIZE, verbose=1)
real_fake, disc_probs = discriminator.predict(fake, batch_size=BATCH_SIZE, verbose=0)
disc_pred = disc_probs.argmax(axis=1)

rows = []
for idx, name in enumerate(CLASS_NAMES):
    mask = labels == idx
    rows.append({
        'label_idx': idx,
        'label': name,
        'disc_class_accuracy': float(np.mean(disc_pred[mask] == labels[mask])),
        'disc_realness_mean': float(np.mean(real_fake[mask])),
        'iq_rms_mean': float(np.mean(np.sqrt(np.mean(np.sum(fake[mask] ** 2, axis=-1), axis=-1)))),
    })

if teacher_model is not None:
    teacher_preds = []
    teacher_conf = []
    for start in range(0, len(fake), max(1, min(BATCH_SIZE, 4))):
        batch_iq = tf.convert_to_tensor(fake[start:start + max(1, min(BATCH_SIZE, 4))], dtype=tf.float32)
        teacher_x = iq_to_teacher_spectrogram_tf(batch_iq)
        probs = teacher_model.predict(teacher_x.numpy(), batch_size=1, verbose=0)
        teacher_preds.extend(probs.argmax(axis=1).tolist())
        teacher_conf.extend(probs.max(axis=1).tolist())
    teacher_pred = np.asarray(teacher_preds, dtype=np.int64)
    teacher_conf = np.asarray(teacher_conf, dtype=np.float32)
    for row in rows:
        mask = labels == row['label_idx']
        row['teacher_class_accuracy'] = float(np.mean(teacher_pred[mask] == labels[mask]))
        row['teacher_confidence_mean'] = float(np.mean(teacher_conf[mask]))
else:
    teacher_pred = None

eval_df = pd.DataFrame(rows)
eval_path = outputs_dir / '63b_noisy_drone_rf_v2_iq_gan_generated_eval.csv'
eval_df.to_csv(eval_path, index=False)
print(eval_df)
print('Saved:', eval_path)

cm = pd.crosstab(pd.Series(labels, name='target'), pd.Series(disc_pred, name='disc_pred')).reindex(index=range(NUM_CLASSES), columns=range(NUM_CLASSES), fill_value=0)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title('Generated I/Q - Discriminator Class Head')
ax.set_xlabel('Predicted generated class')
ax.set_ylabel('Requested generated class')
fig.tight_layout()
cm_path = outputs_dir / '63b_generated_iq_discriminator_confusion_matrix.png'
fig.savefig(cm_path, dpi=150)
_save_current_figure("cell_08_figure_03.png")
plt.close(fig)
print('Saved:', cm_path)

if teacher_pred is not None:
    cm = pd.crosstab(pd.Series(labels, name='target'), pd.Series(teacher_pred, name='teacher_pred')).reindex(index=range(NUM_CLASSES), columns=range(NUM_CLASSES), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title('Generated I/Q - Frozen 33 VGG Teacher')
    ax.set_xlabel('Teacher predicted class')
    ax.set_ylabel('Requested generated class')
    fig.tight_layout()
    teacher_cm_path = outputs_dir / '63b_generated_iq_teacher_confusion_matrix.png'
    fig.savefig(teacher_cm_path, dpi=150)
    _save_current_figure("cell_08_figure_04.png")
    plt.close(fig)
    print('Saved:', teacher_cm_path)

# %% Cell 9
# Cell 9 : Export generated raw I/Q bank for augmentation experiments
if 'fake' not in globals() or 'labels' not in globals():
    export_generator_path = resolve_best_iq_generator_path() if 'resolve_best_iq_generator_path' in globals() else generator_path
    print('Loading export generator:', export_generator_path)
    generator = load_model(export_generator_path, compile=False)
    samples_per_class = int(os.getenv('NOISY_DRONE_IQ_GAN_EXPORT_SAMPLES_PER_CLASS', '256'))
    labels = np.repeat(np.arange(NUM_CLASSES), samples_per_class).astype(np.int32)
    noise = np.random.default_rng(RANDOM_STATE + 1).standard_normal((len(labels), LATENT_DIM)).astype(np.float32)
    fake = generator.predict([noise, labels], batch_size=BATCH_SIZE, verbose=1)

export_path = outputs_dir / '63b_noisy_drone_rf_v2_generated_iq_bank.npz'
np.savez_compressed(
    export_path,
    iq=fake.astype(np.float32),
    labels=labels.astype(np.int64),
    class_names=np.asarray(CLASS_NAMES),
    iq_len=np.asarray([IQ_LEN]),
)
metadata = {
    'generator_path': str(generator_path),
    'iq_shape': [int(IQ_LEN), 2],
    'latent_dim': LATENT_DIM,
    'samples': int(len(labels)),
    'class_names': CLASS_NAMES,
    'notes': 'Synthetic raw I/Q windows generated by conditional GAN.',
}
metadata_path = outputs_dir / '63b_noisy_drone_rf_v2_generated_iq_bank.json'
metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('Saved I/Q bank:', export_path)
print('Saved metadata:', metadata_path)

# %% Cell 10
# Cell 10 : Plot raw-I/Q GAN training curves
if not history_csv_path.exists():
    raise FileNotFoundError(f'Missing history: {history_csv_path}')

history_df = pd.read_csv(history_csv_path).copy()
history_df['global_epoch'] = np.arange(1, len(history_df) + 1)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for col in ['d_loss', 'g_loss']:
    if col in history_df:
        axes[0].plot(history_df['global_epoch'], history_df[col], marker='o', label=col)
axes[0].set_title('Noisy Drone RF v2 Raw-I/Q GAN Loss')
axes[0].set_xlabel('Global epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.35)
axes[0].legend()

for col in ['d_adv', 'd_cls', 'g_adv', 'g_cls', 'teacher_loss']:
    if col in history_df:
        axes[1].plot(history_df['global_epoch'], history_df[col], marker='o', label=col)
axes[1].set_title('Noisy Drone RF v2 Raw-I/Q GAN Loss Components')
axes[1].set_xlabel('Global epoch')
axes[1].set_ylabel('Loss component')
axes[1].grid(True, alpha=0.35)
axes[1].legend()
fig.tight_layout()
plot_path = outputs_dir / '63b_noisy_drone_rf_v2_iq_gan_training_curves.png'
fig.savefig(plot_path, dpi=150)
_save_current_figure("cell_10_figure_05.png")
plt.close(fig)
print('Saved:', plot_path)

# %% Cell 11
# Cell 11 : Teacher-alignment continuation using the frozen discriminator and 33-style complex spectrograms
# Use this when the discriminator class head is good but the frozen 33 VGG teacher predicts everything as Noise.
# This cell loads the existing generator/discriminator, freezes the discriminator, and updates only the generator.

ALIGN_EPOCHS = int(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_EPOCHS', '8'))
ALIGN_STEPS_PER_EPOCH = int(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_STEPS_PER_EPOCH', '120'))
ALIGN_BATCH_SIZE = int(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_BATCH_SIZE', '4'))
ALIGN_LR = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_LR', '2e-5'))
ALIGN_TEACHER_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_TEACHER_WEIGHT', '1.0'))
ALIGN_NOISE_AVOID_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_NOISE_AVOID_WEIGHT', '0.75'))
ALIGN_DISC_CLASS_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_DISC_CLASS_WEIGHT', '1.25'))
ALIGN_DISC_REAL_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_DISC_REAL_WEIGHT', '0.05'))
ALIGN_POWER_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_POWER_WEIGHT', '0.05'))
ALIGN_DIVERSITY_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_DIVERSITY_WEIGHT', '0.02'))
ALIGN_REAL_SPEC_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_REAL_SPEC_WEIGHT', '1.25'))
ALIGN_REAL_IQ_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_REAL_IQ_WEIGHT', '0.02'))
ALIGN_FEATURE_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_FEATURE_WEIGHT', '7.5'))
ALIGN_FEATURE_PROTO_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_FEATURE_PROTO_WEIGHT', '1.5'))
ALIGN_FEATURE_PROTO_CE_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_FEATURE_PROTO_CE_WEIGHT', '0.75'))
ALIGN_FEATURE_PROTO_TEMP = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_FEATURE_PROTO_TEMP', '0.20'))
ALIGN_PRIOR_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_PRIOR_WEIGHT', '0.25'))
ALIGN_DOMINANT_AVOID_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_DOMINANT_AVOID_WEIGHT', '0.75'))
ALIGN_FUTABA_GRAUPNER_AVOID_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_FUTABA_GRAUPNER_AVOID_WEIGHT', '3.0'))
ALIGN_FUTABA_TURNIGY_AVOID_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_FUTABA_TURNIGY_AVOID_WEIGHT', '5.0'))
ALIGN_FUTABA_RAW_TARGET_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_FUTABA_RAW_TARGET_WEIGHT', '8.0'))
ALIGN_FUTABA_PROTO_CE_MULT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_FUTABA_PROTO_CE_MULT', '3.0'))
ALIGN_FUTABA_FEATURE_MULT = float(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_FUTABA_FEATURE_MULT', '2.0'))
ALIGN_REAL_CALIBRATION_PER_CLASS = int(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_REAL_CALIBRATION_PER_CLASS', '12'))
ALIGN_SAVE_EVERY = int(os.getenv('NOISY_DRONE_IQ_GAN_TEACHER_ALIGN_SAVE_EVERY', '2'))

teacher_aligned_generator_path = model_dir / 'noisy_drone_rf_v2_conditional_iq_generator_teacher_aligned.keras'
teacher_aligned_cell10_generator_path = model_dir / 'noisy_drone_rf_v2_conditional_iq_generator_teacher_aligned_cell10.keras'
teacher_align_history_path = outputs_dir / '63b_noisy_drone_rf_v2_iq_gan_teacher_alignment_history.csv'

if not generator_path.exists():
    raise FileNotFoundError(f'Missing existing generator: {generator_path}')
if not discriminator_path.exists():
    raise FileNotFoundError(f'Missing existing discriminator: {discriminator_path}')
if not teacher_model_path.exists():
    raise FileNotFoundError(f'Missing frozen 33 VGG teacher: {teacher_model_path}')

align_source_generator_path = teacher_aligned_generator_path if teacher_aligned_generator_path.exists() else generator_path
align_generator = load_model(align_source_generator_path, compile=False)
align_discriminator = load_model(discriminator_path, compile=False)
align_teacher = load_model(teacher_model_path, compile=False)
align_discriminator.trainable = False
align_teacher.trainable = False
try:
    teacher_embedding_layer = align_teacher.get_layer('vgg_spectrogram_embedding')
except ValueError:
    teacher_embedding_layer = align_teacher.layers[-6]
teacher_feature_model = tf.keras.Model(
    inputs=align_teacher.input,
    outputs=teacher_embedding_layer.output,
    name='frozen_33_vgg_embedding_model',
)
teacher_feature_model.trainable = False

print('Loaded generator:', align_source_generator_path)
print('Loaded frozen discriminator:', discriminator_path)
print('Loaded frozen 33 VGG teacher:', teacher_model_path)
print('Saving teacher-aligned generator to:', teacher_aligned_generator_path)
print('Alignment epochs/steps/batch:', ALIGN_EPOCHS, ALIGN_STEPS_PER_EPOCH, ALIGN_BATCH_SIZE)
print('Alignment prior/dominant/calibration:', ALIGN_PRIOR_WEIGHT, ALIGN_DOMINANT_AVOID_WEIGHT, ALIGN_REAL_CALIBRATION_PER_CLASS)
print('Alignment real-spec/real-IQ weights:', ALIGN_REAL_SPEC_WEIGHT, ALIGN_REAL_IQ_WEIGHT)
print('Alignment teacher feature/prototype weights:', ALIGN_FEATURE_WEIGHT, ALIGN_FEATURE_PROTO_WEIGHT)
print('Alignment teacher prototype CE/temp:', ALIGN_FEATURE_PROTO_CE_WEIGHT, ALIGN_FEATURE_PROTO_TEMP)
print('Alignment Futaba feature/proto/graupner/turnigy/raw-target:', ALIGN_FUTABA_FEATURE_MULT, ALIGN_FUTABA_PROTO_CE_MULT, ALIGN_FUTABA_GRAUPNER_AVOID_WEIGHT, ALIGN_FUTABA_TURNIGY_AVOID_WEIGHT, ALIGN_FUTABA_RAW_TARGET_WEIGHT)


def safe_probs_ce(y_true, y_pred, num_classes=NUM_CLASSES):
    y_pred = tf.cast(y_pred, tf.float32)
    row_sum = tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    looks_like_probs = tf.reduce_all(y_pred >= 0.0) & tf.reduce_all(tf.abs(row_sum - 1.0) < 1e-2)
    probs = tf.cond(looks_like_probs, lambda: y_pred, lambda: tf.nn.softmax(y_pred, axis=-1))
    probs = tf.clip_by_value(probs, 1e-6, 1.0)
    probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
    onehot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    return tf.reduce_mean(-tf.reduce_sum(onehot * tf.math.log(probs), axis=-1))

def iq_to_33_complex_spectrogram_tf(iq):
    """Match notebook 33 preprocessing: complex I/Q -> fftshift FFT -> real/imag channels."""
    iq = tf.cast(iq, tf.float32)
    iq = iq - tf.reduce_mean(iq, axis=1, keepdims=True)
    power = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(iq), axis=-1), axis=1, keepdims=True)) + 1e-6
    iq = iq / power[:, :, None]

    teacher_shape = align_teacher.input_shape[0] if isinstance(align_teacher.input_shape, list) else align_teacher.input_shape
    teacher_h = int(teacher_shape[1])
    teacher_w = int(teacher_shape[2])
    iq_len_static = int(globals().get('IQ_LEN', iq.shape[1] or 4096))
    frame_len = min(int(globals().get('TEACHER_FRAME_LEN', teacher_h)), iq_len_static)
    frame_len = max(16, frame_len)
    frame_step = max(1, (iq_len_static - frame_len) // max(1, teacher_w - 1))

    complex_iq = tf.complex(iq[:, :, 0], iq[:, :, 1])
    frames = tf.signal.frame(complex_iq, frame_length=frame_len, frame_step=frame_step, pad_end=True)
    window = tf.cast(tf.signal.hann_window(frame_len), tf.complex64)
    frames = frames * window[None, None, :]
    fft_complex = tf.signal.fftshift(tf.signal.fft(frames), axes=-1) / tf.cast(frame_len, tf.complex64)
    fft_complex = tf.transpose(fft_complex, [0, 2, 1])

    time_bins = tf.shape(fft_complex)[2]
    pad_t = tf.maximum(0, teacher_w - time_bins)
    fft_complex = tf.pad(fft_complex, [[0, 0], [0, 0], [0, pad_t]])[:, :, :teacher_w]
    spec = tf.stack([tf.math.real(fft_complex), tf.math.imag(fft_complex)], axis=-1)
    spec = tf.cast(spec, tf.float32)
    spec = spec / (tf.math.reduce_std(spec, axis=[1, 2, 3], keepdims=True) + 1e-6)
    spec = tf.clip_by_value(spec, -6.0, 6.0)
    spec = tf.image.resize(spec, [teacher_h, teacher_w], method='bilinear')
    return tf.cast(spec, tf.float32)




def calibrate_teacher_on_real_samples(max_per_class=ALIGN_REAL_CALIBRATION_PER_CLASS):
    """Estimate teacher bias on real I/Q windows so generated teacher logits can be prior-corrected."""
    if 'balanced_df' not in globals():
        print('No balanced_df available; teacher prior calibration uses uniform prior.')
        return np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES, -1
    frames = []
    for label_idx, group in balanced_df.groupby('label_idx', sort=True):
        n = min(int(max_per_class), len(group))
        if n > 0:
            frames.append(group.sample(n=n, random_state=RANDOM_STATE + int(label_idx) + 6300))
    if not frames:
        return np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES, -1
    calib_df = pd.concat(frames, ignore_index=True)
    probs = []
    y_true = []
    for row_idx, row in enumerate(calib_df.itertuples(index=False)):
        rng = np.random.default_rng(RANDOM_STATE + 6400 + row_idx)
        iq = extract_iq_window(row.filepath, rng)[None, ...].astype(np.float32)
        tx = iq_to_33_complex_spectrogram_tf(tf.convert_to_tensor(iq, dtype=tf.float32)).numpy()
        pred = align_teacher.predict(tx, batch_size=1, verbose=0)[0]
        probs.append(pred)
        y_true.append(int(row.label_idx))
    probs = np.asarray(probs, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int64)
    prior = np.maximum(probs.mean(axis=0), 1e-5)
    prior = prior / prior.sum()
    dominant_idx = int(np.argmax(prior))
    acc = float(np.mean(probs.argmax(axis=1) == y_true))
    print('Teacher real-sample calibration accuracy:', acc)
    print('Teacher real-sample mean prior:', dict(zip(CLASS_NAMES, prior.round(4).tolist())))
    print('Teacher dominant class:', CLASS_NAMES[dominant_idx], dominant_idx)
    return prior.astype(np.float32), dominant_idx


teacher_prior_np, teacher_dominant_idx = calibrate_teacher_on_real_samples()
teacher_log_prior = tf.constant(np.log(teacher_prior_np + 1e-5), dtype=tf.float32)
teacher_dominant_idx_tf = tf.constant(teacher_dominant_idx, dtype=tf.int32)


def build_teacher_feature_prototypes(max_per_class=ALIGN_REAL_CALIBRATION_PER_CLASS):
    """Build class prototypes in the frozen 33 VGG embedding space from real I/Q windows."""
    feature_dim = int(teacher_feature_model.output_shape[-1])
    prototypes = np.zeros((NUM_CLASSES, feature_dim), dtype=np.float32)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    if 'balanced_df' not in globals():
        prototypes[:] = 0.0
        return prototypes

    for label_idx, group in balanced_df.groupby('label_idx', sort=True):
        label_idx = int(label_idx)
        n = min(int(max_per_class), len(group))
        if n <= 0:
            continue
        sample_df = group.sample(n=n, random_state=RANDOM_STATE + label_idx + 6500)
        feats = []
        for row_idx, row in enumerate(sample_df.itertuples(index=False)):
            rng = np.random.default_rng(RANDOM_STATE + 6600 + label_idx * 100 + row_idx)
            iq = extract_iq_window(row.filepath, rng)[None, ...].astype(np.float32)
            tx = iq_to_33_complex_spectrogram_tf(tf.convert_to_tensor(iq, dtype=tf.float32)).numpy()
            feat = teacher_feature_model.predict(tx, batch_size=1, verbose=0)[0].astype(np.float32)
            norm = np.linalg.norm(feat) + 1e-6
            feats.append(feat / norm)
        if feats:
            proto = np.mean(np.asarray(feats, dtype=np.float32), axis=0)
            prototypes[label_idx] = proto / (np.linalg.norm(proto) + 1e-6)
            counts[label_idx] = len(feats)
    print('Teacher feature prototype counts:', dict(zip(CLASS_NAMES, counts.tolist())))
    return prototypes


teacher_feature_prototypes_np = build_teacher_feature_prototypes()
teacher_feature_prototypes = tf.constant(teacher_feature_prototypes_np, dtype=tf.float32)


def l2_normalize_features(features):
    return tf.math.l2_normalize(tf.cast(features, tf.float32), axis=-1)


def prior_correct_teacher_probs(teacher_probs):
    logits = tf.math.log(tf.clip_by_value(tf.cast(teacher_probs, tf.float32), 1e-6, 1.0))
    logits = logits - ALIGN_PRIOR_WEIGHT * teacher_log_prior[None, :]
    return tf.nn.softmax(logits, axis=-1)


def target_confidence_loss(labels, probs):
    target = tf.gather(probs, tf.cast(labels, tf.int32), batch_dims=1)
    return tf.reduce_mean(-tf.math.log(tf.clip_by_value(target, 1e-6, 1.0)))




align_optimizer = tf.keras.optimizers.Adam(learning_rate=ALIGN_LR, beta_1=0.5, beta_2=0.999)

@tf.function
def teacher_align_step(real_iq, labels):
    real_iq = tf.cast(real_iq, tf.float32)
    labels = tf.cast(labels, tf.int32)
    batch_size = tf.shape(labels)[0]
    noise = tf.random.normal([batch_size, LATENT_DIM])
    with tf.GradientTape() as tape:
        fake_iq = align_generator([noise, labels], training=True)
        disc_realness, disc_cls = align_discriminator(fake_iq, training=False)
        fake_teacher_x = iq_to_33_complex_spectrogram_tf(fake_iq)
        real_teacher_x = tf.stop_gradient(iq_to_33_complex_spectrogram_tf(real_iq))
        teacher_probs_raw = align_teacher(fake_teacher_x, training=False)
        teacher_probs_raw_norm = tf.cast(teacher_probs_raw, tf.float32)
        teacher_probs_raw_norm = tf.clip_by_value(teacher_probs_raw_norm, 1e-6, 1.0)
        teacher_probs_raw_norm = teacher_probs_raw_norm / tf.reduce_sum(teacher_probs_raw_norm, axis=-1, keepdims=True)
        teacher_probs = prior_correct_teacher_probs(teacher_probs_raw_norm)
        fake_features = l2_normalize_features(teacher_feature_model(fake_teacher_x, training=False))
        real_features = tf.stop_gradient(l2_normalize_features(teacher_feature_model(real_teacher_x, training=False)))
        target_prototypes = tf.gather(teacher_feature_prototypes, labels)

        # Use target confidence after prior correction instead of raw teacher CE; raw teacher
        # logits were observed to collapse generated samples into one dominant class.
        teacher_ce = target_confidence_loss(labels, teacher_probs)
        disc_cls_ce = safe_probs_ce(labels, disc_cls, NUM_CLASSES)
        disc_real_loss = tf.keras.backend.binary_crossentropy(tf.ones_like(disc_realness) * 0.9, tf.clip_by_value(disc_realness, 1e-6, 1.0 - 1e-6))
        disc_real_loss = tf.reduce_mean(disc_real_loss)

        non_noise = tf.cast(tf.not_equal(labels, 4), tf.float32)
        p_noise = tf.clip_by_value(teacher_probs[:, 4], 1e-6, 1.0 - 1e-6)
        noise_avoid = tf.reduce_sum(-tf.math.log(1.0 - p_noise) * non_noise) / (tf.reduce_sum(non_noise) + 1e-6)
        if teacher_dominant_idx >= 0:
            dominant_mask = tf.cast(tf.not_equal(labels, teacher_dominant_idx_tf), tf.float32)
            p_dominant = tf.clip_by_value(teacher_probs[:, teacher_dominant_idx], 1e-6, 1.0 - 1e-6)
            dominant_avoid = tf.reduce_sum(-tf.math.log(1.0 - p_dominant) * dominant_mask) / (tf.reduce_sum(dominant_mask) + 1e-6)
        else:
            dominant_avoid = tf.constant(0.0, dtype=tf.float32)

        # Encourage the generated batch's teacher distribution to be close to balanced.
        mean_teacher = tf.reduce_mean(teacher_probs, axis=0)
        balanced_prior_loss = tf.reduce_mean(tf.square(mean_teacher - (1.0 / float(NUM_CLASSES))))

        real_spec_loss = tf.reduce_mean(tf.abs(fake_teacher_x - real_teacher_x))
        feature_pair_per_sample = tf.reduce_mean(tf.square(fake_features - real_features), axis=-1)
        feature_proto_per_sample = tf.reduce_mean(tf.square(fake_features - target_prototypes), axis=-1)
        # Futaba-family classes are the remaining failure mode; prioritize real same-class manifold matching
        # over global prototype CE for those classes because their prototypes are close to other controllers.
        futaba_mask = tf.cast(tf.logical_or(tf.equal(labels, 1), tf.equal(labels, 2)), tf.float32)
        hard_class_weight = 1.0 + (ALIGN_FUTABA_FEATURE_MULT - 1.0) * futaba_mask
        feature_pair_loss = tf.reduce_mean(feature_pair_per_sample * hard_class_weight)
        feature_proto_loss = tf.reduce_mean(feature_proto_per_sample * hard_class_weight)
        proto_logits = tf.matmul(fake_features, teacher_feature_prototypes, transpose_b=True) / ALIGN_FEATURE_PROTO_TEMP
        proto_ce_per_sample = tf.keras.losses.sparse_categorical_crossentropy(labels, proto_logits, from_logits=True)
        proto_ce_weight = 1.0 + (ALIGN_FUTABA_PROTO_CE_MULT - 1.0) * futaba_mask
        feature_proto_ce = tf.reduce_mean(proto_ce_per_sample * proto_ce_weight)
        p_graupner = tf.clip_by_value(teacher_probs[:, 3], 1e-6, 1.0 - 1e-6)
        futaba_graupner_avoid = tf.reduce_sum(-tf.math.log(1.0 - p_graupner) * futaba_mask) / (tf.reduce_sum(futaba_mask) + 1e-6)
        p_turnigy = tf.clip_by_value(teacher_probs[:, 6], 1e-6, 1.0 - 1e-6)
        futaba_turnigy_avoid = tf.reduce_sum(-tf.math.log(1.0 - p_turnigy) * futaba_mask) / (tf.reduce_sum(futaba_mask) + 1e-6)
        raw_teacher_ce = target_confidence_loss(labels, teacher_probs_raw_norm)
        futaba_raw_target_ce = tf.reduce_sum(raw_teacher_ce * futaba_mask) / (tf.reduce_sum(futaba_mask) + 1e-6)
        # This term is intentionally small: same class, not same sample, but it discourages wild off-manifold I/Q.
        real_iq_loss = tf.reduce_mean(tf.abs(fake_iq - tf.stop_gradient(real_iq)))

        rms = tf.sqrt(tf.reduce_mean(tf.square(fake_iq), axis=[1, 2]) + 1e-6)
        power_loss = tf.reduce_mean(tf.square(rms - 1.0))

        # Encourage different latent vectors to produce non-identical I/Q. Maximizing variance lowers this term.
        diversity_loss = 1.0 / (tf.reduce_mean(tf.math.reduce_std(fake_iq, axis=0)) + 1e-6)

        loss = (
            ALIGN_TEACHER_WEIGHT * teacher_ce
            + ALIGN_NOISE_AVOID_WEIGHT * noise_avoid
            + ALIGN_DOMINANT_AVOID_WEIGHT * dominant_avoid
            + ALIGN_FUTABA_GRAUPNER_AVOID_WEIGHT * futaba_graupner_avoid
            + ALIGN_FUTABA_TURNIGY_AVOID_WEIGHT * futaba_turnigy_avoid
            + ALIGN_FUTABA_RAW_TARGET_WEIGHT * futaba_raw_target_ce
            + ALIGN_PRIOR_WEIGHT * balanced_prior_loss
            + ALIGN_DISC_CLASS_WEIGHT * disc_cls_ce
            + ALIGN_DISC_REAL_WEIGHT * disc_real_loss
            + ALIGN_REAL_SPEC_WEIGHT * real_spec_loss
            + ALIGN_FEATURE_WEIGHT * feature_pair_loss
            + ALIGN_FEATURE_PROTO_WEIGHT * feature_proto_loss
            + ALIGN_FEATURE_PROTO_CE_WEIGHT * feature_proto_ce
            + ALIGN_REAL_IQ_WEIGHT * real_iq_loss
            + ALIGN_POWER_WEIGHT * power_loss
            + ALIGN_DIVERSITY_WEIGHT * diversity_loss
        )
    grads = tape.gradient(loss, align_generator.trainable_variables)
    align_optimizer.apply_gradients(zip(grads, align_generator.trainable_variables))
    return {
        'loss': loss,
        'teacher_ce': teacher_ce,
        'noise_avoid': noise_avoid,
        'dominant_avoid': dominant_avoid,
        'futaba_graupner_avoid': futaba_graupner_avoid,
        'futaba_turnigy_avoid': futaba_turnigy_avoid,
        'futaba_raw_target_ce': futaba_raw_target_ce,
        'balanced_prior_loss': balanced_prior_loss,
        'disc_cls_ce': disc_cls_ce,
        'disc_real_loss': disc_real_loss,
        'real_spec_loss': real_spec_loss,
        'feature_pair_loss': feature_pair_loss,
        'feature_proto_loss': feature_proto_loss,
        'feature_proto_ce': feature_proto_ce,
        'real_iq_loss': real_iq_loss,
        'power_loss': power_loss,
        'diversity_loss': diversity_loss,
        'teacher_noise_mean': tf.reduce_mean(teacher_probs[:, 4]),
        'teacher_target_conf': tf.reduce_mean(tf.gather(teacher_probs, tf.cast(labels, tf.int32), batch_dims=1)),
    }


def next_alignment_batch(iterator):
    """Get a real same-class batch for manifold matching; fallback to synthetic labels if needed."""
    try:
        real_iq, labels = next(iterator)
        real_iq = tf.cast(real_iq, tf.float32)
        labels = tf.cast(labels, tf.int32)
        if tf.shape(real_iq)[0] > ALIGN_BATCH_SIZE:
            real_iq = real_iq[:ALIGN_BATCH_SIZE]
            labels = labels[:ALIGN_BATCH_SIZE]
        return real_iq, labels, iterator
    except Exception as exc:
        print('Alignment real-batch fallback:', type(exc).__name__, exc)
        reps = int(np.ceil(ALIGN_BATCH_SIZE / NUM_CLASSES))
        labels_np = np.tile(np.arange(NUM_CLASSES, dtype=np.int32), reps)[:ALIGN_BATCH_SIZE]
        np.random.default_rng().shuffle(labels_np)
        labels = tf.convert_to_tensor(labels_np, dtype=tf.int32)
        # Tiny fallback noise batch; this path should rarely be used when Cells 1-4 ran.
        real_iq = tf.random.normal([ALIGN_BATCH_SIZE, IQ_LEN, 2])
        return real_iq, labels, iterator


if 'train_ds' not in globals():
    raise RuntimeError('Run Cells 1-4 before Cell 10 so train_ds provides real I/Q manifold batches.')
alignment_iterator = iter(train_ds.repeat())


align_run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
align_rows = []
for epoch in range(1, ALIGN_EPOCHS + 1):
    epoch_rows = []
    for step in range(1, ALIGN_STEPS_PER_EPOCH + 1):
        real_iq, labels, alignment_iterator = next_alignment_batch(alignment_iterator)
        metrics = teacher_align_step(real_iq, labels)
        row = {k: float(v.numpy()) for k, v in metrics.items()}
        epoch_rows.append(row)
        if step == 1 or step % 20 == 0:
            print(
                f"align epoch={epoch:03d} step={step:04d}/{ALIGN_STEPS_PER_EPOCH} "
                f"loss={row['loss']:.4f} teacher_ce={row['teacher_ce']:.4f} "
                f"target_conf={row['teacher_target_conf']:.4f} noise_mean={row['teacher_noise_mean']:.4f} "
                f"real_spec={row.get('real_spec_loss', 0.0):.4f} "
                f"feat_proto={row.get('feature_proto_loss', 0.0):.4f} "
                f"proto_ce={row.get('feature_proto_ce', 0.0):.4f} "
                f"futaba_graupner={row.get('futaba_graupner_avoid', 0.0):.4f} "
                f"futaba_turnigy={row.get('futaba_turnigy_avoid', 0.0):.4f} "
                f"futaba_raw_ce={row.get('futaba_raw_target_ce', 0.0):.4f}"
            )
    summary = {'run_id': align_run_id, 'epoch': epoch}
    for key in epoch_rows[0]:
        summary[key] = float(np.mean([r[key] for r in epoch_rows]))
    align_rows.append(summary)
    print('teacher alignment summary:', summary)
    if epoch == 1 or epoch % ALIGN_SAVE_EVERY == 0 or epoch == ALIGN_EPOCHS:
        align_generator.save(teacher_aligned_generator_path)
        align_generator.save(teacher_aligned_cell10_generator_path)
        print('Saved teacher-aligned generator:', teacher_aligned_generator_path)
        print('Saved stable Cell 10 generator:', teacher_aligned_cell10_generator_path)

align_df = pd.DataFrame(align_rows)
if teacher_align_history_path.exists():
    align_df = pd.concat([pd.read_csv(teacher_align_history_path), align_df], ignore_index=True)
align_df.to_csv(teacher_align_history_path, index=False)
print('Saved teacher alignment history:', teacher_align_history_path)

# Quick evaluation with the same plot style as Cell 7, but using the teacher-aligned generator.
labels = np.repeat(np.arange(NUM_CLASSES), 32).astype(np.int32)
noise = np.random.default_rng(20260704).standard_normal((len(labels), LATENT_DIM)).astype(np.float32)
fake = align_generator.predict([noise, labels], batch_size=max(1, ALIGN_BATCH_SIZE), verbose=0)
disc_realness, disc_probs = align_discriminator.predict(fake, batch_size=max(1, ALIGN_BATCH_SIZE), verbose=0)
teacher_x = iq_to_33_complex_spectrogram_tf(tf.convert_to_tensor(fake, dtype=tf.float32)).numpy()
teacher_probs_raw = align_teacher.predict(teacher_x, batch_size=max(1, min(ALIGN_BATCH_SIZE, 2)), verbose=0)
teacher_probs = prior_correct_teacher_probs(tf.convert_to_tensor(teacher_probs_raw, dtype=tf.float32)).numpy()
disc_pred = disc_probs.argmax(axis=1)
teacher_pred = teacher_probs.argmax(axis=1)

for pred, title, path_name, cmap in [
    (disc_pred, 'Generated I/Q - Frozen Discriminator Class Head After Teacher Alignment', '63b_teacher_aligned_discriminator_confusion_matrix.png', 'Blues'),
    (teacher_pred, 'Generated I/Q - Frozen 33 VGG Teacher After Alignment', '63b_teacher_aligned_teacher_confusion_matrix.png', 'Greens'),
]:
    cm = pd.crosstab(pd.Series(labels, name='target'), pd.Series(pred, name='pred')).reindex(index=range(NUM_CLASSES), columns=range(NUM_CLASSES), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('Requested generated class')
    fig.tight_layout()
    out_path = outputs_dir / path_name
    fig.savefig(out_path, dpi=150)
    _save_current_figure("cell_11_figure_06.png")
    plt.close(fig)
    print('Saved:', out_path)

print('Teacher aligned target accuracy:', float(np.mean(teacher_pred == labels)))
print('Teacher aligned discriminator accuracy:', float(np.mean(disc_pred == labels)))

# %% Cell 12
# Cell 12 : Targeted FutabaT14 repair pass with guarded promotion
from sklearn.metrics import confusion_matrix
# Run after Cell 10 if the global alignment is good except FutabaT14.
# This freezes the discriminator and teacher, updates only the generator, and only saves when diagnostics improve.

REPAIR_CLASS_IDX = int(os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_CLASS_IDX', '1'))  # FutabaT14
REPAIR_EPOCHS = int(os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_EPOCHS', '6'))
REPAIR_STEPS_PER_EPOCH = int(os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_STEPS_PER_EPOCH', '80'))
REPAIR_BATCH_SIZE = int(os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_BATCH_SIZE', '8'))
REPAIR_LR = float(os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_LR', '4e-6'))
REPAIR_FEATURE_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_FEATURE_WEIGHT', '20.0'))
REPAIR_PROTO_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_PROTO_WEIGHT', '8.0'))
REPAIR_TEACHER_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_TEACHER_WEIGHT', '0.50'))
REPAIR_DISC_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_DISC_WEIGHT', '8.0'))
REPAIR_TURNIGY_AVOID_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_TURNIGY_AVOID_WEIGHT', '5.0'))
REPAIR_NOISE_AVOID_WEIGHT = float(os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_NOISE_AVOID_WEIGHT', '6.0'))
REPAIR_DIAG_SAMPLES = int(os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_DIAG_SAMPLES', '64'))
REPAIR_CLASS_EMBEDDING_ONLY = os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_CLASS_EMBEDDING_ONLY', '1') == '1'
REPAIR_PROMOTE_TO_CELL10_PATH = os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_PROMOTE_TO_CELL10_PATH', '0') == '1'

generator_path = globals().get('generator_path', model_dir / 'noisy_drone_rf_v2_conditional_iq_generator.keras')
teacher_aligned_generator_path = globals().get(
    'teacher_aligned_generator_path',
    model_dir / 'noisy_drone_rf_v2_conditional_iq_generator_teacher_aligned.keras',
)
teacher_aligned_cell10_generator_path = globals().get(
    'teacher_aligned_cell10_generator_path',
    model_dir / 'noisy_drone_rf_v2_conditional_iq_generator_teacher_aligned_cell10.keras',
)
repaired_generator_path = model_dir / 'noisy_drone_rf_v2_conditional_iq_generator_teacher_aligned_repaired.keras'
repair_history_path = outputs_dir / '63b_noisy_drone_rf_v2_iq_gan_futabat14_repair_history.csv'
repair_diag_path = outputs_dir / '63b_generated_iq_teacher_alignment_futabat14_repair_confusion.png'

if not discriminator_path.exists():
    raise FileNotFoundError(f'Missing discriminator: {discriminator_path}')
if not teacher_model_path.exists():
    raise FileNotFoundError(f'Missing frozen 33 VGG teacher: {teacher_model_path}')

repair_discriminator = load_model(discriminator_path, compile=False)
repair_teacher = load_model(teacher_model_path, compile=False)
repair_discriminator.trainable = False
repair_teacher.trainable = False
try:
    repair_embedding_layer = repair_teacher.get_layer('vgg_spectrogram_embedding')
except ValueError:
    repair_embedding_layer = repair_teacher.layers[-6]
repair_feature_model = tf.keras.Model(repair_teacher.input, repair_embedding_layer.output, name='repair_33_vgg_embedding_model')
repair_feature_model.trainable = False

def repair_target_confidence_loss(labels, probs):
    target = tf.gather(tf.cast(probs, tf.float32), tf.cast(labels, tf.int32), batch_dims=1)
    return tf.reduce_mean(-tf.math.log(tf.clip_by_value(target, 1e-6, 1.0)))


def repair_safe_probs_ce(y_true, y_pred, num_classes=NUM_CLASSES):
    y_pred = tf.cast(y_pred, tf.float32)
    row_sum = tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    looks_like_probs = tf.reduce_all(y_pred >= 0.0) & tf.reduce_all(tf.abs(row_sum - 1.0) < 1e-2)
    probs = tf.cond(looks_like_probs, lambda: y_pred, lambda: tf.nn.softmax(y_pred, axis=-1))
    probs = tf.clip_by_value(probs, 1e-6, 1.0)
    probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
    onehot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    return tf.reduce_mean(-tf.reduce_sum(onehot * tf.math.log(probs), axis=-1))


if 'balanced_df' not in globals():
    raise RuntimeError('Run Cells 1-4 before Cell 11 so balanced_df is available.')
repair_source_df = balanced_df[balanced_df['label_idx'].astype(int) == REPAIR_CLASS_IDX].reset_index(drop=True)
if repair_source_df.empty:
    raise RuntimeError(f'No real samples available for repair class {REPAIR_CLASS_IDX}.')


def repair_iq_to_teacher_spectrogram(iq):
    """Convert raw I/Q to the 33 VGG full-complex spectrogram input.

    Cell 10 defines iq_to_33_complex_spectrogram_tf, but this final repair cell
    should also run after only setup/data cells. This fallback keeps it self-contained.
    """
    if 'iq_to_33_complex_spectrogram_tf' in globals():
        return iq_to_33_complex_spectrogram_tf(iq)

    iq = tf.cast(iq, tf.float32)
    iq = iq - tf.reduce_mean(iq, axis=1, keepdims=True)
    power = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(iq), axis=-1), axis=1, keepdims=True)) + 1e-6
    iq = iq / power[:, :, None]

    teacher_shape = repair_teacher.input_shape[0] if isinstance(repair_teacher.input_shape, list) else repair_teacher.input_shape
    teacher_h = int(teacher_shape[1])
    teacher_w = int(teacher_shape[2])
    teacher_c = int(teacher_shape[3])
    iq_len_static = int(globals().get('IQ_LEN', iq.shape[1] or 4096))
    frame_len = min(int(os.getenv('NOISY_DRONE_IQ_GAN_REPAIR_FRAME_LEN', str(teacher_h))), iq_len_static)
    frame_len = max(16, frame_len)
    frame_step = max(1, (iq_len_static - frame_len) // max(1, teacher_w - 1))

    complex_iq = tf.complex(iq[:, :, 0], iq[:, :, 1])
    frames = tf.signal.frame(complex_iq, frame_length=frame_len, frame_step=frame_step, pad_end=True)
    window = tf.cast(tf.signal.hann_window(frame_len), tf.complex64)
    frames = frames * window[None, None, :]
    fft_complex = tf.signal.fftshift(tf.signal.fft(frames), axes=-1) / tf.cast(frame_len, tf.complex64)
    fft_complex = tf.transpose(fft_complex, [0, 2, 1])

    time_bins = tf.shape(fft_complex)[2]
    pad_t = tf.maximum(0, teacher_w - time_bins)
    fft_complex = tf.pad(fft_complex, [[0, 0], [0, 0], [0, pad_t]])[:, :, :teacher_w]
    spec = tf.stack([tf.math.real(fft_complex), tf.math.imag(fft_complex)], axis=-1)
    spec = tf.cast(spec, tf.float32)
    spec = spec / (tf.math.reduce_std(spec, axis=[1, 2, 3], keepdims=True) + 1e-6)
    spec = tf.clip_by_value(spec, -6.0, 6.0)
    spec = tf.image.resize(spec, [teacher_h, teacher_w], method='bilinear')
    if teacher_c == 1:
        spec = spec[..., :1]
    elif teacher_c > 2:
        pad = tf.zeros([tf.shape(spec)[0], teacher_h, teacher_w, teacher_c - 2], dtype=spec.dtype)
        spec = tf.concat([spec, pad], axis=-1)
    return tf.cast(spec, tf.float32)


def repair_norm_features(features):
    return tf.math.l2_normalize(tf.cast(features, tf.float32), axis=-1)


def sample_repair_real_batch(batch_size=REPAIR_BATCH_SIZE, seed_offset=0):
    rows = repair_source_df.sample(
        n=batch_size,
        replace=len(repair_source_df) < batch_size,
        random_state=RANDOM_STATE + 9000 + seed_offset,
    ).reset_index(drop=True)
    xs = []
    for row_idx, row in enumerate(rows.itertuples(index=False)):
        rng = np.random.default_rng(RANDOM_STATE + 9100 + seed_offset * 31 + row_idx)
        xs.append(extract_iq_window(row.filepath, rng))
    return tf.convert_to_tensor(np.asarray(xs, dtype=np.float32), dtype=tf.float32)

# Build a clean class prototype for FutabaT14 from real samples.
proto_feats = []
for row_idx, row in enumerate(repair_source_df.sample(n=min(48, len(repair_source_df)), random_state=RANDOM_STATE + 9200).itertuples(index=False)):
    rng = np.random.default_rng(RANDOM_STATE + 9300 + row_idx)
    real_iq = extract_iq_window(row.filepath, rng)[None, ...].astype(np.float32)
    spec = repair_iq_to_teacher_spectrogram(tf.convert_to_tensor(real_iq, dtype=tf.float32))
    feat = repair_norm_features(repair_feature_model(spec, training=False))[0].numpy()
    proto_feats.append(feat)
repair_target_proto = np.mean(np.asarray(proto_feats, dtype=np.float32), axis=0)
repair_target_proto = repair_target_proto / (np.linalg.norm(repair_target_proto) + 1e-6)
repair_target_proto_tf = tf.constant(repair_target_proto[None, :], dtype=tf.float32)


def evaluate_repair_generator(model, samples=REPAIR_DIAG_SAMPLES, seed=RANDOM_STATE + 9400):
    labels_np = np.full(samples, REPAIR_CLASS_IDX, dtype=np.int32)
    noise_np = np.random.default_rng(seed).normal(size=(samples, LATENT_DIM)).astype(np.float32)
    fake_iq = model.predict([noise_np, labels_np], batch_size=32, verbose=0)
    _, disc_cls = repair_discriminator.predict(fake_iq, batch_size=32, verbose=0)
    teacher_x = repair_iq_to_teacher_spectrogram(tf.convert_to_tensor(fake_iq, dtype=tf.float32)).numpy()
    teacher_probs = repair_teacher.predict(teacher_x, batch_size=16, verbose=0)
    disc_pred = disc_cls.argmax(axis=1)
    teacher_pred = teacher_probs.argmax(axis=1)
    disc_hits = int(np.sum(disc_pred == REPAIR_CLASS_IDX))
    teacher_hits = int(np.sum(teacher_pred == REPAIR_CLASS_IDX))
    noise_hits = int(np.sum((disc_pred == 4) | (teacher_pred == 4)))
    turnigy_hits = int(np.sum((disc_pred == 6) | (teacher_pred == 6)))
    target_conf = float(np.mean(teacher_probs[:, REPAIR_CLASS_IDX]))
    score = (3.0 * teacher_hits) + (2.0 * disc_hits) + (samples * target_conf) - (1.5 * noise_hits) - (1.0 * turnigy_hits)
    return {
        'score': float(score),
        'disc_hits': disc_hits,
        'teacher_hits': teacher_hits,
        'noise_hits': noise_hits,
        'turnigy_hits': turnigy_hits,
        'target_conf': target_conf,
        'disc_pred': disc_pred,
        'teacher_pred': teacher_pred,
        'fake_iq': fake_iq,
    }

# Choose the best available source. Previous repair attempts may have promoted a bad checkpoint.
candidate_paths = []
for path in [teacher_aligned_cell10_generator_path, teacher_aligned_generator_path, generator_path, repaired_generator_path]:
    if path.exists() and path not in candidate_paths:
        candidate_paths.append(path)
if not candidate_paths:
    raise FileNotFoundError('No generator candidate exists for repair.')

candidate_scores = []
for path in candidate_paths:
    candidate_model = load_model(path, compile=False)
    diag = evaluate_repair_generator(candidate_model)
    candidate_scores.append((diag['score'], path, diag))
    print('Repair source candidate:', path, {k: diag[k] for k in ['score', 'disc_hits', 'teacher_hits', 'noise_hits', 'turnigy_hits', 'target_conf']})

best_score, source_repair_generator_path, best_diag = max(candidate_scores, key=lambda item: item[0])
repair_generator = load_model(source_repair_generator_path, compile=False)
if REPAIR_CLASS_EMBEDDING_ONLY:
    for layer in repair_generator.layers:
        layer.trainable = layer.name == 'class_embedding'
    if not repair_generator.trainable_variables:
        print('No class_embedding trainable variable found; falling back to full generator repair.')
        repair_generator.trainable = True
    else:
        print('Repair trainable layers:', [layer.name for layer in repair_generator.layers if layer.trainable])
best_weights = repair_generator.get_weights()
repair_optimizer = tf.keras.optimizers.Adam(learning_rate=REPAIR_LR, beta_1=0.5, beta_2=0.999)

print('Repairing class:', CLASS_NAMES[REPAIR_CLASS_IDX], REPAIR_CLASS_IDX)
print('Selected repair generator:', source_repair_generator_path)
print('Initial repair diagnostic:', {k: best_diag[k] for k in ['score', 'disc_hits', 'teacher_hits', 'noise_hits', 'turnigy_hits', 'target_conf']})
print('Saving repaired generator:', repaired_generator_path)


@tf.function
def repair_step(real_iq):
    real_iq = tf.cast(real_iq, tf.float32)
    batch_size = tf.shape(real_iq)[0]
    labels = tf.fill([batch_size], tf.cast(REPAIR_CLASS_IDX, tf.int32))
    noise = tf.random.normal([batch_size, LATENT_DIM])
    with tf.GradientTape() as tape:
        fake_iq = repair_generator([noise, labels], training=True)
        disc_realness, disc_cls = repair_discriminator(fake_iq, training=False)
        fake_spec = repair_iq_to_teacher_spectrogram(fake_iq)
        real_spec = tf.stop_gradient(repair_iq_to_teacher_spectrogram(real_iq))
        teacher_raw = repair_teacher(fake_spec, training=False)
        teacher_probs = tf.cast(teacher_raw, tf.float32)
        teacher_probs = tf.clip_by_value(teacher_probs, 1e-6, 1.0)
        teacher_probs = teacher_probs / tf.reduce_sum(teacher_probs, axis=-1, keepdims=True)
        fake_features = repair_norm_features(repair_feature_model(fake_spec, training=False))
        real_features = tf.stop_gradient(repair_norm_features(repair_feature_model(real_spec, training=False)))

        teacher_target_loss = repair_target_confidence_loss(labels, teacher_probs)
        disc_cls_loss = repair_safe_probs_ce(labels, disc_cls, NUM_CLASSES)
        feature_pair_loss = tf.reduce_mean(tf.square(fake_features - real_features))
        feature_proto_loss = tf.reduce_mean(tf.square(fake_features - repair_target_proto_tf))
        p_noise = tf.clip_by_value(teacher_probs[:, 4], 1e-6, 1.0 - 1e-6)
        p_turnigy = tf.clip_by_value(teacher_probs[:, 6], 1e-6, 1.0 - 1e-6)
        avoid_noise = tf.reduce_mean(-tf.math.log(1.0 - p_noise))
        avoid_turnigy = tf.reduce_mean(-tf.math.log(1.0 - p_turnigy))
        rms = tf.sqrt(tf.reduce_mean(tf.square(fake_iq), axis=[1, 2]) + 1e-6)
        power_loss = tf.reduce_mean(tf.square(rms - 1.0))

        loss = (
            REPAIR_TEACHER_WEIGHT * teacher_target_loss
            + REPAIR_DISC_WEIGHT * disc_cls_loss
            + REPAIR_FEATURE_WEIGHT * feature_pair_loss
            + REPAIR_PROTO_WEIGHT * feature_proto_loss
            + REPAIR_NOISE_AVOID_WEIGHT * avoid_noise
            + REPAIR_TURNIGY_AVOID_WEIGHT * avoid_turnigy
            + 0.05 * power_loss
        )
    grads = tape.gradient(loss, repair_generator.trainable_variables)
    repair_optimizer.apply_gradients(zip(grads, repair_generator.trainable_variables))
    return {
        'loss': loss,
        'teacher_target_loss': teacher_target_loss,
        'disc_cls_loss': disc_cls_loss,
        'feature_pair_loss': feature_pair_loss,
        'feature_proto_loss': feature_proto_loss,
        'avoid_noise': avoid_noise,
        'avoid_turnigy': avoid_turnigy,
        'teacher_target_conf': tf.reduce_mean(teacher_probs[:, REPAIR_CLASS_IDX]),
        'teacher_noise_mean': tf.reduce_mean(teacher_probs[:, 4]),
        'teacher_turnigy_mean': tf.reduce_mean(teacher_probs[:, 6]),
    }

repair_run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
repair_rows = []
for epoch in range(1, REPAIR_EPOCHS + 1):
    epoch_rows = []
    for step in range(1, REPAIR_STEPS_PER_EPOCH + 1):
        real_iq = sample_repair_real_batch(REPAIR_BATCH_SIZE, seed_offset=epoch * 1000 + step)
        metrics = repair_step(real_iq)
        row = {k: float(v.numpy()) for k, v in metrics.items()}
        epoch_rows.append(row)
        if step == 1 or step % 20 == 0:
            print(
                f"repair epoch={epoch:03d} step={step:04d}/{REPAIR_STEPS_PER_EPOCH} "
                f"loss={row['loss']:.4f} target_conf={row['teacher_target_conf']:.4f} "
                f"noise={row['teacher_noise_mean']:.4f} turnigy={row['teacher_turnigy_mean']:.4f}"
            )
    diag = evaluate_repair_generator(repair_generator)
    summary = {
        'run_id': repair_run_id,
        'epoch': epoch,
        'repair_class_idx': REPAIR_CLASS_IDX,
        'repair_class': CLASS_NAMES[REPAIR_CLASS_IDX],
        'diag_score': diag['score'],
        'diag_disc_hits': diag['disc_hits'],
        'diag_teacher_hits': diag['teacher_hits'],
        'diag_noise_hits': diag['noise_hits'],
        'diag_turnigy_hits': diag['turnigy_hits'],
        'diag_target_conf': diag['target_conf'],
    }
    for key in epoch_rows[0]:
        summary[key] = float(np.mean([r[key] for r in epoch_rows]))
    repair_rows.append(summary)
    print('repair summary:', summary)
    if diag['score'] > best_score:
        best_score = diag['score']
        best_weights = repair_generator.get_weights()
        repair_generator.save(repaired_generator_path)
        print('Improved and saved repaired generator:', repaired_generator_path)
        if REPAIR_PROMOTE_TO_CELL10_PATH:
            repair_generator.save(teacher_aligned_generator_path)
            print('Promoted repaired generator:', teacher_aligned_generator_path)
    else:
        print('No diagnostic improvement; not promoting this epoch.')

repair_generator.set_weights(best_weights)
repair_generator.save(repaired_generator_path)
print('Saved best repaired generator:', repaired_generator_path)
if REPAIR_PROMOTE_TO_CELL10_PATH:
    repair_generator.save(teacher_aligned_generator_path)
    print('Promoted best repaired generator:', teacher_aligned_generator_path)
else:
    print('Did not promote repair over Cell 10 generator. Set NOISY_DRONE_IQ_GAN_REPAIR_PROMOTE_TO_CELL10_PATH=1 to promote manually.')

repair_df = pd.DataFrame(repair_rows)
if repair_history_path.exists():
    repair_df = pd.concat([pd.read_csv(repair_history_path), repair_df], ignore_index=True)
repair_df.to_csv(repair_history_path, index=False)
print('Saved repair history:', repair_history_path)

# Final diagnostic across all classes with the best repaired generator.
labels_np = np.repeat(np.arange(NUM_CLASSES, dtype=np.int32), 32)
noise_np = np.random.default_rng(RANDOM_STATE + 9500).normal(size=(len(labels_np), LATENT_DIM)).astype(np.float32)
fake_iq = repair_generator.predict([noise_np, labels_np], batch_size=32, verbose=0)
_, disc_cls = repair_discriminator.predict(fake_iq, batch_size=32, verbose=0)
teacher_x = repair_iq_to_teacher_spectrogram(tf.convert_to_tensor(fake_iq, dtype=tf.float32)).numpy()
teacher_preds = repair_teacher.predict(teacher_x, batch_size=16, verbose=0).argmax(axis=1)
disc_preds = disc_cls.argmax(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, preds, title, cmap in [
    (axes[0], disc_preds, 'Generated I/Q - Frozen Discriminator Class Head After FutabaT14 Repair', 'Blues'),
    (axes[1], teacher_preds, 'Generated I/Q - Frozen 33 VGG Teacher After FutabaT14 Repair', 'Greens'),
]:
    cm = confusion_matrix(labels_np, preds, labels=np.arange(NUM_CLASSES))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('Requested generated class')
    ax.set_title(title)
fig.tight_layout()
fig.savefig(repair_diag_path, dpi=160, bbox_inches='tight')
_save_current_figure("cell_12_figure_07.png")
print('Saved repair diagnostic:', repair_diag_path)
print('Final repaired generator:', repaired_generator_path)

# %% Cell 13
# Cell 13 : Generate raw I/Q by class and evaluate with the noisy-drone classifier
# This is the end-to-end sanity check: GAN -> generated I/Q -> classifier spectrogram input -> classifier prediction.
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

EVAL_SAMPLES_PER_CLASS = int(os.getenv('NOISY_DRONE_IQ_GAN_CLASSIFIER_EVAL_SAMPLES_PER_CLASS', '32'))
EVAL_BATCH_SIZE = int(os.getenv('NOISY_DRONE_IQ_GAN_CLASSIFIER_EVAL_BATCH_SIZE', '4'))
EVAL_SEED = int(os.getenv('NOISY_DRONE_IQ_GAN_CLASSIFIER_EVAL_SEED', str(RANDOM_STATE + 12000)))

repaired_generator_path = model_dir / 'noisy_drone_rf_v2_conditional_iq_generator_teacher_aligned_repaired.keras'
classifier_candidates = [
    model_dir / 'noisy_drone_rf_v2_vgg_full_complex_spectrogram_best.keras',
    model_dir / 'noisy_drone_rf_v2_vgg_complex_spectrogram_best.keras',
    model_dir / 'noisy_drone_rf_v2_vgg_spectrogram_best.keras',
    model_dir / 'noisy_drone_rf_v2_enhanced_complex_spectrogram_cnn_best.keras',
    model_dir / 'noisy_drone_rf_v2_spectrogram_cnn_best.keras',
]

gen_eval_path = resolve_best_iq_generator_path() if 'resolve_best_iq_generator_path' in globals() else repaired_generator_path
classifier_path = next((path for path in classifier_candidates if path.exists()), classifier_candidates[0])
if not gen_eval_path.exists():
    raise FileNotFoundError(f'Missing GAN generator: {gen_eval_path}')
if not classifier_path.exists():
    raise FileNotFoundError(f'Missing noisy-drone classifier: {classifier_path}')

print('Generator:', gen_eval_path)
print('Classifier:', classifier_path)

classifier_model = load_model(classifier_path, compile=False)
gan_generator = load_model(gen_eval_path, compile=False)

classifier_input_shape = classifier_model.input_shape
if isinstance(classifier_input_shape, list):
    classifier_input_shape = classifier_input_shape[0]
classifier_input_shape = tuple(classifier_input_shape[1:])
if len(classifier_input_shape) != 3:
    raise ValueError(f'Expected image-like classifier input, got: {classifier_model.input_shape}')
SPEC_FREQ_BINS, SPEC_TIME_BINS, SPEC_CHANNELS = map(int, classifier_input_shape)
if SPEC_CHANNELS not in (1, 2, 3, 4):
    raise ValueError(f'Unsupported classifier channel count: {SPEC_CHANNELS}')

print('Classifier input shape:', classifier_input_shape)
print('Samples/class:', EVAL_SAMPLES_PER_CLASS)


def generate_iq_by_class(samples_per_class: int):
    labels_np = np.repeat(np.arange(NUM_CLASSES, dtype=np.int32), samples_per_class)
    rng = np.random.default_rng(EVAL_SEED)
    z = rng.standard_normal((len(labels_np), LATENT_DIM)).astype(np.float32)
    try:
        fake_iq = gan_generator.predict({'noise': z, 'label': labels_np}, batch_size=BATCH_SIZE, verbose=0)
    except Exception:
        fake_iq = gan_generator.predict([z, labels_np], batch_size=BATCH_SIZE, verbose=0)
    fake_iq = np.asarray(fake_iq, dtype=np.float32)
    if fake_iq.ndim != 3 or fake_iq.shape[-1] != 2:
        raise ValueError(f'Expected generated I/Q shape (N, T, 2), got {fake_iq.shape}')
    return labels_np, fake_iq


def iq_to_classifier_spectrogram_np(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 2 or iq.shape[-1] != 2:
        raise ValueError(f'Expected one I/Q sample shaped (T, 2), got {iq.shape}')

    complex_iq = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
    complex_iq = complex_iq - np.mean(complex_iq)
    rms = np.sqrt(np.mean(np.abs(complex_iq) ** 2)) + 1e-6
    complex_iq = complex_iq / rms

    nfft = SPEC_FREQ_BINS
    if len(complex_iq) < nfft:
        complex_iq = np.pad(complex_iq, (0, nfft - len(complex_iq)), mode='constant')
    max_start = max(0, len(complex_iq) - nfft)
    starts = np.linspace(0, max_start, SPEC_TIME_BINS).astype(np.int64)
    window = np.hanning(nfft).astype(np.float32)

    frames = np.empty((SPEC_TIME_BINS, nfft), dtype=np.complex64)
    for idx, start in enumerate(starts):
        frames[idx] = complex_iq[start:start + nfft] * window
    spec = np.fft.fftshift(np.fft.fft(frames, n=nfft, axis=1), axes=1).T.astype(np.complex64)

    real = np.real(spec).astype(np.float32)
    imag = np.imag(spec).astype(np.float32)
    mag = np.log1p(np.abs(spec)).astype(np.float32)
    phase = np.angle(spec).astype(np.float32) / np.pi

    if SPEC_CHANNELS == 1:
        out = mag[..., None]
    elif SPEC_CHANNELS == 2:
        out = np.stack([real, imag], axis=-1)
    elif SPEC_CHANNELS == 3:
        out = np.stack([mag, real, imag], axis=-1)
    else:
        out = np.stack([real, imag, mag, phase], axis=-1)

    out = out.astype(np.float32)
    out = out - np.mean(out, axis=(0, 1), keepdims=True)
    out = out / (np.std(out, axis=(0, 1), keepdims=True) + 1e-6)
    return np.clip(out, -6.0, 6.0).astype(np.float32)


def predict_generated_iq(fake_iq: np.ndarray) -> np.ndarray:
    probs = []
    for start in range(0, len(fake_iq), EVAL_BATCH_SIZE):
        batch_iq = fake_iq[start:start + EVAL_BATCH_SIZE]
        batch_x = np.stack([iq_to_classifier_spectrogram_np(sample) for sample in batch_iq], axis=0)
        pred = classifier_model.predict(batch_x, batch_size=EVAL_BATCH_SIZE, verbose=0)
        probs.append(np.asarray(pred, dtype=np.float32))
    return np.concatenate(probs, axis=0)


y_true_generated, generated_iq = generate_iq_by_class(EVAL_SAMPLES_PER_CLASS)
classifier_probs = predict_generated_iq(generated_iq)
y_pred_classifier = classifier_probs.argmax(axis=1)
classifier_acc = accuracy_score(y_true_generated, y_pred_classifier)
print(f'Generated I/Q classifier accuracy: {classifier_acc:.4f}')
print(classification_report(
    y_true_generated,
    y_pred_classifier,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASS_NAMES,
    zero_division=0,
))

cm = confusion_matrix(y_true_generated, y_pred_classifier, labels=np.arange(NUM_CLASSES))
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title('Generated I/Q -> Noisy Drone RF Classifier')
ax.set_xlabel('Classifier predicted class')
ax.set_ylabel('Requested generated class')
plt.tight_layout()

plot_path = outputs_dir / '63b_generated_iq_noisy_drone_classifier_confusion_matrix.png'
fig.savefig(plot_path, dpi=160, bbox_inches='tight')
_save_current_figure("cell_13_figure_08.png")

rows = []
for idx, (true_idx, pred_idx) in enumerate(zip(y_true_generated, y_pred_classifier)):
    rows.append({
        'sample_idx': idx,
        'requested_class_idx': int(true_idx),
        'requested_class': CLASS_NAMES[int(true_idx)],
        'predicted_class_idx': int(pred_idx),
        'predicted_class': CLASS_NAMES[int(pred_idx)],
        'confidence': float(classifier_probs[idx, pred_idx]),
    })
results_df = pd.DataFrame(rows)
results_path = outputs_dir / '63b_generated_iq_noisy_drone_classifier_eval.csv'
results_df.to_csv(results_path, index=False)

metrics = {
    'model': 'noisy_drone_rf_v2_conditional_iq_generator_teacher_aligned_repaired',
    'generator_path': str(gen_eval_path),
    'classifier_path': str(classifier_path),
    'samples_per_class': int(EVAL_SAMPLES_PER_CLASS),
    'accuracy': float(classifier_acc),
    'confusion_matrix': cm.tolist(),
    'class_names': CLASS_NAMES,
    'plot_path': str(plot_path),
    'results_path': str(results_path),
}
metrics_path = outputs_dir / '63b_generated_iq_noisy_drone_classifier_eval_metrics.json'
metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

print('Saved classifier eval CSV:', results_path)
print('Saved classifier eval metrics:', metrics_path)
print('Saved classifier eval plot:', plot_path)
