#!/usr/bin/env python3
"""DST — Dual-Stream Transformer baseline (Jin et al., IEEE TII 2024).

Reference
---------
Jin, B., Wu, H., Zhang, Z., Lian, Z., Zhang, X. & Du, G.
"SRDST: Effective Dynamic Gesture Recognition With Sparse Representation and
 Dual-Stream Transformers in mmWave Radar." IEEE Trans. Ind. Inform. 21,
 604-612 (2024). https://doi.org/10.1109/TII.2024.3455419

This is the Transformer-category SOTA baseline reported in the manuscript
(referred to as **DST**). It is a faithful, parameterised port of the original
research script ``revision/compute_prf1_srdst_tf212.py``; the architecture and
training recipe are unchanged so the published numbers are reproduced.

Architecture (Sec. "DST" of the reference)
-------------------------------------------
1. Dual-stream embedding: a *time* stream (token = frame, feature = channels)
   and a *channel* stream (token = channel, feature = time; the sequence is
   transposed) — each projected to ``embed_dim`` by a Dense layer.
2. Sinusoidal positional encoding on the time stream only (the channel stream
   has no sequential order).
3. ``num_layers`` Post-LN Transformer encoder blocks per stream
   (MHSA + Add&Norm + FFN(Dense-ReLU-Dense) + Add&Norm).
4. Weighted fusion: concat the two pooled stream outputs -> Dense(2, softmax)
   gives per-stream weights -> weighted sum -> classification head.

Reproduction caveats (disclose in the manuscript Methods)
----------------------------------------------------------
* **The sparse-representation (OMP) front-end of the reference is NOT
  implemented.** The original method applies orthogonal-matching-pursuit sparse
  coding per sample before the transformer; that step needs a sparsity level
  tuned to the reference's own pipeline. We feed the raw X/Y/Z-T-D Doppler maps
  directly. This is a deliberate simplification and a known limitation.
* The reference does not report exact layer sizes; hyperparameters were chosen
  to roughly match the reported ~0.17 M parameter budget:
  ``embed_dim=64, num_heads=4, num_layers=2, ffn_dim=128, dropout=0.1``.
* Input adapted to our ``(100, 30, 3)`` spectra (flattened to a length-100
  sequence of 90 features); the reference's 6:2:2 split is replaced by our
  paper-main protocol (9:1 train/val on ``hw``; ``jh``/``ys`` cross-user test).
* Optimizer follows the same recipe used for the other Transformer baselines:
  AdamW (weight_decay=0.05) with a 5-epoch linear warmup + cosine decay
  schedule (base lr 1e-3), 100 epochs, batch size 32, train seed 123.

Usage
-----
    python srdst.py --data-root <root> --out-dir runs/SRDST
    #   <root> must contain dataset_hw_timeshift/sliding_doppler (train)
    #   and dataset_<user>/sliding_doppler for user in {hw, jh, ys} (test)

The model builder is importable for efficiency measurement:
    from srdst import build_srdst         # build_dst is an alias
"""
import os
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

import argparse
import random
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

TRAIN_SEED = 123
N_CLASSES = 6
DATA_TYPE = 'sliding_doppler'
TRAIN_USER = 'hw'
TEST_USERS = ['hw', 'jh', 'ys']


# =====================================================================
# DST architecture — Dual-Stream Transformer
# =====================================================================
class PositionalEncoding(layers.Layer):
    """Standard sinusoidal positional encoding (time stream only)."""

    def __init__(self, seq_len, embed_dim, **kw):
        super().__init__(**kw)
        pos = np.arange(seq_len)[:, None].astype(np.float32)
        i = np.arange(embed_dim)[None, :].astype(np.float32)
        angle = pos / np.power(10000.0, (2 * (i // 2)) / embed_dim)
        pe = np.zeros((seq_len, embed_dim), dtype=np.float32)
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        self.pe = tf.constant(pe[None, ...])  # (1, seq_len, embed_dim)

    def call(self, x):
        return x + self.pe


class TransformerEncoderBlock(layers.Layer):
    """Post-LN encoder: MHSA + Add&Norm + FFN(Dense-ReLU-Dense) + Add&Norm."""

    def __init__(self, embed_dim, num_heads, ffn_dim, drop=0.1, **kw):
        super().__init__(**kw)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=embed_dim // num_heads,
                                             dropout=drop)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ffn_dim, activation='relu'),
            layers.Dropout(drop),
            layers.Dense(embed_dim),
        ])
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(drop)

    def call(self, x, training=False):
        attn = self.mha(x, x, x, training=training)
        x = self.norm1(x + self.drop(attn, training=training))
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + self.drop(ffn_out, training=training))
        return x


def build_srdst(input_shape=(100, 30, 3), num_classes=6,
                embed_dim=64, num_heads=4, num_layers=2, ffn_dim=128, drop=0.1):
    """Dual-Stream Transformer (DST) per Jin et al. 2024."""
    inp = layers.Input(shape=input_shape, name='input')

    # Flatten last 2 dims: (100, 30, 3) -> (100, 90) for the time stream.
    T, H, W = input_shape  # T=time, H=Doppler bins, W=axes
    seq = layers.Reshape((T, H * W), name='to_seq')(inp)  # (B, 100, 90)

    # ---- Time stream: token = time, feature = channels ----
    yt = layers.Dense(embed_dim, name='embed_T')(seq)
    yt = PositionalEncoding(T, embed_dim, name='pe_T')(yt)
    yt = layers.Dropout(drop)(yt)
    for i in range(num_layers):
        yt = TransformerEncoderBlock(embed_dim, num_heads, ffn_dim, drop,
                                     name=f'enc_T{i}')(yt)
    yt = layers.GlobalAveragePooling1D(name='gap_T')(yt)

    # ---- Channel stream: token = channel, feature = time (transposed) ----
    seq_C = layers.Permute((2, 1), name='transpose')(seq)  # (B, 90, 100)
    yc = layers.Dense(embed_dim, name='embed_C')(seq_C)
    # No positional encoding (channel axis has no sequential order).
    yc = layers.Dropout(drop)(yc)
    for i in range(num_layers):
        yc = TransformerEncoderBlock(embed_dim, num_heads, ffn_dim, drop,
                                     name=f'enc_C{i}')(yc)
    yc = layers.GlobalAveragePooling1D(name='gap_C')(yc)

    # ---- Weighted fusion ----
    concat = layers.Concatenate(name='concat_TC')([yt, yc])
    weights = layers.Dense(2, activation='softmax', name='fusion_w')(concat)
    stacked = tf.stack([yt, yc], axis=1)  # (B, 2, embed_dim)
    weights_exp = layers.Reshape((2, 1), name='w_reshape')(weights)
    fused = layers.Multiply(name='weighted')([stacked, weights_exp])
    fused = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1),
                          name='sum_streams')(fused)

    out = layers.Dense(num_classes, activation='softmax', name='head')(fused)
    return tf.keras.Model(inp, out, name='DST')


# Alias matching the manuscript label.
build_dst = build_srdst


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """5-epoch linear warmup followed by cosine decay."""

    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * step / tf.maximum(1.0, float(self.warmup_steps))
        decay_steps = float(max(1, self.total_steps - self.warmup_steps))
        progress = (step - self.warmup_steps) / decay_steps
        cosine_lr = 0.5 * self.base_lr * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return dict(base_lr=self.base_lr, warmup_steps=self.warmup_steps,
                    total_steps=self.total_steps)


# =====================================================================
# Data loaders (paper-main layout)
# =====================================================================
def _set_global_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _load_npy_files(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith('.npy'))


def load_trainval(data_root):
    """Load the time-shift-augmented training subject (500 recordings/class)."""
    directory = os.path.join(data_root, f'dataset_{TRAIN_USER}_timeshift', DATA_TYPE)
    files = _load_npy_files(directory)
    X = np.zeros((6 * 500, 100, 30, 3))
    y = np.zeros((6 * 500,), dtype=int)
    for i in range(6):
        data = np.load(os.path.join(directory, files[i]))
        X[500 * i:500 * (i + 1)] = data[:, :, :, :, :3].reshape(500, 100, 30, 3)
        y[500 * i:500 * (i + 1)] = i
    return X, y


def load_test(data_root, user):
    """Load all 6 classes of one test subject (50/class for hw, else 30)."""
    test_dir = os.path.join(data_root, f'dataset_{user}', DATA_TYPE)
    files = _load_npy_files(test_dir)
    numdata = 50 if user == 'hw' else 30
    X_te = np.zeros((6 * numdata * 10, 100, 30, 3))
    y_te = np.zeros((6 * numdata * 10,), dtype=int)
    for i in range(6):
        data = np.load(os.path.join(test_dir, files[i]))
        reshaped = data[:, :, :, :, :3].reshape(numdata * 10, 100, 30, 3)
        X_te[numdata * 10 * i: numdata * 10 * (i + 1)] = reshaped
        y_te[numdata * 10 * i: numdata * 10 * (i + 1)] = i
    return X_te, y_te


# =====================================================================
# 15-split cross-subject training / evaluation
# =====================================================================
def run(data_root, out_dir, n_splits=15, epochs=100, batch_size=32):
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'srdst_prf1_peruser.csv')
    summary_path = os.path.join(out_dir, 'srdst_prf1_peruser_summary.csv')
    pred_path = os.path.join(out_dir, 'srdst_predictions_peruser.npz')

    print('=' * 80)
    print('DST [Jin TII 2024] — Dual-Stream Transformer for radar gesture recognition')
    print(f'n_splits={n_splits}  epochs={epochs}  TRAIN_SEED={TRAIN_SEED}')
    print('=' * 80, flush=True)

    smoke = build_srdst((100, 30, 3), N_CLASSES)
    print(f'[smoke] total params: {smoke.count_params():,}  (paper target ~0.17M)')

    X_all, y_all = load_trainval(data_root)
    print(f'[data] X_all={X_all.shape}')
    test_data, test_denom = {}, {}
    for u in TEST_USERS:
        X_te, y_te = load_test(data_root, u)
        test_data[u] = (X_te, y_te)
        test_denom[u] = float(np.max(np.abs(X_te))) or 1.0
        print(f'  test_{u}: {X_te.shape}  denom={test_denom[u]:.3f}')

    all_rows, preds_storage = [], {}
    t0 = time.time()

    for split_seed in range(n_splits):
        t_s = time.time()
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_all, y_all, test_size=0.1, random_state=split_seed, stratify=y_all)
        denom = float(np.max(np.abs(X_tr))) or 1.0
        X_tr_n, X_va_n = X_tr / denom, X_va / denom

        _set_global_seed(TRAIN_SEED)
        model = build_srdst((100, 30, 3), N_CLASSES)

        steps_per_epoch = int(np.ceil(X_tr.shape[0] / batch_size))
        schedule = WarmupCosineSchedule(1e-3, 5 * steps_per_epoch,
                                        epochs * steps_per_epoch)
        try:
            from tensorflow.keras.optimizers.experimental import AdamW
        except ImportError:
            from tensorflow.keras.optimizers import AdamW
        optimizer = AdamW(learning_rate=schedule, weight_decay=0.05)

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_tr_n, y_tr, validation_data=(X_va_n, y_va),
                  epochs=epochs, batch_size=batch_size, verbose=0)

        msg = []
        for user in TEST_USERS:
            X_te, y_te = test_data[user]
            X_te_n = X_te / test_denom[user]
            y_pred = np.argmax(model.predict(X_te_n, batch_size=64, verbose=0), axis=1)
            preds_storage[f'DST_{split_seed}_{user}'] = y_pred.astype(np.int8)

            acc = float((y_pred == y_te).mean())
            row = {'model': 'DST', 'split_seed': split_seed, 'user': user,
                   'accuracy': acc,
                   'precision_macro': precision_score(y_te, y_pred, average='macro', zero_division=0),
                   'recall_macro': recall_score(y_te, y_pred, average='macro', zero_division=0),
                   'f1_macro': f1_score(y_te, y_pred, average='macro', zero_division=0),
                   'precision_weighted': precision_score(y_te, y_pred, average='weighted', zero_division=0),
                   'recall_weighted': recall_score(y_te, y_pred, average='weighted', zero_division=0),
                   'f1_weighted': f1_score(y_te, y_pred, average='weighted', zero_division=0)}
            pp = precision_score(y_te, y_pred, average=None, zero_division=0)
            rp = recall_score(y_te, y_pred, average=None, zero_division=0)
            fp = f1_score(y_te, y_pred, average=None, zero_division=0)
            for c in range(N_CLASSES):
                row[f'precision_c{c}'], row[f'recall_c{c}'], row[f'f1_c{c}'] = pp[c], rp[c], fp[c]
            all_rows.append(row)
            msg.append(f'{user}={acc:.4f}')

        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        np.savez_compressed(pred_path, **preds_storage)
        del model
        tf.keras.backend.clear_session()
        elapsed = time.time() - t0
        eta = elapsed / (split_seed + 1) * (n_splits - split_seed - 1)
        print(f'  [split {split_seed+1:2d}/{n_splits}] ' + '  '.join(msg) +
              f'  | this={time.time()-t_s:.0f}s  ETA={eta/60:.1f}min', flush=True)

    # ---- Summary (per-user + cross-user jh+ys) ----
    df = pd.read_csv(csv_path)
    metric_cols = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                   'precision_weighted', 'recall_weighted', 'f1_weighted']
    summary = []
    for u in TEST_USERS:
        sub = df[df['user'] == u]
        row = {'model': 'DST', 'user': u, 'n_splits': len(sub)}
        for col in metric_cols:
            row[f'{col}_mean'], row[f'{col}_std'] = sub[col].mean(), sub[col].std()
        summary.append(row)
    sub = df[df['user'].isin(['jh', 'ys'])]
    cu = sub.groupby('split_seed')[metric_cols].mean()
    row = {'model': 'DST', 'user': 'cross_user(jh+ys)', 'n_splits': len(cu)}
    for col in metric_cols:
        row[f'{col}_mean'], row[f'{col}_std'] = cu[col].mean(), cu[col].std()
    summary.append(row)
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    print('\n' + '=' * 80 + '\nSUMMARY — DST\n' + '=' * 80)
    for u in TEST_USERS + ['cross_user(jh+ys)']:
        s = next((r for r in summary if r['user'] == u), None)
        if s:
            print(f'  {u:25s}  Acc={s["accuracy_mean"]*100:.2f}±{s["accuracy_std"]*100:.2f}  '
                  f'F1={s["f1_macro_mean"]*100:.2f}±{s["f1_macro_std"]*100:.2f}')
    print(f'\nTotal: {(time.time()-t0)/60:.1f} min\nCSV: {csv_path}')


def _parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', default=os.environ.get('STPP_DATA_ROOT', '.'),
                    help='dir holding dataset_<user>[_timeshift]/<representation>/ '
                         '(default: $STPP_DATA_ROOT or current dir)')
    ap.add_argument('--out-dir', default='runs/SRDST')
    ap.add_argument('--n-splits', type=int, default=15)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=32)
    return ap


if __name__ == '__main__':
    a = _parser().parse_args()
    run(a.data_root, a.out_dir, n_splits=a.n_splits, epochs=a.epochs,
        batch_size=a.batch_size)
