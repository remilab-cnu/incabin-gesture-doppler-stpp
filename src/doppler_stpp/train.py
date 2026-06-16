"""Cross-subject training with the paper's 15-split protocol.

Protocol (manuscript, "Experiments"):
  * Train subject: 'hw' (the sliding-window, time-shift-augmented set).
  * 15 independent train/validation splits (random_state 0..14, val = 10%).
  * Optimizer Adam (lr 1e-3 default), sparse categorical cross-entropy.
  * 100 epochs, batch size 32, best-val-loss checkpoint per split.
  * Untrained subjects 'jh' (User B) and 'ys' (User C) are NEVER seen in
    training and are used only for cross-subject evaluation (see evaluate.py).

Determinism: each split sets a fixed TRAIN_SEED for model init/training so the
runs differ only in the data split, isolating split variance.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers

from .data import load_subject, normalize
from .models import build_model

TRAIN_SEED = 123


def train_one_split(model_name, X, y, split_seed, epochs=100, batch_size=32,
                    lr=1e-3, ckpt_path=None, input_shape=(100, 30, 3),
                    num_classes=6, verbose=2):
    """Train a single model on one train/val split; return (model, history)."""
    tf.keras.utils.set_random_seed(TRAIN_SEED)
    x_tr, x_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, random_state=split_seed, stratify=y)

    model = build_model(model_name, input_shape=input_shape, num_classes=num_classes)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = []
    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True, mode="min",
            save_weights_only=True))

    history = model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_val, y_val), callbacks=callbacks,
                        verbose=verbose)
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)
    return model, history


def run(model_name, train_dir, out_dir, n_splits=15, n_recordings=50,
        epochs=100, batch_size=32, lr=1e-3, input_shape=(100, 30, 3),
        num_classes=6):
    """Train ``n_splits`` models and save their weights to ``out_dir``."""
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    X, y = load_subject(train_dir, n_recordings=n_recordings,
                        input_shape=input_shape)
    X, scale = normalize(X)
    print(f"[data] X={X.shape} y={y.shape} norm_scale={scale:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    for s in range(n_splits):
        ckpt = os.path.join(out_dir, f"{model_name}_split{s}.weights.h5")
        print(f"\n=== {model_name} | split {s+1}/{n_splits} ===")
        train_one_split(model_name, X, y, split_seed=s, epochs=epochs,
                        batch_size=batch_size, lr=lr, ckpt_path=ckpt,
                        input_shape=input_shape, num_classes=num_classes)
    # persist the normalisation scale for evaluation reuse if desired
    np.save(os.path.join(out_dir, "norm_scale.npy"), np.array(scale))
    print(f"\n[done] weights + norm_scale saved to {out_dir}")


def _parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-dir", required=True,
                    help="e.g. dataset_hw_timeshift/sliding_doppler")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-splits", type=int, default=15)
    ap.add_argument("--n-recordings", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    return ap


if __name__ == "__main__":
    a = _parser().parse_args()
    run(a.model, a.train_dir, a.out_dir, n_splits=a.n_splits,
        n_recordings=a.n_recordings, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr)
