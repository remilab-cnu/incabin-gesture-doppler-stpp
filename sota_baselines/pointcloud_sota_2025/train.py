"""Common training routine for MSFE-GAM-SPointNet (and SequentialPointNet baseline).

Follows the paper's recipe: AdamW + label smoothing + cosine annealing, 100 epochs, batch 32.
"""
import os
import time

import numpy as np
import tensorflow as tf

from msfe_gam_spointnet import build_msfe_gam_spointnet, build_sequentialpointnet_baseline


def make_model(name, n_classes, n_frames=32, n_points=32, n_features=4):
    common = dict(
        n_frames=n_frames, n_points=n_points,
        n_features=n_features, n_classes=n_classes,
    )
    if name == 'msfe_gam_spointnet':
        return build_msfe_gam_spointnet(**common)
    if name == 'sequentialpointnet':
        return build_sequentialpointnet_baseline(**common)
    if name == 'msfe_only':
        return build_msfe_gam_spointnet(use_msfe=True, use_gam=False, use_separable_mlp=False, **common)
    if name == 'gam_only':
        return build_msfe_gam_spointnet(use_msfe=False, use_gam=True, use_separable_mlp=False, **common)
    raise ValueError(f'Unknown model name: {name}')


def cosine_schedule(initial_lr, total_steps, min_lr=1e-6):
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        alpha=min_lr / initial_lr,
    )


def train_one(model, X_train, y_train, X_val, y_val,
              epochs=100, batch_size=32, lr=1e-4, weight_decay=1e-4,
              label_smoothing=0.1, verbose=1, seed=123):
    tf.keras.utils.set_random_seed(seed)
    steps_per_epoch = max(1, int(np.ceil(len(X_train) / batch_size)))
    total_steps = steps_per_epoch * epochs
    schedule = cosine_schedule(lr, total_steps)
    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=weight_decay)
    except AttributeError:
        opt = tf.keras.optimizers.experimental.AdamW(
            learning_rate=schedule, weight_decay=weight_decay
        )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, label_smoothing=label_smoothing
    ) if 'label_smoothing' in tf.keras.losses.SparseCategoricalCrossentropy.__init__.__code__.co_varnames else (
        tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=label_smoothing)
    )

    # SparseCategoricalCrossentropy in TF 2.21 supports label_smoothing only via Categorical.
    # Wrap labels as one-hot to use Categorical with label_smoothing.
    n_classes = int(model.output.shape[-1])
    y_train_oh = tf.keras.utils.to_categorical(y_train, n_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, n_classes)
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, label_smoothing=label_smoothing
    )
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    t0 = time.time()
    hist = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=epochs, batch_size=batch_size,
        verbose=verbose, shuffle=True,
    )
    elapsed = time.time() - t0
    val_acc = float(hist.history['val_accuracy'][-1])
    val_loss = float(hist.history['val_loss'][-1])
    return {
        'val_accuracy': val_acc,
        'val_loss': val_loss,
        'elapsed_sec': elapsed,
        'history': {k: list(v) for k, v in hist.history.items()},
    }


def evaluate(model, X, y, batch_size=64):
    preds = model.predict(X, batch_size=batch_size, verbose=0)
    pred_cls = preds.argmax(axis=1)
    acc = float((pred_cls == y).mean())
    return acc, pred_cls, preds
