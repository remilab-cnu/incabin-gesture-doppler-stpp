"""Dataset assembly for training and cross-subject evaluation.

The on-disk layout (per subject) produced by the preprocessing step is::

    dataset_<user>/<representation>/<class>_<...>.npy

Each ``.npy`` holds the sliding-window samples for one class:
    shape (n_samples, n_windows, bins, window, 3)   e.g. (50, 10, 100, 30, 3)

These are flattened over the window axis so every window becomes an independent
training/test sample of shape (bins, window, 3) = (100, 30, 3).
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

CLASS_LABELS = {0: "left", 1: "right", 2: "up", 3: "down", 4: "push",
                5: "no_gesture"}
N_CLASSES = len(CLASS_LABELS)
N_WINDOWS = 10  # sliding windows per recording (see preprocessing.sliding_window)


def load_npy_files(directory: str) -> List[str]:
    """Sorted list of .npy filenames in ``directory`` (class order is the sort)."""
    return sorted(f for f in os.listdir(directory) if f.endswith(".npy"))


def load_subject(data_dir: str, n_recordings: int,
                 input_shape: Tuple[int, int, int] = (100, 30, 3)
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Load all 6 classes for one subject into (X, y).

    Parameters
    ----------
    data_dir : str
        e.g. ``dataset_hw/sliding_doppler``.
    n_recordings : int
        Recordings per class for this subject (50 for the train subject 'hw',
        30 for untrained subjects 'jh'/'ys' in the paper's protocol).

    Returns
    -------
    X : (6 * n_recordings * N_WINDOWS, *input_shape)
    y : (6 * n_recordings * N_WINDOWS,)  int labels
    """
    files = load_npy_files(data_dir)
    if len(files) < N_CLASSES:
        raise FileNotFoundError(f"expected >= {N_CLASSES} npy in {data_dir}, "
                                f"found {len(files)}")
    per_class = n_recordings * N_WINDOWS
    X = np.zeros((N_CLASSES * per_class, *input_shape), dtype=np.float64)
    y = np.zeros((N_CLASSES * per_class,), dtype=int)
    for i in range(N_CLASSES):
        data = np.load(os.path.join(data_dir, files[i]))
        X[per_class * i: per_class * (i + 1)] = data.reshape(per_class, *input_shape)
        y[per_class * i: per_class * (i + 1)] = i
    return X, y


def normalize(x: np.ndarray, scale: float | None = None
              ) -> Tuple[np.ndarray, float]:
    """Global max-abs normalisation (the scheme used in the paper notebooks).

    Returns the normalised array and the scale used so the identical scale can
    be reused on other splits if desired.
    """
    if scale is None:
        scale = float(np.max(np.abs(x)))
    if scale == 0:
        return x, 1.0
    return x / scale, scale
