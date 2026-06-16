"""Sliding-window temporal augmentation.

Section "Data augmentation" of the manuscript: a sliding window of width
``window`` frames is slid along the time axis of the original ``(bins, T, 3)``
spectrum with ``stride`` = 1 frame, turning each recording into several
temporally-shifted samples. This preserves the overall gesture pattern while
injecting temporal variation, increasing the number of training samples ~10x
(300 -> 3000) and improving generalisation.

Input  : original spectra of shape (n, bins, T, 3)         e.g. (50, 100, 40, 3)
Output : windowed spectra of shape (n, n_win, bins, window, 3)
                                                            e.g. (50, 10, 100, 30, 3)
"""
from __future__ import annotations

import numpy as np


def sliding_windows(spectra: np.ndarray, window: int = 30, stride: int = 1,
                    max_windows: int | None = 10) -> np.ndarray:
    """Slide a temporal window over the time axis (axis=2) of ``spectra``.

    Parameters
    ----------
    spectra : np.ndarray
        Original spectra, shape ``(n, bins, T, 3)``.
    window : int
        Window width in frames (paper: 30).
    stride : int
        Hop between consecutive windows in frames (paper: 1).
    max_windows : int | None
        Keep at most this many windows per sample (paper uses 10). With
        ``T=40, window=30, stride=1`` there are 11 possible windows; the
        published dataset keeps the first 10. Set to ``None`` to keep all.

    Returns
    -------
    np.ndarray
        Shape ``(n, n_win, bins, window, 3)``.
    """
    if spectra.ndim != 4:
        raise ValueError(f"expected (n, bins, T, 3), got {spectra.shape}")
    n, bins, T, ch = spectra.shape
    if window > T:
        raise ValueError(f"window {window} > available frames {T}")

    starts = list(range(0, T - window + 1, stride))
    if max_windows is not None:
        starts = starts[:max_windows]

    windows = [spectra[:, :, s:s + window, :] for s in starts]
    # stack along new axis=1 -> (n, n_win, bins, window, ch)
    return np.stack(windows, axis=1)


def concat_channels(windowed: np.ndarray) -> np.ndarray:
    """Concatenate the 3 axis-channels along the time axis (CDS representation).

    (n, n_win, bins, window, 3) -> (n, n_win, bins, window*3)

    Implements the "Concatenated Doppler-based Spectra (CDS)" variant used in
    the input-configuration ablation, where axis-specific maps are concatenated
    rather than stacked as channels.
    """
    n, n_win, bins, w, ch = windowed.shape
    # move channel next to time then flatten (window, ch) -> window*ch
    out = np.transpose(windowed, (0, 1, 2, 3, 4)).reshape(n, n_win, bins, w * ch)
    return out
