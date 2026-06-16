"""Doppler-based Spatial-Temporal Point Processing (STPP).

Core preprocessing of the manuscript
    "In-cabin Hand Gesture Recognition Using Doppler-based Spatial Temporal
     Point Processing".

This module converts a raw mmWave-radar **point-cloud recording** (one gesture
instance, stored as a RETINA POINT CLOUD JSON file) into three 2-D
spatial-temporal spectra — XTD, YTD, ZTD — that are stacked along the channel
axis to form a single ``(spatial_bins, time_frames, 3)`` tensor.

Algorithm (per gesture file)
----------------------------
For every frame ``t`` and every detected point ``p`` with cartesian position
``(x, y, z)``, Doppler velocity ``v`` and power ``P``:

1. Each spatial axis is discretised into ``BINS`` cells:

       bin = round(coord / delta_axis) + offset_axis

2. If the bin index falls inside ``[0, BINS)`` the point contributes its value
   to cell ``(bin, t)``. Three representations are accumulated independently:

   * ``count``   – number of points falling in the cell (point density)
   * ``power``   – sum of point powers ``P``           (energy)
   * ``doppler`` – sum of point Doppler velocities ``v`` (the proposed feature)

3. The three per-axis maps are stacked → ``(BINS, T, 3)`` with channels
   ``(XTD, YTD, ZTD)``.

The Doppler representation is the one used for the main results; ``count`` and
``power`` are provided for the input-ablation study (Sec. "Data augmentation").

This is a faithful, parameterised refactor of the original research script
``gesture_datagen_260510_powerDoppler.py``; the binning logic is unchanged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Canonical gesture classes and their integer labels (alphabetical-free, fixed
# order matching the manuscript: left, right, up, down, push, no_gesture).
CLASSES: Tuple[str, ...] = ("left", "right", "up", "down", "push", "no_gesture")

REPRESENTATIONS: Tuple[str, ...] = ("count", "power", "doppler")


@dataclass(frozen=True)
class BinningConfig:
    """Spatial-binning parameters (values from the original research script).

    ``delta_*`` is the metric width of one bin along each axis and ``offset_*``
    re-centres the binned coordinate so that the gesture region lands inside
    ``[0, bins)``. With the defaults below every axis is mapped onto ``bins``
    cells covering roughly: X∈[-0.5, 0.5] m, Y∈[0, 0.6] m, Z∈[-0.8, 0.0] m.
    """

    bins: int = 100
    delta_x: float = 0.01
    delta_y: float = 0.006
    delta_z: float = 0.008
    x_offset: int = 50
    y_offset: int = 0
    z_offset: int = 80
    # Time window [time_start, time_end) extracted from each recording.
    # The manuscript extracts the gesture region from frames 10 to 50 (40
    # frames). These defaults reproduce the published .npy spectra exactly
    # (verified element-wise against dataset_hw/normal_doppler).
    time_start: int = 10
    time_end: int = 50

    @property
    def time_frames(self) -> int:
        return self.time_end - self.time_start


def _read_pointcloud_json(path: str) -> List[dict]:
    """Read a RETINA POINT CLOUD JSON file.

    The first line is a metadata header; every subsequent line is one frame
    with keys ``C`` (flattened xyz), ``V`` (Doppler velocity) and ``P`` (power).
    Returns the list of frame dicts (header excluded).
    """
    frames: List[dict] = []
    with open(path, "r") as f:
        header_seen = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not header_seen:
                header_seen = True  # drop the "FILE TYPE" metadata line
                continue
            frames.append(obj)
    return frames


def file_to_spectra(path: str, cfg: BinningConfig = BinningConfig()
                    ) -> Dict[str, np.ndarray]:
    """Convert one gesture recording into XTD/YTD/ZTD spectra.

    Returns a dict ``{representation: array(bins, T, 3)}`` for
    ``representation`` in ``{count, power, doppler}``. If a recording has fewer
    than ``time_end`` frames the available frames are used and the map is
    zero-padded implicitly (columns beyond the recording stay zero); callers
    that require an exact frame count should check ``spectra['count'].shape``.
    """
    frames = _read_pointcloud_json(path)
    n_frames = len(frames)
    b = cfg.bins

    # Per-axis accumulator maps: (bins, n_frames) for each representation.
    maps = {rep: {ax: np.zeros((b, n_frames), dtype=np.float64)
                  for ax in ("x", "y", "z")} for rep in REPRESENTATIONS}

    deltas = {"x": cfg.delta_x, "y": cfg.delta_y, "z": cfg.delta_z}
    offsets = {"x": cfg.x_offset, "y": cfg.y_offset, "z": cfg.z_offset}

    for t, frame in enumerate(frames):
        xyz = np.asarray(frame["C"], dtype=np.float64).reshape(-1, 3)
        v = np.asarray(frame["V"], dtype=np.float64)   # Doppler
        p = np.asarray(frame["P"], dtype=np.float64)   # power
        coords = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}

        for ax in ("x", "y", "z"):
            idx = np.rint(coords[ax] / deltas[ax]).astype(int) + offsets[ax]
            valid = (idx >= 0) & (idx < b)
            iv = idx[valid]
            # np.add.at performs an unbuffered scatter-add (handles repeats).
            np.add.at(maps["count"][ax][:, t], iv, 1.0)
            np.add.at(maps["power"][ax][:, t], iv, p[valid])
            np.add.at(maps["doppler"][ax][:, t], iv, v[valid])

    t0, t1 = cfg.time_start, min(cfg.time_end, n_frames)
    out: Dict[str, np.ndarray] = {}
    for rep in REPRESENTATIONS:
        x_map = maps[rep]["x"][:, t0:t1]
        y_map = maps[rep]["y"][:, t0:t1]
        z_map = maps[rep]["z"][:, t0:t1]
        out[rep] = np.stack((x_map, y_map, z_map), axis=2)  # (bins, T, 3)
    return out


def generate_dataset(data_root: str, save_root: str, user_tag: str,
                     representations: Tuple[str, ...] = REPRESENTATIONS,
                     cfg: BinningConfig = BinningConfig(),
                     strict_frames: bool = True) -> None:
    """Convert every recording under ``data_root`` into ``.npy`` spectra.

    Expected layout::

        data_root/<class>/*.json          (one JSON per gesture instance)

    Produces, for each class and representation::

        save_root/<class>/<class>_<representation>_<user_tag>.npy
        with shape (n_samples, bins, time_frames, 3)

    Parameters
    ----------
    user_tag : str
        Suffix identifying the subject (e.g. ``hw``, ``jh``, ``ys``).
    strict_frames : bool
        If True, recordings whose extracted window length differs from
        ``cfg.time_frames`` are skipped (matching the original script).
    """
    for cls in CLASSES:
        cls_dir = os.path.join(data_root, cls)
        if not os.path.isdir(cls_dir):
            print(f"[skip] missing class dir: {cls_dir}")
            continue
        files = sorted(f for f in os.listdir(cls_dir) if f.endswith(".json"))

        buckets: Dict[str, List[np.ndarray]] = {r: [] for r in representations}
        for fn in files:
            spectra = file_to_spectra(os.path.join(cls_dir, fn), cfg)
            T = spectra[representations[0]].shape[1]
            if strict_frames and T != cfg.time_frames:
                print(f"[warn] {cls}/{fn}: {T} frames != {cfg.time_frames}; skipped")
                continue
            for rep in representations:
                buckets[rep].append(spectra[rep])

        out_dir = os.path.join(save_root, cls)
        os.makedirs(out_dir, exist_ok=True)
        for rep in representations:
            if not buckets[rep]:
                continue
            arr = np.stack(buckets[rep], axis=0)  # (n, bins, T, 3)
            out_path = os.path.join(out_dir, f"{cls}_{rep}_{user_tag}.npy")
            np.save(out_path, arr)
            print(f"[saved] {out_path}  shape={arr.shape}")


def _build_arg_parser():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True,
                    help="dir containing <class>/*.json point-cloud recordings")
    ap.add_argument("--save-root", required=True, help="output dir for .npy")
    ap.add_argument("--user-tag", required=True, help="subject id, e.g. hw/jh/ys")
    ap.add_argument("--representations", nargs="+", default=list(REPRESENTATIONS),
                    choices=REPRESENTATIONS)
    ap.add_argument("--time-start", type=int, default=BinningConfig.time_start,
                    help="first frame of the extracted gesture window")
    ap.add_argument("--time-end", type=int, default=BinningConfig.time_end,
                    help="last frame (exclusive) of the extracted gesture window")
    return ap


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    config = BinningConfig(time_start=args.time_start, time_end=args.time_end)
    generate_dataset(args.data_root, args.save_root, args.user_tag,
                     tuple(args.representations), config)
