"""Pre-load all JSON point cloud files into a single .npz cache so subsequent
training runs don't have to re-parse JSON each time.

Run once: python -m data.cache  → writes data/pcd_cache_G{G}_F{F}_N{N}_{suffix}.npz

The G prefix encodes frame_group (number of raw frames merged into one super-frame,
per Li et al. 2025 reference). Old caches (G omitted, group=1) are still supported.
"""
import os
import numpy as np

from .pcd_dataset import CLASSES, list_files, load_split

# Raw point-cloud root (pcd_<subject>/<class>/*.json). Obtain from the data
# record and point STPP_PCD_ROOT at it, e.g.
#   STPP_PCD_ROOT=/path/to/pointcloud python -m data.cache
DEFAULT_ROOT = os.environ.get('STPP_PCD_ROOT', 'pointcloud')


def cache_path(n_frames, n_points, normalize, frame_group=1):
    suffix = 'norm' if normalize else 'raw'
    here = os.path.dirname(__file__)
    if frame_group is None or frame_group <= 1:
        return os.path.join(here, f'pcd_cache_F{n_frames}_N{n_points}_{suffix}.npz')
    return os.path.join(
        here, f'pcd_cache_G{frame_group}_F{n_frames}_N{n_points}_{suffix}.npz'
    )


def build_cache(pointcloud_root=DEFAULT_ROOT, n_frames=32, n_points=64,
                frame_group=1, normalize=True):
    path = cache_path(n_frames, n_points, normalize, frame_group=frame_group)
    if os.path.exists(path):
        print(f'Cache already exists: {path}')
        return path
    files = list_files(pointcloud_root)
    print(f'Loading {len(files)} files into cache (G={frame_group}, F={n_frames}, N={n_points})...')
    X, y, users = load_split(
        files, n_frames=n_frames, n_points=n_points, frame_group=frame_group,
        normalize=normalize, verbose=True,
    )
    paths = np.array([p for _, _, p in files])
    np.savez_compressed(path, X=X, y=y, users=users.astype(str), paths=paths)
    print(f'Saved cache: {path}  ({os.path.getsize(path) / 1e6:.1f} MB)')
    return path


def load_cache(n_frames=32, n_points=64, frame_group=1, normalize=True):
    path = cache_path(n_frames, n_points, normalize, frame_group=frame_group)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Cache not found: {path}. Run `python -m data.cache` first.')
    d = np.load(path, allow_pickle=True)
    return d['X'], d['y'], d['users'], d['paths']


if __name__ == '__main__':
    build_cache(n_frames=32, n_points=64, frame_group=1, normalize=True)
