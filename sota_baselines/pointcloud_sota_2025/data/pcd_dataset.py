"""Point cloud dataset loader for MSFE-GAM-SPointNet and other point cloud SOTA models.

Input: JSON files under revision/pointcloud/pcd_{hw,jh,ys}_user{A,B,C}/{class}/*.json
Each JSON line is one frame with keys: T (timestamp), C (xyz flat), V (Doppler vel), P (power), TID, TRK.

Output: tensors of shape (F, N, 4) per sample, where
  F = fixed number of frames (uniform-sampled along time)
  N = fixed number of points per frame (random sample or zero-pad)
  4 = (x, y, z, v)
"""
import json
import os
import numpy as np

USERS = {'hw': 'pcd_hw_userA', 'jh': 'pcd_jh_userB', 'ys': 'pcd_ys_userC'}
CLASSES = ['down', 'left', 'no_gesture', 'push', 'right', 'up']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)


def _parse_frame(line):
    obj = json.loads(line)
    C = np.asarray(obj['C'], dtype=np.float32)
    V = np.asarray(obj['V'], dtype=np.float32)
    n_pts = len(C) // 3
    if n_pts == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xyz = C[: n_pts * 3].reshape(n_pts, 3)
    v = V[:n_pts].reshape(n_pts, 1)
    return np.concatenate([xyz, v], axis=1)


def load_file(path, n_frames=32, n_points=64, frame_group=1, rng=None):
    """Load one JSON file → array shape (n_frames, n_points, 4).

    Following Li et al. 2025 (Electronics 14(2), 371) Section 2.3:
    "three frames of point clouds are combined into one frame, reducing the
    number of frames and increasing the number of point clouds in each frame."

    Pipeline:
      1. Group every `frame_group` consecutive raw frames; concat their points
         into one super-frame. F_raw=54..79 → n_groups ≈ 18..27 when group=3.
      2. Uniform-sample n_frames super-frames along time.
      3. For each chosen super-frame, sample n_points (or zero-pad if fewer).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    with open(path) as f:
        lines = f.readlines()
    frames_raw = [_parse_frame(line) for line in lines[1:]]
    F_raw = len(frames_raw)
    if F_raw == 0:
        return np.zeros((n_frames, n_points, 4), dtype=np.float32)

    # Step 1: group `frame_group` consecutive frames into a single super-frame
    if frame_group is None or frame_group <= 1:
        groups_raw = frames_raw
    else:
        groups_raw = []
        for s in range(0, F_raw, frame_group):
            pts_list = [frames_raw[s + j] for j in range(frame_group) if s + j < F_raw]
            non_empty = [p for p in pts_list if p.shape[0] > 0]
            if non_empty:
                groups_raw.append(np.concatenate(non_empty, axis=0))
            else:
                groups_raw.append(np.zeros((0, 4), dtype=np.float32))
    n_groups = len(groups_raw)
    if n_groups == 0:
        return np.zeros((n_frames, n_points, 4), dtype=np.float32)

    # Step 2: uniform-sample n_frames super-frames
    idx = np.linspace(0, n_groups - 1, n_frames).round().astype(int)
    out = np.zeros((n_frames, n_points, 4), dtype=np.float32)
    for fi, src_i in enumerate(idx):
        pts = groups_raw[src_i]
        n = pts.shape[0]
        if n == 0:
            continue
        if n >= n_points:
            sel = rng.choice(n, n_points, replace=False)
            out[fi] = pts[sel]
        else:
            out[fi, :n] = pts
    return out


def normalize_sample(x):
    """Center xyz on the per-sample mean and scale by per-sample max distance.

    Velocity channel is scaled by per-sample max |v| so all channels are in similar range.
    Operates only on non-zero (real) points.
    """
    mask = np.any(x != 0, axis=-1)
    if not mask.any():
        return x
    xyz = x[..., :3]
    v = x[..., 3:]
    flat_xyz = xyz[mask]
    centroid = flat_xyz.mean(axis=0)
    centered = xyz - centroid
    dist = np.linalg.norm(centered[mask], axis=-1).max()
    if dist < 1e-6:
        dist = 1.0
    centered = centered / dist
    v_abs = np.abs(v[mask]).max()
    if v_abs < 1e-6:
        v_abs = 1.0
    v_norm = v / v_abs
    out = np.concatenate([centered, v_norm], axis=-1)
    out = out * mask[..., None]
    return out.astype(np.float32)


def list_files(pointcloud_root, users=('hw', 'jh', 'ys')):
    """Return list of (user, class_idx, path)."""
    out = []
    for u in users:
        user_dir = os.path.join(pointcloud_root, USERS[u])
        for c in CLASSES:
            class_dir = os.path.join(user_dir, c)
            if not os.path.isdir(class_dir):
                continue
            for fn in sorted(os.listdir(class_dir)):
                if fn.endswith('.json'):
                    out.append((u, CLASS_TO_IDX[c], os.path.join(class_dir, fn)))
    return out


def load_split(file_list, n_frames=32, n_points=64, frame_group=1,
               seed=42, normalize=True, verbose=True):
    """Load all files in `file_list` → (X, y, users) numpy arrays."""
    rng = np.random.default_rng(seed)
    X = np.zeros((len(file_list), n_frames, n_points, 4), dtype=np.float32)
    y = np.zeros(len(file_list), dtype=np.int64)
    users = np.empty(len(file_list), dtype=object)
    for i, (u, ci, p) in enumerate(file_list):
        arr = load_file(p, n_frames=n_frames, n_points=n_points,
                        frame_group=frame_group, rng=rng)
        if normalize:
            arr = normalize_sample(arr)
        X[i] = arr
        y[i] = ci
        users[i] = u
        if verbose and (i + 1) % 100 == 0:
            print(f'  loaded {i+1}/{len(file_list)}')
    return X, y, users


def in_subject_split(pointcloud_root, train_ratio=0.7, seed=42, users=('hw', 'jh', 'ys')):
    """7:3 train/test split within each user-class (MSFE-GAM-SPointNet paper setting)."""
    rng = np.random.default_rng(seed)
    all_files = list_files(pointcloud_root, users=users)
    by_uc = {}
    for u, ci, p in all_files:
        by_uc.setdefault((u, ci), []).append((u, ci, p))
    train, test = [], []
    for (u, ci), files in by_uc.items():
        rng.shuffle(files)
        n_train = max(1, int(round(len(files) * train_ratio)))
        train.extend(files[:n_train])
        test.extend(files[n_train:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def cross_subject_split(pointcloud_root, train_user, test_users=None):
    """Cross-subject: train on one user, test on the others (our paper's setting)."""
    all_files = list_files(pointcloud_root)
    train = [t for t in all_files if t[0] == train_user]
    if test_users is None:
        test_users = [u for u in ('hw', 'jh', 'ys') if u != train_user]
    test = [t for t in all_files if t[0] in test_users]
    return train, test


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        'STPP_PCD_ROOT', 'pointcloud')
    fl = list_files(root)
    print(f'Total files: {len(fl)}')
    per_user = {}
    for u, ci, p in fl:
        per_user.setdefault(u, {}).setdefault(CLASSES[ci], 0)
        per_user[u][CLASSES[ci]] += 1
    for u, dd in per_user.items():
        print(f'  {u}:', dd)

    train, test = in_subject_split(root, train_ratio=0.7, seed=42)
    print(f'\nIn-subject 7:3 split: train={len(train)}, test={len(test)}')

    train_cs, test_cs = cross_subject_split(root, 'hw')
    print(f"Cross-subject (train=hw): train={len(train_cs)}, test={len(test_cs)}")

    print('\nSmoke test: loading first 3 files (defaults F=32, N=64, no grouping)')
    X, y, users = load_split(fl[:3], verbose=False)
    print(f'  X: {X.shape}, dtype={X.dtype}, range=[{X.min():.3f}, {X.max():.3f}]')
    print(f'  y: {y}, users: {list(users)}')
    print(f'  Non-zero point fraction (first sample): {(np.any(X[0] != 0, axis=-1)).mean():.3f}')
