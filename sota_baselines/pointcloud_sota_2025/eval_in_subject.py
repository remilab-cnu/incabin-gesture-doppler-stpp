"""In-subject 7:3 split evaluation (matches MSFE-GAM-SPointNet paper setting).

Each user's files are split 70:30 train/test. Train and evaluate the model.
Repeat with multiple seeds for stability.
"""
import argparse
import csv
import os
import time

import numpy as np
import tensorflow as tf

from data import CLASSES, USERS
from data.cache import load_cache
from train import evaluate, make_model, train_one


def split_indices(users_arr, y_arr, train_ratio=0.7, seed=42, users=('hw', 'jh', 'ys')):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for u in users:
        for c in range(len(CLASSES)):
            mask = (users_arr == u) & (y_arr == c)
            idx = np.where(mask)[0]
            rng.shuffle(idx)
            n_tr = max(1, int(round(len(idx) * train_ratio)))
            train_idx.extend(idx[:n_tr].tolist())
            test_idx.extend(idx[n_tr:].tolist())
    return np.array(train_idx), np.array(test_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='msfe_gam_spointnet',
                    choices=['msfe_gam_spointnet', 'sequentialpointnet', 'msfe_only', 'gam_only'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--n_points', type=int, default=32,
                    help='Subsample N points per frame from cached N=64 to reduce memory')
    ap.add_argument('--out', default='results/in_subject')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    X, y, users_arr, paths = load_cache()
    if X.shape[2] > args.n_points:
        X = X[:, :, :args.n_points, :]
    print(f'Cache loaded: X={X.shape}, n_users={len(set(users_arr))}, classes={len(set(y))}')

    rows = []
    for seed in args.seeds:
        tr_idx, te_idx = split_indices(users_arr, y, train_ratio=0.7, seed=seed)
        print(f'\n=== seed={seed}: train={len(tr_idx)}, test={len(te_idx)} ===')
        model = make_model(args.model, n_classes=len(CLASSES))
        if seed == args.seeds[0]:
            print(f'Model: {model.name}, params={model.count_params():,}')
        t0 = time.time()
        train_one(
            model, X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
            epochs=args.epochs, batch_size=args.batch, lr=args.lr,
            verbose=0, seed=seed + 1000,
        )
        elapsed = time.time() - t0
        # Per-user evaluation
        per_user = {}
        for u in USERS.keys():
            mu = te_idx[users_arr[te_idx] == u]
            if len(mu) == 0:
                continue
            acc, _, _ = evaluate(model, X[mu], y[mu])
            per_user[u] = acc
        acc_all, _, _ = evaluate(model, X[te_idx], y[te_idx])
        print(f'  acc_all={acc_all:.4f} | ' + ' | '.join(f'{u}={a:.4f}' for u, a in per_user.items())
              + f' | time={elapsed:.1f}s')
        rows.append(dict(seed=seed, acc_all=acc_all, time_sec=elapsed, **{f'acc_{u}': v for u, v in per_user.items()}))

    out_csv = os.path.join(args.out, f'{args.model}_in_subject.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    accs = np.array([r['acc_all'] for r in rows])
    print(f'\nSummary ({args.model}): mean={accs.mean():.4f}, std={accs.std():.4f}, '
          f'min={accs.min():.4f}, max={accs.max():.4f}')
    print(f'Saved: {out_csv}')


if __name__ == '__main__':
    main()
