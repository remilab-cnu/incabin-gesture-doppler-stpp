"""cross-subject evaluation (matches our paper's cross_subject_validation setting).

Fold A: train hw, test jh + ys
Fold B: train jh, test hw + ys
Fold C: train ys, test hw + jh
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

FOLDS = {
    'A': {'train': 'hw', 'test': ('jh', 'ys')},
    'B': {'train': 'jh', 'test': ('hw', 'ys')},
    'C': {'train': 'ys', 'test': ('hw', 'jh')},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='msfe_gam_spointnet',
                    choices=['msfe_gam_spointnet', 'sequentialpointnet', 'msfe_only', 'gam_only'])
    ap.add_argument('--folds', nargs='+', default=['A', 'B', 'C'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--n_points', type=int, default=32,
                    help='Subsample N points per frame from cached N=64 to reduce memory')
    ap.add_argument('--out', default='results/cross_subject')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    X, y, users_arr, paths = load_cache()
    if X.shape[2] > args.n_points:
        X = X[:, :, :args.n_points, :]
    print(f'Cache loaded: X={X.shape}, n_users={len(set(users_arr))}')

    rows = []
    for fold in args.folds:
        cfg = FOLDS[fold]
        tr_mask = users_arr == cfg['train']
        for seed in args.seeds:
            print(f'\n=== Fold {fold} (train={cfg["train"]}, test={cfg["test"]}), seed={seed} ===')
            tr_idx = np.where(tr_mask)[0]
            # Use a tiny 10% from train as validation for cosine schedule monitoring
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(tr_idx))
            n_val = max(1, int(0.1 * len(tr_idx)))
            val_idx = tr_idx[perm[:n_val]]
            tr_only_idx = tr_idx[perm[n_val:]]
            model = make_model(args.model, n_classes=len(CLASSES))
            if seed == args.seeds[0] and fold == args.folds[0]:
                print(f'Model: {model.name}, params={model.count_params():,}')
            t0 = time.time()
            train_one(
                model, X[tr_only_idx], y[tr_only_idx], X[val_idx], y[val_idx],
                epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                verbose=0, seed=seed + 1000,
            )
            elapsed = time.time() - t0

            per_user = {}
            for u in cfg['test']:
                mu = np.where(users_arr == u)[0]
                acc, _, _ = evaluate(model, X[mu], y[mu])
                per_user[u] = acc
            avg_test = float(np.mean(list(per_user.values())))
            print(f'  ' + ' | '.join(f'{u}={a:.4f}' for u, a in per_user.items())
                  + f' | avg={avg_test:.4f} | time={elapsed:.1f}s')
            rows.append(dict(fold=fold, seed=seed, train_user=cfg['train'],
                             avg_test=avg_test, time_sec=elapsed,
                             **{f'acc_{u}': v for u, v in per_user.items()}))

    out_csv = os.path.join(args.out, f'{args.model}_cross_subject.csv')
    with open(out_csv, 'w', newline='') as f:
        keys = ['fold', 'seed', 'train_user', 'avg_test', 'acc_hw', 'acc_jh', 'acc_ys', 'time_sec']
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved: {out_csv}')

    # Summary
    print('\nFold averages:')
    for fold in args.folds:
        accs = [r['avg_test'] for r in rows if r['fold'] == fold]
        print(f'  Fold {fold}: mean={np.mean(accs):.4f} std={np.std(accs):.4f}  (n={len(accs)})')


if __name__ == '__main__':
    main()
