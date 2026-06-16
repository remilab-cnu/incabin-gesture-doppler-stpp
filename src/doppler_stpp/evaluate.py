"""Cross-subject evaluation: per-user, per-class accuracy over the 15 splits.

Reproduces Fig. 12 / Table 7 numbers: each of the 15 trained models is
evaluated on every subject; the reported accuracy is the mean over the 15
splits. The train subject 'hw' is in-distribution; 'jh' (User B) and 'ys'
(User C) are untrained (cross-subject).
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from .data import CLASS_LABELS, N_CLASSES, load_subject, normalize
from .models import build_model


def per_class_accuracy(y_true, y_pred, n_classes=N_CLASSES) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.diag(cm) / cm.sum(axis=1)
    return np.nan_to_num(acc)


def evaluate(model_name, weights_dir, test_dirs: Dict[str, int],
             n_splits=15, input_shape=(100, 30, 3), num_classes=N_CLASSES,
             norm_scale: float | None = None) -> Dict[str, dict]:
    """Evaluate every split on every subject.

    Parameters
    ----------
    weights_dir : str
        Directory with ``<model>_split{0..n-1}.weights.h5`` from train.run().
    test_dirs : dict
        Mapping ``{user: n_recordings}`` e.g.
        ``{'hw': 50, 'jh': 30, 'ys': 30}`` -> ``dataset_<user>/sliding_doppler``.
        (Pass the full path via ``--test-root`` on the CLI.)

    Returns a dict ``{user: {overall_mean, overall_std, per_class_mean, ...}}``.
    """
    results: Dict[str, dict] = {}
    for user, (test_dir, n_rec) in test_dirs.items():
        X, y = load_subject(test_dir, n_recordings=n_rec, input_shape=input_shape)
        X, _ = normalize(X, scale=norm_scale)

        overall, per_cls = [], []
        for s in range(n_splits):
            w = os.path.join(weights_dir, f"{model_name}_split{s}.weights.h5")
            model = build_model(model_name, input_shape=input_shape,
                                num_classes=num_classes)
            model.load_weights(w)
            y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
            overall.append(accuracy_score(y, y_pred))
            per_cls.append(per_class_accuracy(y, y_pred, num_classes))

        per_cls = np.asarray(per_cls)
        results[user] = {
            "overall_mean": float(np.mean(overall)),
            "overall_std": float(np.std(overall)),
            "overall_max": float(np.max(overall)),
            "per_class_mean": per_cls.mean(axis=0).tolist(),
            "per_class_std": per_cls.std(axis=0).tolist(),
        }
        print(f"[{user}] acc = {results[user]['overall_mean']*100:.2f} "
              f"± {results[user]['overall_std']*100:.2f} %")
    return results


def save_results_csv(results: Dict[str, dict], path: str) -> None:
    import csv
    cols = ["user", "overall_mean", "overall_std", "overall_max"] + \
           [CLASS_LABELS[c] for c in range(N_CLASSES)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for user, r in results.items():
            w.writerow([user, r["overall_mean"], r["overall_std"], r["overall_max"],
                        *r["per_class_mean"]])
    print(f"[saved] {path}")


def _parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--test-root", required=True,
                    help="dir containing dataset_<user>/<representation>")
    ap.add_argument("--representation", default="sliding_doppler")
    ap.add_argument("--n-splits", type=int, default=15)
    ap.add_argument("--out-csv", default="cross_subject_results.csv")
    return ap


if __name__ == "__main__":
    a = _parser().parse_args()
    # default protocol: hw in-distribution (50), jh/ys untrained (30 each)
    test_dirs = {
        u: (os.path.join(a.test_root, f"dataset_{u}", a.representation), n)
        for u, n in (("hw", 50), ("jh", 30), ("ys", 30))
    }
    res = evaluate(a.model, a.weights_dir, test_dirs, n_splits=a.n_splits)
    save_results_csv(res, a.out_csv)
