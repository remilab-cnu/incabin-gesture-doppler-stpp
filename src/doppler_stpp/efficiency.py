"""Model complexity & latency measurement (Table 7).

Reports trainable parameters, FLOPs (TF profiler, batch size 1) and wall-clock
inference latency. Matches the measurement protocol of the revision scripts
(20 warm-up + 100 timed runs, batch size 1 for real-time deployment).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2)

from .models import MODELS, build_model


def count_flops(model, batch_size=1, dtype=tf.float32):
    """Total float operations via the TF v1 profiler on the frozen graph."""
    input_shape = [batch_size] + list(model.input_shape[1:])
    concrete = tf.function(model).get_concrete_function(
        tf.TensorSpec(input_shape, dtype))
    frozen = convert_variables_to_constants_v2(concrete)
    graph_def = frozen.graph.as_graph_def()
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph, options=opts)
    total = flops.total_float_ops if flops is not None else 0
    return total, total / 2  # (FLOPs, MACs approx.)


def count_params(model, trainable_only=True):
    weights = model.trainable_weights if trainable_only else model.weights
    return int(np.sum([np.prod(v.shape) for v in weights]))


def measure_latency(model, input_shape=(100, 30, 3), warmup=20, runs=100):
    """Mean/median/std/p95 single-sample latency in milliseconds."""
    x = np.random.rand(1, *input_shape).astype("float32")
    for _ in range(warmup):
        model.predict(x, verbose=0)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(x, verbose=0)
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return dict(mean=float(t.mean()), median=float(np.median(t)),
                std=float(t.std()), p95=float(np.percentile(t, 95)))


def profile(model_name, input_shape=(100, 30, 3), num_classes=6,
            warmup=20, runs=100):
    model = build_model(model_name, input_shape=input_shape, num_classes=num_classes)
    flops, macs = count_flops(model, batch_size=1)
    lat = measure_latency(model, input_shape, warmup, runs)
    return {
        "model": model_name,
        "params_M": count_params(model) / 1e6,
        "MFLOPs": flops / 1e6,
        "latency_ms_mean": lat["mean"],
        "latency_ms_p95": lat["p95"],
    }


def _parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--out-csv", default="model_efficiency.csv")
    return ap


if __name__ == "__main__":
    a = _parser().parse_args()
    import csv
    rows = [profile(m, warmup=a.warmup, runs=a.runs) for m in a.models]
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model']:>18}: {r['params_M']:.2f} M params | "
              f"{r['MFLOPs']:.2f} MFLOPs | {r['latency_ms_mean']:.2f} ms")
    print(f"[saved] {a.out_csv}")
