"""CPU-only inference latency for all 9 models (single + multi-thread).

Forces CPU-only execution via CUDA_VISIBLE_DEVICES='' so it does not contend
with any GPU training in progress (and gets clean CPU numbers).

Measures three configurations at batch_size=1:
  - Single-thread  : intra_op=1,  inter_op=1
  - Multi-thread-16: intra_op=16, inter_op=16  (Ryzen 5950X physical cores)
  - Multi-thread-32: intra_op=32, inter_op=32  (SMT threads — full)

Each thread config runs in its own subprocess (TF threading can only be
configured once per process). Within a subprocess, all 9 models are timed
sequentially with 20 warmup + 100 runs each.

Output:
  - all_models_cpu_latency.csv  (9 models × 3 configs = 27 rows)
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import time
import sys
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, activations

# Project root holding the model notebooks. Override with the STPP_ROOT env var,
# e.g.  STPP_ROOT=/path/to/repo python measure_all_models_cpu_latency.py
ROOT = os.environ.get('STPP_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
OUT_CSV = os.path.join(ROOT, 'revision', 'all_models_cpu_latency.csv')
INPUT_SHAPE = (100, 30, 3)
N_CLASSES = 6
BATCH_SIZE = 1
N_WARMUP = 20
N_RUNS = 100

THREAD_CONFIGS = [
    ('single-thread',     1,  1),
    ('multi-thread-16',  16, 16),
    ('multi-thread-32',  32, 32),
]

# Each entry: (display_name, notebook_path_rel, fn_name_or_expr, mode)
# mode = 'lightweight' (builder from one cell, callable as fn(input_shape, n_classes))
#      = 'mobilevit'   (build_mobilevit_100x30_xs(num_classes=...))
#      = 'sota'        (revision notebook with multi-cell defs, callable as fn(input_shape, n_classes))
MODELS = [
    ('MobileNetV3-Small',  '0910_tensorflow_mov3_s_3ch_doppler.ipynb',
     'MobileNetV3Small',          'lightweight'),
    ('MobileNetV3-Large',  '0910_tensorflow_mov3_l_3ch_doppler.ipynb',
     'MobileNetV3Large',          'lightweight'),
    ('MobileNetV2',        '0910_tensorflow_MobileNetV2_3ch_doppler.ipynb',
     'MobileNetV2',               'lightweight'),
    ('GhostNet',           '0910_tensorflow_GhostNet_3ch_doppler.ipynb',
     'GhostNet',                  'lightweight'),
    ('ShuffleNetV2',       '0910_tensorflow_shufflenetV2_3ch_doppler.ipynb',
     'ShuffleNetV2',              'lightweight'),
    ('MobileViT',          '0910_tensorflow_MobileViT_3ch_doppler.ipynb',
     'build_mobilevit_100x30_xs', 'mobilevit'),
    ('CNN-LSTM',           'revision/20260508_sota_comparison_cnnlstm_transformer_hybrid.ipynb',
     'build_cnn_lstm',            'sota'),
    ('CNN-Transformer',    'revision/20260508_sota_comparison_cnnlstm_transformer_hybrid.ipynb',
     'build_cnn_transformer',     'sota'),
]


def load_lightweight_builder(nb_rel, fn_name):
    """Load model-defining cells from a 0910_tensorflow_*.ipynb notebook."""
    nb_path = os.path.join(ROOT, nb_rel)
    nb = json.load(open(nb_path))
    g = {'tf': tf, 'np': np, 'layers': layers, 'models': models,
         'activations': activations, '__name__': '__main__'}
    for c in nb['cells']:
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c.get('source', []))
        if (f'def {fn_name}' in src or 'class ' in src or 'def ghost_' in src or
            'def se_block' in src or 'def conv_bn_act' in src or 'def _make_divisible' in src or
            'def inverted_residual' in src or 'def channel_shuffle' in src or
            'def shuffle_unit' in src or 'def stage' in src):
            try:
                exec(src, g)
            except Exception:
                if fn_name in g:
                    break
    if fn_name not in g:
        raise RuntimeError(f'Could not load {fn_name} from {nb_rel}')
    return g[fn_name]


def load_sota_builder(nb_rel, fn_name):
    """Load the revision SOTA notebook (multiple cells with Transformer components)."""
    nb_path = os.path.join(ROOT, nb_rel)
    nb = json.load(open(nb_path))
    g = {'tf': tf, 'np': np, 'layers': layers, 'models': models,
         'activations': activations, '__name__': '__main__'}
    KEEP_TOKENS = [
        'class AddPositionalEncoding',
        'class TransformerEncoderBlock',
        'def _jin_cnn_branch',
        'def build_cnn_lstm',
        'def build_cnn_transformer',
        'class WarmupCosineSchedule',
    ]
    for c in nb['cells']:
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c.get('source', []))
        if any(tok in src for tok in KEEP_TOKENS):
            try:
                exec(src, g)
            except Exception:
                pass
    if fn_name not in g:
        raise RuntimeError(f'Could not load {fn_name} from {nb_rel}')
    return g[fn_name]


def build_model(display_name, nb_rel, fn_name, mode):
    if mode == 'lightweight':
        fn = load_lightweight_builder(nb_rel, fn_name)
        try:
            return fn(INPUT_SHAPE, N_CLASSES)
        except TypeError:
            return fn(input_shape=INPUT_SHAPE, num_classes=N_CLASSES)
    elif mode == 'mobilevit':
        fn = load_lightweight_builder(nb_rel, fn_name)
        return fn(num_classes=N_CLASSES)
    elif mode == 'sota':
        fn = load_sota_builder(nb_rel, fn_name)
        return fn(INPUT_SHAPE, N_CLASSES)
    else:
        raise ValueError(f'Unknown mode {mode}')


def measure_one_model(model):
    """Time inference with batch_size=1, 20 warmup + 100 runs."""
    x = np.random.randn(BATCH_SIZE, *INPUT_SHAPE).astype(np.float32)
    # Warmup
    for _ in range(N_WARMUP):
        _ = model.predict(x, batch_size=BATCH_SIZE, verbose=0)
    # Timed
    timings = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _ = model.predict(x, batch_size=BATCH_SIZE, verbose=0)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return dict(
        latency_mean_ms=float(np.mean(timings)),
        latency_median_ms=float(np.median(timings)),
        latency_std_ms=float(np.std(timings)),
        latency_p95_ms=float(np.percentile(timings, 95)),
    )


def run_single_config(label, intra, inter):
    """In a fresh subprocess: set threads, measure all 9 models."""
    tf.config.threading.set_intra_op_parallelism_threads(intra)
    tf.config.threading.set_inter_op_parallelism_threads(inter)
    print(f'\n=== {label} (intra={intra}, inter={inter}) ===', flush=True)
    print(f'  intra_op effective: {tf.config.threading.get_intra_op_parallelism_threads()}')
    print(f'  inter_op effective: {tf.config.threading.get_inter_op_parallelism_threads()}')

    results = []
    for display_name, nb_rel, fn_name, mode in MODELS:
        print(f'\n  > {display_name}', flush=True)
        try:
            model = build_model(display_name, nb_rel, fn_name, mode)
        except Exception as e:
            print(f'    ERROR loading {display_name}: {e}', flush=True)
            continue
        params = int(model.count_params())
        timing = measure_one_model(model)
        results.append(dict(
            model=display_name,
            config=label,
            intra_op=intra,
            inter_op=inter,
            params=params,
            **timing,
        ))
        print(f'    params={params:>12,}  mean={timing["latency_mean_ms"]:7.2f} ms  '
              f'median={timing["latency_median_ms"]:7.2f}  std={timing["latency_std_ms"]:5.2f}  '
              f'p95={timing["latency_p95_ms"]:7.2f}', flush=True)
        tf.keras.backend.clear_session()
        del model

    return results


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--single-config':
        label = sys.argv[2]
        intra = int(sys.argv[3])
        inter = int(sys.argv[4])
        rs = run_single_config(label, intra, inter)
        print('---RESULT---')
        print(json.dumps(rs))
        print('---END---')
        sys.exit(0)

    # Parent: subprocess per config
    all_rows = []
    for label, intra, inter in THREAD_CONFIGS:
        print(f'\n>>> Launching subprocess for {label} ...')
        result = subprocess.run(
            [sys.executable, __file__, '--single-config', label, str(intra), str(inter)],
            capture_output=True, text=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                 'TF_CPP_MIN_LOG_LEVEL': '2'},
        )
        # Stream the subprocess output
        print(result.stdout)
        if result.returncode != 0:
            print('STDERR:', result.stderr[-2000:])
            continue
        out = result.stdout
        marker = out.find('---RESULT---')
        end = out.find('---END---')
        if marker == -1 or end == -1:
            print(f'WARNING: could not parse result for {label}')
            continue
        payload = out[marker + len('---RESULT---'):end].strip()
        all_rows.extend(json.loads(payload))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)

    print('\n' + '=' * 100)
    print('All-Models CPU Latency Summary (AMD Ryzen 9 5950X, TF 2.12.1, batch_size=1)')
    print('=' * 100)
    # Pivot for nicer display
    if len(df) > 0:
        pivot = df.pivot_table(index='model', columns='config',
                                values='latency_mean_ms', aggfunc='mean')
        pivot = pivot.reindex([m[0] for m in MODELS])
        cols = [c for c, _, _ in THREAD_CONFIGS if c in pivot.columns]
        print(pivot[cols].round(2).to_string())
    print(f'\nSaved: {OUT_CSV}')
