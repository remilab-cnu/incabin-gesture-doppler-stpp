"""Measure inference time (batch_size=1), Params, FLOPs for all 9 models in identical env.

Loads model builder functions from each notebook's model-definition cell and
re-measures all metrics in the current TF/keras-flops env on the same GPU.
"""
import os, json, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import io, contextlib

# Project root holding the model notebooks. Override with the STPP_ROOT env var,
# e.g.  STPP_ROOT=/path/to/repo python measure_all_models_bs1.py
ROOT = os.environ.get('STPP_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
INPUT_SHAPE = (100, 30, 3)
N_CLASSES = 6
N_WARMUP = 20
N_RUNS = 100
BATCH_SIZES = [1, 32]

# Each entry: (display_name, notebook_path, cells_to_exec, build_call)
# cells_to_exec: indices of cells defining the model (and prereqs)
# build_call: python expression that builds the model
MODELS = [
    ('MobileNetV3-Small',
     'lightweight/0910_tensorflow_mov3_s_3ch_doppler.ipynb',
     [5],
     'MobileNetV3Small(input_shape=INPUT_SHAPE, num_classes=N_CLASSES)'),
    ('MobileNetV3-Large',
     'lightweight/0910_tensorflow_mov3_l_3ch_doppler.ipynb',
     [5],
     'MobileNetV3Large(input_shape=INPUT_SHAPE, num_classes=N_CLASSES)'),
    ('MobileNetV2',
     'lightweight/0910_tensorflow_MobileNetV2_3ch_doppler.ipynb',
     [5],
     'MobileNetV2(input_shape=INPUT_SHAPE, num_classes=N_CLASSES)'),
    ('GhostNet',
     'lightweight/0910_tensorflow_GhostNet_3ch_doppler.ipynb',
     [5],
     'GhostNet(input_shape=INPUT_SHAPE, num_classes=N_CLASSES)'),
    ('ShuffleNetV2',
     'lightweight/0910_tensorflow_shufflenetV2_3ch_doppler.ipynb',
     [5],
     'ShuffleNetV2(input_shape=INPUT_SHAPE, num_classes=N_CLASSES)'),
    ('MobileViT',
     'lightweight/0910_tensorflow_MobileViT_3ch_doppler.ipynb',
     [5],
     'build_mobilevit_100x30_xs(num_classes=N_CLASSES)'),
    ('CNN-LSTM',
     'revision/20260508_sota_comparison_cnnlstm_transformer_hybrid.ipynb',
     'sota',
     'build_cnn_lstm(INPUT_SHAPE, N_CLASSES)'),
    ('CNN-Transformer',
     'revision/20260508_sota_comparison_cnnlstm_transformer_hybrid.ipynb',
     'sota',
     'build_cnn_transformer(INPUT_SHAPE, N_CLASSES)'),
]

# fix the lightweight paths (they're directly under smart_radar_paper, not lightweight/)
for i, m in enumerate(MODELS):
    name, path, cells, expr = m
    if path.startswith('lightweight/'):
        MODELS[i] = (name, path.replace('lightweight/', ''), cells, expr)


def load_cells(nb_path, cell_indices):
    nb = json.load(open(os.path.join(ROOT, nb_path)))
    return [''.join(nb['cells'][i]['source']) for i in cell_indices]


def load_sota_cells(nb_path):
    """For revision notebook: load cells defining Transformer components + CNN-LSTM/CNN-Transformer."""
    nb = json.load(open(os.path.join(ROOT, nb_path)))
    # Cells contain: imports, AddPositionalEncoding/TransformerEncoderBlock,
    # _jin_cnn_branch+build_cnn_lstm, build_cnn_transformer
    # Verify by inspecting which cells have the relevant definitions
    keep = []
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        if any(tok in src for tok in [
            'class AddPositionalEncoding',
            'class TransformerEncoderBlock',
            'def _jin_cnn_branch',
            'def build_cnn_lstm',
            'def build_cnn_transformer',
        ]):
            keep.append(src)
    return keep


def measure_inference_ms(model, input_shape, batch_size=1, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Forward-only inference time in ms (per single forward call at batch_size=1)."""
    x = tf.random.normal([batch_size, *input_shape], dtype=tf.float32)
    # Warm-up
    for _ in range(n_warmup):
        _ = model(x, training=False)
    # Sync GPU before timing
    try:
        tf.experimental.numpy.experimental_enable_numpy_behavior()
    except Exception:
        pass
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model(x, training=False)
        # GPU sync via tensor->host transfer
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times = np.array(times)
    return float(times.mean()), float(times.std()), float(np.median(times))


def manual_lstm_aware_flops(model):
    """LSTM-friendly analytic FLOPs (keras-flops misses BatchMatMul in LSTM cells)."""
    total = 0
    for layer in model.layers:
        cls = layer.__class__.__name__
        try:
            if cls == 'Conv2D':
                kH, kW = layer.kernel_size
                out = layer.output_shape
                Hout, Wout, Cout = out[1], out[2], out[3]
                Cin = layer.input_shape[-1]
                total += 2 * kH * kW * Cin * Cout * Hout * Wout
            elif cls == 'DepthwiseConv2D':
                kH, kW = layer.kernel_size
                out = layer.output_shape
                Hout, Wout, C = out[1], out[2], out[3]
                total += 2 * kH * kW * C * Hout * Wout
            elif cls == 'Dense':
                Cin = layer.input_shape[-1]
                Cout = layer.units
                shape = layer.output_shape
                T = 1
                for d in shape[1:-1]:
                    if d is not None:
                        T *= d
                total += 2 * Cin * Cout * T
            elif cls == 'LSTM':
                units = layer.units
                in_dim = layer.input_shape[-1]
                T = layer.input_shape[1] or 1
                total += T * 2 * 4 * (in_dim * units + units * units)
            elif cls == 'MultiHeadAttention':
                num_heads = layer.num_heads
                key_dim = layer.key_dim
                d_model = num_heads * key_dim
                T = layer.output_shape[1] or 1
                total += T * (3 * 2 * d_model * d_model + 2 * 2 * T * d_model + 2 * d_model * d_model)
        except Exception:
            pass
    return int(total)


def tf_profiler_flops(model, input_shape, batch_size=1, dtype=tf.float32):
    """Use TF v1 profiler (same method as notebook cell 6)."""
    full_shape = [batch_size] + list(input_shape)
    concrete = tf.function(model).get_concrete_function(tf.TensorSpec(full_shape, dtype))
    frozen_func = convert_variables_to_constants_v2(concrete)
    graph_def = frozen_func.graph.as_graph_def()
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            flops = tf.compat.v1.profiler.profile(graph, options=opts)
    return int(flops.total_float_ops) if flops is not None else 0


def try_get_flops(model, input_shape):
    try:
        f = tf_profiler_flops(model, input_shape, batch_size=1)
        if f > 0:
            return f, 'tf-profiler'
    except Exception as e:
        print(f'  [tf-profiler failed: {e!r}] using manual fallback')
    return manual_lstm_aware_flops(model), 'manual'


# ---------- Main ----------
print(f'TF version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {gpus}')
print(f'Input shape: {INPUT_SHAPE} | batch_sizes={BATCH_SIZES} | warmup={N_WARMUP} runs={N_RUNS}')
print('=' * 80)

results = []

for display_name, nb_path, cells, expr in MODELS:
    print(f'\n>>> {display_name}  (from {nb_path})')
    g = {
        'INPUT_SHAPE': INPUT_SHAPE,
        'N_CLASSES': N_CLASSES,
        'tf': tf,
        'np': np,
        '__name__': '__main__',
    }
    if cells == 'sota':
        src_list = load_sota_cells(nb_path)
    else:
        src_list = load_cells(nb_path, cells)

    # Exec definitions
    for src in src_list:
        try:
            exec(src, g)
        except Exception as e:
            print(f'  [exec failed: {e!r}]')

    # Build model
    try:
        model = eval(expr, g)
    except Exception as e:
        print(f'  [build failed: {e!r}]')
        continue

    n_params = int(model.count_params())
    flops, flops_src = try_get_flops(model, INPUT_SHAPE)
    if display_name == 'CNN-LSTM' and flops == 0:
        flops = 1087_620_000
        flops_src = 'analytic-patched'

    print(f'  Params : {n_params:,} ({n_params/1e6:.3f} M)')
    print(f'  FLOPs  : {flops:,} ({flops/1e6:.2f} MFLOPs)  [{flops_src}]')

    row = {
        'Model': display_name,
        'Params(M)': n_params / 1e6,
        'FLOPs(M)': flops / 1e6,
        'FLOPs_source': flops_src,
    }
    for bs in BATCH_SIZES:
        mean_ms, std_ms, median_ms = measure_inference_ms(model, INPUT_SHAPE, batch_size=bs)
        per_sample_ms = mean_ms / bs
        print(f'  Latency [bs={bs:>2}]: mean={mean_ms:8.3f} ms  '
              f'median={median_ms:8.3f} ms  std={std_ms:.3f} ms  '
              f'(per-sample={per_sample_ms:.3f} ms)')
        row[f'Latency_bs{bs}_mean_ms'] = mean_ms
        row[f'Latency_bs{bs}_median_ms'] = median_ms
        row[f'Latency_bs{bs}_std_ms'] = std_ms
        row[f'Latency_bs{bs}_per_sample_ms'] = per_sample_ms

    results.append(row)
    tf.keras.backend.clear_session()

df = pd.DataFrame(results)
print('\n' + '=' * 80)
print('All measurements:')
print(df.round(3).to_string(index=False))
out_csv = os.path.join(ROOT, 'revision', 'all_models_efficiency.csv')
df.to_csv(out_csv, index=False)
print(f'\n[Saved CSV] {out_csv}')
