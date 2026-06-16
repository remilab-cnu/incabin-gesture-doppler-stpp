# Reproduction notes — baseline models

The baselines in this folder were **reimplemented from their original papers and
adapted** to our input representation and protocol; they are **not** verbatim
copies of the authors' (mostly unavailable) code. This document lists, per
baseline, exactly **what was adapted, what the original paper left unspecified,
and how we filled it**, so the comparison is transparent and the published
tables are reproducible.

---

## Summary

| Baseline | Reference | Key adaptation to our setup | Unspecified → filled with | Code |
|---|---|---|---|---|
| **CNN-LSTM** | Jin et al. 2024 [22] | 2 → **3 parallel CNN branches** (RTM+DTM → X/Y/Z-T-D) | LSTM hidden sizes 256→128; standard Post-LN | `cnn_lstm_transformer.ipynb` |
| **CNN-Transformer** | Jin et al. 2024 [22] | 2 → **3 branches**; 1×1 projection to embed-dim 512 | 8 encoders, 8 heads, dropout 0.2, GeLU (paper); LN positions (convention) | `cnn_lstm_transformer.ipynb` |
| **DST** (Dual-Stream Transformer) | Jin et al. 2024 [39] | **OMP sparse front-end omitted**; raw Doppler fed directly | embed 64 / 4 heads / 2 layers / FFN 128 (to match ~0.17 M params) | `srdst.py` |
| **MSFE-GAM-SPointNet** | Li et al. 2025 [49] | 4-D point cloud cached at `(F=32, N=64, 4)`, subsampled to `N=32` for the model input; batch 32 → **8** (memory) | SA channels & k-NN from **PointNet++** (see below) | `pointcloud_sota_2025/` |

---

## Running the baselines

All baselines take the same 3-channel Doppler input and the 15-split protocol
(train seed 123, splits 0–14). Install the environment first
(`pip install -r requirements.txt`, TensorFlow 2.12.1). **No radar data are
bundled** — obtain the spectra and radar point clouds from the data record
(see the main README "Data" section).

**CNN-LSTM and CNN-Transformer** — open the notebook and run all cells top to
bottom; it trains both models over the 15 splits and writes the per-user
accuracy CSVs and comparison figures:

```bash
jupyter notebook cnn_lstm_transformer.ipynb
```

**DST** (Dual-Stream Transformer):

```bash
python srdst.py --data-root <root> --out-dir runs/SRDST
#   <root> holds dataset_<user>[_timeshift]/sliding_doppler
```

**MSFE-GAM-SPointNet** (4-D point cloud) — from `pointcloud_sota_2025/`. The
loader reads the radar point clouds via `data/pcd_dataset.py`; the point-cloud
data are not bundled (data record):

```bash
cd pointcloud_sota_2025
# 1) build the point-cloud cache once (raw JSON location via STPP_PCD_ROOT)
STPP_PCD_ROOT=/path/to/pointcloud python3 -m data.cache
# 2) paper-main protocols, 15 seeds: mixed 7:3 + cross-subject (fold A)
python3 eval_in_subject.py    --model msfe_gam_spointnet --seeds $(seq 0 14) \
    --epochs 100 --batch 8 --out results/in_subject_tf212_15seeds
python3 eval_cross_subject.py --model msfe_gam_spointnet --folds A --seeds $(seq 0 14) \
    --epochs 100 --batch 8 --out results/cross_subject_tf212_15seeds
# ablation (paper Table 5/6): swap --model
#   sequentialpointnet | gam_only | msfe_only | msfe_gam_spointnet
```

**Efficiency / latency** (Table 7):

```bash
# proposed method + lightweight backbones
python -m doppler_stpp.efficiency --models ShuffleNetV2 MobileNetV2 GhostNet
# all 9 models — params / FLOPs / GPU + CPU latency
python efficiency_measurement/measure_all_models_bs1.py
python efficiency_measurement/measure_all_models_cpu_latency.py
```

---

## Per-baseline detail

### CNN-LSTM and CNN-Transformer — Jin et al. 2024 [22]
* **Original input:** two parallel CNN branches for a Range-Time Map (RTM) and a
  Doppler-Time Map (DTM) — i.e. a **2-channel** design.
* **Our adaptation:** **three** parallel CNN branches matching our X-T-D, Y-T-D,
  Z-T-D Doppler channels. Each branch keeps the paper's block structure
  (3×3 convs 64→128→256, BN+ReLU, two 2×2 max-pools). This raises the CNN
  front-end capacity by ~50 % relative to the 2-branch original.
* **CNN-Transformer head:** the three branches are concatenated and projected to
  embed-dim 512 by a 1×1 conv, then 8 Transformer encoder blocks (8 heads,
  dropout 0.2, GeLU) — per [22].
* **Unspecified → convention:** exact LSTM hidden sizes (we use 256→128),
  layer-norm placement (Pre-/Post-LN), and dropout positions that [22] does not
  give were set to standard values.
* **Optimizer (per [22]):** Adam, lr = 1e-4.

### DST (Dual-Stream Transformer) — Jin et al. 2024 [39]
* **Original method** ("SRDST"): an **orthogonal-matching-pursuit (OMP) sparse
  representation** front-end feeds a dual-stream transformer (time + channel
  streams) with weighted fusion.
* **Our adaptation (important):** the **OMP sparse front-end is NOT implemented.**
  We feed the raw X/Y/Z-T-D Doppler maps directly to the dual-stream transformer.
  OMP requires per-sample sparsity-level tuning specific to the reference's
  pipeline; reproducing it faithfully was out of scope. **This is a deliberate
  simplification and a stated limitation.**
* **Unspecified → matched to budget:** the paper reports ~0.17 M parameters but
  not layer sizes; we use embed-dim 64, 4 heads, 2 layers per stream, FFN 128,
  dropout 0.1 to land near that budget.
* **Input/protocol:** input flattened to a length-100 sequence of 90 features;
  the paper's 6:2:2 split is replaced by our 9:1 train/val on `hw` with
  `jh`/`ys` cross-user test.
* **Optimizer:** AdamW (weight_decay 0.05) with 5-epoch linear warmup + cosine
  decay (base lr 1e-3) — the same recipe used for the other Transformer
  baselines.

### MSFE-GAM-SPointNet — Li et al. 2025 [49]
* **Code availability:** the 2025 paper does not release code, so this is a
  reimplementation from its text. The base SequentialPointNet has a public
  implementation, but the paper's MSFE/GAM/Separable-MLP additions and several
  layer sizes are unspecified.
* **Input:** 4-D point cloud `(x, y, z, Doppler v)`, with three raw frames merged
  into one (per the reference's Sec. 2.3). The loader caches `(F=32, N=64, 4)`;
  the evaluation scripts subsample to `N=32` points/frame (`--n_points 32`,
  default) so the model input is `(F=32, N=32, 4)`.
* **Unspecified → PointNet++ standard** (values are from the actual
  implementation in `msfe_gam_spointnet/model.py`):
  * SA1: k = 8, channels = (32, 32, 64)
  * SA2: k = 4, channels = (64, 64, 128)
  * Temporal Conv1D: (128, 128); FC head: (128, 64) + dropout 0.5
* **Optimizer (per paper):** AdamW (weight_decay 1e-4), lr 1e-4, cosine
  annealing, label smoothing 0.1, 100 epochs.
* **Adaptation:** **batch size 32 → 8** because of GPU memory constraints
  (the paper uses 32). 6 classes matched to our dataset.

---

## Lightweight backbones
The lightweight CNNs (ShuffleNetV2, MobileNetV3-S/L, MobileNetV2, GhostNet,
MobileViT) are clean reimplementations of standard published architectures under
`src/doppler_stpp/models/`. In the per-model tuned comparison they were trained
with **each architecture's original-paper optimizer/schedule** (e.g. SGD+momentum
with label smoothing for the MobileNets/GhostNet; AdamW for MobileViT) rather
than a single default — this is hyperparameter *configuration*, not architectural
change.
