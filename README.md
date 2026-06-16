# In-cabin Hand Gesture Recognition — Doppler-based Spatial-Temporal Point Processing (STPP)

Reference implementation for the paper

> **In-cabin Hand Gesture Recognition Using Doppler-based Spatial Temporal Point Processing**
> Youngseo Ji, Jinha Kim, Hyeongwoo Kim, Yewon Jeong, Byungkwan Kim
> Department of Radio and Information Communications Engineering, Chungnam National University

---

## What the method does

A radar **point cloud** (per frame: `(x, y, z)`, Doppler velocity `v`, power `P`)
is reduced to three 2-D **spatial-temporal spectra** — XTD, YTD, ZTD — by
discretising each spatial axis into 100 bins and accumulating, per (bin, frame)
cell, the point Doppler velocity (the proposed feature; point count and power
variants are also produced for ablation). The three maps are stacked into a
`(100, 40, 3)` tensor, then sliding-window augmented to `(100, 30, 3)` samples
that feed a lightweight CNN.

```
raw point-cloud JSON ─▶ axis binning (count/power/doppler) ─▶ (100,40,3) spectrum
                     ─▶ sliding window (w=30, stride=1, ×10) ─▶ (100,30,3) samples
                     ─▶ lightweight CNN (ShuffleNetV2, …)    ─▶ 6-class gesture
```

## Repository layout

```
code_release/
├── src/doppler_stpp/
│   ├── preprocessing/
│   │   ├── pointcloud_to_spectrum.py   # ★ the proposed STPP algorithm
│   │   └── sliding_window.py           # temporal augmentation (SDS / CDS)
│   ├── models/                         # ShuffleNetV2, MobileNetV3-S/L,
│   │   │                               #   MobileNetV2, GhostNet, MobileViT
│   │   └── __init__.py                 # build_model(name) registry
│   ├── data.py                         # dataset assembly + normalisation
│   ├── train.py                        # 15-split cross-subject training
│   ├── evaluate.py                     # per-user / per-class accuracy
│   └── efficiency.py                   # params / FLOPs / latency (Table 7)
├── sota_baselines/                     # CNN-LSTM, DST, CNN-Transformer, SPointNet
├── requirements.txt                    # TensorFlow 2.12.1 (pinned)
├── CITATION.cff   LICENSE (MIT)
```

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # TensorFlow 2.12.1, NumPy, scikit-learn, …
export PYTHONPATH=src
```

## Data

The model consumes raw RETINA point-cloud recordings (one JSON per gesture
instance) laid out as `pcd_<subject>/<class>/*.json`. Six gesture classes:
`left, right, up, down, push, no_gesture`. The radar data are available from the
corresponding author / the accompanying Zenodo **data** record (see the paper's
Data Availability statement); they are not redistributed in this code archive.

## Reproducing the pipeline

All steps are plain `python -m doppler_stpp.<module>` calls (run with
`PYTHONPATH=src`, set in Installation). Replace `<pcd_root>` / `<test_root>`
with the paths to the raw point clouds / processed datasets from the data record.

```bash
# 1) raw point clouds → XTD/YTD/ZTD spectra (count/power/doppler), one call per subject
#    (subject folder → tag: pcd_hw_userA→hw [User A], pcd_jh_userB→jh [User B],
#     pcd_ys_userC→ys [User C])
python -m doppler_stpp.preprocessing.pointcloud_to_spectrum \
    --data-root <pcd_root>/pcd_hw_userA --save-root ./data/dataset_hw_raw --user-tag hw
python -m doppler_stpp.preprocessing.pointcloud_to_spectrum \
    --data-root <pcd_root>/pcd_jh_userB --save-root ./data/dataset_jh_raw --user-tag jh
python -m doppler_stpp.preprocessing.pointcloud_to_spectrum \
    --data-root <pcd_root>/pcd_ys_userC --save-root ./data/dataset_ys_raw --user-tag ys
#    then apply sliding_window.sliding_windows() to obtain the (n,10,100,30,3) tensors

# 2) train one backbone with the 15-split protocol (train subject = hw)
python -m doppler_stpp.train \
    --model ShuffleNetV2 --train-dir dataset_hw_timeshift/sliding_doppler \
    --out-dir runs/ShuffleNetV2 --n-splits 15 --epochs 100 --batch-size 32 --lr 1e-3

# 3) cross-subject evaluation (hw in-distribution; jh=User B, ys=User C untrained)
python -m doppler_stpp.evaluate \
    --model ShuffleNetV2 --weights-dir runs/ShuffleNetV2 --test-root <test_root> \
    --representation sliding_doppler --n-splits 15 --out-csv cross_subject_ShuffleNetV2.csv

# model complexity / latency (Table 7)
python -m doppler_stpp.efficiency --models ShuffleNetV2 MobileNetV2 GhostNet
```

## Training / evaluation protocol (paper)

| Item | Value |
|---|---|
| Input | `(100, 40, 3)` spectrum → sliding window → `(100, 30, 3)` |
| Train subject | `hw` (User A), time-shift + sliding-window augmented |
| Untrained test subjects | `jh` (User B), `ys` (User C) — never seen in training |
| Splits | 15 (`random_state` 0–14, stratified, val = 10%) |
| Optimizer | Adam (lr 1e-3), `sparse_categorical_crossentropy` |
| Epochs / batch | 100 / 32, best-val-loss checkpoint per split |
| Normalisation | global max-abs |
