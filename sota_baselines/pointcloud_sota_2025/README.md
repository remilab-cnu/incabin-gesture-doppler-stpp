# MSFE-GAM-SPointNet — point-cloud SOTA baseline

Reproduction of the point-cloud gesture-recognition method **MSFE-GAM-SPointNet**
(Li et al., *Electronics* **14**(2), 371, 2025) on our in-cabin dataset, used as
the 4-D point-cloud comparison baseline in the manuscript.

> How to run, training settings, and the unspecified-hyperparameter choices are
> documented centrally in [`../REPRODUCTION_NOTES.md`](../REPRODUCTION_NOTES.md).
> This file only covers the input-data format and the folder layout.

## Input data

- Location: data record, `pointcloud/pcd_{hw,jh,ys}_user{A,B,C}/{class}/*.json`
  (loader default `./pointcloud`; override with the `STPP_PCD_ROOT` env var).
- Each JSON line is one frame: `T` (timestamp), `C` (xyz flat), `V` (Doppler
  velocity), `P` (power).
- 3 users × 6 classes (down, left, no_gesture, push, right, up); 589 files total
  (hw 263 / jh 163 / ys 163). Per file: 54–79 frames, 0–887 points/frame
  (median 108).
- Preprocessing → fixed `(F=32, N=64, 4)` tensor: F uniform-sampled to 32 frames;
  N random-sampled / zero-padded; 4 channels = `(x, y, z, v)`, per-sample
  centroid-removed and max-distance normalised.

## Directory layout

```
pointcloud_sota_2025/
├── README.md
├── data/
│   ├── pcd_dataset.py              # JSON → tensor, split utilities
│   ├── cache.py                    # build/load the .npz cache
│   └── pcd_cache_F32_N64_norm.npz  # prebuilt cache (data record)
├── msfe_gam_spointnet/
│   └── model.py                    # MSFE-GAM-SPointNet (TF/Keras)
├── train.py                        # shared training routine (AdamW + cosine + label smoothing)
├── eval_in_subject.py              # mixed-subject 7:3 evaluation
└── eval_cross_subject.py           # cross-subject evaluation (train on one user, test on the others)
```

## Implementation basis

The MSFE-GAM-SPointNet paper (Li et al. 2025 [49]) does not release code, so this
is a reimplementation from the paper text. The base SequentialPointNet has a
public implementation, but the paper's specific additions and several
hyperparameters are not specified, which we fill using PointNet++/PointNet
conventions (MSFE = parallel 1×1 ∥ 3×3 convs; GAM = channel + spatial attention
replacing CBAM; Separable MLP = single-layer neighbour MLP + 2-layer point MLP
with residual). The exact channel counts / k-NN sizes and other filled-in values
are listed in [`../REPRODUCTION_NOTES.md`](../REPRODUCTION_NOTES.md).
