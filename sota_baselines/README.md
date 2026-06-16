# SOTA & point-cloud baselines

The comparison baselines used in the manuscript (Table 7 / Table 12-13). They
are kept as a separate, self-contained set because several use a different input
representation (4-D point clouds) than the main Doppler-spectrum pipeline. All
baseline code is included here; only the cached tensors and raw point clouds are
deferred to the data record.

| Baseline | Code | Input | Reference |
|---|---|---|---|
| CNN-LSTM | `cnn_lstm_transformer.ipynb` | Doppler spectra (100,30,3) | Jin et al. 2024 [22] |
| CNN-Transformer | same notebook | Doppler spectra | Jin et al. 2024 [22] |
| DST (Dual-Stream Transformer) | `srdst.py` | Doppler spectra (100,30,3) | Jin et al. 2024 [39] |
| MSFE-GAM-SPointNet | `pointcloud_sota_2025/` | 4-D point cloud (F,N,4) | Li et al. 2025 [49] |

The lightweight backbones themselves (ShuffleNetV2, MobileNetV3-S/L, MobileNetV2,
GhostNet, MobileViT) are reimplemented under `../src/doppler_stpp/models/`.
Efficiency / latency measurement scripts (all models, batch size 1, TF 2.12.1)
are under `efficiency_measurement/`.

➡️ **See [`REPRODUCTION_NOTES.md`](REPRODUCTION_NOTES.md)** for how to run each
baseline, the adaptation details (what was reimplemented vs. taken from each
original paper), and the unspecified-hyperparameter choices.
