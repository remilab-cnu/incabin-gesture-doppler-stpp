Point-cloud loader code only.

The cached tensors (pcd_cache_*.npz, ~12 MB) and the raw RETINA point-cloud
JSON recordings are NOT included in this code archive — they belong to the
accompanying data record (see the paper's Data Availability statement).

pcd_dataset.py builds the cache directly from the raw point-cloud JSON files;
run it once the raw data are obtained to regenerate pcd_cache_F32_N64_norm.npz.
