# Large-Scale Region-Cache Lazy Patch U-Net Benchmark

This benchmark contains 1 deterministic, solvable maps (1 per size/structure stratum) and 4 algorithm runs. It keeps Manhattan `f=g+h_manhattan` as the primary order; U-Net values are only secondary tie-break keys.

The full-map baseline materializes a complete U-Net prediction once per map. Lazy patch variants score only nodes in active minimum-Manhattan-f tie sets. The no-cache variant may batch independent patches from one tie set, but never reuses a prediction between nodes; `Forwards` is the actual model-call count and `Scored nodes` is the number of distinct patch scores. Region-cache variants reuse the most recently computed local h map covering a query. All timings include search plus their stated neural preparation costs.

## By Map Size

| Size | Algorithm | Expanded | Optimality | Forwards | Scored nodes | Hit rate | Regions | Full map s | Patch s | Lookup s | A* s | Total s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | full_map_unet_tiebreak | 195.00 | 1.000 | 1.0 | 0.0 | 0.000 | 0.0 | 0.0633 | 0.0000 | 0.0000 | 0.0007 | 0.0641 |
| 100 | lazy_patch_32_no_cache | 195.00 | 1.000 | 255.0 | 255.0 | 0.000 | 0.0 | 0.0000 | 0.1266 | 0.0000 | 0.0763 | 0.2330 |
| 100 | region_cache_patch_32 | 195.00 | 1.000 | 3.0 | 258.0 | 0.988 | 3.0 | 0.0000 | 0.0010 | 0.0000 | 0.0019 | 0.0079 |
| 100 | region_cache_patch_64 | 195.00 | 1.000 | 2.0 | 257.0 | 0.992 | 2.0 | 0.0000 | 0.0059 | 0.0000 | 0.0052 | 0.0208 |

## Answers

- 100x100, `lazy_patch_32_no_cache`: total-runtime speedup 0.28x versus full-map; expansion change +0.0%; optimality 100.0%.
- 100x100, `region_cache_patch_32`: total-runtime speedup 8.12x versus full-map; expansion change +0.0%; optimality 100.0%.
- 100x100, `region_cache_patch_64`: total-runtime speedup 3.08x versus full-map; expansion change +0.0%; optimality 100.0%.

`summary_by_structure.csv` and `speedup_analysis.csv` give the corresponding structure-level comparisons. Reported optimality is measured against reverse-BFS cost on each saved map; it is not inferred from the Manhattan primary key.

## Interpretation Limits

The cache changes which local patch prediction supplies an overlapping node's secondary value, so it can legitimately change tie-break trajectories and expansion counts. Results establish measured benchmark associations, not that cache reuse or local context alone causes a given outcome. The U-Net was trained on smaller inputs; patch and full-map inputs therefore remain out-of-distribution at these scales.
