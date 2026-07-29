# Large-Scale Region-Cache Lazy Patch U-Net Benchmark

This benchmark contains 1500 deterministic, solvable maps (100 per size/structure stratum) and 6000 algorithm runs. It keeps Manhattan `f=g+h_manhattan` as the primary order; U-Net values are only secondary tie-break keys.

The full-map baseline materializes a complete U-Net prediction once per map. Lazy patch variants score only nodes in active minimum-Manhattan-f tie sets. The no-cache variant may batch independent patches from one tie set, but never reuses a prediction between nodes; `Forwards` is the actual model-call count and `Scored nodes` is the number of distinct patch scores. Region-cache variants reuse the most recently computed local h map covering a query. All timings include search plus their stated neural preparation costs.

## By Map Size

| Size | Algorithm | Expanded | Optimality | Forwards | Scored nodes | Hit rate | Regions | Full map s | Patch s | Lookup s | A* s | Total s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | full_map_unet_tiebreak | 512.52 | 1.000 | 1.0 | 0.0 | 0.000 | 0.0 | 0.0343 | 0.0000 | 0.0000 | 0.0041 | 0.0384 |
| 100 | lazy_patch_32_no_cache | 463.92 | 1.000 | 496.4 | 496.4 | 0.000 | 0.0 | 0.0000 | 4.5719 | 0.0000 | 0.0216 | 4.6214 |
| 100 | region_cache_patch_32 | 477.76 | 1.000 | 4.5 | 513.9 | 0.880 | 4.5 | 0.0000 | 0.0382 | 0.0002 | 0.0033 | 0.0576 |
| 100 | region_cache_patch_64 | 492.30 | 1.000 | 2.1 | 532.2 | 0.888 | 2.1 | 0.0000 | 0.0370 | 0.0002 | 0.0027 | 0.0634 |
| 1000 | full_map_unet_tiebreak | 55158.60 | 1.000 | 1.0 | 0.0 | 0.000 | 0.0 | 0.8771 | 0.0000 | 0.0000 | 0.2525 | 1.1296 |
| 1000 | lazy_patch_32_no_cache | 48625.98 | 1.000 | 49219.2 | 49219.2 | 0.000 | 0.0 | 0.0000 | 32.4027 | 0.0000 | 0.5346 | 33.5998 |
| 1000 | region_cache_patch_32 | 44790.89 | 1.000 | 200.4 | 45921.0 | 0.867 | 200.4 | 0.0000 | 0.1361 | 0.0110 | 0.1567 | 0.6891 |
| 1000 | region_cache_patch_64 | 47765.89 | 1.000 | 55.0 | 48975.9 | 0.875 | 55.0 | 0.0000 | 0.0812 | 0.0133 | 0.1613 | 0.6649 |
| 500 | full_map_unet_tiebreak | 14816.45 | 1.000 | 1.0 | 0.0 | 0.000 | 0.0 | 0.5666 | 0.0000 | 0.0000 | 0.2431 | 0.8097 |
| 500 | lazy_patch_32_no_cache | 12819.58 | 1.000 | 13059.9 | 13059.9 | 0.000 | 0.0 | 0.0000 | 67.4451 | 0.0000 | 0.5784 | 68.7017 |
| 500 | region_cache_patch_32 | 11887.36 | 1.000 | 58.7 | 12260.1 | 0.882 | 58.7 | 0.0000 | 0.3818 | 0.0136 | 0.1473 | 0.8234 |
| 500 | region_cache_patch_64 | 13342.30 | 1.000 | 17.4 | 13762.8 | 0.889 | 17.4 | 0.0000 | 0.2884 | 0.0133 | 0.1543 | 0.8457 |

## Answers

- 100x100, `lazy_patch_32_no_cache`: total-runtime speedup 0.01x versus full-map; expansion change +9.5%; optimality 100.0%.
- 100x100, `region_cache_patch_32`: total-runtime speedup 0.67x versus full-map; expansion change +6.8%; optimality 100.0%.
- 100x100, `region_cache_patch_64`: total-runtime speedup 0.61x versus full-map; expansion change +3.9%; optimality 100.0%.
- 500x500, `lazy_patch_32_no_cache`: total-runtime speedup 0.01x versus full-map; expansion change +13.5%; optimality 100.0%.
- 500x500, `region_cache_patch_32`: total-runtime speedup 0.98x versus full-map; expansion change +19.8%; optimality 100.0%.
- 500x500, `region_cache_patch_64`: total-runtime speedup 0.96x versus full-map; expansion change +9.9%; optimality 100.0%.
- 1000x1000, `lazy_patch_32_no_cache`: total-runtime speedup 0.03x versus full-map; expansion change +11.8%; optimality 100.0%.
- 1000x1000, `region_cache_patch_32`: total-runtime speedup 1.64x versus full-map; expansion change +18.8%; optimality 100.0%.
- 1000x1000, `region_cache_patch_64`: total-runtime speedup 1.70x versus full-map; expansion change +13.4%; optimality 100.0%.

`summary_by_structure.csv` and `speedup_analysis.csv` give the corresponding structure-level comparisons. Reported optimality is measured against reverse-BFS cost on each saved map; it is not inferred from the Manhattan primary key.

## Interpretation Limits

The cache changes which local patch prediction supplies an overlapping node's secondary value, so it can legitimately change tie-break trajectories and expansion counts. Results establish measured benchmark associations, not that cache reuse or local context alone causes a given outcome. The U-Net was trained on smaller inputs; patch and full-map inputs therefore remain out-of-distribution at these scales.
