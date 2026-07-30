# Large-Scale Region-Cache Lazy Patch U-Net Benchmark

This benchmark contains 1500 deterministic, solvable maps (100 per size/structure stratum) and 6000 algorithm runs. It keeps Manhattan `f=g+h_manhattan` as the primary order; U-Net values are only secondary tie-break keys.

The full-map baseline materializes a complete U-Net prediction once per map. Lazy patch variants score only nodes in active minimum-Manhattan-f tie sets. The no-cache variant may batch independent patches from one tie set, but never reuses a prediction between nodes; `Forwards` is the actual model-call count and `Scored nodes` is the number of distinct patch scores. Region-cache variants reuse the most recently computed local h map covering a query. All timings include search plus their stated neural preparation costs.

## By Map Size

| Size | Algorithm | Expanded | Optimality | Forwards | Scored nodes | Hit rate | Regions | Full map s | Patch s | Lookup s | A* s | Total s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | full_map_unet_tiebreak | 512.54 | 1.000 | 1.0 | 0.0 | 0.000 | 0.0 | 0.0008 | 0.0000 | 0.0000 | 0.0018 | 0.0026 |
| 100 | lazy_patch_32_no_cache | 463.92 | 1.000 | 496.3 | 496.3 | 0.000 | 0.0 | 0.0000 | 0.2050 | 0.0000 | 0.1468 | 0.4002 |
| 100 | region_cache_patch_32 | 477.63 | 1.000 | 4.5 | 513.8 | 0.880 | 4.5 | 0.0000 | 0.0020 | 0.0000 | 0.0027 | 0.0124 |
| 100 | region_cache_patch_64 | 492.33 | 1.000 | 2.1 | 532.2 | 0.888 | 2.1 | 0.0000 | 0.0010 | 0.0000 | 0.0018 | 0.0129 |
| 1000 | full_map_unet_tiebreak | 55143.54 | 1.000 | 1.0 | 0.0 | 0.000 | 0.0 | 0.0615 | 0.0000 | 0.0000 | 0.2235 | 0.2850 |
| 1000 | lazy_patch_32_no_cache | 48623.19 | 1.000 | 49216.3 | 49216.3 | 0.000 | 0.0 | 0.0000 | 19.5044 | 0.0000 | 12.6956 | 36.3931 |
| 1000 | region_cache_patch_32 | 44761.04 | 1.000 | 200.3 | 45891.8 | 0.867 | 200.3 | 0.0000 | 0.1445 | 0.0096 | 0.2258 | 0.7800 |
| 1000 | region_cache_patch_64 | 47700.16 | 1.000 | 54.9 | 48908.1 | 0.875 | 54.9 | 0.0000 | 0.0817 | 0.0106 | 0.1699 | 0.6523 |
| 500 | full_map_unet_tiebreak | 14820.58 | 1.000 | 1.0 | 0.0 | 0.000 | 0.0 | 0.0072 | 0.0000 | 0.0000 | 0.0572 | 0.0644 |
| 500 | lazy_patch_32_no_cache | 12819.60 | 1.000 | 13060.0 | 13060.0 | 0.000 | 0.0 | 0.0000 | 5.2170 | 0.0000 | 3.5071 | 9.8402 |
| 500 | region_cache_patch_32 | 11889.91 | 1.000 | 58.7 | 12262.9 | 0.882 | 58.7 | 0.0000 | 0.0284 | 0.0018 | 0.0502 | 0.1882 |
| 500 | region_cache_patch_64 | 13351.26 | 1.000 | 17.5 | 13771.4 | 0.889 | 17.5 | 0.0000 | 0.0135 | 0.0022 | 0.0393 | 0.1688 |

## Answers

- 100x100, `lazy_patch_32_no_cache`: total-runtime speedup 0.01x versus full-map; expansion change +9.5%; optimality 100.0%.
- 100x100, `region_cache_patch_32`: total-runtime speedup 0.21x versus full-map; expansion change +6.8%; optimality 100.0%.
- 100x100, `region_cache_patch_64`: total-runtime speedup 0.20x versus full-map; expansion change +3.9%; optimality 100.0%.
- 500x500, `lazy_patch_32_no_cache`: total-runtime speedup 0.01x versus full-map; expansion change +13.5%; optimality 100.0%.
- 500x500, `region_cache_patch_32`: total-runtime speedup 0.34x versus full-map; expansion change +19.8%; optimality 100.0%.
- 500x500, `region_cache_patch_64`: total-runtime speedup 0.38x versus full-map; expansion change +9.9%; optimality 100.0%.
- 1000x1000, `lazy_patch_32_no_cache`: total-runtime speedup 0.01x versus full-map; expansion change +11.8%; optimality 100.0%.
- 1000x1000, `region_cache_patch_32`: total-runtime speedup 0.37x versus full-map; expansion change +18.8%; optimality 100.0%.
- 1000x1000, `region_cache_patch_64`: total-runtime speedup 0.44x versus full-map; expansion change +13.5%; optimality 100.0%.

`summary_by_structure.csv` and `speedup_analysis.csv` give the corresponding structure-level comparisons. Reported optimality is measured against reverse-BFS cost on each saved map; it is not inferred from the Manhattan primary key.

## Interpretation Limits

The cache changes which local patch prediction supplies an overlapping node's secondary value, so it can legitimately change tie-break trajectories and expansion counts. Results establish measured benchmark associations, not that cache reuse or local context alone causes a given outcome. The U-Net was trained on smaller inputs; patch and full-map inputs therefore remain out-of-distribution at these scales.
