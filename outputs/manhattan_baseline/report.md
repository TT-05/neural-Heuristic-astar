# CPU Manhattan A* Baseline Benchmark

This benchmark runs pure Manhattan A* on the 1500 saved deterministic cases from `outputs/region_cache_large_benchmark/cases.csv`. It does not regenerate the manifest, load a U-Net checkpoint, or rerun U-Net methods.

Every reconstructed grid is hash-checked against the manifest. For each run, the stored optimal cost is revalidated by a fresh reverse-BFS distance-to-goal computation. Expanded nodes are the non-stale expansions reported by the existing `astar_search` implementation.

## Aggregate Comparison

| Algorithm | Expanded | Saved vs Manhattan | Reduction | Optimality | A* s | Total s | Total runtime vs Manhattan |
|---|---:|---:|---:|---:|---:|---:|---:|
| manhattan_astar | 35120.54 | 0.00 | 0.0% | 100.0% | 0.0737 | 0.0737 | 1.00x (+0.0%) |
| full_map_unet_tiebreak | 23495.86 | 11624.68 | 33.1% | 100.0% | 0.1666 | 0.6592 | 8.94x (+794.0%) |
| region_cache_patch_32 | 19052.00 | 16068.53 | 45.8% | 100.0% | 0.1024 | 0.5234 | 7.10x (+609.7%) |
| region_cache_patch_64 | 20533.50 | 14587.04 | 41.5% | 100.0% | 0.1061 | 0.5247 | 7.12x (+611.5%) |

## By Map Size

| Size | Algorithm | Expanded | Saved vs Manhattan | Reduction | A* s | Total s | Runtime multiplier |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100 | manhattan_astar | 911.17 | 0.00 | 0.0% | 0.0014 | 0.0014 | 1.00x |
| 100 | full_map_unet_tiebreak | 512.52 | 398.64 | 43.8% | 0.0041 | 0.0384 | 28.34x |
| 100 | region_cache_patch_32 | 477.76 | 433.41 | 47.6% | 0.0033 | 0.0576 | 42.48x |
| 100 | region_cache_patch_64 | 492.30 | 418.87 | 46.0% | 0.0027 | 0.0634 | 46.80x |
| 500 | manhattan_astar | 22591.87 | 0.00 | 0.0% | 0.0427 | 0.0427 | 1.00x |
| 500 | full_map_unet_tiebreak | 14816.45 | 7775.42 | 34.4% | 0.2431 | 0.8097 | 18.98x |
| 500 | region_cache_patch_32 | 11887.36 | 10704.51 | 47.4% | 0.1473 | 0.8234 | 19.31x |
| 500 | region_cache_patch_64 | 13342.30 | 9249.57 | 40.9% | 0.1543 | 0.8457 | 19.83x |
| 1000 | manhattan_astar | 81858.57 | 0.00 | 0.0% | 0.1772 | 0.1772 | 1.00x |
| 1000 | full_map_unet_tiebreak | 55158.60 | 26699.97 | 32.6% | 0.2525 | 1.1296 | 6.37x |
| 1000 | region_cache_patch_32 | 44790.89 | 37067.68 | 45.3% | 0.1567 | 0.6891 | 3.89x |
| 1000 | region_cache_patch_64 | 47765.89 | 34092.68 | 41.6% | 0.1613 | 0.6649 | 3.75x |

`summary.csv` also provides the same comparison for every size × structure stratum.

## Interpretation

Expansion savings compare each existing Manhattan-primary U-Net tie-break trajectory against pure Manhattan A* on identical saved cases. Runtime overhead compares the recorded end-to-end CPU timings: neural preparation plus search for U-Net variants, versus search only for Manhattan A*. These CPU measurements do not establish GPU runtime ranking; GPU comparison needs equivalent CUDA execution and synchronized timing.
