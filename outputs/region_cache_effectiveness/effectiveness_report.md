# Region-Cache Lazy-Patch U-Net Effectiveness Analysis

This is a post-hoc analysis of `outputs/region_cache_large_benchmark/results.csv` plus the paired Manhattan baseline `outputs/manhattan_baseline/results.csv` on 1500 identical saved cases. It does not rerun A*, U-Net inference, map generation, or training.

## Aggregate Search-Aware Computation

| Method | Total map cells | U-Net calls | Patch output cells | Exact unique in-map predicted cells | Nominal output cells / map cell | Tie-break queries | Query coverage by predictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| manhattan_astar | 630000000 | 0 | 0 | 0 | 0.000 | 0 | not logged |
| full_map_unet_tiebreak | 630000000 | 1500 | 630000000 | 630000000 | 1.000 | not logged | not logged |
| region_cache_patch_32 | 630000000 | 131815 | 134978560 | not logged | 0.214 | 29347464 | 1.0 |
| region_cache_patch_64 | 630000000 | 37234 | 152510464 | not logged | 0.242 | 31635419 | 1.0 |

For region-cache variants, `queries = hits + misses` and the query-coverage ratio is therefore 1.0. This is aggregate cache accounting combined with the implemented lazy-scoring semantics: a cache miss creates the patch that serves the query, and a hit reads an earlier patch. It is not a spatial trace measurement of where queries or patch footprints occurred.

## Cache Reuse

| Method | Cache queries | Hits | Misses / U-Net calls | Weighted hit rate | Queries per miss |
|---|---:|---:|---:|---:|---:|
| region_cache_patch_32 | 29347464 | 29215649 | 131815 | 99.55% | 222.6 |
| region_cache_patch_64 | 31635419 | 31598185 | 37234 | 99.88% | 849.6 |

The weighted hit rate pools all queries. The CSV also retains the mean of per-case hit rates in `cache_reuse_analysis.csv`; the two summaries answer different questions and should not be conflated.

## Expansion Cost-Benefit

| Method | Expansions saved vs Manhattan | Reduction | Neural inference s | Saved expansions / neural s | Saved expansions / call |
|---|---:|---:|---:|---:|---:|
| full_map_unet_tiebreak | 17437020 | 33.1% | 738.986 | 23595.9 | 11624.7 |
| region_cache_patch_32 | 24102801 | 45.8% | 278.030 | 86691.4 | 182.9 |
| region_cache_patch_64 | 21880560 | 41.5% | 203.338 | 107606.6 | 587.6 |

## Limits of the Completed Logs

The benchmark records counts but not patch centers, the union of in-map patch footprints, or expanded-node traces. Consequently, exact unique predicted in-map cells, exact redundant-prediction ratio, and predicted∩expanded-node overlap are marked `not_logged` rather than estimated. `total_predicted_patch_output_cells` is exact but counts all model output positions, including overlapping patches and out-of-map padded positions. Recovering the unavailable quantities would require rerunning the search with additional trace logging, which this analysis intentionally does not do.

## Conclusion

The completed results support a limited query-level conclusion: the lazy implementation only requests a secondary score when an active Manhattan-f tie can use it, and the aggregate cache counts show that these requests were mostly served by reuse. Cache hit rates, calls, inference time, and expansion reductions are directly recorded. By contrast, the completed logs cannot establish an exact spatial concentration ratio, exact patch-overlap redundancy, or a measured predicted∩expanded-node overlap; those require coordinate-level traces. Runtime conclusions remain CPU-specific.
