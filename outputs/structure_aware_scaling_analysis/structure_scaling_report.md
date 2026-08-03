# Structure-Aware Scaling Analysis

## Scope and method

This analysis uses only completed, optimal rows in scaling_results.csv. Every neural result is paired with Manhattan A* on the same case_id before aggregation. Each size x structure x method stratum contains 50 cases; reported standard deviations describe variation across those cases.

## Observed measurements

- full_map_unet_tiebreak: strongest mean expansion reduction is 51.3% on large_block; weakest is 17.6% on maze_like.
- region_cache_patch_32: strongest mean expansion reduction is 84.3% on large_block; weakest is 17.7% on maze_like.
- region_cache_patch_64: strongest mean expansion reduction is 73.3% on large_block; weakest is 17.7% on maze_like.

- 0 of 60 structure-size-method strata improve both mean expansion reduction and mean wall-clock time.
- 51 strata have at least 20% mean expansion reduction while still being slower than Manhattan.

## Size-dependent observations

- open_random / full_map_unet_tiebreak: 500: 28.8%, 1000: 27.4%, 1500: 27.3%, 2000: 25.6%; 500 to 2000 change: -3.3%.
- open_random / region_cache_patch_32: 500: 32.2%, 1000: 31.9%, 1500: 28.6%, 2000: 30.3%; 500 to 2000 change: -1.9%.
- open_random / region_cache_patch_64: 500: 29.7%, 1000: 30.0%, 1500: 26.5%, 2000: 28.5%; 500 to 2000 change: -1.3%.
- maze_like / full_map_unet_tiebreak: 500: 18.6%, 1000: 16.8%, 1500: 20.2%, 2000: 14.8%; 500 to 2000 change: -3.8%.
- maze_like / region_cache_patch_32: 500: 18.9%, 1000: 16.8%, 1500: 20.3%, 2000: 14.7%; 500 to 2000 change: -4.1%.
- maze_like / region_cache_patch_64: 500: 18.9%, 1000: 16.8%, 1500: 20.5%, 2000: 14.6%; 500 to 2000 change: -4.2%.
- bottleneck / full_map_unet_tiebreak: 500: 40.7%, 1000: 43.7%, 1500: 40.2%, 2000: 50.1%; 500 to 2000 change: +9.3%.
- bottleneck / region_cache_patch_32: 500: 50.4%, 1000: 51.1%, 1500: 49.3%, 2000: 57.0%; 500 to 2000 change: +6.5%.
- bottleneck / region_cache_patch_64: 500: 43.4%, 1000: 46.0%, 1500: 46.6%, 2000: 53.8%; 500 to 2000 change: +10.5%.
- large_block / full_map_unet_tiebreak: 500: 57.4%, 1000: 55.1%, 1500: 45.2%, 2000: 47.6%; 500 to 2000 change: -9.8%.
- large_block / region_cache_patch_32: 500: 82.2%, 1000: 86.0%, 1500: 81.7%, 2000: 87.3%; 500 to 2000 change: +5.1%.
- large_block / region_cache_patch_64: 500: 69.9%, 1000: 73.7%, 1500: 70.6%, 2000: 78.8%; 500 to 2000 change: +9.0%.
- narrow_corridor / full_map_unet_tiebreak: 500: 50.8%, 1000: 39.9%, 1500: 46.6%, 2000: 44.3%; 500 to 2000 change: -6.5%.
- narrow_corridor / region_cache_patch_32: 500: 58.2%, 1000: 48.5%, 1500: 56.9%, 2000: 50.0%; 500 to 2000 change: -8.2%.
- narrow_corridor / region_cache_patch_64: 500: 55.6%, 1000: 51.0%, 1500: 52.8%, 2000: 51.0%; 500 to 2000 change: -4.6%.

## Answers grounded in measurements

1. Neural A* is structure-dependent: structure-specific rows vary materially, so global averages alone are insufficient.
2. The strongest evidence is the highest paired expansion-reduction structure-method rows in structure_comparison.csv; runtime ratios must be considered separately.
3. Region-cache is not asserted to be consistently better than Full-map: every size x structure comparison is reported in the output tables.
4. Rows labelled large_search_gain_runtime_worse are directly measured cases where search reduction did not yield a wall-clock improvement.
5. Measured scenarios most suitable for Neural A* are those labelled improves_search_and_runtime; rows with only search gains are evidence for ordering quality, not runtime competitiveness.

## Possible explanations (not causal claims)

Connectivity, corridor constraints, barriers, local ambiguity, and patch work may contribute to the observed variation. This benchmark does not isolate these factors, so it does not establish causality.
