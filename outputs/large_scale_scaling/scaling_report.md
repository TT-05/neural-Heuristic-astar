# Structure-Aware Neural A* Scaling Report

Manhattan remains the primary A* key. GPU is used only for U-Net inference.

## By size

| Size | Method | Planned | Completed | Skipped | Expanded | Reduction vs Manhattan | Optimality | Total s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1000 | full_map_unet_tiebreak | 250 | 250 | 0 | 51622.48 | 38.2% | 100.0% | 0.3022 |
| 1000 | manhattan_astar | 250 | 250 | 0 | 83588.48 | 0.0% | 100.0% | 0.1188 |
| 1000 | region_cache_patch_32 | 250 | 250 | 0 | 39505.46 | 52.7% | 100.0% | 0.9698 |
| 1000 | region_cache_patch_64 | 250 | 250 | 0 | 43362.23 | 48.1% | 100.0% | 0.6857 |
| 1500 | full_map_unet_tiebreak | 250 | 250 | 0 | 132716.99 | 32.3% | 100.0% | 0.7731 |
| 1500 | manhattan_astar | 250 | 250 | 0 | 196130.66 | 0.0% | 100.0% | 0.3161 |
| 1500 | region_cache_patch_32 | 250 | 250 | 0 | 106981.61 | 45.5% | 100.0% | 2.1770 |
| 1500 | region_cache_patch_64 | 250 | 250 | 0 | 116662.92 | 40.5% | 100.0% | 1.6230 |
| 2000 | full_map_unet_tiebreak | 250 | 250 | 0 | 207476.10 | 38.2% | 100.0% | 1.2474 |
| 2000 | manhattan_astar | 250 | 250 | 0 | 335556.36 | 0.0% | 100.0% | 0.5944 |
| 2000 | region_cache_patch_32 | 250 | 250 | 0 | 168776.36 | 49.7% | 100.0% | 3.7584 |
| 2000 | region_cache_patch_64 | 250 | 250 | 0 | 174283.11 | 48.1% | 100.0% | 2.5924 |
| 500 | full_map_unet_tiebreak | 250 | 250 | 0 | 13744.44 | 37.3% | 100.0% | 0.0724 |
| 500 | manhattan_astar | 250 | 250 | 0 | 21910.14 | 0.0% | 100.0% | 0.0270 |
| 500 | region_cache_patch_32 | 250 | 250 | 0 | 11450.15 | 47.7% | 100.0% | 0.3232 |
| 500 | region_cache_patch_64 | 250 | 250 | 0 | 12692.45 | 42.1% | 100.0% | 0.2001 |

## Evidence-based answers

- full_map_unet_tiebreak expansion reduction by size: 1000: 38.2%, 1500: 32.3%, 2000: 38.2%, 500: 37.3% (not monotonic).
  Structure range: best 59.2% at 500 large_block; worst 6.8% at 1000 maze_like.
- region_cache_patch_32 expansion reduction by size: 1000: 52.7%, 1500: 45.5%, 2000: 49.7%, 500: 47.7% (not monotonic).
  Structure range: best 90.7% at 1000 large_block; worst 4.2% at 2000 maze_like.
- region_cache_patch_64 expansion reduction by size: 1000: 48.1%, 1500: 40.5%, 2000: 48.1%, 500: 42.1% (not monotonic).
  Structure range: best 79.1% at 1000 large_block; worst 0.8% at 2000 maze_like.
- Runtime competitiveness at 500x500: fastest neural method is full_map_unet_tiebreak at 0.0724s, 2.68x Manhattan.
  Full-map is faster than the fastest Region-cache variant (0.0724s versus 0.2001s).
- Runtime competitiveness at 1000x1000: fastest neural method is full_map_unet_tiebreak at 0.3022s, 2.54x Manhattan.
  Full-map is faster than the fastest Region-cache variant (0.3022s versus 0.6857s).
- Runtime competitiveness at 1500x1500: fastest neural method is full_map_unet_tiebreak at 0.7731s, 2.45x Manhattan.
  Full-map is faster than the fastest Region-cache variant (0.7731s versus 1.6230s).
- Runtime competitiveness at 2000x2000: fastest neural method is full_map_unet_tiebreak at 1.2474s, 2.10x Manhattan.
  Full-map is faster than the fastest Region-cache variant (1.2474s versus 2.5924s).
- 1000 region_cache_patch_32: largest measured component is patch extraction at 0.4509s.
- 1000 region_cache_patch_64: largest measured component is patch extraction at 0.3954s.
- 1500 region_cache_patch_32: largest measured component is patch extraction at 1.0144s.
- 1500 region_cache_patch_64: largest measured component is patch extraction at 0.8821s.
- 2000 region_cache_patch_32: largest measured component is patch extraction at 1.7097s.
- 2000 region_cache_patch_64: largest measured component is patch extraction at 1.3794s.
- 500 region_cache_patch_32: largest measured component is patch extraction at 0.1468s.
- 500 region_cache_patch_64: largest measured component is patch extraction at 0.1220s.

Rows marked skipped_infeasible are excluded from means and record a resource limit.
No runtime superiority is claimed unless total runtime is lower than Manhattan on the same completed sample set.
