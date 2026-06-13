# Qualitative Case-Study Analysis

This report summarizes the 20 selected controlled-structured cases. It uses only generated case-study outputs: `selected_cases.csv`, true distance maps, U-Net predictions, U-Net error maps, MLP predictions, and path overlays. It does not rerun training or modify search/model code.

## Concise Group Table

| Group | Typical U-Net behavior | Optimality fails? | Likely mechanism | Evidence from maps/errors |
|---|---|---|---|---|
| `maze_like_unet_wins` | U-Net often produces a useful global field on long maze-like routes and reduces exploration relative to both MLP and Manhattan. | Mostly optimal in this selected group, with one selected case showing a small cost gap. | The selected successes are dominated by underestimation or mild overestimation rather than strong positive barriers; the field can still guide search through the maze layout. | U-Net has lower expanded nodes than MLP by construction, moderate or low large-overestimate rates, and path overlays show the search path following the viable route family. |
| `maze_like_mlp_wins` | U-Net remains optimal but expands much more than MLP in these maze-like cases. | No selected case has a positive U-Net cost gap. | The U-Net field appears less aligned with the useful route ordering: overestimation is higher and the predicted field is rougher/noisier, causing extra exploration without changing final path cost. | The group has high U-Net overestimate rates and large U-Net-minus-MLP expanded gaps while cost gap remains zero. |
| `large_block_unet_nonoptimal` | U-Net can create harmful barriers around large blocks, producing both extra expansions and non-optimal paths. | Optimality fails by definition in this selected group. | Positive error regions are large enough to make the true shortest route look expensive; this is consistent with artificial heuristic barriers near necessary detours/passages. | Selected cases show positive U-Net cost gaps, high overestimate rates, and large positive error regions in the U-Net error maps. |
| `narrow_corridor_unet_fails` | U-Net often overestimates corridor states and does not preserve a smooth corridor gradient. | Some selected cases remain optimal, but at least one selected case is non-optimal. | High overestimation along or near narrow passages makes the corridor less attractive and increases unnecessary expansion; in some cases it changes the selected path. | The selected failures have very high overestimate rates, high local consistency violation/roughness, and U-Net expanded nodes at or above Manhattan. |

## Numeric Group Diagnostics

| Group | Cases | U-Net-MLP expanded | U-Net cost gap | Overestimate rate | Large overestimate rate | true/U-Net corr | true/MLP corr | U-Net roughness | Consistency violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `maze_like_unet_wins` | 5 | -47.20 | 0.40 | 0.275 | 0.047 | 0.838 | 0.810 | 0.92 | 0.199 |
| `maze_like_mlp_wins` | 5 | 56.60 | 0.00 | 0.612 | 0.298 | 0.810 | 0.830 | 0.96 | 0.201 |
| `large_block_unet_nonoptimal` | 5 | 55.00 | 4.40 | 0.763 | 0.184 | 0.900 | 0.951 | 0.93 | 0.198 |
| `narrow_corridor_unet_fails` | 5 | 54.00 | 0.80 | 0.895 | 0.711 | 0.613 | 0.957 | 1.20 | 0.238 |

## maze_like_unet_wins

U-Net often produces a useful global field on long maze-like routes and reduces exploration relative to both MLP and Manhattan. The selected successes are dominated by underestimation or mild overestimation rather than strong positive barriers; the field can still guide search through the maze layout.

Representative image references:

- [path overlay](maze_like_unet_wins/maze_like_rate0.1_seed51_s11-2_g7-18/path_overlay.png)
- [obstacle map](maze_like_unet_wins/maze_like_rate0.1_seed51_s11-2_g7-18/obstacle_map.png)
- [true distance map](maze_like_unet_wins/maze_like_rate0.1_seed51_s11-2_g7-18/true_distance_map.png)
- [U-Net predicted map](maze_like_unet_wins/maze_like_rate0.1_seed51_s11-2_g7-18/unet_predicted_map.png)
- [U-Net error map](maze_like_unet_wins/maze_like_rate0.1_seed51_s11-2_g7-18/unet_error_map.png)
- [MLP predicted map](maze_like_unet_wins/maze_like_rate0.1_seed51_s11-2_g7-18/mlp_predicted_map.png)

Selected cases:

- `maze_like_rate0.1_seed51_s11-2_g7-18`: U-Net-MLP expanded -62, cost gap 0, overestimate 0.230, large-overestimate 0.047, true/U-Net corr 0.850, roughness 0.86
- `maze_like_rate0.1_seed39_s3-16_g16-9`: U-Net-MLP expanded -50, cost gap 0, overestimate 0.110, large-overestimate 0.000, true/U-Net corr 0.937, roughness 0.92
- `maze_like_rate0.3_seed1_s19-8_g1-9`: U-Net-MLP expanded -42, cost gap 0, overestimate 0.286, large-overestimate 0.039, true/U-Net corr 0.897, roughness 0.97
- `maze_like_rate0.4_seed10_s12-17_g6-1`: U-Net-MLP expanded -41, cost gap 2, overestimate 0.265, large-overestimate 0.040, true/U-Net corr 0.912, roughness 0.86
- `maze_like_rate0.4_seed62_s10-13_g19-9`: U-Net-MLP expanded -41, cost gap 0, overestimate 0.482, large-overestimate 0.109, true/U-Net corr 0.594, roughness 0.99

## maze_like_mlp_wins

U-Net remains optimal but expands much more than MLP in these maze-like cases. The U-Net field appears less aligned with the useful route ordering: overestimation is higher and the predicted field is rougher/noisier, causing extra exploration without changing final path cost.

Representative image references:

- [path overlay](maze_like_mlp_wins/maze_like_rate0.2_seed99_s10-0_g0-16/path_overlay.png)
- [obstacle map](maze_like_mlp_wins/maze_like_rate0.2_seed99_s10-0_g0-16/obstacle_map.png)
- [true distance map](maze_like_mlp_wins/maze_like_rate0.2_seed99_s10-0_g0-16/true_distance_map.png)
- [U-Net predicted map](maze_like_mlp_wins/maze_like_rate0.2_seed99_s10-0_g0-16/unet_predicted_map.png)
- [U-Net error map](maze_like_mlp_wins/maze_like_rate0.2_seed99_s10-0_g0-16/unet_error_map.png)
- [MLP predicted map](maze_like_mlp_wins/maze_like_rate0.2_seed99_s10-0_g0-16/mlp_predicted_map.png)

Selected cases:

- `maze_like_rate0.2_seed99_s10-0_g0-16`: U-Net-MLP expanded 79, cost gap 0, overestimate 0.685, large-overestimate 0.272, true/U-Net corr 0.918, roughness 0.88
- `maze_like_rate0.3_seed31_s19-7_g0-7`: U-Net-MLP expanded 76, cost gap 0, overestimate 0.568, large-overestimate 0.318, true/U-Net corr 0.755, roughness 1.16
- `maze_like_rate0.4_seed21_s0-9_g19-9`: U-Net-MLP expanded 49, cost gap 0, overestimate 0.465, large-overestimate 0.119, true/U-Net corr 0.811, roughness 0.96
- `maze_like_rate0.3_seed54_s18-12_g0-3`: U-Net-MLP expanded 40, cost gap 0, overestimate 0.606, large-overestimate 0.311, true/U-Net corr 0.920, roughness 0.88
- `maze_like_rate0.3_seed70_s5-4_g0-19`: U-Net-MLP expanded 39, cost gap 0, overestimate 0.738, large-overestimate 0.469, true/U-Net corr 0.645, roughness 0.90

## large_block_unet_nonoptimal

U-Net can create harmful barriers around large blocks, producing both extra expansions and non-optimal paths. Positive error regions are large enough to make the true shortest route look expensive; this is consistent with artificial heuristic barriers near necessary detours/passages.

Representative image references:

- [path overlay](large_block_unet_nonoptimal/large_block_rate0.3_seed97_s19-3_g1-13/path_overlay.png)
- [obstacle map](large_block_unet_nonoptimal/large_block_rate0.3_seed97_s19-3_g1-13/obstacle_map.png)
- [true distance map](large_block_unet_nonoptimal/large_block_rate0.3_seed97_s19-3_g1-13/true_distance_map.png)
- [U-Net predicted map](large_block_unet_nonoptimal/large_block_rate0.3_seed97_s19-3_g1-13/unet_predicted_map.png)
- [U-Net error map](large_block_unet_nonoptimal/large_block_rate0.3_seed97_s19-3_g1-13/unet_error_map.png)
- [MLP predicted map](large_block_unet_nonoptimal/large_block_rate0.3_seed97_s19-3_g1-13/mlp_predicted_map.png)

Selected cases:

- `large_block_rate0.3_seed97_s19-3_g1-13`: U-Net-MLP expanded 170, cost gap 12, overestimate 0.767, large-overestimate 0.522, true/U-Net corr 0.617, roughness 0.96
- `large_block_rate0.3_seed90_s15-4_g7-12`: U-Net-MLP expanded 21, cost gap 4, overestimate 0.837, large-overestimate 0.139, true/U-Net corr 0.940, roughness 0.93
- `large_block_rate0.1_seed11_s14-16_g9-0`: U-Net-MLP expanded 55, cost gap 2, overestimate 0.468, large-overestimate 0.059, true/U-Net corr 0.979, roughness 0.94
- `large_block_rate0.1_seed12_s5-14_g17-11`: U-Net-MLP expanded 21, cost gap 2, overestimate 0.920, large-overestimate 0.133, true/U-Net corr 0.977, roughness 0.90
- `large_block_rate0.1_seed17_s5-15_g5-3`: U-Net-MLP expanded 8, cost gap 2, overestimate 0.824, large-overestimate 0.068, true/U-Net corr 0.986, roughness 0.94

## narrow_corridor_unet_fails

U-Net often overestimates corridor states and does not preserve a smooth corridor gradient. High overestimation along or near narrow passages makes the corridor less attractive and increases unnecessary expansion; in some cases it changes the selected path.

Representative image references:

- [path overlay](narrow_corridor_unet_fails/narrow_corridor_rate0.4_seed38_s3-15_g15-7/path_overlay.png)
- [obstacle map](narrow_corridor_unet_fails/narrow_corridor_rate0.4_seed38_s3-15_g15-7/obstacle_map.png)
- [true distance map](narrow_corridor_unet_fails/narrow_corridor_rate0.4_seed38_s3-15_g15-7/true_distance_map.png)
- [U-Net predicted map](narrow_corridor_unet_fails/narrow_corridor_rate0.4_seed38_s3-15_g15-7/unet_predicted_map.png)
- [U-Net error map](narrow_corridor_unet_fails/narrow_corridor_rate0.4_seed38_s3-15_g15-7/unet_error_map.png)
- [MLP predicted map](narrow_corridor_unet_fails/narrow_corridor_rate0.4_seed38_s3-15_g15-7/mlp_predicted_map.png)

Selected cases:

- `narrow_corridor_rate0.4_seed38_s3-15_g15-7`: U-Net-MLP expanded 70, cost gap 0, overestimate 1.000, large-overestimate 1.000, true/U-Net corr 0.712, roughness 1.24
- `narrow_corridor_rate0.3_seed35_s8-15_g12-8`: U-Net-MLP expanded 67, cost gap 0, overestimate 1.000, large-overestimate 1.000, true/U-Net corr -0.179, roughness 0.92
- `narrow_corridor_rate0.2_seed31_s16-3_g5-12`: U-Net-MLP expanded 43, cost gap 0, overestimate 0.952, large-overestimate 0.617, true/U-Net corr 0.865, roughness 1.30
- `narrow_corridor_rate0.4_seed21_s1-5_g13-13`: U-Net-MLP expanded 46, cost gap 4, overestimate 0.557, large-overestimate 0.320, true/U-Net corr 0.827, roughness 1.20
- `narrow_corridor_rate0.4_seed79_s15-4_g7-12`: U-Net-MLP expanded 44, cost gap 0, overestimate 0.966, large-overestimate 0.618, true/U-Net corr 0.838, roughness 1.34

## Interpretation Boundaries

These conclusions are limited to the selected top-k cases and should be treated as qualitative evidence, not as a full distributional claim. The evidence supports pattern hypotheses to inspect visually: U-Net success in maze-like cases often comes from useful global route bias, while failures in large-block and narrow-corridor cases are consistent with overestimated passages, rough fields, and artificial heuristic barriers.
