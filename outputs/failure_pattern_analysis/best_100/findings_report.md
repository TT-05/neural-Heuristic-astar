# Failure Pattern Findings

## Case Buckets

- `unet_better_than_manhattan`: 20 cases, mean U-Net-Manhattan expanded -95.65, mean MAE 1.73, mean overestimate rate 0.588, mean consistency violation 0.224
- `unet_worse_than_manhattan`: 13 cases, mean U-Net-Manhattan expanded 58.08, mean MAE 5.69, mean overestimate rate 0.825, mean consistency violation 0.175
- `unet_non_optimal`: 3 cases, mean U-Net-Manhattan expanded -16.00, mean MAE 3.07, mean overestimate rate 0.734, mean consistency violation 0.222

## Structural Patterns

- unet_better_than_manhattan / multiple alternative routes: 20 cases, mean U-Net-Manhattan expanded -95.65, mean MAE 1.73, mean overestimate 0.588
- unet_better_than_manhattan / open space: 20 cases, mean U-Net-Manhattan expanded -95.65, mean MAE 1.73, mean overestimate 0.588
- unet_non_optimal / bottleneck: 2 cases, mean U-Net-Manhattan expanded 1.50, mean MAE 3.44, mean overestimate 0.694
- unet_non_optimal / dense obstacles: 1 cases, mean U-Net-Manhattan expanded -60.00, mean MAE 3.47, mean overestimate 0.878
- unet_non_optimal / maze-like: 1 cases, mean U-Net-Manhattan expanded -60.00, mean MAE 3.47, mean overestimate 0.878
- unet_non_optimal / multiple alternative routes: 3 cases, mean U-Net-Manhattan expanded -16.00, mean MAE 3.07, mean overestimate 0.734
- unet_non_optimal / narrow corridor: 1 cases, mean U-Net-Manhattan expanded -60.00, mean MAE 3.47, mean overestimate 0.878
- unet_non_optimal / sparse obstacles: 2 cases, mean U-Net-Manhattan expanded 6.00, mean MAE 2.87, mean overestimate 0.661
- unet_worse_than_manhattan / bottleneck: 13 cases, mean U-Net-Manhattan expanded 58.08, mean MAE 5.69, mean overestimate 0.825
- unet_worse_than_manhattan / dense obstacles: 4 cases, mean U-Net-Manhattan expanded 54.25, mean MAE 5.36, mean overestimate 0.844
- unet_worse_than_manhattan / large obstacle block: 2 cases, mean U-Net-Manhattan expanded 40.50, mean MAE 6.94, mean overestimate 0.764
- unet_worse_than_manhattan / maze-like: 7 cases, mean U-Net-Manhattan expanded 54.29, mean MAE 6.27, mean overestimate 0.823
- unet_worse_than_manhattan / multiple alternative routes: 6 cases, mean U-Net-Manhattan expanded 62.00, mean MAE 5.20, mean overestimate 0.849
- unet_worse_than_manhattan / narrow corridor: 6 cases, mean U-Net-Manhattan expanded 43.50, mean MAE 6.54, mean overestimate 0.816
- unet_worse_than_manhattan / sparse obstacles: 6 cases, mean U-Net-Manhattan expanded 62.50, mean MAE 5.02, mean overestimate 0.827

## Error Field Regions

- unet_better_than_manhattan / near_goal: mean abs error 1.37, mean max abs error 4.52
- unet_better_than_manhattan / near_obstacle: mean abs error 2.09, mean max abs error 9.19
- unet_better_than_manhattan / corridor: mean abs error 2.55, mean max abs error 8.06
- unet_better_than_manhattan / bottleneck: mean abs error 2.79, mean max abs error 5.24
- unet_better_than_manhattan / optimal_path: mean abs error 2.10, mean max abs error 6.12
- unet_better_than_manhattan / other: mean abs error 1.40, mean max abs error 7.10
- unet_non_optimal / near_goal: mean abs error 1.49, mean max abs error 5.17
- unet_non_optimal / near_obstacle: mean abs error 3.19, mean max abs error 13.09
- unet_non_optimal / corridor: mean abs error 3.55, mean max abs error 13.09
- unet_non_optimal / bottleneck: mean abs error 3.87, mean max abs error 10.54
- unet_non_optimal / optimal_path: mean abs error 2.63, mean max abs error 6.67
- unet_non_optimal / other: mean abs error 2.86, mean max abs error 8.46
- unet_worse_than_manhattan / near_goal: mean abs error 9.77, mean max abs error 14.25
- unet_worse_than_manhattan / near_obstacle: mean abs error 5.80, mean max abs error 16.75
- unet_worse_than_manhattan / corridor: mean abs error 5.96, mean max abs error 16.71
- unet_worse_than_manhattan / bottleneck: mean abs error 6.25, mean max abs error 15.39
- unet_worse_than_manhattan / optimal_path: mean abs error 5.61, mean max abs error 14.50
- unet_worse_than_manhattan / other: mean abs error 5.23, mean max abs error 12.36

## MLP vs U-Net

- obstacle_rate=0.1: paired maps 96, MLP table better 95, U-Net better 1, mean U-Net-MLP expanded 88.44
- obstacle_rate=0.2: paired maps 76, MLP table better 60, U-Net better 14, mean U-Net-MLP expanded 30.63
- obstacle_rate=0.3: paired maps 47, MLP table better 28, U-Net better 17, mean U-Net-MLP expanded 12.23
- obstacle_rate=0.4: paired maps 6, MLP table better 4, U-Net better 2, mean U-Net-MLP expanded 12.50

## Representative Cases

- unet_better_than_manhattan_rate0.1_seed18_size20: unet_better_than_manhattan, structures [open space; multiple alternative routes], MAE 2.20, overestimate 0.777, consistency violation 0.214
- unet_worse_than_manhattan_rate0.2_seed28_size20: unet_worse_than_manhattan, structures [sparse obstacles; bottleneck; multiple alternative routes], MAE 8.56, overestimate 0.958, consistency violation 0.099
- unet_non_optimal_rate0.3_seed23_size20: unet_non_optimal, structures [dense obstacles; narrow corridor; bottleneck; multiple alternative routes; maze-like], MAE 3.47, overestimate 0.878, consistency violation 0.224
- unet_better_than_manhattan_rate0.1_seed12_size20: unet_better_than_manhattan, structures [open space; multiple alternative routes], MAE 1.55, overestimate 0.501, consistency violation 0.238

## Interpretation

U-Net's advantage over Manhattan is clearest on open or sparse maps, where a more aggressive learned field can reduce exploration without forcing many corridor decisions. Its worse and non-optimal cases are concentrated in denser maps with corridor, bottleneck, or route-choice structure, where local overestimation and inconsistency can redirect search or make it expand extra nodes. The largest errors in bad cases are not uniformly distributed: they are amplified near the goal, near obstacles, and along corridor or bottleneck cells. MLP table remains strong because the current benchmark has fixed start-goal geometry and many shortest paths are still well explained by radial distance-to-goal. U-Net receives obstacle information, but this benefit is offset by calibration/admissibility/consistency errors in structured obstacle regions.
