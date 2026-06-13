# Research Synthesis: Learned Heuristics for A* Search

This report synthesizes the benchmark, qualitative, mechanism-validation, and route-critical-cell analyses. It is written as a Results and Discussion narrative and does not propose or evaluate any new search algorithm.

## 1. Experimental Setup

The experiments compare four A* heuristic settings: Dijkstra/zero heuristic, Manhattan distance, an MLP table heuristic, and a U-Net heuristic. The synthesis uses 400 random start-goal maps and 1600 controlled structured maps, for 2000 evaluated maps in the mechanism-level analyses. The structured benchmark contains four controlled map families: maze_like, bottleneck, large_block, and narrow_corridor.

Random start-goal benchmark summary:

| Method | Mean expanded | Optimality | Mean path length | Overestimate rate | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| dijkstra | 137.58 | 1.000 | 14.90 | 0.000 | 14.804 | 16.404 |
| manhattan | 44.99 | 1.000 | 14.90 | 0.000 | 2.450 | 3.732 |
| mlp_table | 28.73 | 1.000 | 14.90 | 0.622 | 2.410 | 3.371 |
| unet | 35.23 | 0.983 | 14.95 | 0.602 | 2.595 | 3.317 |

Controlled structured benchmark summary:

| Method | Mean expanded | Optimality | Mean path length | Overestimate rate | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| dijkstra | 153.40 | 1.000 | 16.88 | 0.000 | 16.612 | 18.663 |
| manhattan | 62.08 | 1.000 | 16.88 | 0.000 | 3.030 | 4.913 |
| mlp_table | 41.63 | 0.998 | 16.89 | 0.656 | 3.225 | 4.623 |
| unet | 52.07 | 0.948 | 17.00 | 0.519 | 3.380 | 4.732 |

## 2. Benchmark Evolution

The evaluation evolved from a fixed start-goal protocol to random start-goal sampling, then to structure-aware analysis and controlled structured maps. This progression matters because the fixed protocol overemphasized global coordinate regularities and made the MLP table heuristic appear stronger than it remained under more varied start-goal conditions.

In the fixed start-goal benchmark, U-Net averaged 152.14 expanded nodes, while MLP table averaged 116.19. This was the setting where the MLP advantage was most pronounced.

Random start-goal evaluation reduced that gap by removing the single-route bias. The controlled structured benchmark then separated geometric regimes that were mixed together in random maps.

The structure-aware benchmark grouped random maps into labels such as open space, sparse obstacles, dense obstacles, maze-like, bottleneck, large obstacle block, narrow corridor, and multiple alternative routes. These categories showed that learned heuristics should not be judged only by aggregate averages: the same heuristic can be useful in one geometry class and fragile in another.

## 3. Main Empirical Findings

Across the random and controlled structured benchmarks, U-Net substantially reduces search relative to Dijkstra: Dijkstra: expanded=150.24, optimality=1.000, path_length=16.49, overestimate=0.000; U-Net: expanded=48.70, optimality=0.955, path_length=16.59, overestimate=0.535. This corresponds to a 67.6% reduction in mean expanded nodes.

Relative to Manhattan, U-Net also reduces expansions on average: Manhattan expanded 58.66 nodes versus U-Net 48.70, a 17.0% reduction. However, unlike Manhattan, U-Net is not guaranteed admissible in these experiments and has a lower optimality rate.

Relative to MLP table, U-Net remains slightly weaker on the aggregate expansion metric: MLP expanded 39.05 nodes versus U-Net 48.70. This confirms that U-Net is competitive but not uniformly dominant. The important result is therefore not that U-Net is always the best heuristic, but that its behavior is strongly conditioned by map structure and by how its predicted field orders route-relevant regions.

## 4. Structure-Dependent Behavior

The structured benchmark shows that learned-heuristic quality is geometry-dependent. The aggregate table below summarizes the controlled structured families.

| Structure | U-Net expanded | MLP expanded | Manhattan expanded | U-Net optimality | U-Net minus MLP expanded |
|---|---:|---:|---:|---:|---:|
| maze_like | 64.60 | 65.24 | 76.77 | 0.975 | -0.64 |
| bottleneck | 58.79 | 45.65 | 70.80 | 1.000 | 13.14 |
| large_block | 45.96 | 29.90 | 61.05 | 0.895 | 16.06 |
| narrow_corridor | 38.91 | 25.74 | 39.70 | 0.920 | 13.18 |

In geometry_easy or open-space-like settings, MLP table often has a strong advantage because coordinate-distance regularities are sufficient and obstacles do not require much global scene interpretation. In obstacle_structured settings, including bottlenecks, large blocks, and corridors, U-Net sometimes benefits from spatial context but also becomes vulnerable to localized overestimation. Maze-like maps are the most favorable controlled structure for U-Net: its expansion gap relative to MLP is near zero and sometimes negative, suggesting useful global route bias. Large-block and narrow-corridor maps expose the clearest failure modes, with higher U-Net expansion and lower optimality.

## 5. Qualitative Case-Study Findings

| Case group | Typical behavior | Mechanism interpretation |
|---|---|---|
| maze_like_unet_wins | U-Net often produces a useful global field on long maze-like routes and reduces exploration relative to both MLP and Manhattan. | The selected successes are dominated by underestimation or mild overestimation rather than strong positive barriers; the field can still guide search through the maze layout. |
| maze_like_mlp_wins | U-Net remains optimal but expands much more than MLP in these maze-like cases. | The U-Net field appears less aligned with the useful route ordering: overestimation is higher and the predicted field is rougher/noisier, causing extra exploration without changing final path cost. |
| large_block_unet_nonoptimal | U-Net can create harmful barriers around large blocks, producing both extra expansions and non-optimal paths. | Positive error regions are large enough to make the true shortest route look expensive; this is consistent with artificial heuristic barriers near necessary detours/passages. |
| narrow_corridor_unet_fails | U-Net often overestimates corridor states and does not preserve a smooth corridor gradient. | High overestimation along or near narrow passages makes the corridor less attractive and increases unnecessary expansion; in some cases it changes the selected path. |

The qualitative cases support a consistent interpretation: U-Net succeeds when its field provides a useful global bias toward the route family that contains the solution, and fails when positive error regions make necessary passages or detours look artificially expensive. Maze-like U-Net wins are mostly route-bias cases; large-block and narrow-corridor failures are mostly barrier-like cases.

## 6. Mechanism Validation Findings

The route-ordering hypothesis is supported: U-Net-win maps have better relative ordering than MLP-win maps by 0.029. This is the clearest mechanism-level result because it directly connects search efficiency with the relative ordering of useful regions rather than only absolute regression error.

The barrier hypothesis is also supported: non-optimal U-Net maps have higher overestimation, with an overestimate gap of 0.122 and a large-overestimate gap of 0.114. This supports the interpretation that localized positive error can make useful routes appear too expensive.

The corridor smoothness hypothesis remains suggestive rather than strongly supported. The roughness relationship in high-corridor maps is weak, with Spearman 0.028, so global smoothness is probably too coarse for corridor-specific failure analysis.

## 7. Route-Critical-Cell Findings

### Q1: U-Net Wins

U-Net-win maps show critical ordering advantage gap 0.038 versus global ordering advantage gap 0.029.

### Q2: U-Net Non-Optimality

On non-optimal U-Net maps, critical minus global overestimate rate is 0.133. On optimal U-Net maps it is 0.103.

### Q3: Predictive Power

For U-Net minus MLP expansion gap, best global Spearman is -0.221 (unet_global_mae); best critical Spearman is -0.107 (unet_critical_overestimate_rate).
For U-Net optimality gap, best global Spearman is 0.161 (unet_global_large_overestimate_rate); best critical Spearman is 0.247 (unet_critical_large_overestimate_rate).

These results refine the mechanism story. Route-critical cells are more informative for U-Net optimality failures, especially through large overestimation near critical regions. They are not yet sufficient to explain expansion reduction: in the current analysis, global MAE remains more predictive of U-Net-minus-MLP expansion gap than the tested route-critical metrics.

## 8. Supported Claims

### Strongly Supported Conclusions

- U-Net substantially improves over Dijkstra and usually improves over Manhattan in expansion efficiency, but sacrifices some optimality.
- The MLP table heuristic remains slightly better than U-Net on aggregate expansion, especially in easy geometric settings.
- U-Net behavior is structure-dependent; maze-like maps are comparatively favorable, while large-block and narrow-corridor maps are more fragile.
- Route ordering is a supported mechanism for U-Net wins, and overestimation barriers are supported for U-Net non-optimality.

### Moderately Supported Conclusions

- U-Net success is better described as useful global route bias than as uniformly better distance regression.
- Route-critical overestimation is more predictive of optimality gap than global overestimation alone.
- Qualitative corridor failures likely involve localized passage barriers, although global roughness does not strongly validate this.

### Speculative Hypotheses

- A learned heuristic may need route-critical calibration more than global calibration to preserve optimality.
- Some expansion advantages may depend on broad global field shape, while optimality failures depend on local high-error cells.
- Corridor failures may require passage-aware local metrics rather than map-wide smoothness or consistency summaries.

## 9. Limitations

- The mechanism metrics are observational and do not prove causality.
- Ordering metrics are computed over map cells, not over the dynamic A* open list.
- Route-critical cells are approximated from one reconstructed optimal path and local structural proxies.
- Controlled structured maps are simplified generators, not a complete path-planning distribution.
- Runtime values are small and implementation-dependent, so expanded nodes are the more stable efficiency measure.

## 10. Future Research Directions

Future analysis should focus on sharper local mechanism measurements rather than algorithm changes. The most useful next questions are whether overestimation concentrates on all optimal paths or only one reconstructed path, whether bottleneck and gap cells can be detected more robustly, and whether A* open-list ordering agrees with the static map-cell ordering metrics. Additional benchmarks should vary map scale, obstacle topology, and start-goal distance while preserving the current separation between global field quality, route-critical calibration, and search behavior.

## Overall Interpretation

The evidence supports a nuanced view of learned heuristics for A*. U-Net is not simply a better distance estimator than MLP table, nor is average prediction error sufficient to explain search behavior. Its value appears when spatial context creates useful route ordering, especially in structured maps where Manhattan is under-informed. Its main risk appears when localized overestimation creates artificial barriers near route-critical cells. The central lesson is that learned heuristic evaluation should measure both global search guidance and local route-critical reliability.
