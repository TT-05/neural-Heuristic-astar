# Structured Benchmark Filtering Findings

## Threshold Assumptions

- `geometry_easy`: low path_stretch, low corridor_rate, and low articulation_count.
- `obstacle_structured`: structure label contains bottleneck, narrow corridor, maze-like, or dense obstacles.
- `high_stretch`: high path_stretch bin.
- `high_corridor`: high corridor_rate bin.
- `high_articulation`: high articulation_count bin.
- Reused thresholds from `difficulty_thresholds.json`: `{'optimal_cost': [10.0, 18.0], 'obstacle_density': [0.19, 0.315], 'path_stretch': [1.0, 1.1481481481481481], 'corridor_rate': [0.22935779816513763, 0.43617021276595747], 'articulation_count': [14.0, 52.0]}`.

## Subset Gaps

- geometry_easy: maps=102, U-Net-MLP expanded=16.84, U-Net-Manhattan expanded=-11.49, MLP-Manhattan expanded=-28.33, U-Net better than MLP count=2, MLP better than U-Net count=84
- obstacle_structured: maps=231, U-Net-MLP expanded=2.21, U-Net-Manhattan expanded=-7.28, MLP-Manhattan expanded=-9.49, U-Net better than MLP count=74, MLP better than U-Net count=105
- high_stretch: maps=132, U-Net-MLP expanded=2.08, U-Net-Manhattan expanded=-9.38, MLP-Manhattan expanded=-11.45, U-Net better than MLP count=56, MLP better than U-Net count=65
- high_corridor: maps=132, U-Net-MLP expanded=2.42, U-Net-Manhattan expanded=-3.56, MLP-Manhattan expanded=-5.98, U-Net better than MLP count=45, MLP better than U-Net count=55
- high_articulation: maps=132, U-Net-MLP expanded=2.25, U-Net-Manhattan expanded=-3.80, MLP-Manhattan expanded=-6.05, U-Net better than MLP count=46, MLP better than U-Net count=52

## Method Summaries


### geometry_easy
- manhattan: maps=102, expanded=48.25, runtime=0.0001261752844869739, path_length=13.80, optimality=1.0, overestimate=0.0
- mlp_table: maps=102, expanded=19.92, runtime=0.0001447463235879196, path_length=13.80, optimality=1.0, overestimate=0.9268598779715924
- unet: maps=102, expanded=36.76, runtime=0.0002349747255007684, path_length=13.80, optimality=1.0, overestimate=0.4751203899681288

### obstacle_structured
- manhattan: maps=231, expanded=43.91, runtime=0.00010162088744251866, path_length=15.63, optimality=1.0, overestimate=0.0
- mlp_table: maps=231, expanded=34.42, runtime=0.00018183320360517762, path_length=15.63, optimality=1.0, overestimate=0.4479961791631468
- unet: maps=231, expanded=36.63, runtime=0.00019219339387805063, path_length=15.69, optimality=0.974025974025974, overestimate=0.660480228529625

### high_stretch
- manhattan: maps=132, expanded=56.52, runtime=0.00013039775748521524, path_length=18.60, optimality=1.0, overestimate=0.0
- mlp_table: maps=132, expanded=45.06, runtime=0.00023218752290411044, path_length=18.60, optimality=1.0, overestimate=0.4299842632618624
- unet: maps=132, expanded=47.14, runtime=0.00024279044696000213, path_length=18.64, optimality=0.9848484848484849, overestimate=0.5793254911320466

### high_corridor
- manhattan: maps=132, expanded=40.32, runtime=9.07601363933557e-05, path_length=15.36, optimality=1.0, overestimate=0.0
- mlp_table: maps=132, expanded=34.33, runtime=0.0001749293411111549, path_length=15.36, optimality=1.0, overestimate=0.3886371133399963
- unet: maps=132, expanded=36.76, runtime=0.00018206508329140806, path_length=15.43, optimality=0.9696969696969697, overestimate=0.7009048204927383

### high_articulation
- manhattan: maps=132, expanded=40.78, runtime=9.14075682169024e-05, path_length=15.51, optimality=1.0, overestimate=0.0
- mlp_table: maps=132, expanded=34.73, runtime=0.0001769337502192877, path_length=15.51, optimality=1.0, overestimate=0.38608311924365285
- unet: maps=132, expanded=36.98, runtime=0.00018368437874662712, path_length=15.55, optimality=0.9772727272727273, overestimate=0.694495984593329

## Answers

MLP dominates most clearly on the `geometry_easy` subset, where radial distance information is enough and obstacle-aware prediction has little room to help. U-Net becomes much more competitive on `obstacle_structured`, `high_corridor`, and `high_articulation` subsets, although MLP still has a small average edge in this run. The strongest evidence for obstacle-aware value is the subset where the U-Net-MLP expanded-node gap is smallest and U-Net wins a large fraction of pairwise comparisons. Aggregate averages are hiding this behavior because the full benchmark mixes geometry-dominated maps with genuinely structured planning maps.
