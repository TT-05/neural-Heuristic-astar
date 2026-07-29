# Expanded Dataset Statistics

## Baseline Dataset Audit

- Maps: 500 total, randomly split into 400 train / 100 validation.
- Structure distribution: open_random only; no maze_like, bottleneck, large_block, or narrow_corridor maps.
- Obstacle-rate distribution: 0.2 only.
- Map size distribution: 20x20 only.
- Goal distribution: one uniformly selected free-cell goal per map. No start is stored because the original U-Net supervises a full distance-to-goal map rather than individual start-goal samples.

## Expanded Dataset

- Maps: 5000
- Unique grid hashes: 5000
- Map size: 20x20 for every map, matching the model and benchmark.
- Map types: open_random, maze_like, bottleneck, large_block, narrow_corridor.
- Obstacle-rate strata: 0.1, 0.2, 0.3, 0.4.
- Split is deterministic and disjoint by generated grid: 4,000 train / 500 validation / 500 test.
- Reference start-goal distance: mean 14.39, min 0, max 56. Reference pairs are metadata; each training example supervises the complete distance map.

| Split | Map type | Obstacle rate | Maps |
|---|---|---:|---:|
| test | bottleneck | 0.1 | 25 |
| test | bottleneck | 0.2 | 25 |
| test | bottleneck | 0.3 | 25 |
| test | bottleneck | 0.4 | 25 |
| test | large_block | 0.1 | 25 |
| test | large_block | 0.2 | 25 |
| test | large_block | 0.3 | 25 |
| test | large_block | 0.4 | 25 |
| test | maze_like | 0.1 | 25 |
| test | maze_like | 0.2 | 25 |
| test | maze_like | 0.3 | 25 |
| test | maze_like | 0.4 | 25 |
| test | narrow_corridor | 0.1 | 25 |
| test | narrow_corridor | 0.2 | 25 |
| test | narrow_corridor | 0.3 | 25 |
| test | narrow_corridor | 0.4 | 25 |
| test | open_random | 0.1 | 25 |
| test | open_random | 0.2 | 25 |
| test | open_random | 0.3 | 25 |
| test | open_random | 0.4 | 25 |
| train | bottleneck | 0.1 | 200 |
| train | bottleneck | 0.2 | 200 |
| train | bottleneck | 0.3 | 200 |
| train | bottleneck | 0.4 | 200 |
| train | large_block | 0.1 | 200 |
| train | large_block | 0.2 | 200 |
| train | large_block | 0.3 | 200 |
| train | large_block | 0.4 | 200 |
| train | maze_like | 0.1 | 200 |
| train | maze_like | 0.2 | 200 |
| train | maze_like | 0.3 | 200 |
| train | maze_like | 0.4 | 200 |
| train | narrow_corridor | 0.1 | 200 |
| train | narrow_corridor | 0.2 | 200 |
| train | narrow_corridor | 0.3 | 200 |
| train | narrow_corridor | 0.4 | 200 |
| train | open_random | 0.1 | 200 |
| train | open_random | 0.2 | 200 |
| train | open_random | 0.3 | 200 |
| train | open_random | 0.4 | 200 |
| val | bottleneck | 0.1 | 25 |
| val | bottleneck | 0.2 | 25 |
| val | bottleneck | 0.3 | 25 |
| val | bottleneck | 0.4 | 25 |
| val | large_block | 0.1 | 25 |
| val | large_block | 0.2 | 25 |
| val | large_block | 0.3 | 25 |
| val | large_block | 0.4 | 25 |
| val | maze_like | 0.1 | 25 |
| val | maze_like | 0.2 | 25 |
| val | maze_like | 0.3 | 25 |
| val | maze_like | 0.4 | 25 |
| val | narrow_corridor | 0.1 | 25 |
| val | narrow_corridor | 0.2 | 25 |
| val | narrow_corridor | 0.3 | 25 |
| val | narrow_corridor | 0.4 | 25 |
| val | open_random | 0.1 | 25 |
| val | open_random | 0.2 | 25 |
| val | open_random | 0.3 | 25 |
| val | open_random | 0.4 | 25 |
