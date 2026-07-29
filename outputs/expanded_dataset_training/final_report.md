# Expanded Dataset Training Report

The U-Net architecture, masked MSE loss, Adam optimizer, batch size, and 50-epoch schedule were unchanged. Only training-data quantity and composition changed.

## Held-out Test Comparison

| Model | MAE | MSE | Mean expanded nodes | Tie-set top-1 accuracy | High-impact events/case | Optimality |
|---|---:|---:|---:|---:|---:|---:|
| expanded | 0.972 | 3.546 | 12.498 | 0.973 | 0.058 | 1.000 |
| old | 3.058 | 27.791 | 12.786 | 0.924 | 0.062 | 1.000 |

## Structure Results

See `structure_performance.csv` and `search_performance_comparison.csv` for all structure-level values.

## Answers

1. Generalization improved on the independent balanced test split: MAE 3.058 -> 0.972.
2. The largest absolute MAE improvement is in `maze_like`.
3. Tie-set oracle-top1 agreement changed from 0.924 to 0.973.
4. Mean U-Net tie-break expansions changed from 12.786 to 12.498; all test paths remained optimal.
5. This experiment isolates data coverage. Remaining errors after improvement are evidence that model capacity and/or the MSE objective may still limit search-relevant ordering, but do not establish that conclusion alone.
