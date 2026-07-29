# Ranking-Loss Lambda Sweep

All runs use the unchanged 5,000-map split, U-Net architecture, optimizer, batch size, 50-epoch schedule, ranking-pair formulation, and 2,000 benchmark cases. Only lambda in `L = MSE + lambda * ranking_loss` changes.

| Lambda | MAE | MSE | Tie-set ordering | Direct expanded | Direct optimality | Tie-break expanded | Tie-break optimality |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 3.338 | 83.893 | 0.983 | 30.63 | 0.9950 | 32.73 | 1.0000 |
| 0.25 | 3.428 | 82.308 | 0.977 | 31.45 | 0.9940 | 32.71 | 1.0000 |
| 0.5 | 3.560 | 80.232 | 0.984 | 31.13 | 0.9975 | 32.70 | 1.0000 |
| 0.75 | 3.552 | 75.781 | 0.981 | 31.94 | 0.9985 | 32.74 | 1.0000 |
| 1.0 | 3.393 | 76.008 | 0.967 | 34.72 | 0.9990 | 32.97 | 1.0000 |
| 2.0 | 3.713 | 68.119 | 0.969 | 36.81 | 0.9990 | 32.91 | 1.0000 |

## Main Question

The lowest Direct U-Net mean expansion count occurs at lambda=0.1 (30.63, optimality 0.9950).
The lowest U-Net tie-break mean expansion count occurs at lambda=0.5 (32.70).
The lowest held-out prediction MSE occurs at lambda=2.0 (68.119).
These may be different lambdas; the sweep reports the observed trade-off rather than defining one universal optimum.

See `results.csv` for per-case data and summary rows.
