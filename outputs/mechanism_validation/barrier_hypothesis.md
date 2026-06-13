# Barrier Hypothesis

Question: are U-Net non-optimal paths associated with severe overestimation?

## Summary Statistics

- all: maps=2000, non_optimal=91, cost_gap=0.102, over=0.535, large_over=0.131, consistency=0.219
- structured: maps=1600, non_optimal=84, cost_gap=0.116, over=0.519, large_over=0.126, consistency=0.216
- random: maps=400, non_optimal=7, cost_gap=0.045, over=0.602, large_over=0.152, consistency=0.231
- non_optimal: maps=91, non_optimal=91, cost_gap=2.242, over=0.652, large_over=0.239, consistency=0.225
- maze_like: maps=400, non_optimal=10, cost_gap=0.055, over=0.368, large_over=0.091, consistency=0.189
- bottleneck: maps=400, non_optimal=0, cost_gap=0.000, over=0.359, large_over=0.015, consistency=0.205
- large_block: maps=400, non_optimal=42, cost_gap=0.240, over=0.584, large_over=0.084, consistency=0.213
- narrow_corridor: maps=400, non_optimal=32, cost_gap=0.170, over=0.764, large_over=0.312, consistency=0.255

## Correlations With Cost Gap

- all overestimate_rate vs cost_gap: Pearson=0.108, Spearman=0.115, n=2000
- all large_overestimate_rate vs cost_gap: Pearson=0.136, Spearman=0.161, n=2000
- all consistency_violation_rate vs cost_gap: Pearson=0.025, Spearman=0.056, n=2000
- structured overestimate_rate vs cost_gap: Pearson=0.124, Spearman=0.131, n=1600
- structured large_overestimate_rate vs cost_gap: Pearson=0.164, Spearman=0.166, n=1600
- structured consistency_violation_rate vs cost_gap: Pearson=0.044, Spearman=0.076, n=1600
- random overestimate_rate vs cost_gap: Pearson=0.080, Spearman=0.101, n=400
- random large_overestimate_rate vs cost_gap: Pearson=0.077, Spearman=0.135, n=400
- random consistency_violation_rate vs cost_gap: Pearson=-0.002, Spearman=0.027, n=400
- non_optimal overestimate_rate vs cost_gap: Pearson=0.061, Spearman=0.025, n=91
- non_optimal large_overestimate_rate vs cost_gap: Pearson=0.149, Spearman=0.136, n=91
- non_optimal consistency_violation_rate vs cost_gap: Pearson=-0.151, Spearman=-0.109, n=91
- maze_like overestimate_rate vs cost_gap: Pearson=0.025, Spearman=0.022, n=400
- maze_like large_overestimate_rate vs cost_gap: Pearson=0.021, Spearman=0.034, n=400
- maze_like consistency_violation_rate vs cost_gap: Pearson=-0.011, Spearman=-0.016, n=400
- bottleneck overestimate_rate vs cost_gap: Pearson=0.000, Spearman=0.000, n=400
- bottleneck large_overestimate_rate vs cost_gap: Pearson=0.000, Spearman=0.000, n=400
- bottleneck consistency_violation_rate vs cost_gap: Pearson=0.000, Spearman=0.000, n=400
- large_block overestimate_rate vs cost_gap: Pearson=0.113, Spearman=0.085, n=400
- large_block large_overestimate_rate vs cost_gap: Pearson=0.253, Spearman=0.161, n=400
- large_block consistency_violation_rate vs cost_gap: Pearson=-0.097, Spearman=-0.014, n=400
- narrow_corridor overestimate_rate vs cost_gap: Pearson=-0.061, Spearman=-0.051, n=400
- narrow_corridor large_overestimate_rate vs cost_gap: Pearson=0.165, Spearman=0.225, n=400
- narrow_corridor consistency_violation_rate vs cost_gap: Pearson=-0.028, Spearman=-0.037, n=400
