# Route-Critical Cell Analysis

This analysis tests whether U-Net search behavior is better explained by prediction behavior on route-critical cells than by averages over all reachable free cells. It does not rerun training or modify model/search code.

Maps analyzed: 2000

## Route-Critical Definition

- Optimal shortest-path cells from BFS distance labels.
- Free cells within configurable path distance k.
- Low-degree narrow-passage cells and articulation points near that path neighborhood.

## Q1: U-Net Wins

U-Net-win maps show critical ordering advantage gap 0.038 versus global ordering advantage gap 0.029.

## Q2: U-Net Non-Optimality

On non-optimal U-Net maps, critical minus global overestimate rate is 0.133. On optimal U-Net maps it is 0.103.

## Q3: Predictive Power

For U-Net minus MLP expansion gap, best global Spearman is -0.221 (unet_global_mae); best critical Spearman is -0.107 (unet_critical_overestimate_rate).
For U-Net optimality gap, best global Spearman is 0.161 (unet_global_large_overestimate_rate); best critical Spearman is 0.247 (unet_critical_large_overestimate_rate).

## Structured Subsets

- maze_like: maps=400, expansion_gap=-0.640, cost_gap=0.055, critical_over=0.591, critical_order_delta=0.003
- bottleneck: maps=400, expansion_gap=13.137, cost_gap=0.000, critical_over=0.498, critical_order_delta=-0.009
- large_block: maps=400, expansion_gap=16.060, cost_gap=0.240, critical_over=0.632, critical_order_delta=-0.020
- narrow_corridor: maps=400, expansion_gap=13.175, cost_gap=0.170, critical_over=0.782, critical_order_delta=-0.055

## Final Comparison

Compare predictive power of global metrics vs route-critical metrics using the predictive_power rows in route_critical_statistics.csv. Higher absolute Spearman means stronger monotonic explanatory power for the target search behavior.
