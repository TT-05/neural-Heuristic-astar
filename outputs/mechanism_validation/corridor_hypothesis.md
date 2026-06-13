# Corridor Smoothness Hypothesis

Question: are corridor failures associated with rough or inconsistent heuristic fields?

High-corridor maps analyzed: 532

- unet_roughness vs unet_expanded: Pearson=-0.211, Spearman=-0.234, n=532
- unet_roughness vs unet_minus_mlp_expanded: Pearson=-0.065, Spearman=0.028, n=532
- unet_roughness vs cost_gap: Pearson=-0.038, Spearman=-0.047, n=532
- unet_gradient_variance vs unet_expanded: Pearson=-0.070, Spearman=-0.053, n=532
- unet_gradient_variance vs unet_minus_mlp_expanded: Pearson=0.030, Spearman=0.201, n=532
- unet_gradient_variance vs cost_gap: Pearson=-0.024, Spearman=-0.003, n=532
- consistency_violation_rate vs unet_expanded: Pearson=-0.314, Spearman=-0.274, n=532
- consistency_violation_rate vs unet_minus_mlp_expanded: Pearson=-0.162, Spearman=-0.070, n=532
- consistency_violation_rate vs cost_gap: Pearson=-0.031, Spearman=-0.034, n=532
- unet_roughness_low: maps=133, mean_metric=1.047, U-Net-MLP expanded=11.504, cost_gap=0.105
- unet_roughness_middle: maps=265, mean_metric=1.200, U-Net-MLP expanded=9.804, cost_gap=0.226
- unet_roughness_high: maps=134, mean_metric=1.417, U-Net-MLP expanded=10.910, cost_gap=0.030
- consistency_violation_rate_low: maps=133, mean_metric=0.215, U-Net-MLP expanded=12.880, cost_gap=0.211
- consistency_violation_rate_middle: maps=265, mean_metric=0.257, U-Net-MLP expanded=10.189, cost_gap=0.121
- consistency_violation_rate_high: maps=134, mean_metric=0.295, U-Net-MLP expanded=8.784, cost_gap=0.134
