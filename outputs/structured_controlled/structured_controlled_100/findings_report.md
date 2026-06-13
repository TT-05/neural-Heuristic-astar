# Controlled Structured Benchmark Findings

Skipped maps by reason: `{}`

## Pairwise Gaps

- bottleneck: maps=400, U-Net-MLP=13.14, U-Net-Manhattan=-12.01, MLP-Manhattan=-25.15, U-Net better than MLP=40, MLP better than U-Net=312, ties=48
- large_block: maps=400, U-Net-MLP=16.06, U-Net-Manhattan=-15.10, MLP-Manhattan=-31.16, U-Net better than MLP=35, MLP better than U-Net=315, ties=50
- maze_like: maps=400, U-Net-MLP=-0.64, U-Net-Manhattan=-12.17, MLP-Manhattan=-11.53, U-Net better than MLP=161, MLP better than U-Net=186, ties=53
- narrow_corridor: maps=400, U-Net-MLP=13.18, U-Net-Manhattan=-0.78, MLP-Manhattan=-13.96, U-Net better than MLP=9, MLP better than U-Net=306, ties=85

## Method Summaries


### bottleneck
- manhattan: maps=400, expanded=70.80, runtime=0.0001732626798639103, path_length=16.84, optimality=1.0, overestimate=0.0
- mlp_table: maps=400, expanded=45.65, runtime=0.00026065406505040303, path_length=16.85, optimality=0.9975, overestimate=0.6919191960014586
- unet: maps=400, expanded=58.79, runtime=0.00033287949491750624, path_length=16.84, optimality=1.0, overestimate=0.3593326821199922

### large_block
- manhattan: maps=400, expanded=61.05, runtime=0.00015313062768655073, path_length=14.77, optimality=1.0, overestimate=0.0
- mlp_table: maps=400, expanded=29.90, runtime=0.00019010314761544578, path_length=14.78, optimality=0.995, overestimate=0.8578139844227735
- unet: maps=400, expanded=45.96, runtime=0.00028580093996424696, path_length=15.01, optimality=0.895, overestimate=0.5840107036266601

### maze_like
- manhattan: maps=400, expanded=76.77, runtime=0.00017057261241461673, path_length=21.62, optimality=1.0, overestimate=0.0
- mlp_table: maps=400, expanded=65.24, runtime=0.000315069090020188, path_length=21.63, optimality=0.9975, overestimate=0.31340467434318797
- unet: maps=400, expanded=64.60, runtime=0.00031821885995668707, path_length=21.68, optimality=0.975, overestimate=0.3675122824694295

### narrow_corridor
- manhattan: maps=400, expanded=39.70, runtime=9.8263659992881e-05, path_length=14.30, optimality=1.0, overestimate=0.0
- mlp_table: maps=400, expanded=25.74, runtime=0.0001581464751234307, path_length=14.30, optimality=1.0, overestimate=0.7620044813438729
- unet: maps=400, expanded=38.91, runtime=0.00022549864264874485, path_length=14.47, optimality=0.92, overestimate=0.7640850184032657
