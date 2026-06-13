# Controlled Structured Benchmark Findings

Skipped maps by reason: `{}`

## Pairwise Gaps

- bottleneck: maps=12, U-Net-MLP=9.08, U-Net-Manhattan=-13.25, MLP-Manhattan=-22.33, U-Net better than MLP=2, MLP better than U-Net=9, ties=1
- large_block: maps=12, U-Net-MLP=24.83, U-Net-Manhattan=-6.33, MLP-Manhattan=-31.17, U-Net better than MLP=0, MLP better than U-Net=9, ties=3
- maze_like: maps=12, U-Net-MLP=-3.25, U-Net-Manhattan=-17.08, MLP-Manhattan=-13.83, U-Net better than MLP=3, MLP better than U-Net=8, ties=1
- narrow_corridor: maps=12, U-Net-MLP=15.50, U-Net-Manhattan=6.33, MLP-Manhattan=-9.17, U-Net better than MLP=0, MLP better than U-Net=9, ties=3

## Method Summaries


### bottleneck
- manhattan: maps=12, expanded=70.83, runtime=0.0001937115833546462, path_length=15.42, optimality=1.0, overestimate=0.0
- mlp_table: maps=12, expanded=48.50, runtime=0.00028970824981418747, path_length=15.42, optimality=1.0, overestimate=0.6382466510375997
- unet: maps=12, expanded=57.58, runtime=0.0003365070003079988, path_length=15.42, optimality=1.0, overestimate=0.35425047562461015

### large_block
- manhattan: maps=12, expanded=56.25, runtime=0.00015667691635220157, path_length=12.83, optimality=1.0, overestimate=0.0
- mlp_table: maps=12, expanded=25.08, runtime=0.00016480533304275014, path_length=12.83, optimality=1.0, overestimate=0.8722760093225707
- unet: maps=12, expanded=49.92, runtime=0.0002996632504922066, path_length=12.83, optimality=1.0, overestimate=0.5491006641219068

### maze_like
- manhattan: maps=12, expanded=83.67, runtime=0.00019132291648323493, path_length=21.92, optimality=1.0, overestimate=0.0
- mlp_table: maps=12, expanded=69.83, runtime=0.000368503583255612, path_length=21.92, optimality=1.0, overestimate=0.27855771361511455
- unet: maps=12, expanded=66.58, runtime=0.0003361807497033927, path_length=21.92, optimality=1.0, overestimate=0.33346032398214015

### narrow_corridor
- manhattan: maps=12, expanded=40.42, runtime=0.00010197225037700264, path_length=15.25, optimality=1.0, overestimate=0.0
- mlp_table: maps=12, expanded=31.25, runtime=0.0001938229161169147, path_length=15.25, optimality=1.0, overestimate=0.7738158836906184
- unet: maps=12, expanded=46.75, runtime=0.000262260500373183, path_length=15.75, optimality=0.8333333333333334, overestimate=0.7826619864388383
