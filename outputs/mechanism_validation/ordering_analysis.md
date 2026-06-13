# Route Ordering Hypothesis

Question: does U-Net preserve useful distance ordering better in cases where it beats MLP?

- all: maps=2000, U-Net-MLP expanded=9.648, U-Net order=0.890, MLP order=0.899, U-Net-MLP order=-0.009, U-Net tau=0.780, MLP tau=0.799
- unet_wins: maps=334, U-Net-MLP expanded=-9.404, U-Net order=0.854, MLP order=0.839, U-Net-MLP order=0.015, U-Net tau=0.708, MLP tau=0.681
- mlp_wins: maps=1352, U-Net-MLP expanded=16.595, U-Net order=0.897, MLP order=0.912, U-Net-MLP order=-0.015, U-Net tau=0.794, MLP tau=0.824
- maze_like: maps=400, U-Net-MLP expanded=-0.640, U-Net order=0.843, MLP order=0.828, U-Net-MLP order=0.015, U-Net tau=0.687, MLP tau=0.659
- bottleneck: maps=400, U-Net-MLP expanded=13.137, U-Net order=0.878, MLP order=0.877, U-Net-MLP order=0.001, U-Net tau=0.756, MLP tau=0.756
- large_block: maps=400, U-Net-MLP expanded=16.060, U-Net order=0.939, MLP order=0.954, U-Net-MLP order=-0.016, U-Net tau=0.877, MLP tau=0.909
- narrow_corridor: maps=400, U-Net-MLP expanded=13.175, U-Net order=0.876, MLP order=0.918, U-Net-MLP order=-0.042, U-Net tau=0.752, MLP tau=0.837
