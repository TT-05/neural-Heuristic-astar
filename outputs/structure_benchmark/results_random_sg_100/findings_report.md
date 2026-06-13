# Structure-Aware Benchmark Findings

## U-Net Advantage Cases

- Clear U-Net advantage cases: 31
- random_rate0.2_seed65_s6-3_g0-16: structures [sparse obstacles], expanded MLP=86.0, U-Net=35.0, Manhattan=105.0, path_stretch=1.11, U-Net overestimate=0.760
- random_rate0.3_seed34_s2-1_g18-11: structures [dense obstacles; narrow corridor; bottleneck; multiple alternative routes; maze-like], expanded MLP=152.0, U-Net=108.0, Manhattan=183.0, path_stretch=1.15, U-Net overestimate=0.422
- random_rate0.3_seed90_s17-8_g8-16: structures [dense obstacles; narrow corridor; bottleneck; multiple alternative routes; maze-like], expanded MLP=66.0, U-Net=29.0, Manhattan=79.0, path_stretch=1.24, U-Net overestimate=0.682
- random_rate0.3_seed58_s3-9_g18-1: structures [dense obstacles; narrow corridor; bottleneck; multiple alternative routes; maze-like], expanded MLP=121.0, U-Net=95.0, Manhattan=148.0, path_stretch=1.35, U-Net overestimate=0.100
- random_rate0.4_seed82_s7-10_g10-12: structures [maze-like; large obstacle block; narrow corridor; bottleneck], expanded MLP=66.0, U-Net=45.0, Manhattan=69.0, path_stretch=3.40, U-Net overestimate=0.760
- random_rate0.4_seed59_s10-8_g3-4: structures [maze-like; large obstacle block; narrow corridor; bottleneck], expanded MLP=69.0, U-Net=49.0, Manhattan=71.0, path_stretch=1.91, U-Net overestimate=0.305
- random_rate0.3_seed36_s3-5_g17-10: structures [dense obstacles; narrow corridor; bottleneck; multiple alternative routes; maze-like], expanded MLP=78.0, U-Net=59.0, Manhattan=115.0, path_stretch=1.42, U-Net overestimate=0.736
- random_rate0.3_seed28_s4-2_g13-14: structures [maze-like; narrow corridor; bottleneck], expanded MLP=65.0, U-Net=47.0, Manhattan=74.0, path_stretch=1.10, U-Net overestimate=0.665
- random_rate0.4_seed16_s17-17_g10-13: structures [maze-like; narrow corridor; bottleneck], expanded MLP=36.0, U-Net=19.0, Manhattan=37.0, path_stretch=1.36, U-Net overestimate=1.000
- random_rate0.3_seed71_s4-16_g17-2: structures [dense obstacles; narrow corridor; bottleneck; multiple alternative routes; maze-like], expanded MLP=122.0, U-Net=105.0, Manhattan=165.0, path_stretch=1.15, U-Net overestimate=0.143

## MLP Advantage Cases

- Clear MLP table advantage cases: 124
- random_rate0.1_seed3_s11-0_g0-19: structures [open space; multiple alternative routes], expanded MLP=64.0, U-Net=140.0, Manhattan=210.0, path_stretch=1.00, U-Net overestimate=0.483
- random_rate0.1_seed55_s0-9_g19-15: structures [open space; multiple alternative routes], expanded MLP=28.0, U-Net=100.0, Manhattan=132.0, path_stretch=1.00, U-Net overestimate=0.173
- random_rate0.1_seed53_s15-9_g4-19: structures [open space; multiple alternative routes], expanded MLP=41.0, U-Net=112.0, Manhattan=121.0, path_stretch=1.00, U-Net overestimate=0.313
- random_rate0.1_seed25_s16-2_g0-17: structures [open space; multiple alternative routes], expanded MLP=102.0, U-Net=169.0, Manhattan=212.0, path_stretch=1.00, U-Net overestimate=0.350
- random_rate0.2_seed17_s18-15_g0-5: structures [sparse obstacles; multiple alternative routes], expanded MLP=38.0, U-Net=102.0, Manhattan=137.0, path_stretch=1.00, U-Net overestimate=0.321
- random_rate0.1_seed79_s5-4_g14-16: structures [open space; multiple alternative routes], expanded MLP=34.0, U-Net=90.0, Manhattan=112.0, path_stretch=1.00, U-Net overestimate=0.501
- random_rate0.2_seed55_s2-10_g13-19: structures [sparse obstacles; multiple alternative routes], expanded MLP=40.0, U-Net=94.0, Manhattan=102.0, path_stretch=1.00, U-Net overestimate=0.614
- random_rate0.1_seed94_s18-9_g4-1: structures [open space; multiple alternative routes], expanded MLP=34.0, U-Net=87.0, Manhattan=106.0, path_stretch=1.00, U-Net overestimate=0.322
- random_rate0.1_seed70_s19-16_g8-6: structures [open space; multiple alternative routes], expanded MLP=40.0, U-Net=89.0, Manhattan=124.0, path_stretch=1.00, U-Net overestimate=0.436
- random_rate0.1_seed96_s17-16_g1-10: structures [open space; multiple alternative routes], expanded MLP=35.0, U-Net=84.0, Manhattan=94.0, path_stretch=1.00, U-Net overestimate=0.297

## Structure Summary

- bottleneck: Manhattan=43.66, MLP=34.19, U-Net=36.47, U-Net-MLP=2.28
- dense obstacles: Manhattan=50.22, MLP=37.46, U-Net=37.47, U-Net-MLP=0.01
- large obstacle block: Manhattan=31.91, MLP=26.73, U-Net=29.59, U-Net-MLP=2.86
- maze-like: Manhattan=44.13, MLP=35.39, U-Net=37.08, U-Net-MLP=1.69
- multiple alternative routes: Manhattan=55.67, MLP=30.92, U-Net=40.37, U-Net-MLP=9.45
- narrow corridor: Manhattan=43.55, MLP=35.53, U-Net=36.96, U-Net-MLP=1.43
- open space: Manhattan=47.93, MLP=19.37, U-Net=36.14, U-Net-MLP=16.77
- sparse obstacles: Manhattan=44.42, MLP=25.00, U-Net=31.13, U-Net-MLP=6.13

## Difficulty Summary


### optimal_cost
- low: Manhattan=12.22, MLP=8.24, U-Net=10.04, U-Net-MLP=1.81
- medium: Manhattan=37.35, MLP=21.08, U-Net=27.74, U-Net-MLP=6.66
- high: Manhattan=90.66, MLP=60.70, U-Net=72.26, U-Net-MLP=11.56

### path_stretch
- low: Manhattan=37.02, MLP=18.05, U-Net=27.62, U-Net-MLP=9.56
- medium: Manhattan=61.68, MLP=46.28, U-Net=46.44, U-Net-MLP=0.16
- high: Manhattan=56.52, MLP=45.06, U-Net=47.14, U-Net-MLP=2.08

### corridor_rate
- low: Manhattan=48.02, MLP=20.91, U-Net=35.17, U-Net-MLP=14.26
- medium: Manhattan=46.60, MLP=30.88, U-Net=33.82, U-Net-MLP=2.94
- high: Manhattan=40.32, MLP=34.33, U-Net=36.76, U-Net-MLP=2.42

### articulation_count
- low: Manhattan=46.00, MLP=20.11, U-Net=34.31, U-Net-MLP=14.20
- medium: Manhattan=48.28, MLP=32.17, U-Net=34.45, U-Net-MLP=2.29
- high: Manhattan=40.78, MLP=34.73, U-Net=36.98, U-Net-MLP=2.25

## Interpretation

This analysis checks whether aggregate means hide map-structure-specific behavior. If U-Net advantage cases cluster in corridor, bottleneck, maze-like, or high-stretch bins, that supports the idea that obstacle-aware predictions help on genuinely structured planning problems. If MLP advantage remains strongest on open or low-stretch maps, that supports the geometry-dominates hypothesis. The tables should be interpreted alongside optimality and overestimate rates because U-Net may reduce expansions while still risking non-optimal paths.
