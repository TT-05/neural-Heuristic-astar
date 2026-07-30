# GPU Runtime Benchmark Report

## Validation

- 6,000 unique case-method records across 1,500 deterministic saved cases.
- CPU neural, GPU neural, and Manhattan result files have identical case ID, map hash, start, and goal tuples.
- All reported runs are optimal (1,500 / 1,500 per method); every path-cost gap is zero.

## Hardware

```csv
device,device_type,cuda_available,gpu_name,cuda_version,pytorch_version
cuda,cuda,True,NVIDIA GeForce RTX 4070 Laptop GPU,12.8,2.11.0+cu128
```

## Overall results

| Method | Device | Expanded | Reduction vs Manhattan | Total (s) | GPU / CPU | Total / Manhattan |
|---|---|---:|---:|---:|---:|---:|
| manhattan_astar | cpu | 35120.54 | 0.0% | 0.0737 | — | 1.00x |
| full_map_unet_tiebreak | cpu | 23495.86 | 33.1% | 0.6592 | — | 8.94x |
| region_cache_patch_32 | cpu | 19052.00 | 45.8% | 0.5234 | — | 7.10x |
| region_cache_patch_64 | cpu | 20533.50 | 41.5% | 0.5247 | — | 7.12x |
| full_map_unet_tiebreak | cuda | 23492.22 | 33.1% | 0.1173 | 5.62x | 1.59x |
| region_cache_patch_32 | cuda | 19042.86 | 45.8% | 0.3269 | 1.60x | 4.43x |
| region_cache_patch_64 | cuda | 20514.58 | 41.6% | 0.2780 | 1.89x | 3.77x |

## Measured conclusions

- CUDA reduced full-map U-Net end-to-end runtime from 0.6592s to 0.1173s (5.62x).
- CUDA reduced Region-cache patch 32 runtime by 1.60x and patch 64 runtime by 1.89x.
- Full-map is the fastest neural method on this GPU (0.1173s). Patch 32 and patch 64 remain 2.79x and 2.37x slower despite reducing expansions more.
- The remaining Region-cache cost is mostly patch extraction (0.1719s for 32; 0.1713s for 64), rather than U-Net inference (0.0583s; 0.0320s).
- Manhattan remains the fastest end-to-end method because it has no neural or patch-preparation work.

## Scope limit

This completed run did not enable per-node expansion or per-patch spatial trajectory logging. The report therefore makes runtime/cache claims only and does not infer optimal-path alignment.
