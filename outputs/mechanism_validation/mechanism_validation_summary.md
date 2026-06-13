# Mechanism Validation Summary

This benchmark-wide analysis checks whether qualitative mechanisms from representative case studies generalize across the random start-goal and controlled structured benchmarks. It does not rerun training or modify search/model code.

Benchmark maps analyzed: 2000
Representative selected cases used as qualitative reference: 20

## Verdicts

1. Barrier hypothesis: **supported**. Non-optimal maps have higher overestimation (over gap 0.122, large-over gap 0.114).
2. Route-ordering hypothesis: **supported**. U-Net-win maps have better relative ordering than MLP-win maps by 0.029.
3. Corridor smoothness hypothesis: **suggestive**. Roughness relationship is weak in high-corridor maps (Spearman 0.028).

## Strength Of Evidence

- Supported evidence means the benchmark-wide metric moves in the predicted direction across the relevant subset.
- Suggestive evidence means the metric is directionally plausible but not strong enough to treat as robust.
- Unsupported means the tested aggregate metric did not match the mechanism in this benchmark.

## Important Limitations

- The metrics are observational and do not establish causality.
- Kendall/order metrics are computed over map cells, not directly over A* open-list states.
- Consistency and roughness are global field summaries and may miss localized barriers near critical passages.
- The controlled structured maps are simple generators, not a complete planning benchmark distribution.

## Most Supported Mechanisms

- **Route ordering is supported**: maps where U-Net beats MLP have better relative U-Net-vs-MLP ordering accuracy than maps where MLP beats U-Net. This is the clearest benchmark-wide support for the route-bias interpretation.
- **Barrier/overestimation is supported**: non-optimal U-Net maps show higher overestimation and large-overestimation rates than optimal maps. The correlations with cost gap are positive but modest, so this should be interpreted as association rather than causality.
- **Corridor smoothness is only suggestive/weak**: high-corridor maps do not show a strong global roughness-to-failure relationship. The representative cases may reflect localized corridor barriers that are diluted by global roughness summaries.
