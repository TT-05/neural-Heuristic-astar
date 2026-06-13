# neural-Heuristic-astar

Neural heuristic learning for A* pathfinding on grid maps.

## Files

- `src/gen_map.py`: generates random grid maps.
- `src/bfs_label.py`: computes shortest-path distance labels with BFS.
- `src/dataset.py`: builds point samples from BFS labels for the MLP baseline.
- `src/astar.py`: runs A* with a caller-provided heuristic function.
- `src/model.py`: defines Manhattan, MLP, and U-Net heuristic helpers.
- `src/train.py`: trains the MLP on BFS distance labels.
- `src/train_unet.py`: trains the U-Net to predict a full distance-to-goal map.
- `src/evaluate.py`: compares Manhattan A*, MLP A*, U-Net A*, and a hybrid heuristic when checkpoints exist.

## Train MLP

Run from the `src` directory:

```bash
cd src
python3 train.py
```

Training saves this checkpoint:

```text
checkpoints/mlp_heuristic.pt
```

## Train U-Net

Run from the `src` directory:

```bash
cd src
python3 train_unet.py
```

Training saves this checkpoint:

```text
checkpoints/unet_heuristic.pt
```

U-Net training also saves:

```text
checkpoints/unet_heuristic_latest.pt
checkpoints/unet_heuristic_best.pt
```

`unet_heuristic.pt` is kept as the compatible checkpoint used by `evaluate.py`
and `experiment.py`; it is updated when a new best validation loss is found.

The U-Net input has two channels:

- obstacle map
- goal-location map

The output is a predicted distance-to-goal value for every grid cell.
During training, U-Net distance labels are normalized by `rows + cols`.
At inference time, predictions are multiplied by the same value before A* uses them.
The training script prints training loss, validation loss, validation MAE,
validation MSE, and validation overestimate rate each epoch.
The current default U-Net training run uses 500 random maps, 50 epochs, and batch size 16.

## Evaluate And Debug

Run from the `src` directory after training:

```bash
cd src
python3 evaluate.py
```

Evaluation reports path length, expanded nodes, runtime, and whether each A* path
matches the BFS shortest-path length.

When a U-Net checkpoint exists, evaluation also writes debug outputs to:

```text
outputs/unet_debug/
```

The debug folder includes:

- `true_distance_map.txt`
- `predicted_distance_map.txt`
- `error_map_pred_minus_true.txt`
- `metrics.txt`
- optional heatmap PNG files if matplotlib is installed

## Research Pipeline

This repository also contains a benchmark and analysis pipeline for studying
learned heuristics for A* without changing the underlying A* implementation.
The current research phase focuses on evaluation, mechanism analysis, and
failure interpretation.

Main benchmark scripts:

- `src/experiment.py`: runs fixed start-goal, random start-goal, and controlled
  structured benchmark experiments.
- `src/structured_maps.py`: generates controlled map families such as
  `maze_like`, `bottleneck`, `large_block`, and `narrow_corridor`.
- `src/filter_structured_benchmark.py`: filters and summarizes structured map
  subsets.

Main analysis scripts:

- `src/analyze_results.py`: summarizes benchmark results and correlations.
- `src/analyze_structure_benchmark.py`: analyzes structure-aware random-map
  behavior.
- `src/analyze_structured_benchmark.py`: analyzes controlled structured-map
  results.
- `src/analyze_failure_patterns.py`: identifies representative U-Net success
  and failure cases.
- `src/analyze_heuristic_error.py`: visualizes and summarizes heuristic error
  fields for selected cases.
- `src/analyze_representative_cases.py`: builds detailed controlled-structure
  case studies.
- `src/summarize_case_study_outputs.py`: creates qualitative summaries from
  generated case-study outputs.
- `src/analyze_mechanism_hypotheses.py`: validates route-ordering, barrier, and
  corridor-smoothness hypotheses across benchmarks.
- `src/analyze_route_critical_cells.py`: compares global heuristic metrics with
  route-critical-cell metrics.
- `src/create_research_synthesis.py`: creates the final research synthesis
  report.

## Benchmarks

The current benchmark progression is:

1. Fixed start-goal evaluation.
2. Random start-goal evaluation.
3. Structure-aware random-map benchmark.
4. Controlled structured benchmark.

Generated benchmark and analysis outputs are stored under:

```text
outputs/experiments/
outputs/structure_benchmark/
outputs/structured_controlled/
outputs/structured_benchmark/
outputs/failure_pattern_analysis/
outputs/case_studies/
outputs/mechanism_validation/
outputs/route_critical_analysis/
outputs/research_synthesis/
```

## Key Findings

- U-Net substantially reduces A* expansions relative to Dijkstra.
- U-Net often improves over Manhattan, but can lose optimality because the
  learned heuristic sometimes overestimates.
- The MLP table heuristic remains slightly stronger on aggregate expansion.
- U-Net is most competitive on maze-like structured maps.
- Large-block and narrow-corridor maps expose the clearest U-Net failure modes.
- The strongest supported success mechanism is route ordering: U-Net wins when
  its predicted field orders useful regions more favorably for A*.
- The strongest supported failure mechanism is localized overestimation:
  non-optimal U-Net paths are associated with artificial heuristic barriers,
  especially near route-critical cells.
- Global roughness is too coarse to strongly explain corridor failures.

## Research Synthesis

The final integrated report is:

```text
outputs/research_synthesis/research_synthesis.md
```

Regenerate it with:

```bash
python3 src/create_research_synthesis.py
```
