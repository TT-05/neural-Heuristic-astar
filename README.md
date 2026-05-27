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
- `src/evaluate.py`: compares Manhattan A*, MLP A*, and U-Net A* when checkpoints exist.

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

The U-Net input has two channels:

- obstacle map
- goal-location map

The output is a predicted distance-to-goal value for every grid cell.

## Evaluate

Run from the `src` directory after training:

```bash
cd src
python3 evaluate.py
```

Evaluation reports path length, expanded nodes, runtime, and whether each A* path
matches the BFS shortest-path length.
