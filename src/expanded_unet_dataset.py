"""Deterministic, stratified map dataset for U-Net distance-map training."""

import argparse
import csv
import hashlib
import json
import os
import random

import numpy as np
from torch.utils.data import Dataset

from bfs_label import compute_distance_to_goal
from model import distance_normalizer_for_grid, grid_goal_tensor
from structured_maps import ALL_MAP_TYPES, generate_structured_map


MAP_TYPES = list(ALL_MAP_TYPES)
OBSTACLE_RATES = [0.1, 0.2, 0.3, 0.4]
SPLIT_COUNTS = {"train": 200, "val": 25, "test": 25}
WIDTH = 20
HEIGHT = 20
SEED = 20260715


def grid_hash(grid):
    return hashlib.sha256(bytes(value for row in grid for value in row)).hexdigest()


def free_cells(grid):
    return [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 0]


def choose_goal_and_reference_start(grid, seed):
    rng = random.Random(seed)
    cells = free_cells(grid)
    if not cells:
        raise ValueError("Map has no free cells.")
    goal = cells[rng.randrange(len(cells))]
    distances = compute_distance_to_goal(grid, goal)
    reachable = [(r, c, distances[r][c]) for r, c in cells if distances[r][c] >= 0]
    if not reachable:
        raise ValueError("Goal has no reachable free cell.")
    distances_only = sorted(value for _, _, value in reachable)
    # This reference pair is metadata only; distance-map supervision still uses every reachable cell.
    quantile = (seed % 3 + 1) / 4
    target_distance = distances_only[int((len(distances_only) - 1) * quantile)]
    candidates = [(r, c) for r, c, value in reachable if value == target_distance]
    return goal, candidates[rng.randrange(len(candidates))], distances


def generate_expanded_dataset(output_dir, seed=SEED):
    os.makedirs(output_dir, exist_ok=True)
    grids, goals, labels, starts, records = [], [], [], [], []
    seen_hashes = set()
    sequence = 0
    for split, count in SPLIT_COUNTS.items():
        for map_type_index, map_type in enumerate(MAP_TYPES):
            for rate_index, obstacle_rate in enumerate(OBSTACLE_RATES):
                accepted = 0
                attempt = 0
                while accepted < count:
                    map_seed = seed + sequence * 1009 + attempt * 7919 + map_type_index * 101 + rate_index * 17
                    grid = generate_structured_map(WIDTH, HEIGHT, map_seed, obstacle_rate, map_type)
                    fingerprint = grid_hash(grid)
                    attempt += 1
                    if fingerprint in seen_hashes or not free_cells(grid):
                        continue
                    goal, reference_start, distance_grid = choose_goal_and_reference_start(grid, map_seed + 1)
                    seen_hashes.add(fingerprint)
                    grids.append(grid)
                    goals.append(goal)
                    labels.append(distance_grid)
                    starts.append(reference_start)
                    records.append(
                        {
                            "index": len(records),
                            "split": split,
                            "map_type": map_type,
                            "obstacle_rate": obstacle_rate,
                            "map_seed": map_seed,
                            "goal_row": goal[0],
                            "goal_col": goal[1],
                            "reference_start_row": reference_start[0],
                            "reference_start_col": reference_start[1],
                            "reference_start_distance": distance_grid[reference_start[0]][reference_start[1]],
                            "grid_sha256": fingerprint,
                        }
                    )
                    accepted += 1
                    sequence += 1

    archive_path = os.path.join(output_dir, "expanded_unet_dataset.npz")
    np.savez_compressed(
        archive_path,
        grids=np.asarray(grids, dtype=np.uint8),
        goals=np.asarray(goals, dtype=np.int16),
        labels=np.asarray(labels, dtype=np.int16),
        reference_starts=np.asarray(starts, dtype=np.int16),
    )
    manifest_path = os.path.join(output_dir, "dataset_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "seed": seed,
        "width": WIDTH,
        "height": HEIGHT,
        "map_types": MAP_TYPES,
        "obstacle_rates": OBSTACLE_RATES,
        "split_counts_per_type_rate": SPLIT_COUNTS,
        "total_maps": len(records),
        "unique_grid_hashes": len(seen_hashes),
    }
    with open(os.path.join(output_dir, "dataset_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return archive_path, manifest_path, metadata


class ExpandedUNetDataset(Dataset):
    def __init__(self, archive_path, manifest_path, split):
        archive = np.load(archive_path)
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            manifest = list(csv.DictReader(handle))
        self.indices = [int(row["index"]) for row in manifest if row["split"] == split]
        self.grids = archive["grids"]
        self.goals = archive["goals"]
        self.labels = archive["labels"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = self.indices[index]
        grid = self.grids[item].tolist()
        goal = tuple(int(value) for value in self.goals[item])
        target = self.labels[item].astype(np.float32)
        mask = (target >= 0).astype(np.float32)
        target = np.clip(target, 0.0, None) / distance_normalizer_for_grid(grid)
        import torch

        return (
            grid_goal_tensor(grid, goal),
            torch.from_numpy(target),
            torch.from_numpy(mask),
            torch.tensor(goal, dtype=torch.long),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a balanced expanded U-Net map dataset.")
    parser.add_argument("--output-dir", default="outputs/expanded_dataset_training/dataset")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    archive, manifest, metadata = generate_expanded_dataset(args.output_dir, args.seed)
    print(f"Saved {metadata['total_maps']} unique maps to {archive}")
    print(f"Manifest: {manifest}")
