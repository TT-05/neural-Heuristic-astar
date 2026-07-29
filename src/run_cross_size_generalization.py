"""Evaluate fixed 20x20 U-Net checkpoints on larger structured maps.

This is an inference-only experiment.  It deliberately leaves the U-Net,
training data, and A* implementation untouched.
"""

import argparse
import csv
import hashlib
import os
import random
import time
from collections import defaultdict

import torch

from astar import astar_search
from bfs_label import compute_distance_to_goal
from model import (
    distance_normalizer_for_grid,
    grid_goal_tensor,
    load_unet_heuristic,
    make_unet_heuristic,
    manhattan_heuristic,
)
from structured_maps import ALL_MAP_TYPES, generate_structured_map


SIZES = (50, 100)
STRUCTURES = ("open_random", "maze_like", "bottleneck", "large_block", "narrow_corridor")
OBSTACLE_RATES = (0.1, 0.2, 0.3, 0.4)
BASE_SEED = 20_260_723


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grid_hash(grid):
    text = "".join("".join(str(cell) for cell in row) for row in grid)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def free_cells(grid):
    return [(row, col) for row, values in enumerate(grid) for col, value in enumerate(values) if value == 0]


def select_solvable_pair(grid, seed):
    """Select a deterministic, solvable pair without using search labels later."""
    cells = free_cells(grid)
    rng = random.Random(seed * 1_000_003 + len(grid) * 1009)
    for _ in range(200):
        start = rng.choice(cells)
        goal = rng.choice(cells)
        if start == goal:
            continue
        distances = compute_distance_to_goal(grid, goal)
        if distances[start[0]][start[1]] >= 0:
            return start, goal, distances
    return None, None, None


def build_cases(cases_per_structure):
    """Create disjoint larger-map cases with deterministic post-training seeds."""
    cases = []
    for size in SIZES:
        for structure_index, structure in enumerate(STRUCTURES):
            accepted = 0
            attempt = 0
            while accepted < cases_per_structure:
                rate = OBSTACLE_RATES[accepted % len(OBSTACLE_RATES)]
                seed = BASE_SEED + size * 100_000 + structure_index * 10_000 + attempt
                grid = generate_structured_map(size, size, seed, rate, structure)
                start, goal, distances = select_solvable_pair(grid, seed)
                attempt += 1
                if start is None:
                    continue
                cases.append(
                    {
                        "case_id": f"size{size}_{structure}_seed{seed}",
                        "map_size": size,
                        "structure_type": structure,
                        "seed": seed,
                        "obstacle_rate": rate,
                        "grid_sha256": grid_hash(grid),
                        "start": f"{start[0]},{start[1]}",
                        "goal": f"{goal[0]},{goal[1]}",
                        "optimal_cost": distances[start[0]][start[1]],
                        "grid": grid,
                        "start_coord": start,
                        "goal_coord": goal,
                    }
                )
                accepted += 1
    return cases


def padded_unet_table(model, grid, goal):
    """Run the fixed U-Net after zero-padding only when a dimension is not /4.

    The architecture downsamples twice, so a 50x50 tensor cannot be decoded
    without this input-shape adaptation.  The predicted map is cropped back to
    the real grid before it is used by the unchanged A* implementation.
    """
    rows = len(grid)
    cols = len(grid[0])
    padded_rows = ((rows + 3) // 4) * 4
    padded_cols = ((cols + 3) // 4) * 4
    padded_grid = [[0 for _ in range(padded_cols)] for _ in range(padded_rows)]
    for row in range(rows):
        padded_grid[row][:cols] = grid[row]
    model_input = grid_goal_tensor(padded_grid, goal).unsqueeze(0)
    with torch.no_grad():
        prediction = model(model_input).squeeze(0).cpu()[:rows, :cols]
    prediction *= distance_normalizer_for_grid(grid)
    return {(row, col): max(0.0, float(prediction[row, col])) for row, col in free_cells(grid)}


def unet_table_and_time(model, grid, goal):
    started = time.perf_counter()
    if len(grid) % 4 == 0 and len(grid[0]) % 4 == 0:
        heuristic = make_unet_heuristic(model, grid, goal)
        table = {cell: heuristic(cell, goal) for cell in free_cells(grid)}
    else:
        table = padded_unet_table(model, grid, goal)
    return table, time.perf_counter() - started


def table_heuristic(table):
    return lambda node, _goal: table[node]


def run_search(grid, start, goal, algorithm, table=None):
    started = time.perf_counter()
    if algorithm == "manhattan":
        result = astar_search(grid, start, goal, manhattan_heuristic)
    elif algorithm.endswith("tiebreak"):
        result = astar_search(grid, start, goal, manhattan_heuristic, secondary_heuristic=table_heuristic(table))
    else:
        result = astar_search(grid, start, goal, table_heuristic(table))
    return result, time.perf_counter() - started


def case_rows(case, models):
    grid = case["grid"]
    start = case["start_coord"]
    goal = case["goal_coord"]
    optimal_cost = case["optimal_cost"]
    tables = {}
    inference_seconds = {}
    for name, model in models.items():
        tables[name], inference_seconds[name] = unet_table_and_time(model, grid, goal)

    specifications = (
        ("manhattan", None, 0.0),
        ("mse_unet_tiebreak", "mse", inference_seconds["mse"]),
        ("ranking_lambda_0.5_unet_tiebreak", "ranking_lambda_0.5", inference_seconds["ranking_lambda_0.5"]),
        ("direct_ranking_lambda_0.1_unet", "ranking_lambda_0.1", inference_seconds["ranking_lambda_0.1"]),
    )
    rows = []
    for algorithm, model_name, inference_time in specifications:
        result, search_time = run_search(grid, start, goal, algorithm, tables.get(model_name))
        found = result["cost"] >= 0
        gap = result["cost"] - optimal_cost if found else ""
        rows.append(
            {
                "case_id": case["case_id"],
                "map_size": case["map_size"],
                "structure_type": case["structure_type"],
                "seed": case["seed"],
                "obstacle_rate": case["obstacle_rate"],
                "grid_sha256": case["grid_sha256"],
                "start": case["start"],
                "goal": case["goal"],
                "algorithm": algorithm,
                "model_checkpoint": model_name or "none",
                "optimal_cost": optimal_cost,
                "path_cost": result["cost"],
                "path_cost_gap": gap,
                "path_found": found,
                "optimal": found and result["cost"] == optimal_cost,
                "expanded_nodes": result["expanded"],
                "search_runtime_seconds": search_time,
                "unet_inference_seconds": inference_time,
                "total_runtime_seconds": search_time + inference_time,
                "input_padding": "none" if case["map_size"] % 4 == 0 else "zero_pad_to_52x52",
            }
        )
    return rows


def summarize(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    summary = []
    for key, group in sorted(grouped.items()):
        item = dict(zip(fields, key))
        item.update(
            {
                "cases": len(group),
                "mean_expanded_nodes": mean(float(row["expanded_nodes"]) for row in group),
                "median_expanded_nodes": sorted(float(row["expanded_nodes"]) for row in group)[len(group) // 2],
                "optimality_rate": mean(float(row["optimal"]) for row in group),
                "path_found_rate": mean(float(row["path_found"]) for row in group),
                "mean_path_cost_gap": mean(float(row["path_cost_gap"]) for row in group if row["path_cost_gap"] != ""),
                "mean_search_runtime_seconds": mean(float(row["search_runtime_seconds"]) for row in group),
                "mean_unet_inference_seconds": mean(float(row["unet_inference_seconds"]) for row in group),
                "mean_total_runtime_seconds": mean(float(row["total_runtime_seconds"]) for row in group),
            }
        )
        summary.append(item)
    return summary


def report(output_dir, by_size, by_structure, cases_per_structure):
    lines = [
        "# Cross-Size Generalization",
        "",
        "This inference-only evaluation applies fixed U-Nets trained on 20x20 maps to new 50x50 and 100x100 maps. It changes neither model weights, model architecture, A* implementation, nor heuristic priority definitions.",
        "",
        f"The dataset contains {cases_per_structure} deterministic, solvable cases for every size/structure stratum ({cases_per_structure * len(SIZES) * len(STRUCTURES)} cases total). All seeds are in a post-training range and maps cannot overlap the 20x20 training maps because their dimensions differ.",
        "",
        "For 50x50 inputs only, the unchanged two-pooling U-Net receives zero-padding to 52x52 and its output is cropped back to 50x50 before A*; 100x100 inputs need no adaptation.",
        "",
        "## By Size",
        "",
        "| Size | Algorithm | Expanded | Optimality | Cost gap | Search s | U-Net inference s | Total s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_size:
        lines.append(f"| {row['map_size']} | {row['algorithm']} | {row['mean_expanded_nodes']:.2f} | {row['optimality_rate']:.3f} | {row['mean_path_cost_gap']:.3f} | {row['mean_search_runtime_seconds']:.4f} | {row['mean_unet_inference_seconds']:.4f} | {row['mean_total_runtime_seconds']:.4f} |")

    comparisons = []
    for size in SIZES:
        size_rows = [row for row in by_size if row["map_size"] == size]
        baseline = next(row for row in size_rows if row["algorithm"] == "manhattan")
        for row in size_rows:
            if row["algorithm"] == "manhattan":
                continue
            comparisons.append((size, row["algorithm"], baseline["mean_expanded_nodes"] - row["mean_expanded_nodes"], row["optimality_rate"]))
    lines += ["", "## Main Question", ""]
    for size, algorithm, reduction, optimality in comparisons:
        direction = "reduced" if reduction > 0 else "increased"
        lines.append(f"- On {size}x{size}, `{algorithm}` {direction} mean expansions versus Manhattan by {abs(reduction):.2f}, with optimality {optimality:.3f}.")
    lines += ["", "Structure-level values are in `summary_by_structure.csv`; case-level measurements and timings are in `results.csv`.", ""]
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fixed-U-Net cross-size generalization experiment.")
    parser.add_argument("--cases-per-structure", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/cross_size_generalization")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cases_per_structure < 1:
        raise ValueError("--cases-per-structure must be positive")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    checkpoints = {
        "mse": os.path.join(root, "outputs/combined_loss_ablation/mse_best.pt"),
        "ranking_lambda_0.5": os.path.join(root, "outputs/ranking_lambda_sweep/lambda_0.5_best.pt"),
        "ranking_lambda_0.1": os.path.join(root, "outputs/ranking_lambda_sweep/lambda_0.1_best.pt"),
    }
    missing = [path for path in checkpoints.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    models = {name: load_unet_heuristic(path) for name, path in checkpoints.items()}
    cases = build_cases(args.cases_per_structure)
    case_records = [{key: value for key, value in case.items() if key not in {"grid", "start_coord", "goal_coord"}} for case in cases]
    write_csv(os.path.join(output_dir, "cases.csv"), case_records)
    rows = []
    for index, case in enumerate(cases, 1):
        rows.extend(case_rows(case, models))
        if index % 10 == 0 or index == len(cases):
            print(f"Evaluated {index}/{len(cases)} maps")
    by_size = summarize(rows, ("map_size", "algorithm"))
    by_structure = summarize(rows, ("map_size", "structure_type", "algorithm"))
    write_csv(os.path.join(output_dir, "results.csv"), rows)
    write_csv(os.path.join(output_dir, "summary_by_size.csv"), by_size)
    write_csv(os.path.join(output_dir, "summary_by_structure.csv"), by_structure)
    report(output_dir, by_size, by_structure, args.cases_per_structure)
    print(f"Saved {len(rows)} runs for {len(cases)} maps to {output_dir}")


if __name__ == "__main__":
    main()
