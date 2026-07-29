"""Measure when fixed U-Net tie-breaking repays its inference cost.

This is an evaluation-only experiment: checkpoints, U-Net architecture, A*,
and heuristic priority definitions are reused without modification.
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
from model import load_unet_heuristic, make_unet_heuristic, manhattan_heuristic
from structured_maps import generate_structured_map


SIZES = (100, 200, 500, 1000)
STRUCTURES = ("open_random", "maze_like", "bottleneck", "large_block", "narrow_corridor")
OBSTACLE_RATES = (0.1, 0.2, 0.3, 0.4)
BASE_SEED = 20_260_724
ALGORITHMS = ("manhattan", "mse_unet_tiebreak", "ranking_lambda_0.5_unet_tiebreak")


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def median(values):
    values = sorted(values)
    if not values:
        return 0.0
    midpoint = len(values) // 2
    return values[midpoint] if len(values) % 2 else (values[midpoint - 1] + values[midpoint]) / 2


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


def choose_solvable_pair(grid, seed):
    cells = free_cells(grid)
    rng = random.Random(seed * 1_000_003 + len(grid) * 1009)
    for _ in range(200):
        start = rng.choice(cells)
        goal = rng.choice(cells)
        if start == goal:
            continue
        distance = compute_distance_to_goal(grid, goal)
        if distance[start[0]][start[1]] >= 0:
            return start, goal, distance
    return None, None, None


def build_cases(cases_per_structure):
    cases = []
    for size in SIZES:
        for structure_index, structure in enumerate(STRUCTURES):
            accepted = 0
            attempt = 0
            while accepted < cases_per_structure:
                obstacle_rate = OBSTACLE_RATES[accepted % len(OBSTACLE_RATES)]
                seed = BASE_SEED + size * 100_000 + structure_index * 10_000 + attempt
                grid = generate_structured_map(size, size, seed, obstacle_rate, structure)
                start, goal, distance = choose_solvable_pair(grid, seed)
                attempt += 1
                if start is None:
                    continue
                cases.append(
                    {
                        "case_id": f"size{size}_{structure}_seed{seed}",
                        "map_size": size,
                        "structure_type": structure,
                        "seed": seed,
                        "obstacle_rate": obstacle_rate,
                        "grid_sha256": grid_hash(grid),
                        "start": f"{start[0]},{start[1]}",
                        "goal": f"{goal[0]},{goal[1]}",
                        "optimal_cost": distance[start[0]][start[1]],
                        "grid": grid,
                        "start_coord": start,
                        "goal_coord": goal,
                    }
                )
                accepted += 1
    return cases


def infer_table(model, grid, goal):
    """Time one complete cached U-Net heuristic preparation per map.

    It includes tensor construction, forward inference, normalization, and
    conversion to the same coordinate lookup consumed by A*.  All tested
    dimensions are divisible by four, so no input padding is used here.
    """
    started = time.perf_counter()
    heuristic = make_unet_heuristic(model, grid, goal)
    table = {cell: heuristic(cell, goal) for cell in free_cells(grid)}
    return table, time.perf_counter() - started


def table_heuristic(table):
    return lambda node, _goal: table[node]


def run_algorithm(grid, start, goal, algorithm, table=None):
    started = time.perf_counter()
    if algorithm == "manhattan":
        result = astar_search(grid, start, goal, manhattan_heuristic)
    else:
        result = astar_search(grid, start, goal, manhattan_heuristic, secondary_heuristic=table_heuristic(table))
    return result, time.perf_counter() - started


def evaluate_case(case, models):
    grid = case["grid"]
    start = case["start_coord"]
    goal = case["goal_coord"]
    optimal_cost = case["optimal_cost"]
    tables = {}
    inference_times = {}
    for name, model in models.items():
        tables[name], inference_times[name] = infer_table(model, grid, goal)

    specifications = (
        ("manhattan", "none", None, 0.0),
        ("mse_unet_tiebreak", "mse", tables["mse"], inference_times["mse"]),
        (
            "ranking_lambda_0.5_unet_tiebreak",
            "ranking_lambda_0.5",
            tables["ranking_lambda_0.5"],
            inference_times["ranking_lambda_0.5"],
        ),
    )
    rows = []
    for algorithm, checkpoint, table, inference_time in specifications:
        result, search_time = run_algorithm(grid, start, goal, algorithm, table)
        path_found = result["cost"] >= 0
        path_gap = result["cost"] - optimal_cost if path_found else ""
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
                "model_checkpoint": checkpoint,
                "optimal_cost": optimal_cost,
                "path_cost": result["cost"],
                "path_cost_gap": path_gap,
                "path_found": path_found,
                "optimal": path_found and result["cost"] == optimal_cost,
                "expanded_nodes": result["expanded"],
                "search_time_seconds": search_time,
                "unet_inference_time_seconds": inference_time,
                "total_runtime_seconds": search_time + inference_time,
            }
        )
    return rows


def summarize(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    result = []
    for key, group in sorted(grouped.items()):
        item = dict(zip(fields, key))
        item.update(
            {
                "cases": len(group),
                "mean_expanded_nodes": mean(float(row["expanded_nodes"]) for row in group),
                "median_expanded_nodes": median(float(row["expanded_nodes"]) for row in group),
                "optimality_rate": mean(float(row["optimal"]) for row in group),
                "mean_path_cost_gap": mean(float(row["path_cost_gap"]) for row in group if row["path_cost_gap"] != ""),
                "mean_search_time_seconds": mean(float(row["search_time_seconds"]) for row in group),
                "mean_unet_inference_time_seconds": mean(float(row["unet_inference_time_seconds"]) for row in group),
                "mean_total_runtime_seconds": mean(float(row["total_runtime_seconds"]) for row in group),
            }
        )
        result.append(item)
    return result


def size_lookup(summary, size, algorithm):
    return next(row for row in summary if row["map_size"] == size and row["algorithm"] == algorithm)


def make_plot(output_dir, by_size):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    import matplotlib.pyplot as plt

    labels = {
        "manhattan": "Manhattan",
        "mse_unet_tiebreak": "MSE U-Net tie-break",
        "ranking_lambda_0.5_unet_tiebreak": "Ranking (lambda=0.5) tie-break",
    }
    colors = {"manhattan": "#374151", "mse_unet_tiebreak": "#d97706", "ranking_lambda_0.5_unet_tiebreak": "#047857"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for algorithm in ALGORITHMS:
        selected = [size_lookup(by_size, size, algorithm) for size in SIZES]
        axes[0].plot(SIZES, [row["mean_expanded_nodes"] for row in selected], marker="o", label=labels[algorithm], color=colors[algorithm])
        axes[1].plot(SIZES, [row["mean_search_time_seconds"] for row in selected], marker="o", label=labels[algorithm], color=colors[algorithm])
        axes[2].plot(SIZES, [row["mean_total_runtime_seconds"] for row in selected], marker="o", label=labels[algorithm], color=colors[algorithm])
    axes[0].set_title("Mean expanded nodes")
    axes[1].set_title("A* search time")
    axes[2].set_title("Total runtime")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Map size (square side length)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Nodes")
    axes[1].set_ylabel("Seconds")
    axes[2].set_ylabel("Seconds")
    axes[2].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "runtime_scaling_plot.png"), dpi=180)
    plt.close(figure)


def write_report(output_dir, by_size, cases_per_structure):
    lines = [
        "# Neural A* Runtime Scaling",
        "",
        "This evaluation uses fixed 20x20-trained U-Net checkpoints without retraining. It compares Manhattan A*, MSE U-Net tie-break, and ranking-loss (lambda=0.5) U-Net tie-break on newly generated maps.",
        "",
        f"Each size/structure stratum has {cases_per_structure} deterministic, solvable maps, for {cases_per_structure * len(SIZES) * len(STRUCTURES)} maps and {cases_per_structure * len(SIZES) * len(STRUCTURES) * len(ALGORITHMS)} A* runs. The saved case manifest provides seeds, start/goal pairs, and grid hashes for reproducibility.",
        "",
        "`unet_inference_time_seconds` includes grid/goal tensor construction, one network forward pass, normalization, and cached heuristic-table construction. `total_runtime_seconds` is this overhead plus A* search time.",
        "",
        "## By Size",
        "",
        "| Size | Algorithm | Expanded | Optimality | Cost gap | Search s | Inference s | Total s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_size:
        lines.append(f"| {row['map_size']} | {row['algorithm']} | {row['mean_expanded_nodes']:.2f} | {row['optimality_rate']:.3f} | {row['mean_path_cost_gap']:.3f} | {row['mean_search_time_seconds']:.4f} | {row['mean_unet_inference_time_seconds']:.4f} | {row['mean_total_runtime_seconds']:.4f} |")

    lines += ["", "## Main Questions", ""]
    for size in SIZES:
        baseline = size_lookup(by_size, size, "manhattan")
        for algorithm in ALGORITHMS[1:]:
            neural = size_lookup(by_size, size, algorithm)
            expansion_delta = baseline["mean_expanded_nodes"] - neural["mean_expanded_nodes"]
            runtime_delta = baseline["mean_total_runtime_seconds"] - neural["mean_total_runtime_seconds"]
            saved_search = baseline["mean_search_time_seconds"] - neural["mean_search_time_seconds"]
            state = "faster" if runtime_delta > 0 else "slower"
            lines.append(f"- {size}x{size}, `{algorithm}` changed expansions by {expansion_delta:.2f} versus Manhattan and was {abs(runtime_delta):.4f}s {state} overall; saved search time was {saved_search:.4f}s versus {neural['mean_unet_inference_time_seconds']:.4f}s neural overhead.")

    for algorithm in ALGORITHMS[1:]:
        faster_sizes = [size for size in SIZES if size_lookup(by_size, size, algorithm)["mean_total_runtime_seconds"] < size_lookup(by_size, size, "manhattan")["mean_total_runtime_seconds"]]
        if faster_sizes:
            lines.append(f"- `{algorithm}` first beats Manhattan total runtime at {min(faster_sizes)}x{min(faster_sizes)} in this measurement.")
        else:
            lines.append(f"- `{algorithm}` does not beat Manhattan total runtime at any tested size.")

    lines.append("- Compare `summary_by_structure.csv` before generalizing an aggregate result: structural family can change both search savings and wall-clock behavior.")
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def warm_up(models):
    """Exclude one-off framework initialization from all recorded inference times."""
    grid = [[0] * 20 for _ in range(20)]
    for model in models.values():
        infer_table(model, grid, (19, 19))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Neural A* runtime scaling across map sizes.")
    parser.add_argument("--cases-per-structure", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/runtime_scaling")
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
    }
    missing = [path for path in checkpoints.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    models = {name: load_unet_heuristic(path) for name, path in checkpoints.items()}
    warm_up(models)
    cases = build_cases(args.cases_per_structure)
    manifest = [{key: value for key, value in case.items() if key not in {"grid", "start_coord", "goal_coord"}} for case in cases]
    write_csv(os.path.join(output_dir, "cases.csv"), manifest)
    rows = []
    for index, case in enumerate(cases, 1):
        rows.extend(evaluate_case(case, models))
        print(f"Evaluated {index}/{len(cases)} maps: {case['case_id']}")
    by_size = summarize(rows, ("map_size", "algorithm"))
    by_structure = summarize(rows, ("map_size", "structure_type", "algorithm"))
    write_csv(os.path.join(output_dir, "results.csv"), rows)
    write_csv(os.path.join(output_dir, "summary_by_size.csv"), by_size)
    write_csv(os.path.join(output_dir, "summary_by_structure.csv"), by_structure)
    make_plot(output_dir, by_size)
    write_report(output_dir, by_size, args.cases_per_structure)
    print(f"Saved {len(rows)} runs for {len(cases)} maps to {output_dir}")


if __name__ == "__main__":
    main()
