import argparse
import csv
import json
import os
import random
import time

from astar import astar_search
from bfs_label import compute_distance_to_goal
from gen_map import gen_map
from model import (
    load_mlp_heuristic,
    load_unet_heuristic,
    make_mlp_heuristic,
    make_mlp_table_heuristic,
    make_unet_heuristic,
    manhattan_heuristic,
)
from oracle_topk_tiebreak import ORACLE_TOPK_METHODS, run_oracle_topk_search
from structured_maps import generate_structured_map, parse_structured_types


MAP_SIZES = [20]
OBSTACLE_RATES = [0.1, 0.2, 0.3, 0.4]
SEEDS = list(range(20))
START = (0, 0)
CORE_HEURISTIC_NAMES = ["dijkstra", "manhattan", "mlp_table", "unet"]
ALL_HEURISTIC_NAMES = ["dijkstra", "manhattan", "mlp", "mlp_table", "unet", "hybrid_max_manhattan_unet"]
EXTENDED_HEURISTIC_NAMES = ALL_HEURISTIC_NAMES + ["manhattan_unet_tiebreak"]
TIEBREAK_CONTROL_HEURISTIC_NAMES = [
    "manhattan",
    "mlp_table",
    "unet",
    "manhattan_unet_tiebreak",
    "manhattan_counter_tiebreak",
    "manhattan_large_g_tiebreak",
    "manhattan_small_g_tiebreak",
    "manhattan_mlp_tiebreak",
    "manhattan_true_distance_tiebreak",
]
ORACLE_TOPK_HEURISTIC_NAMES = [
    "manhattan",
    "manhattan_large_g_tiebreak",
    "manhattan_mlp_tiebreak",
    "manhattan_unet_tiebreak",
    "manhattan_oracle_top1_tiebreak",
    "manhattan_oracle_top2_tiebreak",
    "manhattan_oracle_top4_tiebreak",
    "manhattan_oracle_top8_tiebreak",
    "manhattan_true_distance_tiebreak",
]
CHECKPOINT_CHOICES = {
    "compatible": "unet_heuristic.pt",
    "best": "unet_heuristic_best.pt",
    "latest": "unet_heuristic_latest.pt",
}
DEFAULT_CASE_MIN_DELTA = 10
DEFAULT_CASE_RATIO = 1.2
DEFAULT_MAX_CASES_PER_CATEGORY = 20
DEFAULT_RANDOM_START_GOAL_RETRIES = 100


def dijkstra_heuristic(current, goal):
    return 0.0


def shortest_path_length_from_distance_grid(distance_grid, start):
    start_r, start_c = start
    return distance_grid[start_r][start_c]


def valid_cells_from_distance_grid(distance_grid):
    valid_cells = []
    for r, row in enumerate(distance_grid):
        for c, value in enumerate(row):
            if value >= 0:
                valid_cells.append((r, c, value))
    return valid_cells


def prediction_error_metrics(distance_grid, heuristic, goal):
    valid_cells = valid_cells_from_distance_grid(distance_grid)
    if not valid_cells:
        return {
            "mae": "",
            "mse": "",
            "rmse": "",
            "underestimate_rate": "",
            "overestimate_rate": "",
            "large_underestimate_rate": "",
            "large_overestimate_rate": "",
            "max_overestimate": "",
        }

    errors = []
    for r, c, true_distance in valid_cells:
        predicted = heuristic((r, c), goal)
        errors.append(predicted - true_distance)

    abs_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    mse = sum(squared_errors) / len(squared_errors)
    underestimates = [error for error in errors if error < 0]
    overestimates = [error for error in errors if error > 0]
    large_underestimates = [error for error in errors if error < -3]
    large_overestimates = [error for error in errors if error > 3]

    return {
        "mae": sum(abs_errors) / len(abs_errors),
        "mse": mse,
        "rmse": mse ** 0.5,
        "underestimate_rate": len(underestimates) / len(errors),
        "overestimate_rate": len(overestimates) / len(errors),
        "large_underestimate_rate": len(large_underestimates) / len(errors),
        "large_overestimate_rate": len(large_overestimates) / len(errors),
        "max_overestimate": max(overestimates) if overestimates else 0.0,
    }


def run_search(grid, start, goal, heuristic, secondary_heuristic=None, secondary_priority=None):
    start_time = time.perf_counter()
    result = astar_search(
        grid,
        start,
        goal,
        heuristic,
        secondary_heuristic=secondary_heuristic,
        secondary_priority=secondary_priority,
    )
    runtime = time.perf_counter() - start_time
    return result, runtime


def make_hybrid_heuristic(unet_heuristic):
    def heuristic(current, goal):
        return max(manhattan_heuristic(current, goal), unet_heuristic(current, goal))

    return heuristic


def unet_checkpoint_path(project_root, checkpoint):
    if checkpoint in CHECKPOINT_CHOICES:
        checkpoint_name = CHECKPOINT_CHOICES[checkpoint]
        return os.path.join(project_root, "checkpoints", checkpoint_name)
    return checkpoint


def load_models(project_root, unet_checkpoint):
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    mlp_path = os.path.join(checkpoints_dir, "mlp_heuristic.pt")
    unet_path = unet_checkpoint_path(project_root, unet_checkpoint)

    if not os.path.exists(mlp_path):
        raise FileNotFoundError(f"Missing MLP checkpoint: {mlp_path}")
    if not os.path.exists(unet_path):
        raise FileNotFoundError(f"Missing U-Net checkpoint: {unet_path}")

    return load_mlp_heuristic(mlp_path), load_unet_heuristic(unet_path)


def result_row(
    seed,
    map_size,
    obstacle_rate,
    map_mode,
    structured_type,
    start_goal_mode,
    start,
    goal,
    heuristic_name,
    result,
    runtime,
    optimal_cost,
    metrics,
    tie_diagnostics=None,
):
    path_found = bool(result["path"])
    path_length = result["cost"]
    optimal = path_found and path_length == optimal_cost

    tie_diagnostics = tie_diagnostics or {}
    return {
        "seed": seed,
        "map_size": map_size,
        "obstacle_rate": obstacle_rate,
        "map_mode": map_mode,
        "structured_type": structured_type,
        "start_goal_mode": start_goal_mode,
        "start_row": start[0],
        "start_col": start[1],
        "goal_row": goal[0],
        "goal_col": goal[1],
        "heuristic": heuristic_name,
        "path_found": path_found,
        "optimal": optimal,
        "optimal_cost": optimal_cost,
        "path_length": path_length,
        "expanded_nodes": result["expanded"],
        "runtime_seconds": runtime,
        "expanded_diff_from_manhattan": "",
        "expanded_diff_from_unet_tiebreak": "",
        "expanded_diff_from_true_distance_tiebreak": "",
        "oracle_benefit_capture": "",
        "tie_episode_count": tie_diagnostics.get("tie_episode_count", ""),
        "mean_tie_snapshot_size": tie_diagnostics.get("mean_tie_snapshot_size", ""),
        "max_tie_snapshot_size": tie_diagnostics.get("max_tie_snapshot_size", ""),
        "oracle_corrected_nodes": tie_diagnostics.get("oracle_corrected_nodes", ""),
        "oracle_corrected_expansion_fraction": tie_diagnostics.get("oracle_corrected_expansion_fraction", ""),
        "later_arrivals_into_active_primary_f": tie_diagnostics.get("later_arrivals_into_active_primary_f", ""),
        "skip_reason": "",
        **metrics,
    }


def parse_seeds(seed_text):
    if seed_text is None:
        return SEEDS
    if ":" in seed_text:
        start, end = seed_text.split(":", maxsplit=1)
        return list(range(int(start), int(end)))
    return [int(seed) for seed in seed_text.split(",") if seed.strip()]


def empty_result_row(
    seed,
    map_size,
    obstacle_rate,
    map_mode,
    structured_type,
    start_goal_mode,
    start,
    goal,
    heuristic_name,
    skip_reason,
):
    start_row, start_col = start if start is not None else ("", "")
    goal_row, goal_col = goal if goal is not None else ("", "")
    return {
        "seed": seed,
        "map_size": map_size,
        "obstacle_rate": obstacle_rate,
        "map_mode": map_mode,
        "structured_type": structured_type,
        "start_goal_mode": start_goal_mode,
        "start_row": start_row,
        "start_col": start_col,
        "goal_row": goal_row,
        "goal_col": goal_col,
        "heuristic": heuristic_name,
        "path_found": False,
        "optimal": False,
        "optimal_cost": -1,
        "path_length": -1,
        "expanded_nodes": 0,
        "runtime_seconds": 0.0,
        "expanded_diff_from_manhattan": "",
        "expanded_diff_from_unet_tiebreak": "",
        "expanded_diff_from_true_distance_tiebreak": "",
        "oracle_benefit_capture": "",
        "tie_episode_count": "",
        "mean_tie_snapshot_size": "",
        "max_tie_snapshot_size": "",
        "oracle_corrected_nodes": "",
        "oracle_corrected_expansion_fraction": "",
        "later_arrivals_into_active_primary_f": "",
        "skip_reason": skip_reason,
        "mae": "",
        "mse": "",
        "rmse": "",
        "underestimate_rate": "",
        "overestimate_rate": "",
        "large_underestimate_rate": "",
        "large_overestimate_rate": "",
        "max_overestimate": "",
    }


def make_true_distance_tiebreak(distance_grid):
    def heuristic(current, unused_goal):
        value = distance_grid[current[0]][current[1]]
        return float(value) if value >= 0 else float("inf")

    return heuristic


def build_heuristics(methods, mlp_model, unet_model, grid, goal, distance_grid=None):
    mlp_heuristic = None
    mlp_table_heuristic = None
    unet_heuristic = None
    true_distance_heuristic = None
    heuristics = []

    for heuristic_name in methods:
        if heuristic_name == "dijkstra":
            heuristics.append((heuristic_name, dijkstra_heuristic, None, None))
        elif heuristic_name == "manhattan":
            heuristics.append((heuristic_name, manhattan_heuristic, None, None))
        elif heuristic_name == "mlp":
            if mlp_heuristic is None:
                mlp_heuristic = make_mlp_heuristic(mlp_model)
            heuristics.append((heuristic_name, mlp_heuristic, None, None))
        elif heuristic_name == "mlp_table":
            if mlp_table_heuristic is None:
                mlp_table_heuristic = make_mlp_table_heuristic(mlp_model, grid, goal)
            heuristics.append((heuristic_name, mlp_table_heuristic, None, None))
        elif heuristic_name == "unet":
            if unet_heuristic is None:
                unet_heuristic = make_unet_heuristic(unet_model, grid, goal)
            heuristics.append((heuristic_name, unet_heuristic, None, None))
        elif heuristic_name == "hybrid_max_manhattan_unet":
            if unet_heuristic is None:
                unet_heuristic = make_unet_heuristic(unet_model, grid, goal)
            heuristics.append((heuristic_name, make_hybrid_heuristic(unet_heuristic), None, None))
        elif heuristic_name == "manhattan_unet_tiebreak":
            if unet_heuristic is None:
                unet_heuristic = make_unet_heuristic(unet_model, grid, goal)
            heuristics.append((heuristic_name, manhattan_heuristic, unet_heuristic, None))
        elif heuristic_name == "manhattan_counter_tiebreak":
            heuristics.append((heuristic_name, manhattan_heuristic, None, lambda current, goal, g: 0.0))
        elif heuristic_name == "manhattan_large_g_tiebreak":
            heuristics.append((heuristic_name, manhattan_heuristic, None, lambda current, goal, g: -float(g)))
        elif heuristic_name == "manhattan_small_g_tiebreak":
            heuristics.append((heuristic_name, manhattan_heuristic, None, lambda current, goal, g: float(g)))
        elif heuristic_name == "manhattan_mlp_tiebreak":
            if mlp_table_heuristic is None:
                mlp_table_heuristic = make_mlp_table_heuristic(mlp_model, grid, goal)
            heuristics.append((heuristic_name, manhattan_heuristic, mlp_table_heuristic, None))
        elif heuristic_name == "manhattan_true_distance_tiebreak":
            if distance_grid is None:
                raise ValueError("distance_grid is required for manhattan_true_distance_tiebreak")
            if true_distance_heuristic is None:
                true_distance_heuristic = make_true_distance_tiebreak(distance_grid)
            heuristics.append((heuristic_name, manhattan_heuristic, true_distance_heuristic, None))
        elif heuristic_name in ORACLE_TOPK_METHODS:
            if unet_heuristic is None:
                unet_heuristic = make_unet_heuristic(unet_model, grid, goal)
            heuristics.append(
                (
                    heuristic_name,
                    manhattan_heuristic,
                    None,
                    {"kind": "oracle_topk", "k": ORACLE_TOPK_METHODS[heuristic_name], "unet": unet_heuristic},
                )
            )
        else:
            raise ValueError(f"Unknown heuristic: {heuristic_name}")

    return heuristics


def write_results_csv(output_path, rows):
    fieldnames = [
        "seed",
        "map_size",
        "obstacle_rate",
        "map_mode",
        "structured_type",
        "start_goal_mode",
        "start_row",
        "start_col",
        "goal_row",
        "goal_col",
        "heuristic",
        "path_found",
        "optimal",
        "optimal_cost",
        "path_length",
        "expanded_nodes",
        "runtime_seconds",
        "expanded_diff_from_manhattan",
        "expanded_diff_from_unet_tiebreak",
        "expanded_diff_from_true_distance_tiebreak",
        "oracle_benefit_capture",
        "tie_episode_count",
        "mean_tie_snapshot_size",
        "max_tie_snapshot_size",
        "oracle_corrected_nodes",
        "oracle_corrected_expansion_fraction",
        "later_arrivals_into_active_primary_f",
        "skip_reason",
        "mae",
        "mse",
        "rmse",
        "underestimate_rate",
        "overestimate_rate",
        "large_underestimate_rate",
        "large_overestimate_rate",
        "max_overestimate",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    if not values:
        return ""
    return sum(values) / len(values)


def numeric_value(row, key):
    value = row[key]
    if value == "":
        return None
    return float(value)


def aggregate_results(rows):
    groups = {}
    for row in rows:
        key = (row["map_mode"], row["structured_type"], row["obstacle_rate"], row["heuristic"])
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for map_mode, structured_type, obstacle_rate, heuristic in sorted(groups.keys()):
        group_rows = groups[(map_mode, structured_type, obstacle_rate, heuristic)]
        solvable_rows = [row for row in group_rows if row["optimal_cost"] >= 0]
        found_rows = [row for row in solvable_rows if row["path_found"]]
        overestimate_rates = [
            numeric_value(row, "overestimate_rate")
            for row in solvable_rows
            if numeric_value(row, "overestimate_rate") is not None
        ]
        summary_rows.append(
            {
                "map_mode": map_mode,
                "structured_type": structured_type,
                "obstacle_rate": obstacle_rate,
                "heuristic": heuristic,
                "runs": len(group_rows),
                "solvable_runs": len(solvable_rows),
                "path_found_rate": mean(1.0 if row["path_found"] else 0.0 for row in solvable_rows),
                "optimality_rate": mean(1.0 if row["optimal"] else 0.0 for row in solvable_rows),
                "mean_expanded_nodes": mean(row["expanded_nodes"] for row in solvable_rows),
                "mean_runtime_seconds": mean(row["runtime_seconds"] for row in solvable_rows),
                "mean_path_length": mean(row["path_length"] for row in found_rows),
                "mean_overestimate_rate": mean(overestimate_rates),
            }
        )

    return summary_rows


def add_per_case_expansion_differences(map_results):
    manhattan = map_results.get("manhattan", {}).get("row")
    unet = map_results.get("manhattan_unet_tiebreak", {}).get("row")
    oracle = map_results.get("manhattan_true_distance_tiebreak", {}).get("row")
    if not manhattan:
        return

    manhattan_expanded = manhattan["expanded_nodes"]
    unet_expanded = unet["expanded_nodes"] if unet else None
    oracle_expanded = oracle["expanded_nodes"] if oracle else None
    denominator = None
    if unet_expanded is not None and oracle_expanded is not None:
        denominator = unet_expanded - oracle_expanded

    for payload in map_results.values():
        row = payload["row"]
        expanded = row["expanded_nodes"]
        row["expanded_diff_from_manhattan"] = expanded - manhattan_expanded
        if unet_expanded is not None:
            row["expanded_diff_from_unet_tiebreak"] = expanded - unet_expanded
        if oracle_expanded is not None:
            row["expanded_diff_from_true_distance_tiebreak"] = expanded - oracle_expanded
        if denominator and denominator > 0:
            row["oracle_benefit_capture"] = (unet_expanded - expanded) / denominator
        elif denominator is not None:
            row["oracle_benefit_capture"] = ""


def write_summary_csv(output_path, rows):
    fieldnames = [
        "map_mode",
        "structured_type",
        "obstacle_rate",
        "heuristic",
        "runs",
        "solvable_runs",
        "path_found_rate",
        "optimality_rate",
        "mean_expanded_nodes",
        "mean_runtime_seconds",
        "mean_path_length",
        "mean_overestimate_rate",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def output_file_path(output_dir, base_name, output_tag):
    if not output_tag:
        return os.path.join(output_dir, f"{base_name}.csv")
    return os.path.join(output_dir, f"{base_name}_{output_tag}.csv")


def map_types_for_mode(map_mode, structured_types):
    if map_mode == "random":
        return ["random"]
    if map_mode == "structured":
        return structured_types
    raise ValueError(f"Unknown map mode: {map_mode}")


def generate_experiment_map(map_size, seed, obstacle_rate, map_mode, structured_type):
    if map_mode == "random":
        return gen_map(width=map_size, height=map_size, seed=seed, obstacle_rate=obstacle_rate)
    return generate_structured_map(
        width=map_size,
        height=map_size,
        seed=seed,
        obstacle_rate=obstacle_rate,
        structured_type=structured_type,
    )


def free_cells(grid):
    cells = []
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == 0:
                cells.append((r, c))
    return cells


def pair_rng_seed(seed, map_size, obstacle_rate):
    return seed * 1000003 + map_size * 1009 + int(obstacle_rate * 1000)


def select_fixed_start_goal(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    start = START
    goal = (rows - 1, cols - 1)
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0
    distance_grid = compute_distance_to_goal(grid, goal)
    optimal_cost = shortest_path_length_from_distance_grid(distance_grid, start)
    if optimal_cost == -1:
        return start, goal, distance_grid, optimal_cost, "unsolvable_fixed_start_goal"
    return start, goal, distance_grid, optimal_cost, ""


def select_random_start_goal(grid, seed, map_size, obstacle_rate, max_retries):
    cells = free_cells(grid)
    if len(cells) < 2:
        return None, None, None, -1, "not_enough_free_cells"

    rng = random.Random(pair_rng_seed(seed, map_size, obstacle_rate))
    for _ in range(max_retries):
        start = rng.choice(cells)
        goal = rng.choice(cells)
        if start == goal:
            continue

        distance_grid = compute_distance_to_goal(grid, goal)
        optimal_cost = shortest_path_length_from_distance_grid(distance_grid, start)
        if optimal_cost != -1:
            return start, goal, distance_grid, optimal_cost, ""

    return None, None, None, -1, "no_solvable_random_start_goal_pair"


def select_start_goal(grid, seed, map_size, obstacle_rate, start_goal_mode, max_retries):
    if start_goal_mode == "fixed":
        return select_fixed_start_goal(grid)
    if start_goal_mode == "random":
        return select_random_start_goal(grid, seed, map_size, obstacle_rate, max_retries)
    raise ValueError(f"Unknown start-goal mode: {start_goal_mode}")


def format_grid(grid):
    return "\n".join("".join("#" if value else "." for value in row) for row in grid) + "\n"


def format_path_overlay(grid, start, goal, path):
    path_cells = set(path)
    lines = []
    for r, row in enumerate(grid):
        cells = []
        for c, value in enumerate(row):
            cell = (r, c)
            if cell == start:
                cells.append("S")
            elif cell == goal:
                cells.append("G")
            elif value == 1:
                cells.append("#")
            elif cell in path_cells:
                cells.append("*")
            else:
                cells.append(".")
        lines.append("".join(cells))
    return "\n".join(lines) + "\n"


def save_case(case_dir, category, case_id, grid, start, goal, optimal_cost, map_results):
    os.makedirs(case_dir, exist_ok=True)
    case_name = (
        f"{category}_rate{case_id['obstacle_rate']}_seed{case_id['seed']}"
        f"_size{case_id['map_size']}"
    )
    case_path = os.path.join(case_dir, case_name)
    os.makedirs(case_path, exist_ok=True)

    metadata = {
        "category": category,
        "seed": case_id["seed"],
        "map_size": case_id["map_size"],
        "obstacle_rate": case_id["obstacle_rate"],
        "start": list(start),
        "goal": list(goal),
        "optimal_cost": optimal_cost,
        "heuristics": {},
    }
    for heuristic_name, payload in map_results.items():
        row = payload["row"]
        metadata["heuristics"][heuristic_name] = {
            "heuristic_type": heuristic_name,
            "path_found": row["path_found"],
            "optimal": row["optimal"],
            "path_length": row["path_length"],
            "expanded_nodes": row["expanded_nodes"],
            "runtime_seconds": row["runtime_seconds"],
            "mae": row["mae"],
            "mse": row["mse"],
            "overestimate_rate": row["overestimate_rate"],
            "underestimate_rate": row["underestimate_rate"],
        }

    with open(os.path.join(case_path, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    with open(os.path.join(case_path, "obstacle_map.txt"), "w", encoding="utf-8") as file:
        file.write(format_grid(grid))

    for heuristic_name, payload in map_results.items():
        path = payload["result"]["path"]
        with open(os.path.join(case_path, f"path_{heuristic_name}.txt"), "w", encoding="utf-8") as file:
            file.write(format_path_overlay(grid, start, goal, path))


def collect_interesting_cases(
    case_output_dir,
    case_counts,
    max_cases_per_category,
    min_delta,
    ratio,
    grid,
    start,
    goal,
    optimal_cost,
    case_id,
    map_results,
):
    if "unet" not in map_results or "manhattan" not in map_results:
        return

    unet_row = map_results["unet"]["row"]
    manhattan_row = map_results["manhattan"]["row"]
    unet_expanded = unet_row["expanded_nodes"]
    manhattan_expanded = manhattan_row["expanded_nodes"]

    categories = []
    if unet_expanded - manhattan_expanded >= min_delta and unet_expanded >= manhattan_expanded * ratio:
        categories.append("unet_worse_than_manhattan")
    if manhattan_expanded - unet_expanded >= min_delta and manhattan_expanded >= unet_expanded * ratio:
        categories.append("unet_better_than_manhattan")
    if unet_row["path_found"] and not unet_row["optimal"]:
        categories.append("unet_non_optimal")

    for category in categories:
        if case_counts.get(category, 0) >= max_cases_per_category:
            continue
        category_dir = os.path.join(case_output_dir, category)
        save_case(category_dir, category, case_id, grid, start, goal, optimal_cost, map_results)
        case_counts[category] = case_counts.get(category, 0) + 1


def run_experiments(
    methods,
    unet_checkpoint,
    output_tag,
    seeds,
    collect_cases,
    max_cases_per_category,
    case_min_delta,
    case_ratio,
    start_goal_mode,
    random_start_goal_retries,
    map_mode,
    structured_types,
):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "experiments")
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_file_path(output_dir, "results", output_tag)
    summary_path = output_file_path(output_dir, "summary", output_tag)
    case_output_dir = os.path.join(project_root, "outputs", "failure_cases", output_tag or "default")

    mlp_model, unet_model = load_models(project_root, unet_checkpoint)
    rows = []
    case_counts = {}
    skip_counts = {}

    for map_size in MAP_SIZES:
        for obstacle_rate in OBSTACLE_RATES:
            for structured_type in map_types_for_mode(map_mode, structured_types):
                for seed in seeds:
                    grid = generate_experiment_map(map_size, seed, obstacle_rate, map_mode, structured_type)
                    start, goal, distance_grid, optimal_cost, skip_reason = select_start_goal(
                        grid,
                        seed,
                        map_size,
                        obstacle_rate,
                        start_goal_mode,
                        random_start_goal_retries,
                    )

                    if skip_reason:
                        skip_counts[skip_reason] = skip_counts.get(skip_reason, 0) + 1
                        for heuristic_name in methods:
                            rows.append(
                                empty_result_row(
                                    seed,
                                    map_size,
                                    obstacle_rate,
                                    map_mode,
                                    structured_type,
                                    start_goal_mode,
                                    start,
                                    goal,
                                    heuristic_name,
                                    skip_reason,
                                )
                            )
                        continue

                    heuristics = build_heuristics(methods, mlp_model, unet_model, grid, goal, distance_grid)
                    map_results = {}

                    for heuristic_name, heuristic, secondary_heuristic, secondary_priority in heuristics:
                        if isinstance(secondary_priority, dict) and secondary_priority.get("kind") == "oracle_topk":
                            result, runtime = run_oracle_topk_search(
                                grid,
                                start,
                                goal,
                                secondary_priority["unet"],
                                distance_grid,
                                secondary_priority["k"],
                            )
                        else:
                            result, runtime = run_search(
                                grid, start, goal, heuristic, secondary_heuristic, secondary_priority
                            )
                        metrics = prediction_error_metrics(distance_grid, heuristic, goal)
                        row = result_row(
                            seed,
                            map_size,
                            obstacle_rate,
                            map_mode,
                            structured_type,
                            start_goal_mode,
                            start,
                            goal,
                            heuristic_name,
                            result,
                            runtime,
                            optimal_cost,
                            metrics,
                            result.get("tie_diagnostics"),
                        )
                        rows.append(row)
                        map_results[heuristic_name] = {"row": row, "result": result}

                    add_per_case_expansion_differences(map_results)

                    if collect_cases:
                        case_id = {
                            "seed": seed,
                            "map_size": map_size,
                            "obstacle_rate": obstacle_rate,
                        }
                        collect_interesting_cases(
                            case_output_dir,
                            case_counts,
                            max_cases_per_category,
                            case_min_delta,
                            case_ratio,
                            grid,
                            start,
                            goal,
                            optimal_cost,
                            case_id,
                            map_results,
                        )

    write_results_csv(output_path, rows)
    summary_rows = aggregate_results(rows)
    write_summary_csv(summary_path, summary_rows)

    print(f"Saved experiment results to {output_path}")
    print(f"Saved experiment summary to {summary_path}")
    print(f"Rows: {len(rows)}")
    print(f"Map mode: {map_mode}")
    print(f"Structured types: {structured_types}")
    print(f"Start-goal mode: {start_goal_mode}")
    print(f"U-Net checkpoint: {unet_checkpoint_path(project_root, unet_checkpoint)}")
    print(f"Skipped map/runs by reason: {skip_counts}")
    if collect_cases:
        print(f"Saved interesting cases to {case_output_dir}")
        print(f"Case counts: {case_counts}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run heuristic search experiments.")
    parser.add_argument(
        "--checkpoint",
        default="compatible",
        help="U-Net checkpoint to use: compatible, best, latest, or a checkpoint path.",
    )
    parser.add_argument(
        "--methods",
        choices=["core", "all", "extended", "tiebreak_controls", "oracle_topk"],
        default="core",
        help="core compares Dijkstra, Manhattan, MLP table, and U-Net. all also includes raw MLP and hybrid. extended adds Manhattan+U-Net tie-breaking. tiebreak_controls runs tie-breaking controls. oracle_topk runs partial oracle tie-set budgets.",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix for output files, for example best writes results_best.csv and summary_best.csv.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Seeds to evaluate. Use '0:100' for a range or '0,1,2' for an explicit list. Defaults to 0:20.",
    )
    parser.add_argument(
        "--collect-cases",
        action="store_true",
        help="Save maps where U-Net is much worse/better than Manhattan or returns a non-optimal path.",
    )
    parser.add_argument(
        "--case-min-delta",
        type=int,
        default=DEFAULT_CASE_MIN_DELTA,
        help="Minimum expanded-node difference for better/worse case collection.",
    )
    parser.add_argument(
        "--case-ratio",
        type=float,
        default=DEFAULT_CASE_RATIO,
        help="Minimum expanded-node ratio for better/worse case collection.",
    )
    parser.add_argument(
        "--max-cases-per-category",
        type=int,
        default=DEFAULT_MAX_CASES_PER_CATEGORY,
        help="Maximum number of saved maps per case category.",
    )
    parser.add_argument(
        "--start-goal-mode",
        choices=["fixed", "random"],
        default="fixed",
        help="fixed uses start=(0,0), goal=(rows-1,cols-1). random samples a reproducible solvable start/goal pair.",
    )
    parser.add_argument(
        "--random-start-goal-retries",
        type=int,
        default=DEFAULT_RANDOM_START_GOAL_RETRIES,
        help="Maximum attempts to sample a solvable random start/goal pair per map.",
    )
    parser.add_argument(
        "--map-mode",
        choices=["random", "structured"],
        default="random",
        help="random uses existing Bernoulli obstacle maps. structured uses controlled map templates.",
    )
    parser.add_argument(
        "--structured-types",
        default="all",
        help="Structured map types to run, for example all or narrow_corridor,bottleneck,maze_like,large_block.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.methods == "core":
        methods = CORE_HEURISTIC_NAMES
    elif args.methods == "all":
        methods = ALL_HEURISTIC_NAMES
    elif args.methods == "extended":
        methods = EXTENDED_HEURISTIC_NAMES
    elif args.methods == "tiebreak_controls":
        methods = TIEBREAK_CONTROL_HEURISTIC_NAMES
    else:
        methods = ORACLE_TOPK_HEURISTIC_NAMES
    run_experiments(
        methods,
        args.checkpoint,
        args.output_tag,
        parse_seeds(args.seeds),
        args.collect_cases,
        args.max_cases_per_category,
        args.case_min_delta,
        args.case_ratio,
        args.start_goal_mode,
        args.random_start_goal_retries,
        args.map_mode,
        parse_structured_types(args.structured_types),
    )
