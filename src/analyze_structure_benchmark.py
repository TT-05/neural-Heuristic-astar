import argparse
import csv
import json
import os

from astar import astar_search
from experiment import build_heuristics, dijkstra_heuristic, load_models
from gen_map import gen_map
from model import manhattan_heuristic
from analyze_failure_patterns import (
    articulation_points,
    classify_structure,
    free_cells,
    largest_obstacle_component,
    mean,
    neighbors,
    path_overlap,
    save_path_overlay,
    setup_matplotlib,
    write_csv,
)


METHODS = ["manhattan", "mlp_table", "unet"]
ADVANTAGE_MIN_DELTA = 5
ADVANTAGE_RATIO = 1.15


def read_results(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def to_float(row, key, default=0.0):
    value = row.get(key, "")
    if value == "":
        return default
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    return float(value)


def to_int(row, key, default=0):
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def map_key(row):
    return (
        row["seed"],
        row["map_size"],
        row["obstacle_rate"],
        row.get("start_goal_mode", ""),
        row.get("start_row", ""),
        row.get("start_col", ""),
        row.get("goal_row", ""),
        row.get("goal_col", ""),
    )


def group_rows_by_map(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(map_key(row), {})[row["heuristic"]] = row
    return grouped


def build_grid_for_result(seed, map_size, obstacle_rate, start, goal, start_goal_mode):
    grid = gen_map(width=map_size, height=map_size, seed=seed, obstacle_rate=obstacle_rate)
    if start_goal_mode == "fixed":
        grid[start[0]][start[1]] = 0
        grid[goal[0]][goal[1]] = 0
    return grid


def structure_metrics(grid, start, goal, optimal_path, manhattan_path, unet_path, obstacle_rate):
    cells = free_cells(grid)
    degrees = {cell: sum(1 for _ in neighbors(grid, cell)) for cell in cells}
    corridor_cells = [cell for cell, degree in degrees.items() if degree <= 2]
    articulation = articulation_points(grid)
    path_cells = set(optimal_path)
    path_corridor_cells = [cell for cell in path_cells if degrees.get(cell, 0) <= 2]
    path_articulation_cells = [cell for cell in path_cells if cell in articulation]
    obstacle_cells = sum(sum(1 for value in row if value == 1) for row in grid)
    total_cells = len(grid) * len(grid[0])
    obstacle_density = obstacle_cells / total_cells
    largest_block_rate = largest_obstacle_component(grid) / total_cells
    corridor_rate = len(corridor_cells) / len(cells) if cells else 0.0
    path_corridor_rate = len(path_corridor_cells) / len(optimal_path) if optimal_path else 0.0
    articulation_rate = len(articulation) / len(cells) if cells else 0.0
    path_articulation_rate = len(path_articulation_cells) / len(optimal_path) if optimal_path else 0.0
    manhattan_overlap = path_overlap(optimal_path, manhattan_path)
    unet_overlap = path_overlap(optimal_path, unet_path)
    manhattan_distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    optimal_cost = len(optimal_path) - 1 if optimal_path else -1
    path_stretch = optimal_cost / manhattan_distance if manhattan_distance > 0 and optimal_cost >= 0 else 1.0

    return {
        "obstacle_density": obstacle_density,
        "largest_obstacle_block_rate": largest_block_rate,
        "corridor_rate": corridor_rate,
        "path_corridor_rate": path_corridor_rate,
        "articulation_count": len(articulation),
        "articulation_rate": articulation_rate,
        "path_articulation_count": len(path_articulation_cells),
        "path_articulation_rate": path_articulation_rate,
        "manhattan_optimal_path_overlap": manhattan_overlap,
        "unet_optimal_path_overlap": unet_overlap,
        "manhattan_distance": manhattan_distance,
        "path_stretch": path_stretch,
        "structure_labels": classify_structure(
            obstacle_rate,
            obstacle_density,
            largest_block_rate,
            corridor_rate,
            path_corridor_rate,
            articulation_rate,
            path_articulation_rate,
            manhattan_overlap,
            unet_overlap,
        ),
    }


def quantile_thresholds(values):
    values = sorted(values)
    if not values:
        return 0.0, 0.0
    low_index = int((len(values) - 1) * 0.33)
    high_index = int((len(values) - 1) * 0.67)
    return values[low_index], values[high_index]


def bin_value(value, low, high):
    if value <= low:
        return "low"
    if value <= high:
        return "medium"
    return "high"


def add_difficulty_bins(map_rows):
    metrics = [
        "optimal_cost",
        "obstacle_density",
        "path_stretch",
        "corridor_rate",
        "articulation_count",
    ]
    thresholds = {
        metric: quantile_thresholds(float(row[metric]) for row in map_rows)
        for metric in metrics
    }
    for row in map_rows:
        for metric, (low, high) in thresholds.items():
            row[f"{metric}_bin"] = bin_value(float(row[metric]), low, high)
    return thresholds


def summarize_group(rows, group_keys):
    groups = {}
    for row in rows:
        for heuristic in METHODS:
            result = row["heuristics"].get(heuristic)
            if not result:
                continue
            key = tuple(row[key] for key in group_keys) + (heuristic,)
            groups.setdefault(key, []).append((row, result))

    summary = []
    for key, items in sorted(groups.items()):
        values = [item[1] for item in items]
        row = {group_key: key[index] for index, group_key in enumerate(group_keys)}
        row["heuristic"] = key[-1]
        row["maps"] = len(items)
        row["mean_expanded_nodes"] = mean(to_float(value, "expanded_nodes") for value in values)
        row["mean_runtime_seconds"] = mean(to_float(value, "runtime_seconds") for value in values)
        row["optimality_rate"] = mean(1.0 if value["optimal"] == "True" else 0.0 for value in values)
        row["mean_overestimate_rate"] = mean(to_float(value, "overestimate_rate") for value in values)
        row["mean_path_length"] = mean(to_float(value, "path_length") for value in values if to_float(value, "path_length") >= 0)
        summary.append(row)
    return summary


def summarize_by_structure(map_rows):
    expanded_rows = []
    for row in map_rows:
        for label in row["structure_labels"].split("; "):
            structure_row = dict(row)
            structure_row["structure"] = label
            expanded_rows.append(structure_row)
    return summarize_group(expanded_rows, ["structure"])


def summarize_by_difficulty(map_rows):
    summary_rows = []
    for metric in ["optimal_cost", "obstacle_density", "path_stretch", "corridor_rate", "articulation_count"]:
        scoped_rows = []
        bin_key = f"{metric}_bin"
        for row in map_rows:
            scoped_row = dict(row)
            scoped_row["difficulty_metric"] = metric
            scoped_row["difficulty_bin"] = row[bin_key]
            scoped_rows.append(scoped_row)
        summary_rows.extend(summarize_group(scoped_rows, ["difficulty_metric", "difficulty_bin"]))
    return summary_rows


def clear_advantage(winner_expanded, loser_a_expanded, loser_b_expanded, min_delta, ratio):
    return (
        loser_a_expanded - winner_expanded >= min_delta
        and loser_b_expanded - winner_expanded >= min_delta
        and loser_a_expanded >= winner_expanded * ratio
        and loser_b_expanded >= winner_expanded * ratio
    )


def collect_advantage_cases(map_rows, min_delta, ratio):
    unet_cases = []
    mlp_cases = []
    for row in map_rows:
        heuristics = row["heuristics"]
        if not all(name in heuristics for name in METHODS):
            continue
        manhattan_expanded = to_float(heuristics["manhattan"], "expanded_nodes")
        mlp_expanded = to_float(heuristics["mlp_table"], "expanded_nodes")
        unet_expanded = to_float(heuristics["unet"], "expanded_nodes")

        base = {
            "case_id": row["case_id"],
            "seed": row["seed"],
            "obstacle_rate": row["obstacle_rate"],
            "start_goal_mode": row["start_goal_mode"],
            "start": row["start"],
            "goal": row["goal"],
            "structure_labels": row["structure_labels"],
            "optimal_cost": row["optimal_cost"],
            "path_stretch": row["path_stretch"],
            "corridor_rate": row["corridor_rate"],
            "articulation_count": row["articulation_count"],
            "manhattan_expanded": manhattan_expanded,
            "mlp_table_expanded": mlp_expanded,
            "unet_expanded": unet_expanded,
            "unet_mae": to_float(heuristics["unet"], "mae"),
            "unet_overestimate_rate": to_float(heuristics["unet"], "overestimate_rate"),
            "unet_optimal": heuristics["unet"]["optimal"],
        }

        if clear_advantage(unet_expanded, manhattan_expanded, mlp_expanded, min_delta, ratio):
            unet_cases.append(base)
        if clear_advantage(mlp_expanded, manhattan_expanded, unet_expanded, min_delta, ratio):
            mlp_cases.append(base)

    unet_cases.sort(key=lambda row: row["unet_expanded"] - min(row["manhattan_expanded"], row["mlp_table_expanded"]))
    mlp_cases.sort(key=lambda row: row["mlp_table_expanded"] - min(row["manhattan_expanded"], row["unet_expanded"]))
    return unet_cases, mlp_cases


def create_map_rows(grouped_rows, project_root, checkpoint, max_visual_cases):
    mlp_model, unet_model = load_models(project_root, checkpoint)
    map_rows = []

    for key, heuristics in sorted(grouped_rows.items()):
        if any(heuristics.get(method, {}).get("skip_reason") for method in heuristics):
            continue
        if not all(method in heuristics for method in METHODS):
            continue

        sample = next(iter(heuristics.values()))
        seed = to_int(sample, "seed")
        map_size = to_int(sample, "map_size")
        obstacle_rate = to_float(sample, "obstacle_rate")
        start_goal_mode = sample.get("start_goal_mode", "fixed")
        start = (to_int(sample, "start_row"), to_int(sample, "start_col"))
        goal = (to_int(sample, "goal_row"), to_int(sample, "goal_col"))
        grid = build_grid_for_result(seed, map_size, obstacle_rate, start, goal, start_goal_mode)
        built_heuristics = dict(build_heuristics(METHODS, mlp_model, unet_model, grid, goal))
        optimal_result = astar_search(grid, start, goal, dijkstra_heuristic)
        manhattan_result = astar_search(grid, start, goal, manhattan_heuristic)
        unet_result = astar_search(grid, start, goal, built_heuristics["unet"])
        metrics = structure_metrics(
            grid,
            start,
            goal,
            optimal_result["path"],
            manhattan_result["path"],
            unet_result["path"],
            obstacle_rate,
        )
        case_id = (
            f"{start_goal_mode}_rate{obstacle_rate}_seed{seed}_"
            f"s{start[0]}-{start[1]}_g{goal[0]}-{goal[1]}"
        )

        map_rows.append(
            {
                "case_id": case_id,
                "seed": seed,
                "map_size": map_size,
                "obstacle_rate": obstacle_rate,
                "start_goal_mode": start_goal_mode,
                "start": f"{start[0]},{start[1]}",
                "goal": f"{goal[0]},{goal[1]}",
                "start_tuple": start,
                "goal_tuple": goal,
                "optimal_cost": to_float(sample, "optimal_cost"),
                "heuristics": heuristics,
                "grid": grid,
                "optimal_path": optimal_result["path"],
                "manhattan_path": manhattan_result["path"],
                "unet_path": unet_result["path"],
                **metrics,
            }
        )

    return map_rows


def flatten_map_rows(map_rows):
    output_rows = []
    for row in map_rows:
        output = {key: value for key, value in row.items() if key not in {"heuristics", "grid", "optimal_path", "manhattan_path", "unet_path", "start_tuple", "goal_tuple"}}
        for heuristic in METHODS:
            result = row["heuristics"][heuristic]
            prefix = heuristic
            output[f"{prefix}_expanded_nodes"] = result["expanded_nodes"]
            output[f"{prefix}_runtime_seconds"] = result["runtime_seconds"]
            output[f"{prefix}_optimal"] = result["optimal"]
            output[f"{prefix}_path_length"] = result["path_length"]
            output[f"{prefix}_overestimate_rate"] = result["overestimate_rate"]
        output_rows.append(output)
    return output_rows


def save_advantage_visuals(project_root, output_dir, map_rows, unet_cases, mlp_cases, max_cases):
    plt, np = setup_matplotlib(project_root)
    if plt is None:
        return
    by_case_id = {row["case_id"]: row for row in map_rows}
    for category, cases in [("unet_advantage", unet_cases), ("mlp_advantage", mlp_cases)]:
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        for case in cases[:max_cases]:
            row = by_case_id[case["case_id"]]
            save_path_overlay(
                os.path.join(category_dir, f"{case['case_id']}.png"),
                row["grid"],
                row["start_tuple"],
                row["goal_tuple"],
                row["optimal_path"],
                row["manhattan_path"],
                row["unet_path"],
                case["case_id"],
                plt,
                np,
            )


def write_report(path, structure_summary, difficulty_summary, unet_cases, mlp_cases):
    def rows_for(summary, key, value):
        return [row for row in summary if row.get(key) == value]

    with open(path, "w", encoding="utf-8") as file:
        file.write("# Structure-Aware Benchmark Findings\n\n")
        file.write("## U-Net Advantage Cases\n\n")
        file.write(f"- Clear U-Net advantage cases: {len(unet_cases)}\n")
        for case in unet_cases[:10]:
            file.write(
                f"- {case['case_id']}: structures [{case['structure_labels']}], "
                f"expanded MLP={case['mlp_table_expanded']}, U-Net={case['unet_expanded']}, "
                f"Manhattan={case['manhattan_expanded']}, path_stretch={case['path_stretch']:.2f}, "
                f"U-Net overestimate={case['unet_overestimate_rate']:.3f}\n"
            )

        file.write("\n## MLP Advantage Cases\n\n")
        file.write(f"- Clear MLP table advantage cases: {len(mlp_cases)}\n")
        for case in mlp_cases[:10]:
            file.write(
                f"- {case['case_id']}: structures [{case['structure_labels']}], "
                f"expanded MLP={case['mlp_table_expanded']}, U-Net={case['unet_expanded']}, "
                f"Manhattan={case['manhattan_expanded']}, path_stretch={case['path_stretch']:.2f}, "
                f"U-Net overestimate={case['unet_overestimate_rate']:.3f}\n"
            )

        file.write("\n## Structure Summary\n\n")
        for structure in sorted({row["structure"] for row in structure_summary}):
            scoped = rows_for(structure_summary, "structure", structure)
            means = {row["heuristic"]: float(row["mean_expanded_nodes"]) for row in scoped}
            if all(name in means for name in METHODS):
                file.write(
                    f"- {structure}: Manhattan={means['manhattan']:.2f}, "
                    f"MLP={means['mlp_table']:.2f}, U-Net={means['unet']:.2f}, "
                    f"U-Net-MLP={means['unet'] - means['mlp_table']:.2f}\n"
                )

        file.write("\n## Difficulty Summary\n\n")
        for metric in ["optimal_cost", "path_stretch", "corridor_rate", "articulation_count"]:
            file.write(f"\n### {metric}\n")
            for difficulty_bin in ["low", "medium", "high"]:
                scoped = [
                    row for row in difficulty_summary
                    if row["difficulty_metric"] == metric and row["difficulty_bin"] == difficulty_bin
                ]
                means = {row["heuristic"]: float(row["mean_expanded_nodes"]) for row in scoped}
                if all(name in means for name in METHODS):
                    file.write(
                        f"- {difficulty_bin}: Manhattan={means['manhattan']:.2f}, "
                        f"MLP={means['mlp_table']:.2f}, U-Net={means['unet']:.2f}, "
                        f"U-Net-MLP={means['unet'] - means['mlp_table']:.2f}\n"
                    )

        file.write("\n## Interpretation\n\n")
        file.write(
            "This analysis checks whether aggregate means hide map-structure-specific behavior. "
            "If U-Net advantage cases cluster in corridor, bottleneck, maze-like, or high-stretch bins, "
            "that supports the idea that obstacle-aware predictions help on genuinely structured planning "
            "problems. If MLP advantage remains strongest on open or low-stretch maps, that supports the "
            "geometry-dominates hypothesis. The tables should be interpreted alongside optimality and "
            "overestimate rates because U-Net may reduce expansions while still risking non-optimal paths.\n"
        )


def analyze_structure_benchmark(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = args.input
    if input_path is None:
        input_path = os.path.join(project_root, "outputs", "experiments", "results_random_sg_100.csv")
    output_dir = args.output_dir
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(project_root, "outputs", "structure_benchmark", base_name)
    os.makedirs(output_dir, exist_ok=True)

    rows = read_results(input_path)
    grouped = group_rows_by_map(rows)
    map_rows = create_map_rows(grouped, project_root, args.checkpoint, args.max_visual_cases)
    thresholds = add_difficulty_bins(map_rows)

    flat_rows = flatten_map_rows(map_rows)
    map_table_path = os.path.join(output_dir, "map_structure_metrics.csv")
    write_csv(map_table_path, flat_rows, list(flat_rows[0].keys()) if flat_rows else [])

    structure_summary = summarize_by_structure(map_rows)
    structure_summary_path = os.path.join(output_dir, "structure_summary.csv")
    write_csv(structure_summary_path, structure_summary, list(structure_summary[0].keys()) if structure_summary else [])

    difficulty_summary = summarize_by_difficulty(map_rows)
    difficulty_summary_path = os.path.join(output_dir, "difficulty_summary.csv")
    write_csv(difficulty_summary_path, difficulty_summary, list(difficulty_summary[0].keys()) if difficulty_summary else [])

    unet_cases, mlp_cases = collect_advantage_cases(map_rows, args.advantage_min_delta, args.advantage_ratio)
    unet_path = os.path.join(output_dir, "unet_advantage_cases.csv")
    mlp_path = os.path.join(output_dir, "mlp_advantage_cases.csv")
    case_fields = list(unet_cases[0].keys()) if unet_cases else list(mlp_cases[0].keys()) if mlp_cases else []
    write_csv(unet_path, unet_cases, case_fields)
    write_csv(mlp_path, mlp_cases, case_fields)

    with open(os.path.join(output_dir, "difficulty_thresholds.json"), "w", encoding="utf-8") as file:
        json.dump(thresholds, file, indent=2)

    visuals_dir = os.path.join(output_dir, "advantage_overlays")
    save_advantage_visuals(project_root, visuals_dir, map_rows, unet_cases, mlp_cases, args.max_visual_cases)

    report_path = os.path.join(output_dir, "findings_report.md")
    write_report(report_path, structure_summary, difficulty_summary, unet_cases, mlp_cases)

    print(f"Saved map structure metrics to {map_table_path}")
    print(f"Saved structure summary to {structure_summary_path}")
    print(f"Saved difficulty summary to {difficulty_summary_path}")
    print(f"Saved U-Net advantage cases to {unet_path}")
    print(f"Saved MLP advantage cases to {mlp_path}")
    print(f"Saved advantage overlays to {visuals_dir}")
    print(f"Saved findings report to {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run structure-aware benchmark analysis over experiment results.")
    parser.add_argument("--input", default=None, help="Experiment results CSV. Defaults to random start-goal best-100 results.")
    parser.add_argument("--output-dir", default=None, help="Output analysis directory.")
    parser.add_argument("--checkpoint", default="best", help="compatible, best, latest, or checkpoint path for path overlays.")
    parser.add_argument("--advantage-min-delta", type=int, default=ADVANTAGE_MIN_DELTA)
    parser.add_argument("--advantage-ratio", type=float, default=ADVANTAGE_RATIO)
    parser.add_argument("--max-visual-cases", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    analyze_structure_benchmark(parse_args())
