import argparse
import csv
import json
import os
from collections import deque

from astar import astar_search
from bfs_label import compute_distance_to_goal
from gen_map import gen_map
from model import load_unet_heuristic, make_unet_heuristic, manhattan_heuristic


CHECKPOINT_CHOICES = {
    "compatible": "unet_heuristic.pt",
    "best": "unet_heuristic_best.pt",
    "latest": "unet_heuristic_latest.pt",
}


def dijkstra_heuristic(current, goal):
    return 0.0


def checkpoint_path(project_root, checkpoint):
    if checkpoint in CHECKPOINT_CHOICES:
        return os.path.join(project_root, "checkpoints", CHECKPOINT_CHOICES[checkpoint])
    return checkpoint


def case_dirs(root_dir):
    for category in sorted(os.listdir(root_dir)):
        category_dir = os.path.join(root_dir, category)
        if not os.path.isdir(category_dir):
            continue
        for name in sorted(os.listdir(category_dir)):
            path = os.path.join(category_dir, name)
            if os.path.exists(os.path.join(path, "metadata.json")):
                yield category, path


def load_metadata(case_dir):
    with open(os.path.join(case_dir, "metadata.json"), "r", encoding="utf-8") as file:
        return json.load(file)


def build_grid(metadata):
    map_size = int(metadata["map_size"])
    start = tuple(metadata["start"])
    goal = tuple(metadata["goal"])
    grid = gen_map(
        width=map_size,
        height=map_size,
        seed=int(metadata["seed"]),
        obstacle_rate=float(metadata["obstacle_rate"]),
    )
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0
    return grid


def neighbors(grid, cell):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    r, c = cell
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + dr
        nc = c + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            continue
        if grid[nr][nc] == 0:
            yield (nr, nc)


def free_cells(grid):
    cells = []
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == 0:
                cells.append((r, c))
    return cells


def largest_obstacle_component(grid):
    seen = set()
    best = 0
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 or (r, c) in seen:
                continue
            queue = deque([(r, c)])
            seen.add((r, c))
            size = 0
            while queue:
                cell_r, cell_c = queue.popleft()
                size += 1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = cell_r + dr
                    nc = cell_c + dc
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        continue
                    if grid[nr][nc] == 0 or (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    queue.append((nr, nc))
            best = max(best, size)
    return best


def articulation_points(grid):
    cells = free_cells(grid)
    cell_set = set(cells)
    if not cells:
        return set()

    index = {}
    low = {}
    parent = {}
    points = set()
    counter = [0]

    def dfs(cell):
        index[cell] = counter[0]
        low[cell] = counter[0]
        counter[0] += 1
        child_count = 0

        for neighbor in neighbors(grid, cell):
            if neighbor not in cell_set:
                continue
            if neighbor not in index:
                parent[neighbor] = cell
                child_count += 1
                dfs(neighbor)
                low[cell] = min(low[cell], low[neighbor])
                if cell not in parent and child_count > 1:
                    points.add(cell)
                if cell in parent and low[neighbor] >= index[cell]:
                    points.add(cell)
            elif parent.get(cell) != neighbor:
                low[cell] = min(low[cell], index[neighbor])

    for cell in cells:
        if cell not in index:
            dfs(cell)
    return points


def path_overlap(path_a, path_b):
    if not path_a or not path_b:
        return 0.0
    a = set(path_a)
    b = set(path_b)
    return len(a & b) / len(a | b)


def local_consistency_metrics(predicted_grid, valid_mask):
    total_edges = 0
    violations = 0
    magnitudes = []
    rows = len(predicted_grid)
    cols = len(predicted_grid[0]) if rows else 0

    for r in range(rows):
        for c in range(cols):
            if not valid_mask[r][c]:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if not valid_mask[nr][nc]:
                    continue
                total_edges += 1
                violation = predicted_grid[r][c] - predicted_grid[nr][nc] - 1.0
                if violation > 0:
                    violations += 1
                    magnitudes.append(violation)

    return {
        "local_consistency_violation_rate": violations / total_edges if total_edges else 0.0,
        "local_consistency_mean_violation": sum(magnitudes) / len(magnitudes) if magnitudes else 0.0,
        "local_consistency_max_violation": max(magnitudes) if magnitudes else 0.0,
    }


def prediction_grid_from_heuristic(grid, goal, heuristic):
    values = []
    for r, row in enumerate(grid):
        value_row = []
        for c, cell in enumerate(row):
            value_row.append(0.0 if cell == 1 else heuristic((r, c), goal))
        values.append(value_row)
    return values


def error_metrics(true_grid, predicted_grid):
    errors = []
    for r, row in enumerate(true_grid):
        for c, true_value in enumerate(row):
            if true_value < 0:
                continue
            errors.append(predicted_grid[r][c] - true_value)
    abs_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    overestimates = [error for error in errors if error > 0]
    underestimates = [error for error in errors if error < 0]
    mse = sum(squared_errors) / len(squared_errors) if squared_errors else 0.0
    return {
        "mae": sum(abs_errors) / len(abs_errors) if abs_errors else 0.0,
        "mse": mse,
        "overestimate_rate": len(overestimates) / len(errors) if errors else 0.0,
        "underestimate_rate": len(underestimates) / len(errors) if errors else 0.0,
        "max_overestimate": max(overestimates) if overestimates else 0.0,
        "max_underestimate": min(underestimates) if underestimates else 0.0,
    }


def error_region_metrics(grid, true_grid, predicted_grid, optimal_path, articulation):
    path_set = set(optimal_path)
    region_errors = {
        "near_goal": [],
        "near_obstacle": [],
        "corridor": [],
        "bottleneck": [],
        "optimal_path": [],
        "other": [],
    }

    for r, row in enumerate(true_grid):
        for c, true_value in enumerate(row):
            if true_value < 0:
                continue
            cell = (r, c)
            error = abs(predicted_grid[r][c] - true_value)
            degree = sum(1 for _ in neighbors(grid, cell))
            near_obstacle = degree < 4

            if true_value <= 5:
                region_errors["near_goal"].append(error)
            if near_obstacle:
                region_errors["near_obstacle"].append(error)
            if degree <= 2:
                region_errors["corridor"].append(error)
            if cell in articulation:
                region_errors["bottleneck"].append(error)
            if cell in path_set:
                region_errors["optimal_path"].append(error)
            if true_value > 5 and not near_obstacle and degree > 2 and cell not in articulation and cell not in path_set:
                region_errors["other"].append(error)

    summary = {}
    for region, values in region_errors.items():
        summary[f"{region}_mean_abs_error"] = sum(values) / len(values) if values else 0.0
        summary[f"{region}_max_abs_error"] = max(values) if values else 0.0
        summary[f"{region}_cells"] = len(values)
    return summary


def classify_structure(obstacle_rate, obstacle_density, largest_block_rate, corridor_rate, path_corridor_rate, articulation_rate, path_articulation_rate, manhattan_overlap, unet_overlap):
    labels = []
    if obstacle_density < 0.14:
        labels.append("open space")
    elif obstacle_density < 0.24:
        labels.append("sparse obstacles")
    elif obstacle_density < 0.34:
        labels.append("dense obstacles")
    else:
        labels.append("maze-like")

    if largest_block_rate >= 0.05:
        labels.append("large obstacle block")
    if path_corridor_rate >= 0.45 or corridor_rate >= 0.35:
        labels.append("narrow corridor")
    if path_articulation_rate >= 0.08 or articulation_rate >= 0.08:
        labels.append("bottleneck")
    if manhattan_overlap < 0.75 or unet_overlap < 0.75:
        labels.append("multiple alternative routes")
    if obstacle_rate >= 0.3 and corridor_rate >= 0.25:
        labels.append("maze-like")

    return "; ".join(dict.fromkeys(labels))


def setup_matplotlib(project_root):
    matplotlib_cache = os.path.join(project_root, "outputs", "matplotlib_cache")
    os.makedirs(matplotlib_cache, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_cache)
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        return None, None
    return plt, np


def save_path_overlay(path, grid, start, goal, optimal_path, manhattan_path, unet_path, title, plt, np):
    if plt is None:
        return
    image = np.ones((len(grid), len(grid[0]), 3), dtype=float)
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == 1:
                image[r, c] = [0.05, 0.05, 0.05]

    plt.figure(figsize=(6, 6))
    plt.imshow(image, interpolation="nearest")

    def draw(path_values, color, label, linewidth):
        if not path_values:
            return
        ys = [cell[0] for cell in path_values]
        xs = [cell[1] for cell in path_values]
        plt.plot(xs, ys, color=color, linewidth=linewidth, label=label, alpha=0.85)

    draw(optimal_path, "#1f77b4", "optimal", 3.0)
    draw(manhattan_path, "#2ca02c", "manhattan", 2.0)
    draw(unet_path, "#d62728", "u-net", 2.0)
    plt.scatter([start[1]], [start[0]], c="#9467bd", marker="o", s=45, label="start")
    plt.scatter([goal[1]], [goal[0]], c="#ff7f0e", marker="*", s=85, label="goal")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_error_triptych(path, grid, true_grid, predicted_grid, title, plt, np):
    if plt is None:
        return
    valid = np.array([[value >= 0 for value in row] for row in true_grid])
    true_values = np.array([[value if value >= 0 else np.nan for value in row] for row in true_grid], dtype=float)
    predicted_values = np.array(
        [[predicted_grid[r][c] if valid[r, c] else np.nan for c in range(len(grid[0]))] for r in range(len(grid))],
        dtype=float,
    )
    error_values = predicted_values - true_values

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [
        (true_values, "BFS true distance", "viridis"),
        (predicted_values, "U-Net predicted distance", "viridis"),
        (error_values, "prediction - true", "coolwarm"),
    ]
    for axis, (values, panel_title, cmap) in zip(axes, panels):
        image = axis.imshow(values, cmap=cmap)
        axis.set_title(panel_title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize_taxonomy(case_rows):
    groups = {}
    for row in case_rows:
        for label in row["structure_labels"].split("; "):
            groups.setdefault((row["case_category"], label), []).append(row)

    summary_rows = []
    for (case_category, structure), rows in sorted(groups.items()):
        summary_rows.append(
            {
                "case_category": case_category,
                "structure": structure,
                "cases": len(rows),
                "mean_obstacle_rate": mean(float(row["obstacle_rate"]) for row in rows),
                "mean_unet_expanded": mean(float(row["unet_expanded_nodes"]) for row in rows),
                "mean_manhattan_expanded": mean(float(row["manhattan_expanded_nodes"]) for row in rows),
                "mean_unet_minus_manhattan_expanded": mean(float(row["unet_minus_manhattan_expanded"]) for row in rows),
                "mean_unet_minus_mlp_expanded": mean(float(row["unet_minus_mlp_expanded"]) for row in rows),
                "mean_mae": mean(float(row["mae"]) for row in rows),
                "mean_overestimate_rate": mean(float(row["overestimate_rate"]) for row in rows),
                "mean_underestimate_rate": mean(float(row["underestimate_rate"]) for row in rows),
                "mean_local_consistency_violation_rate": mean(
                    float(row["local_consistency_violation_rate"]) for row in rows
                ),
            }
        )
    return summary_rows


def summarize_error_regions(case_rows):
    regions = ["near_goal", "near_obstacle", "corridor", "bottleneck", "optimal_path", "other"]
    groups = {}
    for row in case_rows:
        groups.setdefault(row["case_category"], []).append(row)

    summary_rows = []
    for case_category, rows in sorted(groups.items()):
        for region in regions:
            scoped_rows = [row for row in rows if float(row[f"{region}_cells"]) > 0]
            summary_rows.append(
                {
                    "case_category": case_category,
                    "region": region,
                    "cases": len(scoped_rows),
                    "mean_abs_error": mean(float(row[f"{region}_mean_abs_error"]) for row in scoped_rows),
                    "mean_max_abs_error": mean(float(row[f"{region}_max_abs_error"]) for row in scoped_rows),
                    "mean_cells": mean(float(row[f"{region}_cells"]) for row in scoped_rows),
                }
            )
    return summary_rows


def select_representatives(case_rows):
    representatives = []
    seen = set()
    for category in ["unet_better_than_manhattan", "unet_worse_than_manhattan", "unet_non_optimal"]:
        candidates = [row for row in case_rows if row["case_category"] == category]
        if not candidates:
            continue
        if category == "unet_better_than_manhattan":
            chosen = min(candidates, key=lambda row: float(row["unet_minus_manhattan_expanded"]))
        else:
            chosen = max(candidates, key=lambda row: float(row["mae"]))
        representatives.append(chosen)
        seen.add(chosen["case_id"])

    for row in sorted(case_rows, key=lambda item: float(item["local_consistency_violation_rate"]), reverse=True):
        if row["case_id"] not in seen:
            representatives.append(row)
            break
    return representatives


def write_report(path, case_rows, taxonomy_rows, error_region_rows, mlp_rows, representative_rows):
    categories = {}
    for row in case_rows:
        categories.setdefault(row["case_category"], []).append(row)

    def category_line(category):
        rows = categories.get(category, [])
        if not rows:
            return f"- `{category}`: 0 cases"
        return (
            f"- `{category}`: {len(rows)} cases, "
            f"mean U-Net-Manhattan expanded {mean(float(row['unet_minus_manhattan_expanded']) for row in rows):.2f}, "
            f"mean MAE {mean(float(row['mae']) for row in rows):.2f}, "
            f"mean overestimate rate {mean(float(row['overestimate_rate']) for row in rows):.3f}, "
            f"mean consistency violation {mean(float(row['local_consistency_violation_rate']) for row in rows):.3f}"
        )

    with open(path, "w", encoding="utf-8") as file:
        file.write("# Failure Pattern Findings\n\n")
        file.write("## Case Buckets\n\n")
        for category in ["unet_better_than_manhattan", "unet_worse_than_manhattan", "unet_non_optimal"]:
            file.write(category_line(category) + "\n")

        file.write("\n## Structural Patterns\n\n")
        for row in taxonomy_rows:
            file.write(
                f"- {row['case_category']} / {row['structure']}: "
                f"{row['cases']} cases, mean U-Net-Manhattan expanded "
                f"{float(row['mean_unet_minus_manhattan_expanded']):.2f}, "
                f"mean MAE {float(row['mean_mae']):.2f}, "
                f"mean overestimate {float(row['mean_overestimate_rate']):.3f}\n"
            )

        file.write("\n## Error Field Regions\n\n")
        for row in error_region_rows:
            file.write(
                f"- {row['case_category']} / {row['region']}: "
                f"mean abs error {float(row['mean_abs_error']):.2f}, "
                f"mean max abs error {float(row['mean_max_abs_error']):.2f}\n"
            )

        file.write("\n## MLP vs U-Net\n\n")
        for row in mlp_rows:
            file.write(
                f"- obstacle_rate={row['obstacle_rate']}: paired maps {row['paired_maps']}, "
                f"MLP table better {row['mlp_table_better_count']}, U-Net better {row['unet_better_count']}, "
                f"mean U-Net-MLP expanded {float(row['mean_unet_minus_mlp_expanded']):.2f}\n"
            )

        file.write("\n## Representative Cases\n\n")
        for row in representative_rows:
            file.write(
                f"- {row['case_id']}: {row['case_category']}, structures [{row['structure_labels']}], "
                f"MAE {float(row['mae']):.2f}, overestimate {float(row['overestimate_rate']):.3f}, "
                f"consistency violation {float(row['local_consistency_violation_rate']):.3f}\n"
            )

        file.write("\n## Interpretation\n\n")
        file.write(
            "U-Net's advantage over Manhattan is clearest on open or sparse maps, where a more aggressive learned "
            "field can reduce exploration without forcing many corridor decisions. Its worse and non-optimal cases "
            "are concentrated in denser maps with corridor, bottleneck, or route-choice structure, where local "
            "overestimation and inconsistency can redirect search or make it expand extra nodes. The largest errors "
            "in bad cases are not uniformly distributed: they are amplified near the goal, near obstacles, and along "
            "corridor or bottleneck cells. MLP table remains "
            "strong because the current benchmark has fixed start-goal geometry and many shortest paths are still "
            "well explained by radial distance-to-goal. U-Net receives obstacle information, but this benefit is "
            "offset by calibration/admissibility/consistency errors in structured obstacle regions.\n"
        )


def load_mlp_gap_rows(project_root):
    path = os.path.join(project_root, "outputs", "analysis", "results_best_100", "unet_vs_mlp_table.csv")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def analyze_failure_patterns(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = args.cases_dir
    if root_dir is None:
        root_dir = os.path.join(project_root, "outputs", "failure_cases", args.tag)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(project_root, "outputs", "failure_pattern_analysis", args.tag)
    os.makedirs(output_dir, exist_ok=True)
    overlay_dir = os.path.join(output_dir, "path_overlays")
    error_dir = os.path.join(output_dir, "error_fields")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    plt, np = setup_matplotlib(project_root)
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint))
    case_rows = []

    for category, case_dir in case_dirs(root_dir):
        metadata = load_metadata(case_dir)
        grid = build_grid(metadata)
        start = tuple(metadata["start"])
        goal = tuple(metadata["goal"])
        case_id = os.path.basename(case_dir)
        distance_grid = compute_distance_to_goal(grid, goal)
        valid_mask = [[value >= 0 for value in row] for row in distance_grid]

        unet_heuristic = make_unet_heuristic(unet_model, grid, goal)
        optimal_result = astar_search(grid, start, goal, dijkstra_heuristic)
        manhattan_result = astar_search(grid, start, goal, manhattan_heuristic)
        unet_result = astar_search(grid, start, goal, unet_heuristic)
        predicted_grid = prediction_grid_from_heuristic(grid, goal, unet_heuristic)
        metrics = error_metrics(distance_grid, predicted_grid)
        consistency = local_consistency_metrics(predicted_grid, valid_mask)

        cells = free_cells(grid)
        articulation = articulation_points(grid)
        degrees = {cell: sum(1 for _ in neighbors(grid, cell)) for cell in cells}
        corridor_cells = [cell for cell, degree in degrees.items() if degree <= 2]
        optimal_path = optimal_result["path"]
        manhattan_path = manhattan_result["path"]
        unet_path = unet_result["path"]
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
        structure_labels = classify_structure(
            float(metadata["obstacle_rate"]),
            obstacle_density,
            largest_block_rate,
            corridor_rate,
            path_corridor_rate,
            articulation_rate,
            path_articulation_rate,
            manhattan_overlap,
            unet_overlap,
        )
        region_metrics = error_region_metrics(grid, distance_grid, predicted_grid, optimal_path, articulation)
        mlp_data = metadata["heuristics"].get("mlp_table", {})

        row = {
            "case_id": case_id,
            "case_category": category,
            "seed": metadata["seed"],
            "obstacle_rate": metadata["obstacle_rate"],
            "path_length": unet_result["cost"],
            "optimal_cost": metadata["optimal_cost"],
            "manhattan_path_length": manhattan_result["cost"],
            "unet_expanded_nodes": unet_result["expanded"],
            "manhattan_expanded_nodes": manhattan_result["expanded"],
            "mlp_table_expanded_nodes": mlp_data.get("expanded_nodes", ""),
            "unet_minus_manhattan_expanded": unet_result["expanded"] - manhattan_result["expanded"],
            "unet_minus_mlp_expanded": unet_result["expanded"] - mlp_data.get("expanded_nodes", 0),
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "overestimate_rate": metrics["overestimate_rate"],
            "underestimate_rate": metrics["underestimate_rate"],
            "max_overestimate": metrics["max_overestimate"],
            "max_underestimate": metrics["max_underestimate"],
            "local_consistency_violation_rate": consistency["local_consistency_violation_rate"],
            "local_consistency_mean_violation": consistency["local_consistency_mean_violation"],
            "local_consistency_max_violation": consistency["local_consistency_max_violation"],
            "obstacle_density": obstacle_density,
            "largest_obstacle_block_rate": largest_block_rate,
            "corridor_rate": corridor_rate,
            "path_corridor_rate": path_corridor_rate,
            "articulation_rate": articulation_rate,
            "path_articulation_rate": path_articulation_rate,
            "manhattan_optimal_path_overlap": manhattan_overlap,
            "unet_optimal_path_overlap": unet_overlap,
            "structure_labels": structure_labels,
            **region_metrics,
        }
        case_rows.append(row)

        save_path_overlay(
            os.path.join(overlay_dir, f"{case_id}.png"),
            grid,
            start,
            goal,
            optimal_path,
            manhattan_path,
            unet_path,
            case_id,
            plt,
            np,
        )

    fieldnames = list(case_rows[0].keys()) if case_rows else []
    case_table_path = os.path.join(output_dir, "case_taxonomy.csv")
    write_csv(case_table_path, case_rows, fieldnames)

    taxonomy_rows = summarize_taxonomy(case_rows)
    taxonomy_fieldnames = list(taxonomy_rows[0].keys()) if taxonomy_rows else []
    taxonomy_path = os.path.join(output_dir, "taxonomy_summary.csv")
    write_csv(taxonomy_path, taxonomy_rows, taxonomy_fieldnames)

    error_region_rows = summarize_error_regions(case_rows)
    error_region_fieldnames = list(error_region_rows[0].keys()) if error_region_rows else []
    error_region_path = os.path.join(output_dir, "error_region_summary.csv")
    write_csv(error_region_path, error_region_rows, error_region_fieldnames)

    representative_rows = select_representatives(case_rows)
    with open(os.path.join(output_dir, "representative_cases.json"), "w", encoding="utf-8") as file:
        json.dump(representative_rows, file, indent=2)

    for row in representative_rows:
        metadata = load_metadata(os.path.join(root_dir, row["case_category"], row["case_id"]))
        grid = build_grid(metadata)
        goal = tuple(metadata["goal"])
        distance_grid = compute_distance_to_goal(grid, goal)
        unet_heuristic = make_unet_heuristic(unet_model, grid, goal)
        predicted_grid = prediction_grid_from_heuristic(grid, goal, unet_heuristic)
        save_error_triptych(
            os.path.join(error_dir, f"{row['case_id']}.png"),
            grid,
            distance_grid,
            predicted_grid,
            row["case_id"],
            plt,
            np,
        )

    mlp_rows = load_mlp_gap_rows(project_root)
    report_path = os.path.join(output_dir, "findings_report.md")
    write_report(report_path, case_rows, taxonomy_rows, error_region_rows, mlp_rows, representative_rows)

    print(f"Saved case taxonomy to {case_table_path}")
    print(f"Saved taxonomy summary to {taxonomy_path}")
    print(f"Saved error-region summary to {error_region_path}")
    print(f"Saved path overlays to {overlay_dir}")
    print(f"Saved representative error fields to {error_dir}")
    print(f"Saved findings report to {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze saved U-Net failure/success case patterns.")
    parser.add_argument("--tag", default="best_100", help="Failure-case output tag.")
    parser.add_argument("--cases-dir", default=None, help="Path to outputs/failure_cases/<tag>.")
    parser.add_argument("--output-dir", default=None, help="Analysis output directory.")
    parser.add_argument("--checkpoint", default="best", help="compatible, best, latest, or a checkpoint path.")
    return parser.parse_args()


if __name__ == "__main__":
    analyze_failure_patterns(parse_args())
