import argparse
import csv
import os
from collections import deque

from analyze_cross_structure_route_bias import case_id, compactness, expansion_summary
from analyze_maze_visual_mechanisms import distance_to_path_table
from analyze_route_bias_mechanisms import recovery_times
from analyze_tie_set_counterfactual_penalty import result_expanded
from analyze_tie_set_ordering import build_grid, checkpoint_path, group_maps, mean, pearson, prediction_table, read_csv, spearman
from analyze_tie_set_weighted_ordering import reconstruct_optimal_path
from bfs_label import compute_distance_to_goal
from generate_maze_case_studies import simulate_secondary_expansion
from model import load_mlp_heuristic, load_unet_heuristic, make_mlp_table_heuristic, make_unet_heuristic


METRICS = [
    "off_path_count_gap",
    "off_path_fraction_gap",
    "cumulative_off_path_gap",
    "frontier_compactness_gap",
    "mean_true_distance_gap",
    "early_true_distance_gap",
    "true_distance_progress_slope_gap",
    "mean_obstacle_boundary_distance_gap",
    "near_boundary_count_gap",
    "near_boundary_fraction_gap",
    "boundary_following_run_gap",
    "mean_recovery_time_gap",
]


def write_csv(path, rows):
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def obstacle_boundary_cells(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    boundary = set()
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value != 0:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    boundary.add((r, c))
                    break
    return boundary


def distance_from_sources(grid, sources):
    distances = {cell: 0 for cell in sources}
    queue = deque(sources)
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = r + dr
            nc = c + dc
            cell = (nr, nc)
            if nr < 0 or nr >= len(grid) or nc < 0 or nc >= len(grid[0]):
                continue
            if grid[nr][nc] != 0 or cell in distances:
                continue
            distances[cell] = distances[(r, c)] + 1
            queue.append(cell)
    return distances


def slope(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    xs = list(range(len(values)))
    mx = mean(xs)
    my = mean(values)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / denom


def longest_run(flags):
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def method_metrics(expanded, path_distance, true_distance, boundary_distance):
    path_summary = expansion_summary(expanded, path_distance)
    true_values = [true_distance.get(cell, 0.0) for cell in expanded]
    boundary_values = [boundary_distance.get(cell, 999.0) for cell in expanded]
    near_boundary = [1 if value <= 1 else 0 for value in boundary_values]
    early_count = max(1, len(expanded) // 4) if expanded else 0
    output = dict(path_summary)
    output.update(
        {
            "mean_true_distance": mean(true_values),
            "early_true_distance": mean(true_values[:early_count]),
            "true_distance_progress_slope": slope(true_values),
            "mean_obstacle_boundary_distance": mean(boundary_values),
            "near_boundary_count": sum(near_boundary),
            "near_boundary_fraction": sum(near_boundary) / len(near_boundary) if near_boundary else 0.0,
            "boundary_following_run": longest_run(near_boundary),
        }
    )
    return output


def analyze_map(methods, mlp_model, unet_model):
    sample = methods["manhattan"]
    grid, start, goal = build_grid(sample)
    distance_grid = compute_distance_to_goal(grid, goal)
    true_distance = {
        (r, c): float(value)
        for r, row in enumerate(distance_grid)
        for c, value in enumerate(row)
        if value >= 0
    }
    path = reconstruct_optimal_path(grid, distance_grid, start, goal)
    path_distance = distance_to_path_table(grid, path)
    boundary = obstacle_boundary_cells(grid)
    boundary_distance = distance_from_sources(grid, boundary)
    mlp_table = prediction_table(grid, goal, make_mlp_table_heuristic(mlp_model, grid, goal))
    unet_table = prediction_table(grid, goal, make_unet_heuristic(unet_model, grid, goal))
    mlp_expanded = simulate_secondary_expansion(grid, start, goal, mlp_table)[0]
    unet_expanded = simulate_secondary_expansion(grid, start, goal, unet_table)[0]
    mlp = method_metrics(mlp_expanded, path_distance, true_distance, boundary_distance)
    unet = method_metrics(unet_expanded, path_distance, true_distance, boundary_distance)

    row = {
        "row_type": "map_summary",
        "map_id": case_id(sample),
        "structured_type": sample.get("structured_type"),
        "seed": sample["seed"],
        "obstacle_rate": sample["obstacle_rate"],
        "start": f"{sample['start_row']},{sample['start_col']}",
        "goal": f"{sample['goal_row']},{sample['goal_col']}",
        "path_length": len(path) - 1 if path else -1,
        "mlp_expanded": result_expanded(methods, "manhattan_mlp_tiebreak"),
        "unet_expanded": result_expanded(methods, "manhattan_unet_tiebreak"),
    }
    row["expanded_gap"] = row["unet_expanded"] - row["mlp_expanded"]
    for key in mlp:
        row[f"mlp_{key}"] = mlp[key]
        row[f"unet_{key}"] = unet[key]
        row[f"{key}_gap"] = unet[key] - mlp[key]
    row["off_path_count_gap"] = row["unet_off_path_count"] - row["mlp_off_path_count"]
    row["off_path_fraction_gap"] = row["unet_off_path_fraction"] - row["mlp_off_path_fraction"]
    row["cumulative_off_path_gap"] = row["unet_cumulative_off_path"] - row["mlp_cumulative_off_path"]
    row["frontier_compactness_gap"] = row["unet_frontier_compactness"] - row["mlp_frontier_compactness"]
    row["mean_recovery_time_gap"] = row["unet_mean_recovery_time"] - row["mlp_mean_recovery_time"]
    return row


def aggregate_row(map_rows):
    row = {"row_type": "aggregate", "structured_type": "large_block", "maps": len(map_rows)}
    for key in ["expanded_gap", *METRICS]:
        row[key] = mean(item[key] for item in map_rows)
    return row


def correlation_rows(map_rows):
    rows = []
    ys = [row["expanded_gap"] for row in map_rows]
    for metric in METRICS:
        xs = [row[metric] for row in map_rows]
        rows.append(
            {
                "row_type": "correlation",
                "structured_type": "large_block",
                "metric": metric,
                "y": "expanded_gap",
                "n": len(map_rows),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
        )
    return rows


def save_plots(path, map_rows, correlations):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(path), "matplotlib_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter([row["near_boundary_fraction_gap"] for row in map_rows], [row["expanded_gap"] for row in map_rows], s=15)
    axes[0].set_title("Boundary expansion gap")
    axes[0].set_xlabel("U-Net - MLP near-boundary fraction")
    axes[0].set_ylabel("expanded gap")

    axes[1].scatter([row["true_distance_progress_slope_gap"] for row in map_rows], [row["expanded_gap"] for row in map_rows], s=15)
    axes[1].set_title("Goal-distance progress gap")
    axes[1].set_xlabel("U-Net - MLP slope")
    axes[1].set_ylabel("expanded gap")

    metrics = ["near_boundary_fraction_gap", "mean_obstacle_boundary_distance_gap", "true_distance_progress_slope_gap", "off_path_count_gap"]
    selected = [next(row for row in correlations if row["metric"] == metric) for metric in metrics]
    axes[2].bar([row["metric"].replace("_gap", "") for row in selected], [row["spearman"] for row in selected])
    axes[2].set_title("Spearman with expanded gap")
    axes[2].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def find_corr(correlations, metric):
    return next(row for row in correlations if row["metric"] == metric)


def write_summary(path, aggregate, correlations):
    boundary_corr = find_corr(correlations, "near_boundary_fraction_gap")
    boundary_distance_corr = find_corr(correlations, "mean_obstacle_boundary_distance_gap")
    progress_corr = find_corr(correlations, "true_distance_progress_slope_gap")
    off_path_corr = find_corr(correlations, "off_path_count_gap")
    compact_corr = find_corr(correlations, "frontier_compactness_gap")
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Large-Block Failure Mechanism Analysis\n\n")
        file.write(
            "This analysis focuses on why U-Net loses to MLP on large_block maps when off-path exploration alone is a weak explanation. "
            "It is analysis-only and does not modify production A*, models, checkpoints, training, or benchmark outputs.\n\n"
        )
        file.write("## Aggregate Metrics\n\n")
        file.write(
            f"- maps: {aggregate['maps']}\n"
            f"- expanded_gap: {aggregate['expanded_gap']:.3f}\n"
            f"- off_path_count_gap: {aggregate['off_path_count_gap']:.3f}\n"
            f"- off_path_fraction_gap: {aggregate['off_path_fraction_gap']:.3f}\n"
            f"- near_boundary_fraction_gap: {aggregate['near_boundary_fraction_gap']:.3f}\n"
            f"- mean_obstacle_boundary_distance_gap: {aggregate['mean_obstacle_boundary_distance_gap']:.3f}\n"
            f"- true_distance_progress_slope_gap: {aggregate['true_distance_progress_slope_gap']:.3f}\n"
            f"- frontier_compactness_gap: {aggregate['frontier_compactness_gap']:.3f}\n\n"
        )
        file.write("## Correlations With Expanded Gap\n\n")
        file.write("| Metric | Pearson | Spearman |\n")
        file.write("|---|---:|---:|\n")
        for row in correlations:
            file.write(f"| {row['metric']} | {row['pearson']:.3f} | {row['spearman']:.3f} |\n")
        file.write("\n## Answers\n\n")
        file.write(
            f"1. Off-path count only partially explains large_block failure: Spearman={off_path_corr['spearman']:.3f}. "
            "The mean off-path gap is small, so U-Net's loss is not mainly broad wrong-branch exploration.\n"
        )
        file.write(
            f"2. Boundary attraction evidence: near-boundary fraction gap has Spearman={boundary_corr['spearman']:.3f}; "
            f"mean boundary-distance gap has Spearman={boundary_distance_corr['spearman']:.3f}. "
            "Positive near-boundary gaps indicate U-Net spends relatively more expansion mass near obstacle edges.\n"
        )
        file.write(
            f"3. Geometric progress evidence: true-distance progress slope gap has Spearman={progress_corr['spearman']:.3f}. "
            "A positive harmful slope gap suggests U-Net makes slower progress around the block than MLP.\n"
        )
        file.write(
            f"4. Compactness evidence: frontier compactness gap has Spearman={compact_corr['spearman']:.3f}. "
            "This tests whether U-Net's expansions are less cleanly organized around the obstacle.\n\n"
        )
        file.write("## Interpretation\n\n")
        file.write(
            "Large-block failure appears more structure-specific than maze-like off-path failure. The key candidate mechanism is poor obstacle-edge ordering: "
            "U-Net can spend extra search effort around large obstacle boundaries without proportionally improving goal-directed progress. "
            "MLP's simpler geometry-biased ordering may be cleaner when the dominant task is skirting a large block rather than choosing among maze branches.\n"
        )


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "large_block_failure_mechanisms")
    os.makedirs(output_dir, exist_ok=True)
    groups = group_maps(read_csv(args.structured_results), "structured")
    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint, "unet"))
    map_rows = []
    for methods in groups:
        if methods["manhattan"].get("structured_type") == "large_block":
            map_rows.append(analyze_map(methods, mlp_model, unet_model))
    aggregate = aggregate_row(map_rows)
    correlations = correlation_rows(map_rows)
    write_csv(os.path.join(output_dir, "large_block_failure_statistics.csv"), map_rows + [aggregate] + correlations)
    save_plots(os.path.join(output_dir, "large_block_failure_plots.png"), map_rows, correlations)
    write_summary(os.path.join(output_dir, "large_block_failure_summary.md"), aggregate, correlations)
    print(f"Saved large-block failure mechanism outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze large-block U-Net failure mechanisms.")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--checkpoint", default="best")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
