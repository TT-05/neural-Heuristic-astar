import argparse
import csv
import math
import os

from analyze_maze_visual_mechanisms import distance_to_path_table
from analyze_route_bias_mechanisms import longest_on_path_streak, recovery_times
from analyze_tie_set_counterfactual_penalty import collect_tie_events, result_expanded
from analyze_tie_set_ordering import (
    STRUCTURED_TYPES,
    build_grid,
    checkpoint_path,
    group_maps,
    mean,
    pearson,
    prediction_table,
    read_csv,
    spearman,
)
from analyze_tie_set_weighted_ordering import reconstruct_optimal_path, route_critical_cells
from bfs_label import compute_distance_to_goal
from generate_maze_case_studies import simulate_secondary_expansion
from model import load_mlp_heuristic, load_unet_heuristic, make_mlp_table_heuristic, make_unet_heuristic


METRICS = [
    "off_path_count_gap",
    "off_path_fraction_gap",
    "cumulative_off_path_gap",
    "mean_recovery_time_gap",
    "frontier_compactness_gap",
    "first_disagreement_step",
    "disagreement_count",
]


def write_csv(path, rows):
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def case_id(sample):
    return (
        f"{sample.get('structured_type')}_rate{sample['obstacle_rate']}_seed{sample['seed']}"
        f"_s{sample['start_row']}-{sample['start_col']}_g{sample['goal_row']}-{sample['goal_col']}"
    )


def compactness(expanded):
    if not expanded:
        return 0.0
    rows = [cell[0] for cell in expanded]
    cols = [cell[1] for cell in expanded]
    area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    return len(expanded) / area if area else 0.0


def expansion_summary(expanded, distance_table):
    distances = [distance_table.get(cell, 0) for cell in expanded]
    off_flags = [1 if value > 1 else 0 for value in distances]
    recoveries = recovery_times(distances, radius=1)
    return {
        "off_path_count": sum(off_flags),
        "off_path_fraction": sum(off_flags) / len(off_flags) if off_flags else 0.0,
        "cumulative_off_path": sum(off_flags),
        "mean_recovery_time": mean(recoveries),
        "max_consecutive_on_path": longest_on_path_streak(distances, radius=1),
        "frontier_compactness": compactness(expanded),
    }


def first_disagreement(events):
    for event in events:
        if event["mlp_selected_node"] != event["unet_selected_node"]:
            return event["expanded_step"]
    return -1


def disagreement_count(events):
    return sum(1 for event in events if event["mlp_selected_node"] != event["unet_selected_node"])


def analyze_map(methods, mlp_model, unet_model):
    sample = methods["manhattan"]
    grid, start, goal = build_grid(sample)
    distance_grid = compute_distance_to_goal(grid, goal)
    true_table = {
        (r, c): float(value)
        for r, row in enumerate(distance_grid)
        for c, value in enumerate(row)
        if value >= 0
    }
    path = reconstruct_optimal_path(grid, distance_grid, start, goal)
    path_distance = distance_to_path_table(grid, path)
    critical = route_critical_cells(grid, path, 2)
    mlp_table = prediction_table(grid, goal, make_mlp_table_heuristic(mlp_model, grid, goal))
    unet_table = prediction_table(grid, goal, make_unet_heuristic(unet_model, grid, goal))
    mlp_expanded_order = simulate_secondary_expansion(grid, start, goal, mlp_table)[0]
    unet_expanded_order = simulate_secondary_expansion(grid, start, goal, unet_table)[0]
    events, _ = collect_tie_events(grid, start, goal, true_table, mlp_table, unet_table, critical)
    mlp_summary = expansion_summary(mlp_expanded_order, path_distance)
    unet_summary = expansion_summary(unet_expanded_order, path_distance)
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
    for key in mlp_summary:
        row[f"mlp_{key}"] = mlp_summary[key]
        row[f"unet_{key}"] = unet_summary[key]
        row[f"{key}_gap"] = unet_summary[key] - mlp_summary[key]
    row["off_path_count_gap"] = row["unet_off_path_count"] - row["mlp_off_path_count"]
    row["off_path_fraction_gap"] = row["unet_off_path_fraction"] - row["mlp_off_path_fraction"]
    row["cumulative_off_path_gap"] = row["unet_cumulative_off_path"] - row["mlp_cumulative_off_path"]
    row["mean_recovery_time_gap"] = row["unet_mean_recovery_time"] - row["mlp_mean_recovery_time"]
    row["frontier_compactness_gap"] = row["unet_frontier_compactness"] - row["mlp_frontier_compactness"]
    row["first_disagreement_step"] = first_disagreement(events)
    row["disagreement_count"] = disagreement_count(events)
    return row


def aggregate_rows(map_rows):
    rows = []
    for structured_type in STRUCTURED_TYPES:
        scoped = [row for row in map_rows if row["structured_type"] == structured_type]
        out = {"row_type": "aggregate", "structured_type": structured_type, "maps": len(scoped)}
        for key in ["expanded_gap", *METRICS]:
            out[key] = mean(row[key] for row in scoped)
        rows.append(out)
    return rows


def correlation_rows(map_rows):
    rows = []
    for structured_type in STRUCTURED_TYPES:
        scoped = [row for row in map_rows if row["structured_type"] == structured_type]
        ys = [row["expanded_gap"] for row in scoped]
        for metric in METRICS:
            xs = [row[metric] for row in scoped]
            rows.append(
                {
                    "row_type": "correlation",
                    "structured_type": structured_type,
                    "metric": metric,
                    "y": "expanded_gap",
                    "n": len(scoped),
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                }
            )
    return rows


def save_plots(path, aggregates, correlations):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(path), "matplotlib_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = STRUCTURED_TYPES
    axes[0].bar(labels, [next(row for row in aggregates if row["structured_type"] == label)["expanded_gap"] for label in labels])
    axes[0].set_title("Mean expanded gap")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, [next(row for row in aggregates if row["structured_type"] == label)["off_path_count_gap"] for label in labels])
    axes[1].set_title("Mean off-path count gap")
    axes[1].tick_params(axis="x", rotation=20)

    metric = "off_path_count_gap"
    axes[2].bar(
        labels,
        [next(row for row in correlations if row["structured_type"] == label and row["metric"] == metric)["spearman"] for label in labels],
    )
    axes[2].set_title("Off-path gap Spearman")
    axes[2].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def corr(correlations, structured_type, metric):
    return next(row for row in correlations if row["structured_type"] == structured_type and row["metric"] == metric)


def write_summary(path, aggregates, correlations):
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Cross-Structure Route-Bias Analysis\n\n")
        file.write(
            "This analysis compares MLP and U-Net tie-break trajectory behavior across structured map types. "
            "It is analysis-only and does not modify production A*, models, checkpoints, training, or benchmark outputs.\n\n"
        )
        file.write("## Aggregate Metrics\n\n")
        file.write("| Structure | maps | expanded gap | off-path count gap | off-path fraction gap | recovery-time gap | compactness gap |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in aggregates:
            file.write(
                f"| {row['structured_type']} | {row['maps']} | {row['expanded_gap']:.3f} | "
                f"{row['off_path_count_gap']:.3f} | {row['off_path_fraction_gap']:.3f} | "
                f"{row['mean_recovery_time_gap']:.3f} | {row['frontier_compactness_gap']:.3f} |\n"
            )
        file.write("\n## Correlations With Expanded Gap\n\n")
        file.write("| Structure | off-path count | off-path fraction | recovery time | compactness | disagreement count |\n")
        file.write("|---|---:|---:|---:|---:|---:|\n")
        for structured_type in STRUCTURED_TYPES:
            file.write(
                f"| {structured_type} | "
                f"{corr(correlations, structured_type, 'off_path_count_gap')['spearman']:.3f} | "
                f"{corr(correlations, structured_type, 'off_path_fraction_gap')['spearman']:.3f} | "
                f"{corr(correlations, structured_type, 'mean_recovery_time_gap')['spearman']:.3f} | "
                f"{corr(correlations, structured_type, 'frontier_compactness_gap')['spearman']:.3f} | "
                f"{corr(correlations, structured_type, 'disagreement_count')['spearman']:.3f} |\n"
            )
        file.write("\n## Answers\n\n")
        maze_off = corr(correlations, "maze_like", "off_path_count_gap")["spearman"]
        file.write(
            f"1. Maze-like route bias generalizes only if off-path gaps align with expanded gaps in other structures. "
            f"Maze-like off-path Spearman is {maze_off:.3f}; compare the table above for other structures.\n"
        )
        file.write(
            "2. U-Net advantage is most directly explained by reduced off-path exploration where both mean off-path gap and "
            "correlation are positive with expanded gap. Structures with weak or inconsistent values need separate mechanisms.\n"
        )
        file.write(
            "3. Large-block and narrow-corridor MLP wins are supported by off-path harm if U-Net has positive mean off-path gaps "
            "and those gaps correlate with positive expanded gaps.\n"
        )
        file.write(
            "4. Recovery-time explains failures better than simple off-path count only where recovery-time Spearman exceeds off-path-count Spearman.\n"
        )
        file.write(
            "5. Structure-specific mechanisms dominate when the strongest explanatory metric changes across map types; route bias is not a universal explanation.\n\n"
        )
        file.write("## Design Implication\n\n")
        file.write(
            "Future learned-heuristic algorithms should preserve U-Net-style route bias for structures where it reduces off-path exploration, "
            "but should include structure-aware diagnostics because the same learned bias can become harmful in corridor/block settings.\n"
        )


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "cross_structure_route_bias")
    os.makedirs(output_dir, exist_ok=True)
    groups = group_maps(read_csv(args.structured_results), "structured")
    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint, "unet"))
    map_rows = []
    for methods in groups:
        sample = methods["manhattan"]
        if sample.get("structured_type") in STRUCTURED_TYPES:
            map_rows.append(analyze_map(methods, mlp_model, unet_model))
    aggregates = aggregate_rows(map_rows)
    correlations = correlation_rows(map_rows)
    write_csv(os.path.join(output_dir, "cross_structure_route_bias_statistics.csv"), map_rows + aggregates + correlations)
    save_plots(os.path.join(output_dir, "cross_structure_route_bias_plots.png"), aggregates, correlations)
    write_summary(os.path.join(output_dir, "cross_structure_route_bias_summary.md"), aggregates, correlations)
    print(f"Saved cross-structure route-bias outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze route-bias mechanisms across structured map types.")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--checkpoint", default="best")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
