import argparse
import csv
import heapq
import math
import os

from analyze_tie_set_ordering import (
    STRUCTURED_TYPES,
    build_grid,
    checkpoint_path,
    group_maps,
    manhattan,
    mean,
    neighbors,
    pairwise_ordering_accuracy,
    pearson,
    prediction_table,
    read_csv,
    spearman,
    tie_set_metrics,
    to_float,
)
from analyze_tie_set_weighted_ordering import reconstruct_optimal_path, route_critical_cells
from bfs_label import compute_distance_to_goal
from model import load_mlp_heuristic, load_unet_heuristic, make_mlp_table_heuristic, make_unet_heuristic


EPS = 1e-9
METHODS = ["mlp", "unet"]
CORRELATION_METRICS = [
    "mean_regret_gap",
    "mean_normalized_regret_gap",
    "high_10_a_regret_gap",
    "high_5_a_regret_gap",
    "early_high_impact_10_a_regret_gap",
    "high_10_b_regret_gap",
    "pairwise_gap",
    "top1_gap",
    "route_critical_top1_gap",
]


def write_csv(path, rows):
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def selected_node(nodes, secondary_table):
    return min(nodes, key=lambda node: (secondary_table[node], node))


def comparable_pairs(true_values):
    count = 0
    for i in range(len(true_values)):
        for j in range(i + 1, len(true_values)):
            if true_values[i] != true_values[j]:
                count += 1
    return count


def map_identity(sample):
    return {
        "source": sample["source"],
        "seed": sample["seed"],
        "map_size": sample["map_size"],
        "obstacle_rate": sample["obstacle_rate"],
        "map_mode": sample.get("map_mode", "random"),
        "structured_type": sample.get("structured_type", "random"),
        "start": f"{sample['start_row']},{sample['start_col']}",
        "goal": f"{sample['goal_row']},{sample['goal_col']}",
        "case_id": (
            f"{sample.get('structured_type', 'random')}_rate{sample['obstacle_rate']}_seed{sample['seed']}"
            f"_s{sample['start_row']}-{sample['start_col']}_g{sample['goal_row']}-{sample['goal_col']}"
        ),
    }


def result_expanded(methods, method):
    return to_float(methods[method], "expanded_nodes")


def percentile_subset(events, key, fraction):
    if not events:
        return []
    ordered = sorted(events, key=lambda event: event[key], reverse=True)
    size = max(1, math.ceil(len(ordered) * fraction))
    return ordered[:size]


def average(events, key):
    return mean(event[key] for event in events)


def replay_tie_decisions(grid, start, goal, true_table, mlp_table, unet_table, critical):
    open_set = []
    heapq.heappush(open_set, (manhattan(start, goal), 0, start))
    g_score = {start: 0}
    events = []
    expanded = 0

    while open_set:
        f_primary, current_g, current = heapq.heappop(open_set)
        if current_g > g_score.get(current, float("inf")):
            continue

        nodes = [current]
        for queued_f, queued_g, queued_node in open_set:
            if queued_f == f_primary and queued_g == g_score.get(queued_node, float("inf")):
                nodes.append(queued_node)

        if len(nodes) >= 2:
            true_values = [true_table[node] for node in nodes]
            min_true = min(true_values)
            max_true = max(true_values)
            true_range = max_true - min_true
            pair_count = comparable_pairs(true_values)
            mlp_selected = selected_node(nodes, mlp_table)
            unet_selected = selected_node(nodes, unet_table)
            mlp_regret = true_table[mlp_selected] - min_true
            unet_regret = true_table[unet_selected] - min_true
            mlp_metrics = tie_set_metrics(nodes, true_table, mlp_table)
            unet_metrics = tie_set_metrics(nodes, true_table, unet_table)
            route_overlap = sum(1 for node in nodes if node in critical)
            event = {
                "expanded_step": expanded,
                "f_primary": f_primary,
                "tie_set_size": len(nodes),
                "true_distance_min": min_true,
                "true_distance_max": max_true,
                "true_distance_range": true_range,
                "comparable_pair_count": pair_count,
                "criticality_a": len(nodes) * true_range,
                "criticality_b": pair_count * true_range,
                "route_critical_overlap": route_overlap,
                "route_critical_overlap_fraction": route_overlap / len(nodes),
                "mlp_regret": mlp_regret,
                "unet_regret": unet_regret,
                "regret_gap": unet_regret - mlp_regret,
                "mlp_normalized_regret": mlp_regret / (true_range + EPS),
                "unet_normalized_regret": unet_regret / (true_range + EPS),
                "normalized_regret_gap": (unet_regret - mlp_regret) / (true_range + EPS),
                "mlp_pairwise": mlp_metrics["pairwise_ordering_accuracy"],
                "unet_pairwise": unet_metrics["pairwise_ordering_accuracy"],
                "pairwise_gap": unet_metrics["pairwise_ordering_accuracy"] - mlp_metrics["pairwise_ordering_accuracy"],
                "mlp_top1": mlp_metrics["top1_accuracy"],
                "unet_top1": unet_metrics["top1_accuracy"],
                "top1_gap": unet_metrics["top1_accuracy"] - mlp_metrics["top1_accuracy"],
            }
            events.append(event)

        expanded += 1
        if current == goal:
            break

        for neighbor in neighbors(grid, current):
            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g + manhattan(neighbor, goal), tentative_g, neighbor))

    for event in events:
        event["expansion_fraction"] = event["expanded_step"] / expanded if expanded else 0.0
    return events, expanded


def subset_mean(events, key, selector):
    subset = selector(events)
    return average(subset, key)


def analyze_map(methods, mlp_model, unet_model, route_radius):
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
    critical = route_critical_cells(grid, path, route_radius)

    mlp_table = prediction_table(grid, goal, make_mlp_table_heuristic(mlp_model, grid, goal))
    unet_table = prediction_table(grid, goal, make_unet_heuristic(unet_model, grid, goal))
    events, expanded_steps = replay_tie_decisions(grid, start, goal, true_table, mlp_table, unet_table, critical)

    route_events = [event for event in events if event["route_critical_overlap"] > 0]
    early_events = [event for event in events if event["expansion_fraction"] < 0.25]
    high_10_a = percentile_subset(events, "criticality_a", 0.10)
    high_5_a = percentile_subset(events, "criticality_a", 0.05)
    high_10_b = percentile_subset(events, "criticality_b", 0.10)
    high_5_b = percentile_subset(events, "criticality_b", 0.05)
    early_high_10_a = percentile_subset(early_events, "criticality_a", 0.10)
    early_high_10_b = percentile_subset(early_events, "criticality_b", 0.10)

    row = {
        "row_type": "map_summary",
        **map_identity(sample),
        "expanded_steps": expanded_steps,
        "tie_set_count": len(events),
        "tie_step_fraction": len(events) / expanded_steps if expanded_steps else 0.0,
        "mean_tie_set_size": average(events, "tie_set_size"),
        "mean_true_distance_range": average(events, "true_distance_range"),
        "mean_criticality_a": average(events, "criticality_a"),
        "mean_criticality_b": average(events, "criticality_b"),
        "mean_mlp_regret": average(events, "mlp_regret"),
        "mean_unet_regret": average(events, "unet_regret"),
        "mean_regret_gap": average(events, "regret_gap"),
        "mean_normalized_regret_gap": average(events, "normalized_regret_gap"),
        "high_10_a_regret_gap": average(high_10_a, "regret_gap"),
        "high_5_a_regret_gap": average(high_5_a, "regret_gap"),
        "high_10_b_regret_gap": average(high_10_b, "regret_gap"),
        "high_5_b_regret_gap": average(high_5_b, "regret_gap"),
        "early_regret_gap": average(early_events, "regret_gap"),
        "early_high_impact_10_a_regret_gap": average(early_high_10_a, "regret_gap"),
        "early_high_impact_10_b_regret_gap": average(early_high_10_b, "regret_gap"),
        "pairwise_gap": average(events, "pairwise_gap"),
        "top1_gap": average(events, "top1_gap"),
        "route_critical_pairwise_gap": average(route_events, "pairwise_gap"),
        "route_critical_top1_gap": average(route_events, "top1_gap"),
        "route_critical_regret_gap": average(route_events, "regret_gap"),
        "mlp_expanded": result_expanded(methods, "manhattan_mlp_tiebreak"),
        "unet_expanded": result_expanded(methods, "manhattan_unet_tiebreak"),
        "true_distance_expanded": result_expanded(methods, "manhattan_true_distance_tiebreak"),
    }
    row["expanded_gap"] = row["unet_expanded"] - row["mlp_expanded"]
    return row


def aggregate_rows(map_rows):
    rows = []
    scopes = [("random", "all"), ("structured", "all")] + [("structured", item) for item in STRUCTURED_TYPES]
    numeric_keys = [
        "tie_set_count",
        "tie_step_fraction",
        "mean_tie_set_size",
        "mean_true_distance_range",
        "mean_criticality_a",
        "mean_criticality_b",
        "mean_mlp_regret",
        "mean_unet_regret",
        "mean_regret_gap",
        "mean_normalized_regret_gap",
        "high_10_a_regret_gap",
        "high_5_a_regret_gap",
        "high_10_b_regret_gap",
        "high_5_b_regret_gap",
        "early_regret_gap",
        "early_high_impact_10_a_regret_gap",
        "early_high_impact_10_b_regret_gap",
        "pairwise_gap",
        "top1_gap",
        "route_critical_pairwise_gap",
        "route_critical_top1_gap",
        "route_critical_regret_gap",
        "expanded_gap",
    ]
    for source, structured_type in scopes:
        scoped = [row for row in map_rows if row["source"] == source]
        if structured_type != "all":
            scoped = [row for row in scoped if row["structured_type"] == structured_type]
        aggregate = {"row_type": "aggregate", "source": source, "structured_type": structured_type, "maps": len(scoped)}
        for key in numeric_keys:
            aggregate[key] = average(scoped, key)
        rows.append(aggregate)
    return rows


def correlation_rows(map_rows):
    rows = []
    scopes = [("structured", "all"), ("structured", "bottleneck")] + [("structured", item) for item in STRUCTURED_TYPES if item != "bottleneck"]
    for source, structured_type in scopes:
        scoped = [row for row in map_rows if row["source"] == source]
        if structured_type != "all":
            scoped = [row for row in scoped if row["structured_type"] == structured_type]
        ys = [row["expanded_gap"] for row in scoped]
        for metric in CORRELATION_METRICS:
            xs = [row[metric] for row in scoped]
            rows.append(
                {
                    "source": source,
                    "structured_type": structured_type,
                    "metric": metric,
                    "y": "expanded_gap",
                    "n": len(scoped),
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                }
            )
    return rows


def bottleneck_cases(map_rows):
    rows = [row for row in map_rows if row["source"] == "structured" and row["structured_type"] == "bottleneck"]
    keys = [
        "case_id",
        "start",
        "goal",
        "tie_set_count",
        "mean_tie_set_size",
        "mean_true_distance_range",
        "mean_regret_gap",
        "mean_normalized_regret_gap",
        "high_10_a_regret_gap",
        "high_5_a_regret_gap",
        "early_high_impact_10_a_regret_gap",
        "pairwise_gap",
        "top1_gap",
        "route_critical_top1_gap",
        "mlp_expanded",
        "unet_expanded",
        "true_distance_expanded",
        "expanded_gap",
    ]
    output = [{key: row.get(key, "") for key in keys} for row in sorted(rows, key=lambda item: item["expanded_gap"])]
    return output


def save_plots(path, correlations, bottleneck_rows):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(path), "matplotlib_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    corr_metrics = [
        "mean_regret_gap",
        "high_10_a_regret_gap",
        "early_high_impact_10_a_regret_gap",
        "pairwise_gap",
        "top1_gap",
        "route_critical_top1_gap",
    ]
    bottleneck_corr = [
        next(row for row in correlations if row["structured_type"] == "bottleneck" and row["metric"] == metric)
        for metric in corr_metrics
    ]
    axes[0].bar([row["metric"].replace("_gap", "") for row in bottleneck_corr], [float(row["spearman"]) for row in bottleneck_corr])
    axes[0].set_title("Bottleneck Spearman vs expanded gap")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].scatter(
        [row["high_10_a_regret_gap"] for row in bottleneck_rows],
        [row["expanded_gap"] for row in bottleneck_rows],
        s=12,
    )
    axes[1].set_title("High-impact regret gap")
    axes[1].set_xlabel("U-Net - MLP regret")
    axes[1].set_ylabel("U-Net - MLP expanded")

    axes[2].scatter(
        [row["pairwise_gap"] for row in bottleneck_rows],
        [row["expanded_gap"] for row in bottleneck_rows],
        s=12,
    )
    axes[2].set_title("Pairwise gap")
    axes[2].set_xlabel("U-Net - MLP pairwise")
    axes[2].set_ylabel("U-Net - MLP expanded")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def lookup(rows, source, structured_type):
    return next(row for row in rows if row["source"] == source and row["structured_type"] == structured_type)


def corr_lookup(correlations, structured_type, metric):
    return next(row for row in correlations if row["structured_type"] == structured_type and row["metric"] == metric)


def write_summary(path, aggregates, correlations):
    bottleneck = lookup(aggregates, "structured", "bottleneck")
    structured = lookup(aggregates, "structured", "all")
    metrics = [
        "mean_regret_gap",
        "mean_normalized_regret_gap",
        "high_10_a_regret_gap",
        "high_5_a_regret_gap",
        "early_high_impact_10_a_regret_gap",
        "pairwise_gap",
        "top1_gap",
        "route_critical_top1_gap",
    ]
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Tie-Set Importance Analysis\n\n")
        file.write(
            "This analysis tests whether a small number of high-impact Manhattan-primary tie decisions explain "
            "expanded-node differences better than average tie-set ordering metrics. It replays search decisions "
            "offline and does not modify production A* or benchmark outputs.\n\n"
        )
        file.write("## Bottleneck Summary\n\n")
        file.write(
            f"- Mean expanded gap (U-Net - MLP): {bottleneck['expanded_gap']:.3f}.\n"
            f"- Mean regret gap: {bottleneck['mean_regret_gap']:.3f}.\n"
            f"- Top 10% criticality-A regret gap: {bottleneck['high_10_a_regret_gap']:.3f}.\n"
            f"- Top 5% criticality-A regret gap: {bottleneck['high_5_a_regret_gap']:.3f}.\n"
            f"- Early top 10% criticality-A regret gap: {bottleneck['early_high_impact_10_a_regret_gap']:.3f}.\n"
            f"- Pairwise gap: {bottleneck['pairwise_gap']:.3f}.\n"
            f"- Top1 gap: {bottleneck['top1_gap']:.3f}.\n"
            f"- Route-critical top1 gap: {bottleneck['route_critical_top1_gap']:.3f}.\n\n"
        )
        file.write("## Correlation With Expanded Gap\n\n")
        file.write("| Scope | Metric | Pearson | Spearman | n |\n")
        file.write("|---|---|---:|---:|---:|\n")
        for structured_type in ["all", "bottleneck"]:
            for metric in metrics:
                row = corr_lookup(correlations, structured_type, metric)
                file.write(
                    f"| {structured_type} | {metric} | {row['pearson']:.3f} | {row['spearman']:.3f} | {row['n']} |\n"
                )
        file.write("\n")
        bottleneck_corr = {metric: corr_lookup(correlations, "bottleneck", metric) for metric in metrics}
        best_metric = max(metrics, key=lambda metric: abs(float(bottleneck_corr[metric]["spearman"])))
        file.write("## Structure-Level Means\n\n")
        file.write("| Structure | Expanded gap | Mean regret gap | High10 regret gap | Early high10 regret gap | Pairwise gap | Top1 gap |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for structured_type in STRUCTURED_TYPES:
            row = lookup(aggregates, "structured", structured_type)
            file.write(
                f"| {structured_type} | {row['expanded_gap']:.3f} | {row['mean_regret_gap']:.3f} | "
                f"{row['high_10_a_regret_gap']:.3f} | {row['early_high_impact_10_a_regret_gap']:.3f} | "
                f"{row['pairwise_gap']:.3f} | {row['top1_gap']:.3f} |\n"
            )
        file.write("\n")
        file.write("## Key Answers\n\n")
        file.write(
            "- Are all tie sets equally important? No. Criticality varies by tie-set size, true-distance range, "
            "and comparable-pair count, so aggregating all decisions equally loses search-relevant detail.\n"
        )
        file.write(
            "- Do high-impact tie sets explain expanded nodes better? Not consistently. In bottleneck, "
            f"top-10% criticality-A regret Spearman is {bottleneck_corr['high_10_a_regret_gap']['spearman']:.3f}, "
            f"while mean regret is {bottleneck_corr['mean_regret_gap']['spearman']:.3f} and route-critical top1 has "
            f"absolute Spearman {abs(float(bottleneck_corr['route_critical_top1_gap']['spearman'])):.3f}.\n"
        )
        file.write(
            "- Does regret explain bottleneck better than pairwise ordering? Slightly for mean/normalized regret, "
            f"but not decisively: bottleneck mean regret Spearman is {bottleneck_corr['mean_regret_gap']['spearman']:.3f}, "
            f"normalized regret is {bottleneck_corr['mean_normalized_regret_gap']['spearman']:.3f}, and pairwise has "
            f"absolute Spearman {abs(float(bottleneck_corr['pairwise_gap']['spearman'])):.3f}.\n"
        )
        file.write(
            "- Do early high-impact decisions dominate search efficiency? The current evidence says no: bottleneck "
            f"early high-impact Spearman is {bottleneck_corr['early_high_impact_10_a_regret_gap']['spearman']:.3f}, "
            "weaker than mean regret and route-critical top1.\n"
        )
        file.write(
            f"- Strongest bottleneck predictor among the tested metrics by absolute Spearman: {best_metric} "
            f"({bottleneck_corr[best_metric]['spearman']:.3f}).\n\n"
        )
        file.write("## Interpretation\n\n")
        file.write(
            "High-impact regret metrics are more decision-focused than average pairwise ordering because they measure "
            "how far the actually selected node is from the best true-distance node inside each active tie set. "
            "The results should be read observationally: stronger correlations indicate better explanatory alignment, "
            "not causal proof. Bottleneck remains the key diagnostic structure for deciding whether a few important "
            "tie decisions explain small aggregate expansion differences.\n\n"
        )
        file.write(
            f"Across structured maps, mean regret gap is {structured['mean_regret_gap']:.3f} and mean expanded gap is "
            f"{structured['expanded_gap']:.3f}. Compare the correlation table above to judge whether high-impact regret "
            "outperforms pairwise, top1, and route-critical top1 as a predictor of expanded-node reduction.\n"
        )


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "tie_set_importance")
    os.makedirs(output_dir, exist_ok=True)

    random_groups = group_maps(read_csv(args.random_results), "random")
    structured_groups = group_maps(read_csv(args.structured_results), "structured")
    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint, "unet"))

    map_rows = []
    for methods in random_groups + structured_groups:
        map_rows.append(analyze_map(methods, mlp_model, unet_model, args.route_radius))

    aggregates = aggregate_rows(map_rows)
    correlations = correlation_rows(map_rows)
    bottleneck = bottleneck_cases(map_rows)

    write_csv(os.path.join(output_dir, "tie_set_importance_statistics.csv"), map_rows + aggregates)
    write_csv(os.path.join(output_dir, "tie_set_importance_correlations.csv"), correlations)
    write_csv(os.path.join(output_dir, "bottleneck_decision_critical_cases.csv"), bottleneck)
    save_plots(os.path.join(output_dir, "tie_set_importance_plots.png"), correlations, [row for row in map_rows if row["structured_type"] == "bottleneck"])
    write_summary(os.path.join(output_dir, "tie_set_importance_summary.md"), aggregates, correlations)
    print(f"Saved tie-set importance outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze high-impact tie-set decision regret.")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_tiebreak_controls_100.csv")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--route-radius", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
