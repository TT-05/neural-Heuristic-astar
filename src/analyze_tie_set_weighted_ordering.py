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
from bfs_label import compute_distance_to_goal
from model import load_mlp_heuristic, load_unet_heuristic, make_mlp_table_heuristic, make_unet_heuristic


SECONDARY_METHODS = ["mlp", "unet"]
ORDERING_METHODS = ["mlp", "unet", "oracle"]
FILTERS = ["all", "route_critical", "non_route_critical", "route_critical_fraction_ge_025"]
WEIGHT_MODES = ["unweighted", "tie_set_size", "comparable_pairs", "early_weight"]


def write_csv(path, rows):
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_optimal_path(grid, distance_grid, start, goal):
    if distance_grid[start[0]][start[1]] < 0:
        return []
    path = [start]
    current = start
    while current != goal:
        current_distance = distance_grid[current[0]][current[1]]
        candidates = [cell for cell in neighbors(grid, current) if distance_grid[cell[0]][cell[1]] == current_distance - 1]
        if not candidates:
            return []
        current = sorted(candidates)[0]
        path.append(current)
    return path


def route_critical_cells(grid, path, radius):
    critical = set(path)
    frontier = set(path)
    for _ in range(radius):
        next_frontier = set()
        for cell in frontier:
            for neighbor in neighbors(grid, cell):
                if neighbor not in critical:
                    next_frontier.add(neighbor)
        critical.update(next_frontier)
        frontier = next_frontier
    return critical


def comparable_pairs(true_values):
    total = 0
    for i in range(len(true_values)):
        for j in range(i + 1, len(true_values)):
            if true_values[i] != true_values[j]:
                total += 1
    return total


def selected_stats(nodes, true_table, secondary_table):
    selected = min(nodes, key=lambda node: (secondary_table[node], node))
    true_values = [true_table[node] for node in nodes]
    selected_true = true_table[selected]
    lower = sum(1 for value in true_values if value < selected_true)
    equal = sum(1 for value in true_values if value == selected_true)
    rank = lower + 1
    percentile = lower / max(1, len(nodes) - 1)
    return {
        "selected_true_distance": selected_true,
        "selected_true_rank": rank,
        "selected_true_percentile": percentile,
        "selected_true_tie_count": equal,
    }


def event_weight(event, mode):
    if mode == "unweighted":
        return 1.0
    if mode == "tie_set_size":
        return float(event["tie_set_size"])
    if mode == "comparable_pairs":
        return float(event["comparable_pairs"])
    if mode == "early_weight":
        return 1.0 / (1.0 + event["expanded_step"])
    raise ValueError(f"Unknown weight mode: {mode}")


def weighted_mean(events, key, mode):
    pairs = [(event[key], event_weight(event, mode)) for event in events if event_weight(event, mode) > 0]
    denom = sum(weight for _, weight in pairs)
    if denom == 0:
        return 0.0
    return sum(value * weight for value, weight in pairs) / denom


def timing_bucket(event, expanded_steps):
    if expanded_steps <= 0:
        return "all"
    fraction = event["expanded_step"] / expanded_steps
    if fraction < 0.25:
        return "early_25"
    if fraction < 0.75:
        return "middle_50"
    return "late_25"


def include_event(event, filter_name):
    if filter_name == "all":
        return True
    if filter_name == "route_critical":
        return event["route_critical_overlap"] > 0
    if filter_name == "non_route_critical":
        return event["route_critical_overlap"] == 0
    if filter_name == "route_critical_fraction_ge_025":
        return event["route_critical_overlap_fraction"] >= 0.25
    raise ValueError(f"Unknown filter: {filter_name}")


def simulate_events(grid, start, goal, true_table, mlp_table, unet_table, critical):
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
            overlap = sum(1 for node in nodes if node in critical)
            base = {
                "expanded_step": expanded,
                "f_primary": f_primary,
                "tie_set_size": len(nodes),
                "comparable_pairs": comparable_pairs(true_values),
                "route_critical_overlap": overlap,
                "route_critical_overlap_fraction": overlap / len(nodes),
            }
            for method, table in [("mlp", mlp_table), ("unet", unet_table), ("oracle", true_table)]:
                metrics = tie_set_metrics(nodes, true_table, table)
                event = {"method": method, **base, **metrics, **selected_stats(nodes, true_table, table)}
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
        event["timing_bucket"] = timing_bucket(event, expanded)
    return events, expanded


def result_expanded(methods, method):
    return to_float(methods[method], "expanded_nodes")


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


def summarize_events(events, method, filter_name, weight_mode, timing="all"):
    scoped = [event for event in events if event["method"] == method and include_event(event, filter_name)]
    if timing != "all":
        scoped = [event for event in scoped if event["timing_bucket"] == timing]
    return {
        "tie_sets": len(scoped),
        "mean_tie_set_size": mean(event["tie_set_size"] for event in scoped),
        "weighted_pairwise_accuracy": weighted_mean(scoped, "pairwise_ordering_accuracy", weight_mode),
        "weighted_top1_accuracy": weighted_mean(scoped, "top1_accuracy", weight_mode),
        "weighted_spearman": weighted_mean(scoped, "spearman", weight_mode),
        "weighted_kendall": weighted_mean(scoped, "kendall_tau", weight_mode),
        "mean_selected_true_distance": weighted_mean(scoped, "selected_true_distance", weight_mode),
        "mean_selected_true_rank": weighted_mean(scoped, "selected_true_rank", weight_mode),
        "mean_selected_true_percentile": weighted_mean(scoped, "selected_true_percentile", weight_mode),
    }


def analyze_map(methods, mlp_model, unet_model, radius):
    sample = methods["manhattan"]
    grid, start, goal = build_grid(sample)
    distance_grid = compute_distance_to_goal(grid, goal)
    true_table = {}
    for r, row in enumerate(distance_grid):
        for c, value in enumerate(row):
            if value >= 0:
                true_table[(r, c)] = float(value)
    path = reconstruct_optimal_path(grid, distance_grid, start, goal)
    critical = route_critical_cells(grid, path, radius)

    mlp_h = make_mlp_table_heuristic(mlp_model, grid, goal)
    unet_h = make_unet_heuristic(unet_model, grid, goal)
    mlp_table = prediction_table(grid, goal, mlp_h)
    unet_table = prediction_table(grid, goal, unet_h)
    events, expanded_steps = simulate_events(grid, start, goal, true_table, mlp_table, unet_table, critical)

    identity = map_identity(sample)
    rows = []
    for method in ORDERING_METHODS:
        for filter_name in FILTERS:
            for weight_mode in WEIGHT_MODES:
                stats = summarize_events(events, method, filter_name, weight_mode)
                rows.append(
                    {
                        "row_type": "map_summary",
                        "method": method,
                        "filter": filter_name,
                        "weight_mode": weight_mode,
                        "timing_bucket": "all",
                        "expanded_steps": expanded_steps,
                        "tie_step_fraction": stats["tie_sets"] / expanded_steps if expanded_steps else 0.0,
                        **identity,
                        **stats,
                    }
                )
            for timing in ["early_25", "middle_50", "late_25"]:
                stats = summarize_events(events, method, filter_name, "unweighted", timing)
                rows.append(
                    {
                        "row_type": "map_summary",
                        "method": method,
                        "filter": filter_name,
                        "weight_mode": "unweighted",
                        "timing_bucket": timing,
                        "expanded_steps": expanded_steps,
                        "tie_step_fraction": stats["tie_sets"] / expanded_steps if expanded_steps else 0.0,
                        **identity,
                        **stats,
                    }
                )

    for row in rows:
        row["mlp_expanded"] = result_expanded(methods, "manhattan_mlp_tiebreak")
        row["unet_expanded"] = result_expanded(methods, "manhattan_unet_tiebreak")
        row["true_distance_expanded"] = result_expanded(methods, "manhattan_true_distance_tiebreak")
        row["unet_minus_mlp_expanded"] = row["unet_expanded"] - row["mlp_expanded"]
    return rows


def aggregate_rows(map_rows):
    rows = []
    scopes = [("random", "all"), ("structured", "all")] + [("structured", structured_type) for structured_type in STRUCTURED_TYPES]
    for source, structured_type in scopes:
        scoped_source = [row for row in map_rows if row["source"] == source]
        if structured_type != "all":
            scoped_source = [row for row in scoped_source if row["structured_type"] == structured_type]
        for method in ORDERING_METHODS:
            for filter_name in FILTERS:
                for weight_mode in WEIGHT_MODES:
                    scoped = [
                        row
                        for row in scoped_source
                        if row["method"] == method
                        and row["filter"] == filter_name
                        and row["weight_mode"] == weight_mode
                        and row["timing_bucket"] == "all"
                    ]
                    rows.append(
                        {
                            "row_type": "aggregate",
                            "source": source,
                            "structured_type": structured_type,
                            "method": method,
                            "filter": filter_name,
                            "weight_mode": weight_mode,
                            "timing_bucket": "all",
                            "maps": len(scoped),
                            "tie_sets": sum(row["tie_sets"] for row in scoped),
                            "mean_tie_set_size": mean(row["mean_tie_set_size"] for row in scoped),
                            "mean_tie_step_fraction": mean(row["tie_step_fraction"] for row in scoped),
                            "weighted_pairwise_accuracy": mean(row["weighted_pairwise_accuracy"] for row in scoped),
                            "weighted_top1_accuracy": mean(row["weighted_top1_accuracy"] for row in scoped),
                            "weighted_spearman": mean(row["weighted_spearman"] for row in scoped),
                            "weighted_kendall": mean(row["weighted_kendall"] for row in scoped),
                            "mean_selected_true_distance": mean(row["mean_selected_true_distance"] for row in scoped),
                            "mean_selected_true_rank": mean(row["mean_selected_true_rank"] for row in scoped),
                            "mean_selected_true_percentile": mean(row["mean_selected_true_percentile"] for row in scoped),
                        }
                    )
                for timing in ["early_25", "middle_50", "late_25"]:
                    scoped = [
                        row
                        for row in scoped_source
                        if row["method"] == method
                        and row["filter"] == filter_name
                        and row["weight_mode"] == "unweighted"
                        and row["timing_bucket"] == timing
                    ]
                    rows.append(
                        {
                            "row_type": "aggregate",
                            "source": source,
                            "structured_type": structured_type,
                            "method": method,
                            "filter": filter_name,
                            "weight_mode": "unweighted",
                            "timing_bucket": timing,
                            "maps": len(scoped),
                            "tie_sets": sum(row["tie_sets"] for row in scoped),
                            "mean_tie_set_size": mean(row["mean_tie_set_size"] for row in scoped),
                            "mean_tie_step_fraction": mean(row["tie_step_fraction"] for row in scoped),
                            "weighted_pairwise_accuracy": mean(row["weighted_pairwise_accuracy"] for row in scoped),
                            "weighted_top1_accuracy": mean(row["weighted_top1_accuracy"] for row in scoped),
                            "weighted_spearman": mean(row["weighted_spearman"] for row in scoped),
                            "weighted_kendall": mean(row["weighted_kendall"] for row in scoped),
                            "mean_selected_true_distance": mean(row["mean_selected_true_distance"] for row in scoped),
                            "mean_selected_true_rank": mean(row["mean_selected_true_rank"] for row in scoped),
                            "mean_selected_true_percentile": mean(row["mean_selected_true_percentile"] for row in scoped),
                        }
                    )
    return rows


def pair_maps(map_rows, source="structured", structured_type=None, filter_name="all", weight_mode="unweighted", timing="all"):
    scoped = [
        row
        for row in map_rows
        if row["source"] == source
        and row["filter"] == filter_name
        and row["weight_mode"] == weight_mode
        and row["timing_bucket"] == timing
    ]
    if structured_type:
        scoped = [row for row in scoped if row["structured_type"] == structured_type]
    paired = {}
    for row in scoped:
        paired.setdefault(row["case_id"], {})[row["method"]] = row
    return [pair for pair in paired.values() if "mlp" in pair and "unet" in pair]


def correlation_rows(map_rows):
    specs = [
        ("A_unweighted_pairwise", "all", "unweighted", "weighted_pairwise_accuracy"),
        ("B_weighted_pairwise_tie_size", "all", "tie_set_size", "weighted_pairwise_accuracy"),
        ("B_weighted_pairwise_comparable_pairs", "all", "comparable_pairs", "weighted_pairwise_accuracy"),
        ("C_top1", "all", "unweighted", "weighted_top1_accuracy"),
        ("D_route_critical_pairwise", "route_critical", "unweighted", "weighted_pairwise_accuracy"),
        ("E_route_critical_top1", "route_critical", "unweighted", "weighted_top1_accuracy"),
    ]
    rows = []
    for source, structured_type in [("structured", None)] + [("structured", item) for item in STRUCTURED_TYPES]:
        for label, filter_name, weight_mode, metric in specs:
            pairs = pair_maps(map_rows, source, structured_type, filter_name, weight_mode)
            xs = [pair["unet"][metric] - pair["mlp"][metric] for pair in pairs]
            ys = [pair["unet"]["unet_minus_mlp_expanded"] for pair in pairs]
            rows.append(
                {
                    "row_type": "correlation",
                    "source": source,
                    "structured_type": structured_type or "all",
                    "analysis": label,
                    "filter": filter_name,
                    "weight_mode": weight_mode,
                    "metric": metric,
                    "x": f"unet_minus_mlp_{metric}",
                    "y": "unet_minus_mlp_expanded",
                    "n": len(xs),
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                }
            )
    return rows


def bottleneck_diagnostics(map_rows):
    rows = []
    paired = pair_maps(map_rows, "structured", "bottleneck", "all", "unweighted")
    rc_paired = {pair["mlp"]["case_id"]: pair for pair in pair_maps(map_rows, "structured", "bottleneck", "route_critical", "unweighted")}
    for pair in paired:
        mlp = pair["mlp"]
        unet = pair["unet"]
        rc = rc_paired.get(mlp["case_id"], {})
        rc_mlp = rc.get("mlp", {})
        rc_unet = rc.get("unet", {})
        row = {
            "case_id": mlp["case_id"],
            "start": mlp["start"],
            "goal": mlp["goal"],
            "tie_set_count": mlp["tie_sets"],
            "mean_tie_set_size": mlp["mean_tie_set_size"],
            "mlp_pairwise": mlp["weighted_pairwise_accuracy"],
            "unet_pairwise": unet["weighted_pairwise_accuracy"],
            "mlp_top1": mlp["weighted_top1_accuracy"],
            "unet_top1": unet["weighted_top1_accuracy"],
            "mlp_route_critical_pairwise": rc_mlp.get("weighted_pairwise_accuracy", 0.0),
            "unet_route_critical_pairwise": rc_unet.get("weighted_pairwise_accuracy", 0.0),
            "mlp_route_critical_top1": rc_mlp.get("weighted_top1_accuracy", 0.0),
            "unet_route_critical_top1": rc_unet.get("weighted_top1_accuracy", 0.0),
            "mlp_expanded": mlp["mlp_expanded"],
            "unet_expanded": mlp["unet_expanded"],
            "true_distance_expanded": mlp["true_distance_expanded"],
            "unet_minus_mlp_pairwise": unet["weighted_pairwise_accuracy"] - mlp["weighted_pairwise_accuracy"],
            "unet_minus_mlp_top1": unet["weighted_top1_accuracy"] - mlp["weighted_top1_accuracy"],
            "unet_minus_mlp_route_critical_pairwise": rc_unet.get("weighted_pairwise_accuracy", 0.0)
            - rc_mlp.get("weighted_pairwise_accuracy", 0.0),
            "unet_minus_mlp_route_critical_top1": rc_unet.get("weighted_top1_accuracy", 0.0)
            - rc_mlp.get("weighted_top1_accuracy", 0.0),
            "unet_minus_mlp_expanded": mlp["unet_minus_mlp_expanded"],
        }
        rows.append(row)
    return sorted(rows, key=lambda row: row["unet_minus_mlp_expanded"])


def save_plots(path, aggregates, correlations):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(path)), "matplotlib_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, metric in enumerate(["weighted_pairwise_accuracy", "weighted_top1_accuracy", "mean_selected_true_percentile"]):
        ax = axes[idx]
        labels = STRUCTURED_TYPES
        mlp = []
        unet = []
        for label in labels:
            mlp.append(
                next(
                    row
                    for row in aggregates
                    if row["structured_type"] == label
                    and row["method"] == "mlp"
                    and row["filter"] == "route_critical"
                    and row["weight_mode"] == "unweighted"
                    and row["timing_bucket"] == "all"
                )[metric]
            )
            unet.append(
                next(
                    row
                    for row in aggregates
                    if row["structured_type"] == label
                    and row["method"] == "unet"
                    and row["filter"] == "route_critical"
                    and row["weight_mode"] == "unweighted"
                    and row["timing_bucket"] == "all"
                )[metric]
            )
        x = range(len(labels))
        ax.bar([value - 0.2 for value in x], mlp, width=0.4, label="MLP")
        ax.bar([value + 0.2 for value in x], unet, width=0.4, label="U-Net")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(f"route-critical {metric}")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def lookup_aggregate(aggregates, source, structured_type, method, filter_name, weight_mode, timing="all"):
    return next(
        row
        for row in aggregates
        if row["source"] == source
        and row["structured_type"] == structured_type
        and row["method"] == method
        and row["filter"] == filter_name
        and row["weight_mode"] == weight_mode
        and row["timing_bucket"] == timing
    )


def lookup_corr(correlations, structured_type, analysis):
    return next(row for row in correlations if row["structured_type"] == structured_type and row["analysis"] == analysis)


def write_summary(path, aggregates, correlations):
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Weighted Tie-Set Ordering Analysis\n\n")
        file.write(
            "This analysis extends tie-set ordering with weighted aggregation, route-critical tie-set filters, timing buckets, "
            "and top-1 selected-node diagnostics. It is analysis-only and does not modify production A* or benchmark outputs.\n\n"
        )
        file.write("## Bottleneck Discrepancy\n\n")
        for filter_name in ["all", "route_critical"]:
            mlp = lookup_aggregate(aggregates, "structured", "bottleneck", "mlp", filter_name, "unweighted")
            unet = lookup_aggregate(aggregates, "structured", "bottleneck", "unet", filter_name, "unweighted")
            file.write(
                f"- {filter_name}: pairwise MLP={mlp['weighted_pairwise_accuracy']:.3f}, "
                f"U-Net={unet['weighted_pairwise_accuracy']:.3f}; top1 MLP={mlp['weighted_top1_accuracy']:.3f}, "
                f"U-Net={unet['weighted_top1_accuracy']:.3f}; selected percentile MLP={mlp['mean_selected_true_percentile']:.3f}, "
                f"U-Net={unet['mean_selected_true_percentile']:.3f}.\n"
            )
        file.write(
            "\nBottleneck remains close: MLP is slightly better on average pairwise/top1 metrics, while earlier expanded-node results "
            "show U-Net slightly ahead. This suggests average tie-set ordering is useful but still too coarse for this structure.\n\n"
        )

        file.write("## Weighted Metrics By Structure\n\n")
        file.write("| Structure | Filter | Weight | MLP pairwise | U-Net pairwise | MLP top1 | U-Net top1 |\n")
        file.write("|---|---|---|---:|---:|---:|---:|\n")
        for structured_type in STRUCTURED_TYPES:
            for filter_name in ["all", "route_critical"]:
                for weight_mode in ["unweighted", "tie_set_size", "comparable_pairs", "early_weight"]:
                    mlp = lookup_aggregate(aggregates, "structured", structured_type, "mlp", filter_name, weight_mode)
                    unet = lookup_aggregate(aggregates, "structured", structured_type, "unet", filter_name, weight_mode)
                    file.write(
                        f"| {structured_type} | {filter_name} | {weight_mode} | "
                        f"{mlp['weighted_pairwise_accuracy']:.3f} | {unet['weighted_pairwise_accuracy']:.3f} | "
                        f"{mlp['weighted_top1_accuracy']:.3f} | {unet['weighted_top1_accuracy']:.3f} |\n"
                    )
        file.write("\n")

        file.write("## Timing Buckets\n\n")
        file.write("| Structure | Method | Early pairwise | Middle pairwise | Late pairwise | Early top1 | Middle top1 | Late top1 |\n")
        file.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for structured_type in STRUCTURED_TYPES:
            for method in ORDERING_METHODS:
                early = lookup_aggregate(aggregates, "structured", structured_type, method, "all", "unweighted", "early_25")
                middle = lookup_aggregate(aggregates, "structured", structured_type, method, "all", "unweighted", "middle_50")
                late = lookup_aggregate(aggregates, "structured", structured_type, method, "all", "unweighted", "late_25")
                file.write(
                    f"| {structured_type} | {method} | {early['weighted_pairwise_accuracy']:.3f} | "
                    f"{middle['weighted_pairwise_accuracy']:.3f} | {late['weighted_pairwise_accuracy']:.3f} | "
                    f"{early['weighted_top1_accuracy']:.3f} | {middle['weighted_top1_accuracy']:.3f} | "
                    f"{late['weighted_top1_accuracy']:.3f} |\n"
                )
        file.write("\n")

        file.write("## Correlations With Expanded Gap\n\n")
        for analysis in [
            "A_unweighted_pairwise",
            "B_weighted_pairwise_tie_size",
            "C_top1",
            "D_route_critical_pairwise",
            "E_route_critical_top1",
        ]:
            row = lookup_corr(correlations, "all", analysis)
            file.write(f"- {analysis}: Pearson={row['pearson']:.3f}, Spearman={row['spearman']:.3f}, n={row['n']}.\n")
        file.write(
            "\nNegative correlations mean better U-Net-minus-MLP ordering is associated with fewer U-Net-minus-MLP expanded nodes.\n\n"
        )

        file.write("## Interpretation\n\n")
        file.write(
            "Weighted and route-critical metrics confirm that tie-set ordering is a meaningful mechanism, but bottleneck shows that "
            "simple average pairwise accuracy can miss small expansion differences. Top-1 and route-critical summaries are closer "
            "to the actual expansion decision, but even they do not fully explain every structure. Conclusions remain observational.\n"
        )


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "tie_set_ordering")
    os.makedirs(output_dir, exist_ok=True)

    random_groups = group_maps(read_csv(args.random_results), "random")
    structured_groups = group_maps(read_csv(args.structured_results), "structured")
    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint, "unet"))

    map_rows = []
    for methods in random_groups + structured_groups:
        map_rows.extend(analyze_map(methods, mlp_model, unet_model, args.route_radius))

    aggregates = aggregate_rows(map_rows)
    correlations = correlation_rows(map_rows)
    diagnostics = bottleneck_diagnostics(map_rows)
    write_csv(os.path.join(output_dir, "tie_set_weighted_ordering_statistics.csv"), aggregates + correlations)
    write_csv(os.path.join(output_dir, "bottleneck_tie_set_diagnostics.csv"), diagnostics)
    save_plots(os.path.join(output_dir, "tie_set_weighted_ordering_plots.png"), aggregates, correlations)
    write_summary(os.path.join(output_dir, "tie_set_weighted_ordering_summary.md"), aggregates, correlations)
    print(f"Saved weighted tie-set ordering outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze weighted and route-critical tie-set ordering.")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_tiebreak_controls_100.csv")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--route-radius", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
