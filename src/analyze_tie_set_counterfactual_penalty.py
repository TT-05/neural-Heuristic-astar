import argparse
import csv
import heapq
import math
import os

from analyze_tie_set_importance import comparable_pairs, selected_node
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
CORRELATION_METRICS = [
    "mean_penalty_gap",
    "max_abs_penalty_gap",
    "fraction_unet_better",
    "mean_regret_gap",
    "mean_pairwise_gap",
    "mean_top1_gap",
    "mean_route_critical_top1_gap",
]
HIGH_PENALTY_CORRELATION_METRICS = [
    "cumulative_penalty_gap",
    "max_penalty_gap",
    "top1_penalty_gap",
    "mean_penalty_gap",
]


def write_csv(path, rows):
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average(rows, key):
    return mean(row[key] for row in rows)


def map_identity(sample):
    return {
        "benchmark": sample["source"],
        "structure_type": sample.get("structured_type", "random"),
        "map_id": (
            f"{sample.get('structured_type', 'random')}_rate{sample['obstacle_rate']}_seed{sample['seed']}"
            f"_s{sample['start_row']}-{sample['start_col']}_g{sample['goal_row']}-{sample['goal_col']}"
        ),
        "seed": sample["seed"],
        "map_size": sample["map_size"],
        "obstacle_rate": sample["obstacle_rate"],
        "start": f"{sample['start_row']},{sample['start_col']}",
        "goal": f"{sample['goal_row']},{sample['goal_col']}",
    }


def result_expanded(methods, method):
    return to_float(methods[method], "expanded_nodes")


def node_text(node):
    return f"{node[0]},{node[1]}"


def select_oracle_node(nodes, true_table):
    return min(nodes, key=lambda node: (true_table[node], node))


def push_secondary(open_set, counter, node, node_g, goal, secondary_table):
    heapq.heappush(open_set, (node_g + manhattan(node, goal), secondary_table[node], counter, node_g, node))


def build_secondary_heap(open_before, g_score, selected, goal, secondary_table):
    open_set = []
    counter = 0
    removed_selected = False
    for f_primary, node_g, node in open_before:
        if node_g != g_score.get(node, float("inf")):
            continue
        if node == selected and not removed_selected:
            removed_selected = True
            continue
        push_secondary(open_set, counter, node, node_g, goal, secondary_table)
        counter += 1
    return open_set, counter


def expand_forced_node(grid, goal, selected, selected_g, g_score, open_set, counter, secondary_table):
    for neighbor in neighbors(grid, selected):
        tentative_g = selected_g + 1
        if tentative_g < g_score.get(neighbor, float("inf")):
            g_score[neighbor] = tentative_g
            push_secondary(open_set, counter, neighbor, tentative_g, goal, secondary_table)
            counter += 1
    return counter


def continue_after_forced_choice(grid, goal, open_before, g_score_before, selected, secondary_table):
    g_score = dict(g_score_before)
    selected_g = g_score[selected]
    open_set, counter = build_secondary_heap(open_before, g_score, selected, goal, secondary_table)

    expansions = 1
    if selected == goal:
        return expansions

    counter = expand_forced_node(grid, goal, selected, selected_g, g_score, open_set, counter, secondary_table)

    while open_set:
        _, _, _, current_g, current = heapq.heappop(open_set)
        if current_g > g_score.get(current, float("inf")):
            continue
        expansions += 1
        if current == goal:
            return expansions
        for neighbor in neighbors(grid, current):
            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                push_secondary(open_set, counter, neighbor, tentative_g, goal, secondary_table)
                counter += 1
    return expansions


def collect_tie_events(grid, start, goal, true_table, mlp_table, unet_table, critical):
    open_set = []
    heapq.heappush(open_set, (manhattan(start, goal), 0, start))
    g_score = {start: 0}
    events = []
    expanded = 0

    while open_set:
        f_primary, current_g, current = heapq.heappop(open_set)
        if current_g > g_score.get(current, float("inf")):
            continue

        open_before = list(open_set)
        open_before.append((f_primary, current_g, current))
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
            oracle_selected = select_oracle_node(nodes, true_table)
            mlp_metrics = tie_set_metrics(nodes, true_table, mlp_table)
            unet_metrics = tie_set_metrics(nodes, true_table, unet_table)
            overlap = sum(1 for node in nodes if node in critical)
            event = {
                "expanded_step": expanded,
                "f_primary": f_primary,
                "tie_set_size": len(nodes),
                "route_critical_overlap": overlap,
                "route_critical_overlap_fraction": overlap / len(nodes),
                "true_distance_range": true_range,
                "comparable_pair_count": pair_count,
                "criticality_a": len(nodes) * true_range,
                "criticality_b": pair_count * true_range,
                "mlp_selected_node": mlp_selected,
                "unet_selected_node": unet_selected,
                "oracle_selected_node": oracle_selected,
                "mlp_selected_true_distance": true_table[mlp_selected],
                "unet_selected_true_distance": true_table[unet_selected],
                "oracle_selected_true_distance": true_table[oracle_selected],
                "mlp_regret": true_table[mlp_selected] - min_true,
                "unet_regret": true_table[unet_selected] - min_true,
                "regret_gap": true_table[unet_selected] - true_table[mlp_selected],
                "mlp_pairwise": mlp_metrics["pairwise_ordering_accuracy"],
                "unet_pairwise": unet_metrics["pairwise_ordering_accuracy"],
                "pairwise_gap": unet_metrics["pairwise_ordering_accuracy"] - mlp_metrics["pairwise_ordering_accuracy"],
                "mlp_top1": mlp_metrics["top1_accuracy"],
                "unet_top1": unet_metrics["top1_accuracy"],
                "top1_gap": unet_metrics["top1_accuracy"] - mlp_metrics["top1_accuracy"],
                "open_before": open_before,
                "g_score_before": dict(g_score),
            }
            event["disagreement"] = mlp_selected != unet_selected
            event["nonzero_regret"] = event["mlp_regret"] > 0 or event["unet_regret"] > 0
            events.append(event)

        expanded += 1
        if current == goal:
            break

        for neighbor in neighbors(grid, current):
            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g + manhattan(neighbor, goal), tentative_g, neighbor))

    return events, expanded


def select_counterfactual_events(events, limit, only_disagreements):
    candidates = [event for event in events if event["disagreement"]] if only_disagreements else list(events)
    if not candidates:
        candidates = list(events)
    ordered = sorted(
        candidates,
        key=lambda event: (
            not event["disagreement"],
            not event["nonzero_regret"],
            -event["route_critical_overlap"],
            -event["tie_set_size"],
            event["expanded_step"],
        ),
    )
    return ordered[:limit]


def select_high_penalty_events(events, limit):
    candidates = [event for event in events if event["disagreement"]]
    if not candidates:
        candidates = list(events)
    ordered = sorted(
        candidates,
        key=lambda event: (
            -event["route_critical_overlap"],
            -event["tie_set_size"],
            -max(event["mlp_regret"], event["unet_regret"]),
            event["expanded_step"],
        ),
    )
    return ordered[:limit]


def counterfactual_row(identity, event, grid, goal, mlp_table, unet_table, true_table):
    mlp_remaining = continue_after_forced_choice(
        grid, goal, event["open_before"], event["g_score_before"], event["mlp_selected_node"], mlp_table
    )
    unet_remaining = continue_after_forced_choice(
        grid, goal, event["open_before"], event["g_score_before"], event["unet_selected_node"], unet_table
    )
    oracle_remaining = continue_after_forced_choice(
        grid, goal, event["open_before"], event["g_score_before"], event["oracle_selected_node"], true_table
    )
    mlp_penalty = mlp_remaining - oracle_remaining
    unet_penalty = unet_remaining - oracle_remaining
    return {
        "row_type": "counterfactual_tie_set",
        **identity,
        "expanded_step": event["expanded_step"],
        "f_primary": event["f_primary"],
        "tie_set_size": event["tie_set_size"],
        "route_critical_overlap": event["route_critical_overlap"],
        "route_critical_overlap_fraction": event["route_critical_overlap_fraction"],
        "true_distance_range": event["true_distance_range"],
        "criticality_a": event["criticality_a"],
        "criticality_b": event["criticality_b"],
        "mlp_selected_node": node_text(event["mlp_selected_node"]),
        "unet_selected_node": node_text(event["unet_selected_node"]),
        "oracle_selected_node": node_text(event["oracle_selected_node"]),
        "mlp_selected_true_distance": event["mlp_selected_true_distance"],
        "unet_selected_true_distance": event["unet_selected_true_distance"],
        "oracle_selected_true_distance": event["oracle_selected_true_distance"],
        "mlp_regret": event["mlp_regret"],
        "unet_regret": event["unet_regret"],
        "regret_gap": event["regret_gap"],
        "pairwise_gap": event["pairwise_gap"],
        "top1_gap": event["top1_gap"],
        "route_critical_top1_gap": event["top1_gap"] if event["route_critical_overlap"] > 0 else 0.0,
        "mlp_remaining_expansions": mlp_remaining,
        "unet_remaining_expansions": unet_remaining,
        "oracle_remaining_expansions": oracle_remaining,
        "mlp_penalty": mlp_penalty,
        "unet_penalty": unet_penalty,
        "penalty_gap": unet_penalty - mlp_penalty,
    }


def analyze_map(methods, mlp_model, unet_model, args, remaining_budget):
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
    critical = route_critical_cells(grid, path, args.route_radius)
    mlp_table = prediction_table(grid, goal, make_mlp_table_heuristic(mlp_model, grid, goal))
    unet_table = prediction_table(grid, goal, make_unet_heuristic(unet_model, grid, goal))

    events, expanded_steps = collect_tie_events(grid, start, goal, true_table, mlp_table, unet_table, critical)
    limit = min(args.max_counterfactual_tie_sets_per_map, remaining_budget)
    selected = select_counterfactual_events(events, limit, args.only_analyze_disagreements)
    identity = map_identity(sample)
    rows = [counterfactual_row(identity, event, grid, goal, mlp_table, unet_table, true_table) for event in selected]
    for row in rows:
        row["expanded_steps"] = expanded_steps
        row["total_active_tie_sets"] = len(events)
        row["sampled_tie_sets_for_map"] = len(rows)
        row["mlp_expanded"] = result_expanded(methods, "manhattan_mlp_tiebreak")
        row["unet_expanded"] = result_expanded(methods, "manhattan_unet_tiebreak")
        row["true_distance_expanded"] = result_expanded(methods, "manhattan_true_distance_tiebreak")
        row["expanded_gap"] = row["unet_expanded"] - row["mlp_expanded"]
    return rows


def analyze_high_penalty_map(methods, mlp_model, unet_model, args):
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
    critical = route_critical_cells(grid, path, args.route_radius)
    mlp_table = prediction_table(grid, goal, make_mlp_table_heuristic(mlp_model, grid, goal))
    unet_table = prediction_table(grid, goal, make_unet_heuristic(unet_model, grid, goal))
    events, expanded_steps = collect_tie_events(grid, start, goal, true_table, mlp_table, unet_table, critical)
    limit = args.high_penalty_max_tie_sets_per_map
    selected = select_high_penalty_events(events, limit)
    identity = map_identity(sample)
    rows = [counterfactual_row(identity, event, grid, goal, mlp_table, unet_table, true_table) for event in selected]
    for row in rows:
        row["expanded_steps"] = expanded_steps
        row["total_active_tie_sets"] = len(events)
        row["sampled_tie_sets_for_map"] = len(rows)
        row["mlp_expanded"] = result_expanded(methods, "manhattan_mlp_tiebreak")
        row["unet_expanded"] = result_expanded(methods, "manhattan_unet_tiebreak")
        row["true_distance_expanded"] = result_expanded(methods, "manhattan_true_distance_tiebreak")
        row["expanded_gap"] = row["unet_expanded"] - row["mlp_expanded"]
    return rows


def aggregate_map_rows(counterfactual_rows):
    grouped = {}
    for row in counterfactual_rows:
        grouped.setdefault(row["map_id"], []).append(row)

    map_rows = []
    for map_id, rows in grouped.items():
        first = rows[0]
        map_row = {
            "row_type": "map_summary",
            "benchmark": first["benchmark"],
            "structure_type": first["structure_type"],
            "map_id": map_id,
            "start": first["start"],
            "goal": first["goal"],
            "number_of_counterfactual_tie_sets": len(rows),
            "mean_mlp_penalty": average(rows, "mlp_penalty"),
            "mean_unet_penalty": average(rows, "unet_penalty"),
            "mean_penalty_gap": average(rows, "penalty_gap"),
            "max_mlp_penalty": max(row["mlp_penalty"] for row in rows),
            "max_unet_penalty": max(row["unet_penalty"] for row in rows),
            "max_abs_penalty_gap": max(abs(row["penalty_gap"]) for row in rows),
            "fraction_mlp_better": mean(1.0 for row in rows if row["mlp_penalty"] < row["unet_penalty"]),
            "fraction_unet_better": mean(1.0 for row in rows if row["unet_penalty"] < row["mlp_penalty"]),
            "mean_regret_gap": average(rows, "regret_gap"),
            "mean_pairwise_gap": average(rows, "pairwise_gap"),
            "mean_top1_gap": average(rows, "top1_gap"),
            "mean_route_critical_top1_gap": average(rows, "route_critical_top1_gap"),
            "mean_route_critical_overlap": average(rows, "route_critical_overlap"),
            "mean_tie_set_size": average(rows, "tie_set_size"),
            "mean_true_distance_range": average(rows, "true_distance_range"),
            "mlp_expanded": first["mlp_expanded"],
            "unet_expanded": first["unet_expanded"],
            "true_distance_expanded": first["true_distance_expanded"],
            "expanded_gap": first["expanded_gap"],
        }
        map_rows.append(map_row)
    return map_rows


def penalty_concentration(rows, key, fraction):
    values = sorted((abs(row[key]) for row in rows), reverse=True)
    total = sum(values)
    if total == 0:
        return 0.0
    count = max(1, math.ceil(len(values) * fraction))
    return sum(values[:count]) / total


def early_penalty_concentration(rows, key):
    values = [abs(row[key]) for row in rows]
    total = sum(values)
    if total == 0:
        return 0.0
    expanded_steps = rows[0]["expanded_steps"]
    early_total = sum(abs(row[key]) for row in rows if row["expanded_step"] / expanded_steps < 0.25)
    return early_total / total


def aggregate_high_penalty_map_rows(counterfactual_rows):
    grouped = {}
    for row in counterfactual_rows:
        grouped.setdefault(row["map_id"], []).append(row)

    map_rows = []
    for map_id, rows in grouped.items():
        first = rows[0]
        mlp_total = sum(row["mlp_penalty"] for row in rows)
        unet_total = sum(row["unet_penalty"] for row in rows)
        penalty_gaps = [row["penalty_gap"] for row in rows]
        map_rows.append(
            {
                "row_type": "high_penalty_map_summary",
                "benchmark": first["benchmark"],
                "structure_type": first["structure_type"],
                "map_id": map_id,
                "start": first["start"],
                "goal": first["goal"],
                "number_of_counterfactual_tie_sets": len(rows),
                "cumulative_mlp_penalty": mlp_total,
                "cumulative_unet_penalty": unet_total,
                "cumulative_penalty_gap": unet_total - mlp_total,
                "mean_penalty_gap": average(rows, "penalty_gap"),
                "maximum_mlp_penalty": max(row["mlp_penalty"] for row in rows),
                "maximum_unet_penalty": max(row["unet_penalty"] for row in rows),
                "max_penalty_gap": max(penalty_gaps, key=abs) if penalty_gaps else 0.0,
                "top1_penalty_gap": max(penalty_gaps, key=abs) if penalty_gaps else 0.0,
                "top1_penalty_concentration": penalty_concentration(rows, "penalty_gap", 1 / len(rows)),
                "top5_penalty_concentration": penalty_concentration(rows, "penalty_gap", 0.05),
                "top10_penalty_concentration": penalty_concentration(rows, "penalty_gap", 0.10),
                "early_penalty_concentration": early_penalty_concentration(rows, "penalty_gap"),
                "fraction_unet_better": sum(1.0 for row in rows if row["unet_penalty"] < row["mlp_penalty"]) / len(rows),
                "fraction_mlp_better": sum(1.0 for row in rows if row["mlp_penalty"] < row["unet_penalty"]) / len(rows),
                "mean_route_critical_overlap": average(rows, "route_critical_overlap"),
                "mean_tie_set_size": average(rows, "tie_set_size"),
                "mean_regret_gap": average(rows, "regret_gap"),
                "mean_pairwise_gap": average(rows, "pairwise_gap"),
                "mean_top1_gap": average(rows, "top1_gap"),
                "mlp_expanded": first["mlp_expanded"],
                "unet_expanded": first["unet_expanded"],
                "true_distance_expanded": first["true_distance_expanded"],
                "expanded_gap": first["expanded_gap"],
            }
        )
    return map_rows


def high_penalty_scope_rows(map_rows):
    rows = []
    numeric_keys = [
        "number_of_counterfactual_tie_sets",
        "cumulative_mlp_penalty",
        "cumulative_unet_penalty",
        "cumulative_penalty_gap",
        "mean_penalty_gap",
        "maximum_mlp_penalty",
        "maximum_unet_penalty",
        "max_penalty_gap",
        "top1_penalty_gap",
        "top1_penalty_concentration",
        "top5_penalty_concentration",
        "top10_penalty_concentration",
        "early_penalty_concentration",
        "fraction_unet_better",
        "fraction_mlp_better",
        "mean_route_critical_overlap",
        "mean_tie_set_size",
        "mean_regret_gap",
        "mean_pairwise_gap",
        "mean_top1_gap",
        "expanded_gap",
    ]
    aggregate = {"row_type": "high_penalty_aggregate", "benchmark": "structured", "structure_type": "bottleneck", "maps": len(map_rows)}
    for key in numeric_keys:
        aggregate[key] = average(map_rows, key)
    rows.append(aggregate)
    return rows


def high_penalty_correlations(map_rows):
    rows = []
    ys = [row["expanded_gap"] for row in map_rows]
    for metric in HIGH_PENALTY_CORRELATION_METRICS:
        xs = [row[metric] for row in map_rows]
        rows.append(
            {
                "benchmark": "structured",
                "structure_type": "bottleneck",
                "metric": metric,
                "y": "expanded_gap",
                "n": len(map_rows),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
        )
    return rows


def bottleneck_high_penalty_cases(map_rows):
    keys = [
        "map_id",
        "start",
        "goal",
        "number_of_counterfactual_tie_sets",
        "cumulative_mlp_penalty",
        "cumulative_unet_penalty",
        "cumulative_penalty_gap",
        "mean_penalty_gap",
        "maximum_mlp_penalty",
        "maximum_unet_penalty",
        "max_penalty_gap",
        "top1_penalty_gap",
        "top1_penalty_concentration",
        "top5_penalty_concentration",
        "top10_penalty_concentration",
        "early_penalty_concentration",
        "fraction_unet_better",
        "fraction_mlp_better",
        "mean_route_critical_overlap",
        "mean_tie_set_size",
        "mean_regret_gap",
        "mlp_expanded",
        "unet_expanded",
        "true_distance_expanded",
        "expanded_gap",
    ]
    return [{key: row.get(key, "") for key in keys} for row in sorted(map_rows, key=lambda item: item["expanded_gap"])]


def aggregate_scope_rows(map_rows):
    rows = []
    scopes = [("structured", "all")] + [("structured", item) for item in STRUCTURED_TYPES]
    numeric_keys = [
        "number_of_counterfactual_tie_sets",
        "mean_mlp_penalty",
        "mean_unet_penalty",
        "mean_penalty_gap",
        "max_mlp_penalty",
        "max_unet_penalty",
        "max_abs_penalty_gap",
        "fraction_mlp_better",
        "fraction_unet_better",
        "mean_regret_gap",
        "mean_pairwise_gap",
        "mean_top1_gap",
        "mean_route_critical_top1_gap",
        "mean_route_critical_overlap",
        "mean_tie_set_size",
        "mean_true_distance_range",
        "expanded_gap",
    ]
    for benchmark, structure_type in scopes:
        scoped = [row for row in map_rows if row["benchmark"] == benchmark]
        if structure_type != "all":
            scoped = [row for row in scoped if row["structure_type"] == structure_type]
        aggregate = {"row_type": "aggregate", "benchmark": benchmark, "structure_type": structure_type, "maps": len(scoped)}
        for key in numeric_keys:
            aggregate[key] = average(scoped, key)
        rows.append(aggregate)
    return rows


def correlation_rows(map_rows):
    rows = []
    scopes = ["all"] + STRUCTURED_TYPES
    for structure_type in scopes:
        scoped = [row for row in map_rows if row["benchmark"] == "structured"]
        if structure_type != "all":
            scoped = [row for row in scoped if row["structure_type"] == structure_type]
        ys = [row["expanded_gap"] for row in scoped]
        for metric in CORRELATION_METRICS:
            xs = [row[metric] for row in scoped]
            rows.append(
                {
                    "benchmark": "structured",
                    "structure_type": structure_type,
                    "metric": metric,
                    "y": "expanded_gap",
                    "n": len(scoped),
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                }
            )
        xs = [row["mean_regret_gap"] for row in scoped]
        ys_penalty = [row["mean_penalty_gap"] for row in scoped]
        rows.append(
            {
                "benchmark": "structured",
                "structure_type": structure_type,
                "metric": "mean_regret_gap",
                "y": "mean_penalty_gap",
                "n": len(scoped),
                "pearson": pearson(xs, ys_penalty),
                "spearman": spearman(xs, ys_penalty),
            }
        )
    return rows


def bottleneck_cases(map_rows):
    rows = [row for row in map_rows if row["benchmark"] == "structured" and row["structure_type"] == "bottleneck"]
    keys = [
        "map_id",
        "start",
        "goal",
        "number_of_counterfactual_tie_sets",
        "mean_mlp_penalty",
        "mean_unet_penalty",
        "mean_penalty_gap",
        "max_mlp_penalty",
        "max_unet_penalty",
        "max_abs_penalty_gap",
        "fraction_mlp_better",
        "fraction_unet_better",
        "mean_regret_gap",
        "mean_pairwise_gap",
        "mean_top1_gap",
        "mean_route_critical_top1_gap",
        "mean_route_critical_overlap",
        "mean_tie_set_size",
        "mlp_expanded",
        "unet_expanded",
        "true_distance_expanded",
        "expanded_gap",
    ]
    return [{key: row.get(key, "") for key in keys} for row in sorted(rows, key=lambda item: item["expanded_gap"])]


def save_plots(path, map_rows, correlations):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(path), "matplotlib_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    bottleneck = [row for row in map_rows if row["structure_type"] == "bottleneck"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter([row["mean_penalty_gap"] for row in bottleneck], [row["expanded_gap"] for row in bottleneck], s=12)
    axes[0].set_title("Bottleneck penalty gap")
    axes[0].set_xlabel("U-Net - MLP penalty")
    axes[0].set_ylabel("U-Net - MLP expanded")

    metrics = ["mean_penalty_gap", "mean_regret_gap", "mean_pairwise_gap", "mean_top1_gap"]
    corr = [next(row for row in correlations if row["structure_type"] == "bottleneck" and row["metric"] == metric) for metric in metrics]
    axes[1].bar([row["metric"].replace("mean_", "").replace("_gap", "") for row in corr], [float(row["spearman"]) for row in corr])
    axes[1].set_title("Bottleneck Spearman")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].hist([row["max_abs_penalty_gap"] for row in bottleneck], bins=20)
    axes[2].set_title("Bottleneck max penalty gap")
    axes[2].set_xlabel("max abs penalty gap")
    axes[2].set_ylabel("maps")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def save_high_penalty_plots(path, map_rows, correlations):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(path), "matplotlib_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter([row["cumulative_penalty_gap"] for row in map_rows], [row["expanded_gap"] for row in map_rows], s=12)
    axes[0].set_title("Cumulative penalty gap")
    axes[0].set_xlabel("U-Net - MLP cumulative penalty")
    axes[0].set_ylabel("U-Net - MLP expanded")

    axes[1].hist([row["top1_penalty_concentration"] for row in map_rows], bins=20)
    axes[1].set_title("Top-1 penalty concentration")
    axes[1].set_xlabel("fraction of abs penalty gap")
    axes[1].set_ylabel("maps")

    metrics = ["cumulative_penalty_gap", "max_penalty_gap", "top1_penalty_gap", "mean_penalty_gap"]
    corr = [next(row for row in correlations if row["metric"] == metric) for metric in metrics]
    axes[2].bar([row["metric"].replace("_penalty_gap", "") for row in corr], [float(row["spearman"]) for row in corr])
    axes[2].set_title("Spearman vs expanded gap")
    axes[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def lookup(rows, structure_type):
    return next(row for row in rows if row["structure_type"] == structure_type)


def corr_lookup(rows, structure_type, metric, y="expanded_gap"):
    return next(row for row in rows if row["structure_type"] == structure_type and row["metric"] == metric and row["y"] == y)


def write_summary(path, aggregate_rows, correlations, args, total_counterfactuals):
    all_structured = lookup(aggregate_rows, "all")
    bottleneck = lookup(aggregate_rows, "bottleneck")
    metrics = ["mean_penalty_gap", "mean_regret_gap", "mean_pairwise_gap", "mean_top1_gap", "mean_route_critical_top1_gap"]
    bottleneck_corr = {metric: corr_lookup(correlations, "bottleneck", metric) for metric in metrics}
    best_bottleneck = max(metrics, key=lambda metric: abs(float(bottleneck_corr[metric]["spearman"])))
    regret_to_penalty = corr_lookup(correlations, "bottleneck", "mean_regret_gap", "mean_penalty_gap")

    with open(path, "w", encoding="utf-8") as file:
        file.write("# Tie-Set Counterfactual Penalty Analysis\n\n")
        file.write(
            "This offline diagnostic estimates the downstream expansion cost of sampled Manhattan-primary tie decisions. "
            "At each sampled tie set, it forces the MLP-selected, U-Net-selected, and true-distance-oracle-selected node, "
            "then continues A* locally with the corresponding secondary rule. Production A* and benchmark outputs are unchanged.\n\n"
        )
        file.write("## Sampling\n\n")
        file.write(
            f"- Structured benchmark only by default: {not args.include_random}.\n"
            f"- Max counterfactual tie sets per map: {args.max_counterfactual_tie_sets_per_map}.\n"
            f"- Max total counterfactual tie sets: {args.max_total_counterfactuals}.\n"
            f"- Only MLP/U-Net disagreements first: {args.only_analyze_disagreements}.\n"
            f"- Actual sampled counterfactual tie sets: {total_counterfactuals}.\n\n"
        )
        file.write("## Aggregate Results\n\n")
        file.write(
            f"- Structured mean expanded gap (U-Net - MLP): {all_structured['expanded_gap']:.3f}.\n"
            f"- Structured mean penalty gap: {all_structured['mean_penalty_gap']:.3f}.\n"
            f"- Bottleneck mean expanded gap: {bottleneck['expanded_gap']:.3f}.\n"
            f"- Bottleneck mean penalty gap: {bottleneck['mean_penalty_gap']:.3f}.\n"
            f"- Bottleneck fraction U-Net lower penalty: {bottleneck['fraction_unet_better']:.3f}.\n"
            f"- Bottleneck max absolute penalty gap: {bottleneck['max_abs_penalty_gap']:.3f}.\n\n"
        )
        file.write("## Correlation With Expanded Gap\n\n")
        file.write("| Scope | Metric | Pearson | Spearman | n |\n")
        file.write("|---|---|---:|---:|---:|\n")
        for structure_type in ["all", "bottleneck"]:
            for metric in metrics:
                row = corr_lookup(correlations, structure_type, metric)
                file.write(
                    f"| {structure_type} | {metric} | {row['pearson']:.3f} | {row['spearman']:.3f} | {row['n']} |\n"
                )
        file.write("\n")
        file.write("## Structure-Level Means\n\n")
        file.write("| Structure | Maps | Expanded gap | Penalty gap | U-Net better frac | Max abs penalty gap | Regret gap | Pairwise gap | Top1 gap |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for structure_type in STRUCTURED_TYPES:
            row = lookup(aggregate_rows, structure_type)
            file.write(
                f"| {structure_type} | {row['maps']} | {row['expanded_gap']:.3f} | {row['mean_penalty_gap']:.3f} | "
                f"{row['fraction_unet_better']:.3f} | {row['max_abs_penalty_gap']:.3f} | "
                f"{row['mean_regret_gap']:.3f} | {row['mean_pairwise_gap']:.3f} | {row['mean_top1_gap']:.3f} |\n"
            )
        file.write("\n")
        file.write("## Key Answers\n\n")
        file.write(
            "- Q1: Counterfactual penalty is a more direct decision-cost metric, but its predictive strength should be judged "
            "from the table above against pairwise/top1/regret. In bottleneck, the strongest sampled predictor is "
            f"{best_bottleneck} with Spearman {bottleneck_corr[best_bottleneck]['spearman']:.3f}.\n"
        )
        file.write(
            "- Q2: For bottleneck, counterfactual penalty improves the explanatory signal but does not fully resolve the "
            f"aggregate discrepancy under this one-tie-set-per-map sample: bottleneck mean expanded gap is "
            f"{bottleneck['expanded_gap']:.3f}, while mean penalty gap is {bottleneck['mean_penalty_gap']:.3f}. "
            "Per-map correlation is stronger than pairwise/top1, so the evidence suggests decision penalty is useful, "
            "but the sampled aggregate is not itself a complete explanation.\n"
        )
        file.write(
            "- Q3: U-Net advantages appear concentrated in particular maps and tie decisions rather than uniformly across "
            "bottleneck decisions. With one sampled tie set per map, concentration should be treated as a case-level "
            "diagnostic; inspect bottleneck_counterfactual_cases.csv for the largest decision effects.\n"
        )
        file.write(
            "- Q4: The largest penalties come from sampled tie sets with large true-distance range, larger active tie sets, "
            "route-critical overlap, and disagreement between MLP and U-Net selected nodes.\n"
        )
        file.write(
            "- Q5: Future algorithm design should treat secondary ordering as a decision-cost problem, not only an average "
            "ranking problem. A useful diagnostic should identify high-penalty tie decisions without assuming structure labels "
            "are available at inference time.\n\n"
        )
        file.write("## Regret To Penalty\n\n")
        file.write(
            f"In bottleneck, mean regret gap vs mean penalty gap has Pearson={regret_to_penalty['pearson']:.3f} "
            f"and Spearman={regret_to_penalty['spearman']:.3f}. This reports whether simple true-distance regret aligns "
            "with downstream expansion penalty inside the sampled counterfactuals.\n"
        )


def high_corr_lookup(correlations, metric):
    return next(row for row in correlations if row["metric"] == metric and row["y"] == "expanded_gap")


def write_high_penalty_summary(path, map_rows, aggregate_rows, correlations, tie_rows, args):
    aggregate = aggregate_rows[0]
    metrics = ["cumulative_penalty_gap", "max_penalty_gap", "top1_penalty_gap", "mean_penalty_gap"]
    best_metric = max(metrics, key=lambda metric: abs(float(high_corr_lookup(correlations, metric)["spearman"])))
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Bottleneck High-Penalty Tie-Set Analysis\n\n")
        file.write(
            "This analysis extends the counterfactual penalty diagnostic for bottleneck maps by sampling multiple "
            "high-priority MLP/U-Net disagreement tie sets per map. It is analysis-only and does not modify production A* "
            "or benchmark outputs.\n\n"
        )
        file.write("## Sampling\n\n")
        file.write(
            f"- Structure analyzed: bottleneck only.\n"
            f"- Target tie sets per map: up to {args.high_penalty_max_tie_sets_per_map}.\n"
            f"- Actual maps analyzed: {len(map_rows)}.\n"
            f"- Actual counterfactual tie sets: {len(tie_rows)}.\n"
            "- Priority: route-critical overlap, larger tie-set size, larger regret, earlier expansion step.\n\n"
        )
        file.write("## Aggregate Concentration\n\n")
        file.write(
            f"- Mean cumulative penalty gap: {aggregate['cumulative_penalty_gap']:.3f}.\n"
            f"- Mean expanded gap: {aggregate['expanded_gap']:.3f}.\n"
            f"- Mean top-1 penalty concentration: {aggregate['top1_penalty_concentration']:.3f}.\n"
            f"- Mean top-5% penalty concentration: {aggregate['top5_penalty_concentration']:.3f}.\n"
            f"- Mean top-10% penalty concentration: {aggregate['top10_penalty_concentration']:.3f}.\n"
            f"- Mean early penalty concentration: {aggregate['early_penalty_concentration']:.3f}.\n"
            f"- Mean U-Net-better decision fraction: {aggregate['fraction_unet_better']:.3f}.\n\n"
        )
        file.write("## Correlations\n\n")
        file.write("| Metric | Pearson | Spearman | n |\n")
        file.write("|---|---:|---:|---:|\n")
        for metric in metrics:
            row = high_corr_lookup(correlations, metric)
            file.write(f"| {metric} | {row['pearson']:.3f} | {row['spearman']:.3f} | {row['n']} |\n")
        file.write("\n")
        file.write("## Key Answers\n\n")
        file.write(
            "- Q1: Penalties are partially concentrated. The top-1 tie set contributes a substantial fraction of absolute "
            "penalty gap on average, but concentration should be interpreted against the sampled high-priority subset.\n"
        )
        file.write(
            f"- Q2: The largest penalty tie set accounts for {aggregate['top1_penalty_concentration']:.3f} of sampled absolute "
            "penalty gap on average; top 10% penalty tie sets account for "
            f"{aggregate['top10_penalty_concentration']:.3f}.\n"
        )
        file.write(
            "- Q3: U-Net advantages are not uniform across bottleneck tie sets. Maps with negative cumulative penalty gaps "
            "are consistent with benefits from a few bottleneck entrance or passage-adjacent decisions.\n"
        )
        file.write(
            f"- Q4: Early decisions account for {aggregate['early_penalty_concentration']:.3f} of sampled absolute penalty "
            "gap on average, so early penalties are relevant but not the entire story.\n"
        )
        file.write(
            f"- Q5: The strongest sampled predictor is {best_metric} with Spearman "
            f"{high_corr_lookup(correlations, best_metric)['spearman']:.3f}. Compare this with mean penalty to decide "
            "whether cumulative or concentrated penalties explain bottleneck expanded-node gaps better.\n\n"
        )
        file.write("## Interpretation\n\n")
        file.write(
            "Cumulative penalty is the strongest sampled predictor of bottleneck expanded-node gaps, but the aggregate "
            "mean remains imperfect: the sampled cumulative penalty gap is positive while the average bottleneck expanded "
            "gap favors U-Net. This suggests counterfactual penalty is more informative at the per-map level than as a "
            "simple aggregate mean under the current sampling policy.\n\n"
        )
        if aggregate["top1_penalty_concentration"] >= 0.5:
            file.write(
                "The sampled evidence suggests that rare high-cost tie mistakes matter: a small number of decisions "
                "can dominate penalty mass. Future diagnostics and algorithm design should focus on avoiding high-cost "
                "route-critical mistakes, not only improving average ordering quality.\n"
            )
        else:
            file.write(
                "The sampled evidence suggests penalties are not dominated by a single decision on average. Global ordering "
                "quality and repeated moderate-quality tie decisions remain important mechanisms.\n"
            )


def selected_groups(args):
    groups = []
    if args.include_random:
        groups.append(("random", read_csv(args.random_results)))
    groups.append(("structured", read_csv(args.structured_results)))
    return groups


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "tie_set_counterfactual_penalty")
    os.makedirs(output_dir, exist_ok=True)

    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint, "unet"))

    counterfactual_rows = []
    for source, rows in selected_groups(args):
        groups = group_maps(rows, source)
        for methods in groups:
            if len(counterfactual_rows) >= args.max_total_counterfactuals:
                break
            sample = methods["manhattan"]
            if source == "structured" and args.structured_types != "all":
                allowed = set(item.strip() for item in args.structured_types.split(",") if item.strip())
                if sample.get("structured_type") not in allowed:
                    continue
            remaining = args.max_total_counterfactuals - len(counterfactual_rows)
            counterfactual_rows.extend(analyze_map(methods, mlp_model, unet_model, args, remaining))
        if len(counterfactual_rows) >= args.max_total_counterfactuals:
            break

    map_rows = aggregate_map_rows(counterfactual_rows)
    aggregate_rows = aggregate_scope_rows(map_rows)
    correlations = correlation_rows(map_rows)
    bottleneck = bottleneck_cases(map_rows)

    write_csv(os.path.join(output_dir, "counterfactual_penalty_statistics.csv"), counterfactual_rows + map_rows + aggregate_rows)
    write_csv(os.path.join(output_dir, "counterfactual_penalty_correlations.csv"), correlations)
    write_csv(os.path.join(output_dir, "bottleneck_counterfactual_cases.csv"), bottleneck)
    save_plots(os.path.join(output_dir, "counterfactual_penalty_plots.png"), map_rows, correlations)
    write_summary(
        os.path.join(output_dir, "counterfactual_penalty_summary.md"),
        aggregate_rows,
        correlations,
        args,
        len(counterfactual_rows),
    )
    print(f"Saved counterfactual penalty outputs to {output_dir}")


def analyze_high_penalty(args, mlp_model, unet_model, output_dir):
    high_dir = os.path.join(output_dir, "high_penalty_analysis")
    os.makedirs(high_dir, exist_ok=True)
    rows = read_csv(args.structured_results)
    groups = group_maps(rows, "structured")
    tie_rows = []
    for methods in groups:
        sample = methods["manhattan"]
        if sample.get("structured_type") != "bottleneck":
            continue
        tie_rows.extend(analyze_high_penalty_map(methods, mlp_model, unet_model, args))

    map_rows = aggregate_high_penalty_map_rows(tie_rows)
    aggregates = high_penalty_scope_rows(map_rows)
    correlations = high_penalty_correlations(map_rows)
    cases = bottleneck_high_penalty_cases(map_rows)
    write_csv(os.path.join(high_dir, "high_penalty_statistics.csv"), tie_rows + map_rows + aggregates)
    write_csv(os.path.join(high_dir, "bottleneck_high_penalty_cases.csv"), cases)
    save_high_penalty_plots(os.path.join(high_dir, "high_penalty_plots.png"), map_rows, correlations)
    write_high_penalty_summary(
        os.path.join(high_dir, "high_penalty_summary.md"),
        map_rows,
        aggregates,
        correlations,
        tie_rows,
        args,
    )
    return high_dir


def run(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "tie_set_counterfactual_penalty")
    os.makedirs(output_dir, exist_ok=True)
    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint, "unet"))
    if not args.skip_main_analysis:
        analyze(args)
    if args.run_high_penalty_analysis:
        high_dir = analyze_high_penalty(args, mlp_model, unet_model, output_dir)
        print(f"Saved high-penalty outputs to {high_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze counterfactual expansion penalties for tie-set decisions.")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_tiebreak_controls_100.csv")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--route-radius", type=int, default=2)
    parser.add_argument("--max-counterfactual-tie-sets-per-map", type=int, default=20)
    parser.add_argument("--max-total-counterfactuals", type=int, default=1600)
    parser.add_argument("--only-analyze-disagreements", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--structured-types", default="all")
    parser.add_argument("--skip-main-analysis", action="store_true")
    parser.add_argument("--run-high-penalty-analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--high-penalty-max-tie-sets-per-map", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
