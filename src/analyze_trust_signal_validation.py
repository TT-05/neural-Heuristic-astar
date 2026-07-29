import argparse
import csv
import heapq
import math
import os
import random
from collections import Counter, defaultdict

from analyze_maze_visual_mechanisms import distance_to_path_table
from analyze_tie_set_ordering import (
    build_grid,
    checkpoint_path,
    manhattan,
    mean,
    neighbors,
    pearson,
    prediction_table,
    read_csv,
    spearman,
    to_float,
)
from analyze_tie_set_weighted_ordering import reconstruct_optimal_path, route_critical_cells
from bfs_label import compute_distance_to_goal
from model import load_mlp_heuristic, load_unet_heuristic, make_mlp_table_heuristic, make_unet_heuristic


METHODS = ["manhattan_large_g_tiebreak", "manhattan_mlp_tiebreak", "manhattan_unet_tiebreak"]
SHORT = {
    "manhattan_large_g_tiebreak": "large_g",
    "manhattan_mlp_tiebreak": "mlp",
    "manhattan_unet_tiebreak": "unet",
}
SHORT_TO_METHOD = {value: key for key, value in SHORT.items()}
ONLINE_EXCLUDED_COLUMNS = {
    "benchmark",
    "map_id",
    "seed",
    "map_mode",
    "structured_type",
    "start",
    "goal",
    "best_method",
    "best_methods",
    "best_expanded",
    "is_best_tie",
    "large_g_expanded",
    "mlp_expanded",
    "unet_expanded",
    "unet_minus_mlp_expanded",
    "unet_minus_large_g_expanded",
    "mlp_minus_large_g_expanded",
}


def write_csv(path, rows):
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def std(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / len(values))


def value_range(values):
    values = list(values)
    return max(values) - min(values) if values else 0.0


def as_numeric(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def map_key(row, source):
    return (
        source,
        row.get("seed", ""),
        row.get("map_size", ""),
        row.get("obstacle_rate", ""),
        row.get("map_mode", "random"),
        row.get("structured_type", "random"),
        row.get("start_goal_mode", "random"),
        row.get("start_row", ""),
        row.get("start_col", ""),
        row.get("goal_row", ""),
        row.get("goal_col", ""),
    )


def group_result_maps(paths):
    grouped = {}
    for source, path in paths:
        if not os.path.exists(path):
            continue
        for row in read_csv(path):
            heuristic = row.get("heuristic")
            if heuristic not in {"manhattan", *METHODS, "manhattan_true_distance_tiebreak"}:
                continue
            row = dict(row)
            row["source"] = source
            row.setdefault("map_mode", "random")
            row.setdefault("structured_type", "random")
            grouped.setdefault(map_key(row, source), {})[heuristic] = row
    output = []
    for methods in grouped.values():
        if "manhattan" not in methods:
            continue
        if all(method in methods for method in METHODS):
            output.append(methods)
    return output


def map_id(sample):
    return (
        f"{sample.get('map_mode', 'random')}_{sample.get('structured_type', 'random')}"
        f"_rate{sample['obstacle_rate']}_seed{sample['seed']}"
        f"_s{sample['start_row']}-{sample['start_col']}_g{sample['goal_row']}-{sample['goal_col']}"
    )


def result_expanded(methods, method):
    return to_float(methods[method], "expanded_nodes")


def best_label(methods):
    expanded = {method: result_expanded(methods, method) for method in METHODS}
    best_value = min(expanded.values())
    best_methods = sorted(method for method, value in expanded.items() if value == best_value)
    return {
        "best_method": SHORT[best_methods[0]],
        "best_methods": ";".join(SHORT[method] for method in best_methods),
        "best_expanded": best_value,
        "is_best_tie": int(len(best_methods) > 1),
        "large_g_expanded": expanded["manhattan_large_g_tiebreak"],
        "mlp_expanded": expanded["manhattan_mlp_tiebreak"],
        "unet_expanded": expanded["manhattan_unet_tiebreak"],
        "unet_minus_mlp_expanded": expanded["manhattan_unet_tiebreak"] - expanded["manhattan_mlp_tiebreak"],
        "unet_minus_large_g_expanded": expanded["manhattan_unet_tiebreak"]
        - expanded["manhattan_large_g_tiebreak"],
        "mlp_minus_large_g_expanded": expanded["manhattan_mlp_tiebreak"]
        - expanded["manhattan_large_g_tiebreak"],
    }


def secondary_value(method, node, g_score, mlp_table, unet_table):
    if method == "large_g":
        return -float(g_score)
    if method == "mlp":
        return mlp_table[node]
    if method == "unet":
        return unet_table[node]
    raise ValueError(f"Unknown method: {method}")


def selected_by(nodes, method, g_score, mlp_table, unet_table):
    return min(nodes, key=lambda node: (secondary_value(method, node, g_score[node], mlp_table, unet_table), node))


def pairwise_disagreement(nodes, method_a, method_b, g_score, mlp_table, unet_table):
    total = 0
    disagree = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a_i = secondary_value(method_a, nodes[i], g_score[nodes[i]], mlp_table, unet_table)
            a_j = secondary_value(method_a, nodes[j], g_score[nodes[j]], mlp_table, unet_table)
            b_i = secondary_value(method_b, nodes[i], g_score[nodes[i]], mlp_table, unet_table)
            b_j = secondary_value(method_b, nodes[j], g_score[nodes[j]], mlp_table, unet_table)
            a_diff = a_i - a_j
            b_diff = b_i - b_j
            if a_diff == 0 or b_diff == 0:
                continue
            total += 1
            if a_diff * b_diff < 0:
                disagree += 1
    return disagree, total


def update_tie_stats(stats, active_nodes, current_g_score, mlp_table, unet_table, expanded_step):
    stats["tie_event_count"] += 1
    stats["tie_set_sizes"].append(len(active_nodes))
    stats["tie_steps"].append(expanded_step)
    for left, right in [("unet", "mlp"), ("unet", "large_g"), ("mlp", "large_g")]:
        left_choice = selected_by(active_nodes, left, current_g_score, mlp_table, unet_table)
        right_choice = selected_by(active_nodes, right, current_g_score, mlp_table, unet_table)
        if left_choice != right_choice:
            stats[f"{left}_{right}_choice_disagreement"] += 1
        disagree, total = pairwise_disagreement(active_nodes, left, right, current_g_score, mlp_table, unet_table)
        stats[f"{left}_{right}_pairwise_disagree"] += disagree
        stats[f"{left}_{right}_pairwise_total"] += total


def replay_early(grid, start, goal, policy, mlp_table, unet_table, max_steps):
    open_set = []
    counter = 0
    g_score = {start: 0}
    heapq.heappush(
        open_set,
        (
            manhattan(start, goal),
            secondary_value(policy, start, 0, mlp_table, unet_table),
            counter,
            0,
            start,
        ),
    )
    expanded_order = []
    frontier_snapshots = []
    tie_stats = {
        "tie_event_count": 0,
        "tie_set_sizes": [],
        "tie_steps": [],
        "unet_mlp_choice_disagreement": 0,
        "unet_large_g_choice_disagreement": 0,
        "mlp_large_g_choice_disagreement": 0,
        "unet_mlp_pairwise_disagree": 0,
        "unet_mlp_pairwise_total": 0,
        "unet_large_g_pairwise_disagree": 0,
        "unet_large_g_pairwise_total": 0,
        "mlp_large_g_pairwise_disagree": 0,
        "mlp_large_g_pairwise_total": 0,
    }

    while open_set and len(expanded_order) < max_steps:
        f_primary, _, _, current_g, current = heapq.heappop(open_set)
        if current_g > g_score.get(current, float("inf")):
            continue

        active_nodes = [current]
        active_g = {current: current_g}
        for queued_f, _, _, queued_g, queued_node in open_set:
            if queued_f == f_primary and queued_g == g_score.get(queued_node, float("inf")):
                active_nodes.append(queued_node)
                active_g[queued_node] = queued_g
        if len(active_nodes) >= 2:
            update_tie_stats(tie_stats, active_nodes, active_g, mlp_table, unet_table, len(expanded_order))

        expanded_order.append(current)
        frontier_snapshots.append([item[4] for item in open_set])
        if current == goal:
            break

        for neighbor in neighbors(grid, current):
            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                counter += 1
                heapq.heappush(
                    open_set,
                    (
                        tentative_g + manhattan(neighbor, goal),
                        secondary_value(policy, neighbor, tentative_g, mlp_table, unet_table),
                        counter,
                        tentative_g,
                        neighbor,
                    ),
                )

    return expanded_order, frontier_snapshots, tie_stats


def spatial_spread(cells):
    if not cells:
        return {
            "row_std": 0.0,
            "col_std": 0.0,
            "bbox_area": 0.0,
            "compactness": 0.0,
        }
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    return {
        "row_std": std(rows),
        "col_std": std(cols),
        "bbox_area": float(bbox_area),
        "compactness": len(cells) / bbox_area if bbox_area else 0.0,
    }


def direction_consistency(cells):
    if len(cells) < 3:
        return 0.0
    directions = []
    for prev, current in zip(cells, cells[1:]):
        dr = current[0] - prev[0]
        dc = current[1] - prev[1]
        length = math.sqrt(dr * dr + dc * dc)
        if length > 0:
            directions.append((dr / length, dc / length))
    if len(directions) < 2:
        return 0.0
    dots = [a[0] * b[0] + a[1] * b[1] for a, b in zip(directions, directions[1:])]
    return mean(dots)


def heuristic_smoothness(cells, table):
    if len(cells) < 2:
        return 0.0
    diffs = [abs(table[a] - table[b]) for a, b in zip(cells, cells[1:]) if a in table and b in table]
    return mean(diffs)


def trajectory_online_features(prefix, expanded, frontier_snapshots, goal, mlp_table, unet_table):
    manhattan_values = [manhattan(cell, goal) for cell in expanded]
    unet_values = [unet_table[cell] for cell in expanded if cell in unet_table]
    mlp_values = [mlp_table[cell] for cell in expanded if cell in mlp_table]
    first = manhattan_values[0] if manhattan_values else 0.0
    last = manhattan_values[-1] if manhattan_values else first
    deltas = [a - b for a, b in zip(manhattan_values, manhattan_values[1:])]
    frontier_sizes = [len(frontier) for frontier in frontier_snapshots]
    last_frontier = frontier_snapshots[-1] if frontier_snapshots else []
    spread = spatial_spread(expanded)
    frontier_spread = spatial_spread(last_frontier)
    features = {
        f"{prefix}_early_expanded_count": float(len(expanded)),
        f"{prefix}_early_goal_progress": first - last,
        f"{prefix}_early_goal_progress_rate": (first - last) / max(1, len(expanded) - 1),
        f"{prefix}_early_mean_manhattan": mean(manhattan_values),
        f"{prefix}_early_final_manhattan": last,
        f"{prefix}_early_mean_step_progress": mean(deltas),
        f"{prefix}_early_progress_std": std(deltas),
        f"{prefix}_early_direction_consistency": direction_consistency(expanded),
        f"{prefix}_early_row_std": spread["row_std"],
        f"{prefix}_early_col_std": spread["col_std"],
        f"{prefix}_early_bbox_area": spread["bbox_area"],
        f"{prefix}_early_compactness": spread["compactness"],
        f"{prefix}_early_frontier_mean_size": mean(frontier_sizes),
        f"{prefix}_early_frontier_final_size": frontier_sizes[-1] if frontier_sizes else 0.0,
        f"{prefix}_early_frontier_row_std": frontier_spread["row_std"],
        f"{prefix}_early_frontier_col_std": frontier_spread["col_std"],
        f"{prefix}_early_frontier_bbox_area": frontier_spread["bbox_area"],
        f"{prefix}_unet_h_mean": mean(unet_values),
        f"{prefix}_unet_h_std": std(unet_values),
        f"{prefix}_unet_h_range": value_range(unet_values),
        f"{prefix}_mlp_h_mean": mean(mlp_values),
        f"{prefix}_mlp_h_std": std(mlp_values),
        f"{prefix}_mlp_h_range": value_range(mlp_values),
        f"{prefix}_unet_mlp_h_absdiff_mean": mean(abs(unet_table[cell] - mlp_table[cell]) for cell in expanded),
        f"{prefix}_unet_smoothness": heuristic_smoothness(expanded, unet_table),
        f"{prefix}_mlp_smoothness": heuristic_smoothness(expanded, mlp_table),
    }
    return features


def tie_online_features(prefix, stats, max_steps):
    tie_count = stats["tie_event_count"]
    features = {
        f"{prefix}_tie_event_count": float(tie_count),
        f"{prefix}_tie_event_fraction": tie_count / max(1, max_steps),
        f"{prefix}_tie_set_mean_size": mean(stats["tie_set_sizes"]),
        f"{prefix}_tie_set_max_size": max(stats["tie_set_sizes"]) if stats["tie_set_sizes"] else 0.0,
        f"{prefix}_tie_set_step_mean": mean(stats["tie_steps"]),
    }
    for left, right in [("unet", "mlp"), ("unet", "large_g"), ("mlp", "large_g")]:
        choice_key = f"{left}_{right}_choice_disagreement"
        disagree_key = f"{left}_{right}_pairwise_disagree"
        total_key = f"{left}_{right}_pairwise_total"
        features[f"{prefix}_{left}_{right}_choice_disagreement_rate"] = stats[choice_key] / max(1, tie_count)
        features[f"{prefix}_{left}_{right}_pairwise_disagreement_rate"] = stats[disagree_key] / max(1, stats[total_key])
    return features


def online_features_for_map(methods, mlp_model, unet_model, max_steps):
    sample = methods["manhattan"]
    grid, start, goal = build_grid(sample)
    mlp_table = prediction_table(grid, goal, make_mlp_table_heuristic(mlp_model, grid, goal))
    unet_table = prediction_table(grid, goal, make_unet_heuristic(unet_model, grid, goal))
    labels = best_label(methods)
    row = {
        "benchmark": sample["source"],
        "map_id": map_id(sample),
        "seed": sample["seed"],
        "map_mode": sample.get("map_mode", "random"),
        "structured_type": sample.get("structured_type", "random"),
        "start": f"{sample['start_row']},{sample['start_col']}",
        "goal": f"{sample['goal_row']},{sample['goal_col']}",
        **labels,
    }
    replay_data = {}
    for policy in ["large_g", "mlp", "unet"]:
        expanded, frontier_snapshots, tie_stats = replay_early(grid, start, goal, policy, mlp_table, unet_table, max_steps)
        replay_data[policy] = expanded
        row.update(trajectory_online_features(policy, expanded, frontier_snapshots, goal, mlp_table, unet_table))
        row.update(tie_online_features(policy, tie_stats, max_steps))

    row["unet_minus_mlp_early_goal_progress_rate"] = (
        row["unet_early_goal_progress_rate"] - row["mlp_early_goal_progress_rate"]
    )
    row["unet_minus_large_g_early_goal_progress_rate"] = (
        row["unet_early_goal_progress_rate"] - row["large_g_early_goal_progress_rate"]
    )
    row["mlp_minus_large_g_early_goal_progress_rate"] = (
        row["mlp_early_goal_progress_rate"] - row["large_g_early_goal_progress_rate"]
    )
    row["unet_minus_mlp_early_spread"] = row["unet_early_bbox_area"] - row["mlp_early_bbox_area"]
    row["unet_minus_large_g_early_spread"] = row["unet_early_bbox_area"] - row["large_g_early_bbox_area"]
    row["unet_minus_mlp_h_smoothness"] = row["unet_unet_smoothness"] - row["mlp_unet_smoothness"]
    return row, replay_data, grid, start, goal, mlp_table, unet_table


def oracle_features_for_map(online_row, replay_data, grid, start, goal, mlp_table, unet_table):
    distance_grid = compute_distance_to_goal(grid, goal)
    path = reconstruct_optimal_path(grid, distance_grid, start, goal)
    path_dist = distance_to_path_table(grid, path)
    critical = route_critical_cells(grid, path, 2) if path else set()
    true_table = {
        (r, c): float(value)
        for r, line in enumerate(distance_grid)
        for c, value in enumerate(line)
        if value >= 0
    }
    row = {
        key: online_row[key]
        for key in [
            "benchmark",
            "map_id",
            "seed",
            "map_mode",
            "structured_type",
            "start",
            "goal",
            "best_method",
            "best_methods",
        ]
    }
    for policy, expanded in replay_data.items():
        distances = [path_dist.get(cell, 0.0) for cell in expanded]
        off_path = [1.0 if value > 1 else 0.0 for value in distances]
        route_near = [1.0 if value <= 1 else 0.0 for value in distances]
        true_values = [true_table.get(cell, 0.0) for cell in expanded]
        row[f"{policy}_oracle_early_off_path_ratio"] = mean(off_path)
        row[f"{policy}_oracle_early_route_concentration"] = mean(route_near)
        row[f"{policy}_oracle_mean_distance_to_path"] = mean(distances)
        row[f"{policy}_oracle_mean_true_distance"] = mean(true_values)
    for name, table in [("mlp", mlp_table), ("unet", unet_table)]:
        critical_values = [table[cell] - true_table[cell] for cell in critical if cell in table and cell in true_table]
        row[f"{name}_oracle_critical_overestimate_rate"] = mean(1.0 if value > 0 else 0.0 for value in critical_values)
        row[f"{name}_oracle_critical_large_overestimate_rate"] = mean(
            1.0 if value > 2.0 else 0.0 for value in critical_values
        )
        row[f"{name}_oracle_critical_mean_error"] = mean(critical_values)
    row["unet_minus_mlp_oracle_early_off_path_ratio"] = (
        row["unet_oracle_early_off_path_ratio"] - row["mlp_oracle_early_off_path_ratio"]
    )
    row["unet_minus_mlp_oracle_route_concentration"] = (
        row["unet_oracle_early_route_concentration"] - row["mlp_oracle_early_route_concentration"]
    )
    row["unet_minus_mlp_oracle_critical_large_overestimate_rate"] = (
        row["unet_oracle_critical_large_overestimate_rate"] - row["mlp_oracle_critical_large_overestimate_rate"]
    )
    return row


def feature_names(rows, excluded):
    names = []
    for key in sorted(set().union(*(row.keys() for row in rows))):
        if key in excluded:
            continue
        values = [as_numeric(row.get(key)) for row in rows]
        if any(value is not None for value in values):
            names.append(key)
    return names


def matrix(rows, names):
    output = []
    for row in rows:
        output.append([as_numeric(row.get(name)) or 0.0 for name in names])
    return output


def stratified_folds(labels, folds, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[label].append(index)
    fold_indices = [[] for _ in range(folds)]
    for indices in by_label.values():
        rng.shuffle(indices)
        for i, index in enumerate(indices):
            fold_indices[i % folds].append(index)
    return fold_indices


def standardize(train_x, test_x):
    cols = len(train_x[0]) if train_x else 0
    means = []
    scales = []
    for col in range(cols):
        values = [row[col] for row in train_x]
        m = mean(values)
        s = std(values) or 1.0
        means.append(m)
        scales.append(s)
    return (
        [[(value - means[i]) / scales[i] for i, value in enumerate(row)] for row in train_x],
        [[(value - means[i]) / scales[i] for i, value in enumerate(row)] for row in test_x],
    )


def majority_label(labels):
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def nearest_centroid_predict(train_x, train_y, test_x):
    train_x, test_x = standardize(train_x, test_x)
    centroids = {}
    for label in sorted(set(train_y)):
        rows = [row for row, row_label in zip(train_x, train_y) if row_label == label]
        centroids[label] = [mean(row[col] for row in rows) for col in range(len(train_x[0]))]
    predictions = []
    for row in test_x:
        predictions.append(
            min(
                centroids,
                key=lambda label: sum((row[col] - centroids[label][col]) ** 2 for col in range(len(row))),
            )
        )
    return predictions, {}


def decision_stump_predict(train_x, train_y, test_x, names):
    best = None
    majority = majority_label(train_y)
    for col, name in enumerate(names):
        values = sorted(set(row[col] for row in train_x))
        if len(values) < 2:
            continue
        thresholds = [(left + right) / 2.0 for left, right in zip(values, values[1:])]
        if len(thresholds) > 100:
            step = max(1, len(thresholds) // 100)
            thresholds = thresholds[::step]
        for threshold in thresholds:
            left_labels = [label for row, label in zip(train_x, train_y) if row[col] <= threshold]
            right_labels = [label for row, label in zip(train_x, train_y) if row[col] > threshold]
            if not left_labels or not right_labels:
                continue
            left_majority = majority_label(left_labels)
            right_majority = majority_label(right_labels)
            predictions = [left_majority if row[col] <= threshold else right_majority for row in train_x]
            accuracy = sum(int(pred == label) for pred, label in zip(predictions, train_y)) / len(train_y)
            candidate = (accuracy, name, col, threshold, left_majority, right_majority)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return [majority for _ in test_x], {"selected_feature": "", "threshold": ""}
    _, name, col, threshold, left_majority, right_majority = best
    return [left_majority if row[col] <= threshold else right_majority for row in test_x], {
        "selected_feature": name,
        "threshold": threshold,
    }


def macro_f1(labels, predictions, classes):
    scores = []
    for cls in classes:
        tp = sum(1 for y, pred in zip(labels, predictions) if y == cls and pred == cls)
        fp = sum(1 for y, pred in zip(labels, predictions) if y != cls and pred == cls)
        fn = sum(1 for y, pred in zip(labels, predictions) if y == cls and pred != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return mean(scores)


def evaluate_predictions(labels, predictions):
    classes = sorted(set(labels) | set(predictions))
    return {
        "accuracy": sum(int(y == pred) for y, pred in zip(labels, predictions)) / len(labels) if labels else 0.0,
        "macro_f1": macro_f1(labels, predictions, classes),
    }


def cross_validate(rows, names, model_name, folds, seed):
    labels = [row["best_method"] for row in rows]
    x = matrix(rows, names)
    fold_indices = stratified_folds(labels, folds, seed)
    predictions = [None] * len(rows)
    metadata = []
    for fold, test_indices in enumerate(fold_indices):
        if not test_indices:
            continue
        train_indices = [index for index in range(len(rows)) if index not in set(test_indices)]
        train_x = [x[index] for index in train_indices]
        train_y = [labels[index] for index in train_indices]
        test_x = [x[index] for index in test_indices]
        if model_name == "nearest_centroid":
            fold_predictions, fold_meta = nearest_centroid_predict(train_x, train_y, test_x)
        elif model_name == "decision_stump":
            fold_predictions, fold_meta = decision_stump_predict(train_x, train_y, test_x, names)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        for index, prediction in zip(test_indices, fold_predictions):
            predictions[index] = prediction
        metadata.append({"fold": fold, **fold_meta})
    metrics = evaluate_predictions(labels, predictions)
    return metrics, predictions, metadata


def baseline_predictions(labels, name, seed):
    if name == "random_guess":
        rng = random.Random(seed)
        classes = sorted(set(labels))
        return [rng.choice(classes) for _ in labels]
    if name.startswith("always_"):
        return [name.replace("always_", "") for _ in labels]
    raise ValueError(f"Unknown baseline: {name}")


def confusion_rows(labels, predictions, model_name):
    classes = sorted(set(labels) | set(predictions))
    rows = []
    for true_label in classes:
        for predicted_label in classes:
            rows.append(
                {
                    "model": model_name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": sum(
                        1 for y, prediction in zip(labels, predictions) if y == true_label and prediction == predicted_label
                    ),
                }
            )
    return rows


def feature_correlations(rows, names):
    targets = {
        "is_unet_best": [1.0 if row["best_method"] == "unet" else 0.0 for row in rows],
        "is_mlp_best": [1.0 if row["best_method"] == "mlp" else 0.0 for row in rows],
        "is_large_g_best": [1.0 if row["best_method"] == "large_g" else 0.0 for row in rows],
        "unet_minus_mlp_expanded": [as_numeric(row["unet_minus_mlp_expanded"]) or 0.0 for row in rows],
        "unet_minus_large_g_expanded": [as_numeric(row["unet_minus_large_g_expanded"]) or 0.0 for row in rows],
        "mlp_minus_large_g_expanded": [as_numeric(row["mlp_minus_large_g_expanded"]) or 0.0 for row in rows],
    }
    output = []
    for name in names:
        xs = [as_numeric(row.get(name)) or 0.0 for row in rows]
        for target, ys in targets.items():
            output.append(
                {
                    "feature": name,
                    "target": target,
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                    "mean": mean(xs),
                    "std": std(xs),
                }
            )
    return output


def evaluate_by_structure(rows, predictions, model_name):
    output = []
    labels = [row["best_method"] for row in rows]
    for structure in sorted(set(row.get("structured_type", "unknown") for row in rows)):
        indices = [i for i, row in enumerate(rows) if row.get("structured_type") == structure]
        if not indices:
            continue
        scoped_labels = [labels[i] for i in indices]
        scoped_predictions = [predictions[i] for i in indices]
        metrics = evaluate_predictions(scoped_labels, scoped_predictions)
        output.append({"model": model_name, "structured_type": structure, "n": len(indices), **metrics})
    return output


def save_feature_importance(output_path, metadata):
    counts = Counter(item.get("selected_feature") for item in metadata if item.get("selected_feature"))
    if not counts:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    labels = [item[0] for item in counts.most_common(12)]
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], values[::-1])
    ax.set_xlabel("CV folds selected by decision stump")
    ax.set_title("Online Feature Importance Proxy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def write_summary(path, online_rows, oracle_rows, model_rows, baseline_rows, corr_rows, structure_rows, unavailable):
    best_counts = Counter(row["best_method"] for row in online_rows)
    online_model_rows = [row for row in model_rows if row.get("feature_group") == "online"]
    oracle_model_rows = [row for row in model_rows if row.get("feature_group") == "oracle_only"]
    best_model = max(online_model_rows, key=lambda row: float(row["accuracy"])) if online_model_rows else {}
    best_oracle_model = max(oracle_model_rows, key=lambda row: float(row["accuracy"])) if oracle_model_rows else {}
    best_baseline = max(baseline_rows, key=lambda row: float(row["accuracy"])) if baseline_rows else {}
    top_corrs = sorted(
        [
            row
            for row in corr_rows
            if row.get("feature_group") == "online"
            and row["target"] in {"is_unet_best", "unet_minus_mlp_expanded"}
        ],
        key=lambda row: abs(float(row["spearman"])),
        reverse=True,
    )[:10]
    top_oracle_corrs = sorted(
        [
            row
            for row in corr_rows
            if row.get("feature_group") == "oracle_only"
            and row["target"] in {"is_unet_best", "unet_minus_mlp_expanded"}
        ],
        key=lambda row: abs(float(row["spearman"])),
        reverse=True,
    )[:8]
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Trust-Signal Validation\n\n")
        file.write(
            "This offline analysis tests whether structure-agnostic, online-feasible early-search signals can predict which "
            "Manhattan-primary secondary guidance should be trusted. It does not implement an adaptive search algorithm.\n\n"
        )
        file.write("## Feature Separation\n\n")
        file.write(
            "Online-feasible features use only early expanded coordinates, Manhattan distance, MLP/U-Net predicted values, "
            "frontier/tie-set statistics, and ordering disagreement among nodes sharing the same primary f-value.\n\n"
        )
        file.write(
            "Oracle-only diagnostic features use true shortest-path or true-distance information and are reported separately. "
            "They are not mixed into the main predictive model.\n\n"
        )
        if unavailable:
            file.write("Unavailable or intentionally excluded signals:\n\n")
            for item in unavailable:
                file.write(f"- {item}\n")
            file.write("\n")

        file.write("## Label Distribution\n\n")
        file.write("| best method | maps |\n|---|---:|\n")
        for label, count in sorted(best_counts.items()):
            file.write(f"| {label} | {count} |\n")
        file.write("\n")

        file.write("## Baselines\n\n")
        file.write("| baseline | accuracy | macro-F1 |\n|---|---:|---:|\n")
        for row in baseline_rows:
            file.write(f"| {row['model']} | {float(row['accuracy']):.3f} | {float(row['macro_f1']):.3f} |\n")
        file.write("\n")

        file.write("## Diagnostic Models Using Online Features Only\n\n")
        file.write("| model | feature group | accuracy | macro-F1 | features |\n|---|---|---:|---:|---:|\n")
        for row in online_model_rows:
            file.write(
                f"| {row['model']} | {row['feature_group']} | {float(row['accuracy']):.3f} "
                f"| {float(row['macro_f1']):.3f} | {row['features']} |\n"
            )
        file.write("\n")

        if oracle_model_rows:
            file.write("## Oracle-Only Diagnostic Models\n\n")
            file.write(
                "These models use true-distance or optimal-path features and are reported only as diagnostic upper-bound evidence.\n\n"
            )
            file.write("| model | feature group | accuracy | macro-F1 | features |\n|---|---|---:|---:|---:|\n")
            for row in oracle_model_rows:
                file.write(
                    f"| {row['model']} | {row['feature_group']} | {float(row['accuracy']):.3f} "
                    f"| {float(row['macro_f1']):.3f} | {row['features']} |\n"
                )
            file.write("\n")

        file.write("## Strongest Online Correlations\n\n")
        file.write("| feature | target | Spearman | Pearson |\n|---|---|---:|---:|\n")
        for row in top_corrs:
            file.write(
                f"| {row['feature']} | {row['target']} | {float(row['spearman']):.3f} "
                f"| {float(row['pearson']):.3f} |\n"
            )
        file.write("\n")

        if top_oracle_corrs:
            file.write("## Strongest Oracle-Only Correlations\n\n")
            file.write("| feature | target | Spearman | Pearson |\n|---|---|---:|---:|\n")
            for row in top_oracle_corrs:
                file.write(
                    f"| {row['feature']} | {row['target']} | {float(row['spearman']):.3f} "
                    f"| {float(row['pearson']):.3f} |\n"
                )
            file.write("\n")

        if structure_rows:
            file.write("## Post-Hoc Structure-Stratified Performance\n\n")
            file.write("Structure labels are used only here for reporting, not as prediction features.\n\n")
            file.write("| model | structure | n | accuracy | macro-F1 |\n|---|---|---:|---:|---:|\n")
            for row in structure_rows:
                file.write(
                    f"| {row['model']} | {row['structured_type']} | {row['n']} | "
                    f"{float(row['accuracy']):.3f} | {float(row['macro_f1']):.3f} |\n"
                )
            file.write("\n")

        file.write("## Critical Answers\n\n")
        margin = float(best_model.get("accuracy", 0.0)) - float(best_baseline.get("accuracy", 0.0))
        file.write(
            f"- Best online model: {best_model.get('model', 'n/a')} with accuracy "
            f"{float(best_model.get('accuracy', 0.0)):.3f} and macro-F1 {float(best_model.get('macro_f1', 0.0)):.3f}.\n"
        )
        file.write(
            f"- Best simple baseline: {best_baseline.get('model', 'n/a')} with accuracy "
            f"{float(best_baseline.get('accuracy', 0.0)):.3f}; online-model margin is {margin:.3f}.\n"
        )
        if margin >= 0.05:
            file.write(
                "- Evidence for online trust signals: positive but still diagnostic. The result suggests early-search behavior contains "
                "some structure-agnostic signal for choosing guidance.\n"
            )
        else:
            file.write(
                "- Evidence for online trust signals: weak. The online models do not beat simple baselines by a meaningful margin, "
                "so adaptive trust-based tie-breaking should not be prioritized yet.\n"
            )
        file.write(
            "- Map structure labels were not used as input features. Any structure-specific results are post-hoc diagnostics only.\n"
        )
        if oracle_rows:
            oracle_delta = float(best_oracle_model.get("accuracy", 0.0)) - float(best_model.get("accuracy", 0.0))
            file.write(
                "- Oracle-only features are available in a separate file for interpretation and upper-bound diagnostics, but they are not "
                "part of the main online-feasible model.\n"
            )
            file.write(
                f"- Best oracle-only diagnostic accuracy is {float(best_oracle_model.get('accuracy', 0.0)):.3f}, "
                f"which differs from the best online model by {oracle_delta:.3f}.\n"
            )
        file.write(
            "- Interpretation: this is a go/no-go validation for adaptive neural guidance, not an algorithm proposal.\n"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Validate online-feasible trust signals for tie-breaking guidance.")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_tiebreak_controls_100.csv")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--early-steps", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/trust_signal_validation")
    parser.add_argument("--skip-oracle", action="store_true")
    return parser.parse_args()


def main(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    result_paths = [
        ("random", os.path.join(project_root, args.random_results)),
        ("structured", os.path.join(project_root, args.structured_results)),
    ]
    grouped = group_result_maps(result_paths)
    if not grouped:
        raise RuntimeError("No complete tie-breaking result groups found.")

    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint, "unet"))

    online_rows = []
    oracle_rows = []
    for index, methods in enumerate(grouped):
        online_row, replay_data, grid, start, goal, mlp_table, unet_table = online_features_for_map(
            methods, mlp_model, unet_model, args.early_steps
        )
        online_rows.append(online_row)
        if not args.skip_oracle:
            oracle_rows.append(oracle_features_for_map(online_row, replay_data, grid, start, goal, mlp_table, unet_table))
        if (index + 1) % 250 == 0:
            print(f"Analyzed {index + 1}/{len(grouped)} maps")

    online_path = os.path.join(output_dir, "online_features.csv")
    oracle_path = os.path.join(output_dir, "oracle_diagnostic_features.csv")
    write_csv(online_path, online_rows)
    if oracle_rows:
        write_csv(oracle_path, oracle_rows)

    online_names = feature_names(online_rows, ONLINE_EXCLUDED_COLUMNS)
    oracle_names = feature_names(
        oracle_rows,
        {
            "benchmark",
            "map_id",
            "seed",
            "map_mode",
            "structured_type",
            "start",
            "goal",
            "best_method",
            "best_methods",
        },
    )

    labels = [row["best_method"] for row in online_rows]
    baseline_rows = []
    all_confusion = []
    for baseline in ["always_mlp", "always_unet", "always_large_g", "random_guess"]:
        predictions = baseline_predictions(labels, baseline, args.seed)
        metrics = evaluate_predictions(labels, predictions)
        baseline_rows.append({"model": baseline, **metrics})
        all_confusion.extend(confusion_rows(labels, predictions, baseline))

    model_rows = []
    model_predictions = {}
    model_metadata = {}
    for model_name in ["nearest_centroid", "decision_stump"]:
        metrics, predictions, metadata = cross_validate(online_rows, online_names, model_name, args.folds, args.seed)
        model_rows.append({"model": model_name, "feature_group": "online", "features": len(online_names), **metrics})
        model_predictions[model_name] = predictions
        model_metadata[model_name] = metadata
        all_confusion.extend(confusion_rows(labels, predictions, model_name))

    if oracle_rows and oracle_names:
        for model_name in ["nearest_centroid", "decision_stump"]:
            metrics, predictions, _ = cross_validate(oracle_rows, oracle_names, model_name, args.folds, args.seed)
            model_rows.append({"model": model_name, "feature_group": "oracle_only", "features": len(oracle_names), **metrics})
            all_confusion.extend(confusion_rows(labels, predictions, f"{model_name}_oracle_only"))

    corr_rows = feature_correlations(online_rows, online_names)
    if oracle_rows:
        for row in feature_correlations([{**online, **oracle} for online, oracle in zip(online_rows, oracle_rows)], oracle_names):
            row["feature_group"] = "oracle_only"
            corr_rows.append(row)
    for row in corr_rows:
        row.setdefault("feature_group", "online")

    best_online_model = max(
        [row for row in model_rows if row["feature_group"] == "online"],
        key=lambda row: float(row["accuracy"]),
    )["model"]
    structure_rows = evaluate_by_structure(online_rows, model_predictions[best_online_model], best_online_model)

    write_csv(os.path.join(output_dir, "model_results.csv"), model_rows)
    write_csv(os.path.join(output_dir, "baseline_results.csv"), baseline_rows)
    write_csv(os.path.join(output_dir, "confusion_matrix.csv"), all_confusion)
    write_csv(os.path.join(output_dir, "feature_correlations.csv"), corr_rows)
    write_csv(os.path.join(output_dir, "posthoc_structure_results.csv"), structure_rows)
    save_feature_importance(
        os.path.join(output_dir, "feature_importance.png"),
        model_metadata.get("decision_stump", []),
    )

    unavailable = [
        "Final expanded-node gaps are not used as online features because they would leak the target.",
        "Map structure labels are not used as model inputs; they are only used for post-hoc reporting.",
        "True optimal-path and true-distance features are written separately as oracle diagnostics.",
    ]
    write_summary(
        os.path.join(output_dir, "trust_signal_summary.md"),
        online_rows,
        oracle_rows,
        model_rows,
        baseline_rows,
        corr_rows,
        structure_rows,
        unavailable,
    )
    print(f"Saved trust-signal validation outputs to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
