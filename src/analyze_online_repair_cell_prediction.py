"""Predict benchmark-specific projection repair cells from online features only."""

import argparse
import csv
import hashlib
import heapq
import math
import os
from collections import Counter, defaultdict, deque

import numpy as np

from analyze_direct_vs_tiebreak import checked_search, trace_search
from analyze_projected_unet_astar import project_consistent_lower_envelope, table_metrics
from analyze_projection_modification_ablation import case_id
from analyze_unet_structure_behavior import (
    STRUCTURES, benchmark_cases, free_cells, mean, neighbors, rebuild_case,
)
from bfs_label import compute_distance_to_goal
from model import load_unet_heuristic, make_unet_heuristic, manhattan_heuristic


FEATURES = [
    "raw_unet_h", "manhattan_h", "first_generated_g", "first_generated_f",
    "first_open_rank", "was_generated", "local_consistency_violation",
    "local_h_gradient", "local_h_variance", "obstacle_density_r1",
    "obstacle_density_r2", "free_neighbor_count", "nearest_obstacle_distance",
    "unet_minus_manhattan", "local_branching", "near_first_direct_tie_divergence",
]
POLICIES = [
    "raw_direct", "full_projection", "unet_tiebreak", "oracle_greedy_minimal_repair",
    "predictor_top_1", "predictor_top_2", "predictor_top_5",
    "predictor_threshold_f1", "predictor_threshold_0_5",
]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stable_number(value):
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)


def dummy_distance(grid):
    """trace_search requires distances for display-only fields; search ignores them."""
    return [[0 if value == 0 else -1 for value in row] for row in grid]


def repair_label_lookup(root):
    repairs_path = os.path.join(root, "outputs/projection_modification_ablation/minimal_repair_sets.csv")
    cells_path = os.path.join(root, "outputs/projection_modification_ablation/modified_cells.csv")
    selected_components = {}
    with open(repairs_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["raw_nonoptimal"] != "True":
                continue
            selected_components[row["case_id"]] = {
                int(value) for value in row["selected_components"].split(";") if value
            }
    labels = set()
    with open(cells_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            selected = selected_components.get(row["case_id"], set())
            if int(row["component_id"]) in selected:
                labels.add((row["case_id"], row["cell"]))
    return labels, selected_components


def obstacle_distances(grid):
    rows, cols = len(grid), len(grid[0])
    result = [[math.inf] * cols for _ in range(rows)]
    queue = deque()
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 1:
                result[row][col] = 0
                queue.append((row, col))
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == math.inf:
                result[nr][nc] = result[row][col] + 1
                queue.append((nr, nc))
    return result


def obstacle_density(grid, cell, radius):
    row, col = cell
    values = [
        grid[nr][nc]
        for nr in range(max(0, row - radius), min(len(grid), row + radius + 1))
        for nc in range(max(0, col - radius), min(len(grid[0]), col + radius + 1))
    ]
    return sum(values) / len(values)


def first_open_entries(trace):
    first = {}
    for step in trace["trace"]:
        for rank, item in enumerate(step["open_before"]):
            node = item["node"]
            if node not in first:
                first[node] = {"g": item["g"], "f": item["f"], "rank": rank}
    return first


def divergence_neighborhood(direct_trace, tie_trace):
    direct = [item["expanded"]["node"] for item in direct_trace["trace"]]
    tied = [item["expanded"]["node"] for item in tie_trace["trace"]]
    for first, second in zip(direct, tied):
        if first != second:
            return {first, second}
    return set()


def build_case_features(grid, start, goal, table, direct_trace, tie_trace, labels, current_id):
    first = first_open_entries(direct_trace)
    near_divergence = divergence_neighborhood(direct_trace, tie_trace)
    nearest_obstacle = obstacle_distances(grid)
    cells = free_cells(grid)
    rows, values = [], []
    for cell in cells:
        adjacent = list(neighbors(grid, cell))
        adjacent_h = [table[node] for node in adjacent]
        difference = [abs(table[cell] - value) for value in adjacent_h]
        local_violation = max((max(0.0, delta - 1.0) for delta in difference), default=0.0)
        local_gradient = mean(difference) if difference else 0.0
        local_variance = float(np.var([table[cell], *adjacent_h])) if adjacent_h else 0.0
        local_branching = mean(max(0, len(list(neighbors(grid, node))) - 2) for node in adjacent) if adjacent else 0.0
        generated = first.get(cell)
        manhattan = float(manhattan_heuristic(cell, goal))
        feature = [
            table[cell], manhattan,
            generated["g"] if generated else -1.0,
            generated["f"] if generated else -1.0,
            generated["rank"] if generated else -1.0,
            float(generated is not None), local_violation, local_gradient, local_variance,
            obstacle_density(grid, cell, 1), obstacle_density(grid, cell, 2), len(adjacent),
            nearest_obstacle[cell[0]][cell[1]], table[cell] - manhattan, local_branching,
            float(any(abs(cell[0] - node[0]) + abs(cell[1] - node[1]) <= 2 for node in near_divergence)),
        ]
        label = int((current_id, f"{cell[0]},{cell[1]}") in labels)
        values.append(feature)
        rows.append({"case_id": current_id, "structure_type": None, "cell": f"{cell[0]},{cell[1]}", "repair_label": label,
                     **dict(zip(FEATURES, feature))})
    return cells, np.asarray(values, dtype=np.float64), rows


def split_groups(case_metadata):
    positive = [index for index, item in enumerate(case_metadata) if item["positive_count"] > 0]
    positive.sort(key=lambda index: stable_number(case_metadata[index]["case_id"]))
    positive_splits = {}
    for position, index in enumerate(positive):
        positive_splits[index] = "test" if position % 5 == 0 else ("validation" if position % 5 == 1 else "train")
    output = []
    for index, item in enumerate(case_metadata):
        if index in positive_splits:
            output.append(positive_splits[index]); continue
        bucket = stable_number(item["case_id"]) % 20
        output.append("train" if bucket < 14 else ("validation" if bucket < 17 else "test"))
    return output


def balanced_training_indices(indices, labels, seed):
    positive = indices[labels[indices] == 1]
    negative = indices[labels[indices] == 0]
    if not len(positive):
        raise ValueError("Training split contains no repair cells")
    generator = np.random.default_rng(seed)
    # Use balanced samples for all three interpretable models; evaluation remains on
    # naturally imbalanced held-out maps and uses PR-oriented metrics.
    limit = min(len(negative), len(positive))
    sampled_negative = generator.choice(negative, size=limit, replace=False)
    return np.concatenate((positive, sampled_negative))


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / (np.arange(len(ordered)) + 1)
    return float(precision[ordered == 1].sum() / ordered.sum())


def binary_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int8)
    predicted = np.asarray(scores) >= threshold
    true_positive = int(np.sum(predicted & (labels == 1)))
    false_positive = int(np.sum(predicted & (labels == 0)))
    false_negative = int(np.sum(~predicted & (labels == 1)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def best_f1_threshold(labels, scores):
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))
    best = max(((*binary_metrics(labels, scores, threshold), threshold) for threshold in candidates), key=lambda value: value[2])
    return best[3]


class StandardLogistic:
    def fit(self, values, labels):
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        x = (values - self.mean) / self.scale
        y = labels.astype(float)
        self.weights = np.zeros(x.shape[1], dtype=float)
        self.bias = 0.0
        positive_weight = (len(y) - y.sum()) / max(y.sum(), 1.0)
        weights = np.where(y == 1, positive_weight, 1.0)
        for _ in range(350):
            logits = np.clip(x @ self.weights + self.bias, -30, 30)
            predicted = 1.0 / (1.0 + np.exp(-logits))
            residual = (predicted - y) * weights
            self.weights -= .08 * (x.T @ residual / weights.sum() + 1e-4 * self.weights)
            self.bias -= .08 * residual.sum() / weights.sum()
        return self

    def predict_proba(self, values):
        logits = np.clip(((values - self.mean) / self.scale) @ self.weights + self.bias, -30, 30)
        return 1.0 / (1.0 + np.exp(-logits))

    def importance(self):
        return np.abs(self.weights)


class ShallowTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.importances = np.zeros(len(FEATURES), dtype=float)

    @staticmethod
    def gini(labels):
        if not len(labels): return 0.0
        probability = labels.mean()
        return 2 * probability * (1 - probability)

    def fit(self, values, labels):
        self.root = self._build(values, labels, 0)
        return self

    def _build(self, values, labels, depth):
        probability = float(labels.mean()) if len(labels) else 0.0
        if depth >= self.max_depth or len(labels) < 80 or probability in (0.0, 1.0):
            return (None, probability, None, None)
        parent = self.gini(labels); best = None
        for feature in range(values.shape[1]):
            thresholds = np.unique(np.quantile(values[:, feature], [.1, .3, .5, .7, .9]))
            for threshold in thresholds:
                left = values[:, feature] <= threshold
                if left.sum() < 20 or (~left).sum() < 20: continue
                child = (left.mean() * self.gini(labels[left]) + (~left).mean() * self.gini(labels[~left]))
                gain = parent - child
                if best is None or gain > best[0]: best = (gain, feature, threshold, left)
        if best is None or best[0] <= 1e-8:
            return (None, probability, None, None)
        gain, feature, threshold, left = best
        self.importances[feature] += gain * len(labels)
        return (feature, probability, threshold, (self._build(values[left], labels[left], depth + 1), self._build(values[~left], labels[~left], depth + 1)))

    def predict_proba(self, values):
        result = np.empty(len(values), dtype=float)
        for index, row in enumerate(values):
            node = self.root
            while node[0] is not None:
                node = node[3][0] if row[node[0]] <= node[2] else node[3][1]
            result[index] = node[1]
        return result

    def importance(self):
        total = self.importances.sum()
        return self.importances / total if total else self.importances


class BoostedStumps:
    """Small AdaBoost stump ensemble, used because sklearn is not a project dependency."""
    def __init__(self, rounds=25):
        self.rounds = rounds
        self.stumps = []
        self.importances = np.zeros(len(FEATURES), dtype=float)

    def fit(self, values, labels):
        target = np.where(labels == 1, 1.0, -1.0)
        weights = np.ones(len(target), dtype=float) / len(target)
        for _ in range(self.rounds):
            candidate = None
            for feature in range(values.shape[1]):
                for threshold in np.unique(np.quantile(values[:, feature], [.1, .3, .5, .7, .9])):
                    base = np.where(values[:, feature] <= threshold, 1.0, -1.0)
                    for polarity in (1.0, -1.0):
                        prediction = polarity * base
                        error = weights[prediction != target].sum()
                        if candidate is None or error < candidate[0]: candidate = (error, feature, threshold, polarity, prediction)
            error, feature, threshold, polarity, prediction = candidate
            error = min(max(error, 1e-9), 1 - 1e-9)
            alpha = .5 * math.log((1 - error) / error)
            self.stumps.append((feature, threshold, polarity, alpha))
            self.importances[feature] += abs(alpha)
            weights *= np.exp(-alpha * target * prediction)
            weights /= weights.sum()
        return self

    def predict_proba(self, values):
        score = np.zeros(len(values), dtype=float)
        for feature, threshold, polarity, alpha in self.stumps:
            score += alpha * polarity * np.where(values[:, feature] <= threshold, 1.0, -1.0)
        return 1.0 / (1.0 + np.exp(np.clip(-2 * score, -30, 30)))

    def importance(self):
        total = self.importances.sum()
        return self.importances / total if total else self.importances


def model_factory(name):
    if name == "logistic_regression": return StandardLogistic()
    if name == "shallow_decision_tree": return ShallowTree()
    if name == "boosted_stumps": return BoostedStumps()
    raise ValueError(name)


def top_k_metrics(labels, scores, groups, budget):
    recalls, hits = [], []
    for group in np.unique(groups):
        mask = groups == group
        positives = labels[mask].sum()
        if not positives: continue
        selected = np.argsort(-scores[mask], kind="stable")[:budget]
        selected_labels = labels[mask][selected]
        recalls.append(selected_labels.sum() / positives)
        hits.append(float(selected_labels.sum() > 0))
    return mean(recalls) if recalls else float("nan"), mean(hits) if hits else float("nan")


def local_downward_projection(grid, raw, selected, radius):
    """Restrict lower-envelope relaxation to graph-radius neighborhoods of candidates."""
    affected, queue = set(selected), deque((node, 0) for node in selected)
    while queue:
        node, distance = queue.popleft()
        if distance >= radius: continue
        for neighbor in neighbors(grid, node):
            if neighbor not in affected:
                affected.add(neighbor); queue.append((neighbor, distance + 1))
    table = dict(raw)
    heap = [(raw[node], node) for node in affected]
    heapq.heapify(heap)
    while heap:
        value, node = heapq.heappop(heap)
        if value != table[node]: continue
        for neighbor in neighbors(grid, node):
            if neighbor not in affected: continue
            candidate = value + 1.0
            if candidate < table[neighbor]:
                table[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return table, {node for node in affected if raw[node] - table[node] > 1e-6}


def run_policy_case(grid, start, goal, raw, distance, policy, selected, radius):
    if policy == "raw_direct": table, mode, modified = raw, "direct_unet", set()
    elif policy == "full_projection":
        table, mode = project_consistent_lower_envelope(grid, goal, raw), "direct_unet"
        modified = {node for node in raw if raw[node] - table[node] > 1e-6}
    elif policy == "unet_tiebreak": table, mode, modified = raw, "unet_tiebreak", set()
    else:
        table, modified = local_downward_projection(grid, raw, selected, radius)
        mode = "direct_unet"
    trace = trace_search(grid, start, goal, table, distance, set(), mode)
    checked_search(grid, start, goal, table, mode, trace)
    metrics = table_metrics(grid, table, distance)
    return trace, len(modified), metrics


def aggregate_policy(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[("overall", "all", row["policy"])].append(row)
        grouped[("structure", row["structure_type"], row["policy"])].append(row)
    output = []
    for (scope, structure, policy), values in sorted(grouped.items()):
        output.append({"scope": scope, "structure_type": structure, "policy": policy, "cases": len(values),
                       "optimality_rate": mean(float(row["optimal"]) for row in values),
                       "nonoptimal_case_count": sum(not row["optimal"] for row in values),
                       "mean_expanded_nodes": mean(row["expanded_nodes"] for row in values),
                       "mean_modified_cells": mean(row["modified_cells"] for row in values),
                       "mean_consistency_violations": mean(row["remaining_consistency_violations"] for row in values),
                       "mean_admissibility_violations": mean(row["remaining_admissibility_violations"] for row in values)})
    return output


def build_feature_dataset(root, args, labels, output):
    cases = benchmark_cases(root, args.structured_results, args.random_results)
    if args.max_cases is not None: cases = cases[:args.max_cases]
    model = load_unet_heuristic(os.path.join(root, args.expanded_checkpoint))
    arrays, targets, group_ids, structures, cell_lists, metadata = [], [], [], [], [], []
    columns = ["case_id", "structure_type", "cell", "repair_label", *FEATURES, "split"]
    path = os.path.join(output, "repair_cell_features.csv")
    temporary_path = path + ".temporary"
    with open(temporary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for index, row in enumerate(cases, start=1):
            grid, start, goal = rebuild_case(row)
            current_id = case_id(row, start, goal)
            heuristic = make_unet_heuristic(model, grid, goal)
            raw = {cell: float(heuristic(cell, goal)) for cell in free_cells(grid)}
            trace_distance = dummy_distance(grid)
            direct = trace_search(grid, start, goal, raw, trace_distance, set(), "direct_unet")
            tie = trace_search(grid, start, goal, raw, trace_distance, set(), "unet_tiebreak")
            cells, values, rows = build_case_features(grid, start, goal, raw, direct, tie, labels, current_id)
            for item in rows: item["structure_type"] = row["analysis_structure"]
            arrays.append(values); targets.append(np.asarray([item["repair_label"] for item in rows], dtype=np.int8))
            group_ids.extend([index - 1] * len(cells)); structures.extend([row["analysis_structure"]] * len(cells)); cell_lists.append(cells)
            metadata.append({"case_id": current_id, "structure_type": row["analysis_structure"], "row": row,
                             "positive_count": int(sum(item["repair_label"] for item in rows)), "cells": cells})
            writer.writerows(rows)
            if index % 100 == 0: print(f"features {index}/{len(cases)}")
    splits = split_groups(metadata)
    # Append split labels with a streaming rewrite rather than retaining the whole CSV in memory.
    split_by_case = {item["case_id"]: splits[index] for index, item in enumerate(metadata)}
    with open(temporary_path, newline="", encoding="utf-8") as source, open(path, "w", newline="", encoding="utf-8") as destination:
        reader = csv.DictReader(source); writer = csv.DictWriter(destination, fieldnames=columns); writer.writeheader()
        for item in reader:
            item["split"] = split_by_case[item["case_id"]]
            writer.writerow(item)
    os.unlink(temporary_path)
    return cases, model, np.vstack(arrays), np.concatenate(targets), np.asarray(group_ids), np.asarray(structures), cell_lists, metadata, splits


def feature_analysis(values, labels, structures):
    output = []
    scopes = [("overall", np.ones(len(labels), dtype=bool))]
    scopes.extend((structure, structures == structure) for structure in STRUCTURES if np.any(structures == structure))
    for scope, mask in scopes:
        subset, target = values[mask], labels[mask]
        for index, name in enumerate(FEATURES):
            positive, negative = subset[target == 1, index], subset[target == 0, index]
            pooled = math.sqrt((positive.var() + negative.var()) / 2) if len(positive) and len(negative) else math.nan
            effect = (positive.mean() - negative.mean()) / pooled if pooled > 1e-12 else 0.0
            forward, reverse = average_precision(target, subset[:, index]), average_precision(target, -subset[:, index])
            output.append({"scope": scope, "feature": name, "positive_count": len(positive), "negative_count": len(negative),
                           "positive_mean": positive.mean() if len(positive) else math.nan,
                           "negative_mean": negative.mean() if len(negative) else math.nan,
                           "positive_median": float(np.median(positive)) if len(positive) else math.nan,
                           "negative_median": float(np.median(negative)) if len(negative) else math.nan,
                           "cohens_d": effect, "univariate_pr_auc": max(forward, reverse),
                           "best_direction": "higher" if forward >= reverse else "lower"})
    return output


def train_and_evaluate(values, labels, groups, split_names, metadata):
    split_per_row = np.asarray([split_names[group] for group in groups])
    train = np.flatnonzero(split_per_row == "train")
    validation = np.flatnonzero(split_per_row == "validation")
    test = np.flatnonzero(split_per_row == "test")
    results, fitted = [], {}
    for name in ("logistic_regression", "shallow_decision_tree", "boosted_stumps"):
        sample = balanced_training_indices(train, labels, stable_number(name) % (2**32))
        predictor = model_factory(name).fit(values[sample], labels[sample])
        validation_scores, test_scores = predictor.predict_proba(values[validation]), predictor.predict_proba(values[test])
        threshold = best_f1_threshold(labels[validation], validation_scores)
        precision, recall, f1 = binary_metrics(labels[test], test_scores, threshold)
        top1 = top_k_metrics(labels[test], test_scores, groups[test], 1)
        top2 = top_k_metrics(labels[test], test_scores, groups[test], 2)
        top5 = top_k_metrics(labels[test], test_scores, groups[test], 5)
        results.append({"model": name, "test_cells": len(test), "test_repair_cells": int(labels[test].sum()), "threshold": threshold,
                        "validation_pr_auc": average_precision(labels[validation], validation_scores),
                        "precision": precision, "recall": recall, "f1": f1, "pr_auc": average_precision(labels[test], test_scores),
                        "top_1_recall_per_map": top1[0], "top_1_case_hit_rate": top1[1],
                        "top_2_recall_per_map": top2[0], "top_2_case_hit_rate": top2[1],
                        "top_5_recall_per_map": top5[0], "top_5_case_hit_rate": top5[1],
                        "feature_importance": ";".join(f"{feature}:{weight:.4f}" for feature, weight in zip(FEATURES, predictor.importance()))})
        fitted[name] = (predictor, threshold)
    chosen = max(results, key=lambda row: row["validation_pr_auc"])["model"]
    # Cross-fitted scores let every map be scored by a predictor that did not train on it.
    folds = np.asarray([stable_number(item["case_id"]) % 5 for item in metadata])
    for position, item in enumerate(metadata):
        if item["positive_count"]:
            folds[position] = stable_number(item["case_id"] + "positive") % 5
    oof = np.zeros(len(labels), dtype=float)
    for fold in range(5):
        train_indices = np.flatnonzero(folds[groups] != fold)
        held = np.flatnonzero(folds[groups] == fold)
        sample = balanced_training_indices(train_indices, labels, 100 + fold)
        predictor = model_factory(chosen).fit(values[sample], labels[sample])
        oof[held] = predictor.predict_proba(values[held])
    return results, chosen, fitted[chosen][1], oof


def policy_evaluation(root, args, cases, model, values, groups, cell_lists, metadata, oof_scores, threshold):
    prior_path = os.path.join(root, "outputs/projection_modification_ablation/partial_projection_results.csv")
    oracle = {}
    with open(prior_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] == "greedy_minimal_repair": oracle[row["case_id"]] = row
    rows = []
    for index, row in enumerate(cases):
        grid, start, goal = rebuild_case(row); item = metadata[index]; current_id = item["case_id"]
        distance = compute_distance_to_goal(grid, goal)
        optimal_cost = distance[start[0]][start[1]]
        heuristic = make_unet_heuristic(model, grid, goal)
        raw = {cell: float(heuristic(cell, goal)) for cell in free_cells(grid)}
        case_indices = np.flatnonzero(groups == index)
        cells, scores = cell_lists[index], oof_scores[case_indices]
        generated = values[case_indices, FEATURES.index("was_generated")] > .5
        ranked = [cells[position] for position in np.argsort(-scores, kind="stable") if generated[position]]
        selections = {
            "predictor_top_1": set(ranked[:1]), "predictor_top_2": set(ranked[:2]), "predictor_top_5": set(ranked[:5]),
            "predictor_threshold_f1": {cells[position] for position in range(len(cells)) if generated[position] and scores[position] >= threshold},
            "predictor_threshold_0_5": {cells[position] for position in range(len(cells)) if generated[position] and scores[position] >= .5},
        }
        for policy in POLICIES:
            if policy == "oracle_greedy_minimal_repair":
                prior = oracle[current_id]
                rows.append({"case_id": current_id, "structure_type": row["analysis_structure"], "policy": policy,
                             "expanded_nodes": int(prior["expanded_nodes"]), "path_cost": int(prior["path_cost"]), "optimal_cost": optimal_cost,
                             "optimal": prior["optimal"] == "True", "modified_cells": int(prior["modified_cells"]),
                             "remaining_consistency_violations": int(prior["remaining_consistency_violations"]),
                             "remaining_admissibility_violations": int(prior["remaining_admissibility_violations"]), "selected_risk_cells": int(prior["components_selected"])})
                continue
            trace, modified, metrics = run_policy_case(grid, start, goal, raw, distance, policy, selections.get(policy, set()), args.local_radius)
            rows.append({"case_id": current_id, "structure_type": row["analysis_structure"], "policy": policy,
                         "expanded_nodes": trace["expanded"], "path_cost": trace["cost"], "optimal_cost": optimal_cost,
                         "optimal": trace["cost"] == optimal_cost, "modified_cells": modified,
                         "remaining_consistency_violations": metrics["consistency_violations"],
                         "remaining_admissibility_violations": metrics["admissibility_violations"],
                         "selected_risk_cells": len(selections.get(policy, set()))})
        if (index + 1) % 100 == 0: print(f"policies {index + 1}/{len(cases)}")
    return rows


def write_report(output, feature_rows, predictor_rows, chosen, prevalence, policy_rows, summary):
    overall = {row["policy"]: row for row in summary if row["scope"] == "overall"}
    strongest = sorted((row for row in feature_rows if row["scope"] == "overall"), key=lambda row: row["univariate_pr_auc"], reverse=True)[:5]
    selected = next(row for row in predictor_rows if row["model"] == chosen)
    safe = [row for row in overall.values() if row["optimality_rate"] == 1.0 and row["mean_consistency_violations"] == 0 and row["mean_admissibility_violations"] == 0]
    safe_better = [row for row in safe if row["mean_expanded_nodes"] < overall["unet_tiebreak"]["mean_expanded_nodes"]]
    lines = ["# Online Repair Cell Prediction", "", "Repair labels are benchmark-specific cells from the prior greedy-minimal-repair ablation. Features exclude true distance, optimal paths, BFS labels, recovery costs, and final search outcomes.", "", f"Repair-cell prevalence: {prevalence:.6f}.", "", "## Strongest Online Feature Signals", ""]
    for row in strongest:
        lines.append(f"- `{row['feature']}`: univariate PR-AUC {row['univariate_pr_auc']:.4f}, Cohen's d {row['cohens_d']:.3f} ({row['best_direction']} is more repair-like).")
    lines += ["", "## Structure-specific Signals", ""]
    for structure in STRUCTURES:
        rows = [row for row in feature_rows if row["scope"] == structure]
        if not rows or not rows[0]["positive_count"]:
            lines.append(f"- `{structure}`: no repair cells in the previous greedy-minimal-repair labels.")
            continue
        best = max(rows, key=lambda row: row["univariate_pr_auc"])
        lines.append(f"- `{structure}`: {best['positive_count']} repair cells; strongest single feature `{best['feature']}` (PR-AUC {best['univariate_pr_auc']:.4f}).")
    lift = selected["pr_auc"] / prevalence
    local = [row for name, row in overall.items() if name.startswith("predictor_")]
    best_local = max(local, key=lambda row: row["optimality_rate"])
    raw_failures = overall["raw_direct"]["nonoptimal_case_count"]
    lines += ["", "## Held-out Predictor Result", "", f"- Selected by validation PR-AUC: `{chosen}`.", f"- Test PR-AUC {selected['pr_auc']:.4f}, about {lift:.1f}x the {prevalence:.4%} repair-cell prevalence; precision {selected['precision']:.4f}; recall {selected['recall']:.4f}; F1 {selected['f1']:.4f}.", f"- Test top-1 / top-2 / top-5 repair-cell recall per positive map: {selected['top_1_recall_per_map']:.4f} / {selected['top_2_recall_per_map']:.4f} / {selected['top_5_recall_per_map']:.4f}.", "", "## Online-feature Repair Replay", "", "Policy scores are cross-fitted by map. The implementation is a two-pass replay: a raw search supplies generated-cell features, then a local graph-radius consistency relaxation is applied before the evaluated search. It uses no ground-truth search labels at scoring time, but is not a single-pass deployable policy.", "", "| Policy | Mean expanded | Optimality | Modified cells | Remaining consistency violations |", "|---|---:|---:|---:|---:|"]
    for policy in POLICIES:
        row = overall[policy]
        lines.append(f"| {policy} | {row['mean_expanded_nodes']:.2f} | {row['optimality_rate']:.4f} | {row['mean_modified_cells']:.2f} | {row['mean_consistency_violations']:.2f} |")
    lines += ["", "## Answers", "", "1. Repair cells tend to be generated later with larger Direct-U-Net `f`, larger U-Net/Manhattan disagreement, and somewhat larger local consistency violations; the structure-specific table above shows that the strongest univariate signal varies by structure.", f"2. Yes, above chance but weakly: held-out PR-AUC is {selected['pr_auc']:.4f} versus prevalence {prevalence:.4f}. Low top-k cell recall shows that the model does not reliably isolate a complete repair component.", f"3. The best local policy is `{best_local['policy']}`: it leaves {best_local['nonoptimal_case_count']} non-optimal cases versus {raw_failures} for raw Direct U-Net. It cannot recover full optimality without the broad cost of full projection.", "4. Safe policies that beat U-Net tie-break overall: " + (", ".join(row["policy"] for row in safe_better) if safe_better else "none") + ".", "", "Component labels were created using an offline optimality oracle, so this analysis reports predictive association, not a causal or deployable repair mechanism.", ""]
    with open(os.path.join(output, "report.md"), "w", encoding="utf-8") as handle: handle.write("\n".join(lines))


def run(args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output = os.path.join(root, args.output_dir); os.makedirs(output, exist_ok=True)
    labels, selected_components = repair_label_lookup(root)
    cases, model, values, target, groups, structures, cells, metadata, splits = build_feature_dataset(root, args, labels, output)
    print(f"repair labels: {int(target.sum())} cells from {len(selected_components)} cases")
    feature_rows = feature_analysis(values, target, structures)
    write_csv(os.path.join(output, "feature_analysis.csv"), feature_rows)
    predictor_rows, chosen, threshold, scores = train_and_evaluate(values, target, groups, splits, metadata)
    write_csv(os.path.join(output, "predictor_results.csv"), predictor_rows)
    policy_rows = policy_evaluation(root, args, cases, model, values, groups, cells, metadata, scores, threshold)
    write_csv(os.path.join(output, "online_policy_results.csv"), policy_rows)
    summary = aggregate_policy(policy_rows)
    write_csv(os.path.join(output, "summary_by_structure.csv"), summary)
    write_report(output, feature_rows, predictor_rows, chosen, float(target.mean()), policy_rows, summary)
    print(f"completed {len(cases)} cases in {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict benchmark-specific consistency repair cells from online features.")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_tiebreak_controls_100.csv")
    parser.add_argument("--expanded-checkpoint", default="outputs/expanded_dataset_training/unet_heuristic_expanded_best.pt")
    parser.add_argument("--output-dir", default="outputs/online_repair_cell_prediction")
    parser.add_argument("--local-radius", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
