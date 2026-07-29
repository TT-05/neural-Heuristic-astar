import argparse
import csv
import math
import os
import random
from collections import Counter, defaultdict


TARGET_LABELS = {"unet", "manhattan_unet_tiebreak"}
NON_FEATURE_COLUMNS = {
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
THRESHOLD_FEATURES = [
    "unet_minus_mlp_early_goal_progress_rate",
    "unet_minus_large_g_early_goal_progress_rate",
]


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value, default=None):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def std(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / len(values))


def pearson(xs, ys):
    xs = list(xs)
    ys = list(ys)
    if len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom_x = math.sqrt(sum(x * x for x in dx))
    denom_y = math.sqrt(sum(y * y for y in dy))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            output[indexed[k][0]] = rank
        i = j
    return output


def spearman(xs, ys):
    if len(xs) < 2:
        return 0.0
    return pearson(ranks(xs), ranks(ys))


def is_unet_best(row):
    return 1 if row.get("best_method") in TARGET_LABELS else 0


def feature_names(rows, excluded):
    names = []
    for key in sorted(set().union(*(row.keys() for row in rows))):
        if key in excluded:
            continue
        values = [to_float(row.get(key)) for row in rows]
        if any(value is not None for value in values):
            names.append(key)
    return names


def matrix(rows, names):
    return [[to_float(row.get(name), 0.0) for name in names] for row in rows]


def stratified_folds(labels, folds, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[label].append(index)
    split = [[] for _ in range(folds)]
    for indices in by_label.values():
        rng.shuffle(indices)
        for i, index in enumerate(indices):
            split[i % folds].append(index)
    return split


def standardizer(train_x):
    cols = len(train_x[0]) if train_x else 0
    means = []
    scales = []
    for col in range(cols):
        values = [row[col] for row in train_x]
        means.append(mean(values))
        scales.append(std(values) or 1.0)
    return means, scales


def apply_standardizer(rows, means, scales):
    return [[(value - means[i]) / scales[i] for i, value in enumerate(row)] for row in rows]


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def fit_logistic_balanced(train_x, train_y, epochs=220, lr=0.08, l2=0.001):
    means, scales = standardizer(train_x)
    x = apply_standardizer(train_x, means, scales)
    n = len(x)
    cols = len(x[0]) if x else 0
    weights = [0.0] * cols
    bias = 0.0
    positives = sum(train_y)
    negatives = len(train_y) - positives
    class_weight = {
        1: len(train_y) / (2.0 * positives) if positives else 1.0,
        0: len(train_y) / (2.0 * negatives) if negatives else 1.0,
    }
    for _ in range(epochs):
        grad_w = [0.0] * cols
        grad_b = 0.0
        for row, label in zip(x, train_y):
            pred = sigmoid(sum(w * value for w, value in zip(weights, row)) + bias)
            error = (pred - label) * class_weight[label]
            for col, value in enumerate(row):
                grad_w[col] += error * value
            grad_b += error
        for col in range(cols):
            grad_w[col] = grad_w[col] / n + l2 * weights[col]
            weights[col] -= lr * grad_w[col]
        bias -= lr * grad_b / n
    return {"weights": weights, "bias": bias, "means": means, "scales": scales}


def predict_logistic(model, test_x):
    x = apply_standardizer(test_x, model["means"], model["scales"])
    return [sigmoid(sum(w * value for w, value in zip(model["weights"], row)) + model["bias"]) for row in x]


def weighted_gini(labels, sample_weights):
    total = sum(sample_weights)
    if total <= 0:
        return 0.0
    p = sum(weight for label, weight in zip(labels, sample_weights) if label == 1) / total
    return 1.0 - p * p - (1.0 - p) * (1.0 - p)


def class_weights(labels):
    positives = sum(labels)
    negatives = len(labels) - positives
    return [len(labels) / (2.0 * (positives if label else negatives)) if (positives if label else negatives) else 1.0 for label in labels]


def leaf_probability(labels, weights):
    denom = sum(weights)
    return sum(weight for label, weight in zip(labels, weights) if label == 1) / denom if denom else 0.0


def candidate_thresholds(values, max_thresholds=20):
    unique = sorted(set(values))
    if len(unique) < 2:
        return []
    thresholds = [(left + right) / 2.0 for left, right in zip(unique, unique[1:])]
    if len(thresholds) <= max_thresholds:
        return thresholds
    step = max(1, len(thresholds) // max_thresholds)
    return thresholds[::step]


def best_split(rows, labels, weights, feature_indices):
    parent_weight = sum(weights)
    best = None
    for feature in feature_indices:
        values = [row[feature] for row in rows]
        for threshold in candidate_thresholds(values):
            left = [i for i, value in enumerate(values) if value <= threshold]
            right = [i for i, value in enumerate(values) if value > threshold]
            if not left or not right:
                continue
            left_gini = weighted_gini([labels[i] for i in left], [weights[i] for i in left])
            right_gini = weighted_gini([labels[i] for i in right], [weights[i] for i in right])
            score = (
                sum(weights[i] for i in left) * left_gini + sum(weights[i] for i in right) * right_gini
            ) / parent_weight
            candidate = (score, feature, threshold, left, right)
            if best is None or candidate[0] < best[0]:
                best = candidate
    return best


def build_tree(rows, labels, weights, feature_indices, depth):
    if depth <= 0 or len(set(labels)) <= 1 or len(rows) < 8:
        return {"prob": leaf_probability(labels, weights)}
    split = best_split(rows, labels, weights, feature_indices)
    if split is None:
        return {"prob": leaf_probability(labels, weights)}
    _, feature, threshold, left_idx, right_idx = split
    return {
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(
            [rows[i] for i in left_idx],
            [labels[i] for i in left_idx],
            [weights[i] for i in left_idx],
            feature_indices,
            depth - 1,
        ),
        "right": build_tree(
            [rows[i] for i in right_idx],
            [labels[i] for i in right_idx],
            [weights[i] for i in right_idx],
            feature_indices,
            depth - 1,
        ),
    }


def predict_tree_one(tree, row):
    node = tree
    while "prob" not in node:
        node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
    return node["prob"]


def predict_tree(tree, rows):
    return [predict_tree_one(tree, row) for row in rows]


def fit_tree(train_x, train_y, depth=3, feature_indices=None):
    weights = class_weights(train_y)
    features = feature_indices or list(range(len(train_x[0])))
    return build_tree(train_x, train_y, weights, features, depth)


def fit_forest(train_x, train_y, seed, trees=16, depth=2):
    rng = random.Random(seed)
    cols = len(train_x[0])
    feature_count = max(1, int(math.sqrt(cols)))
    forest = []
    for _ in range(trees):
        indices = [rng.randrange(len(train_x)) for _ in range(len(train_x))]
        features = sorted(rng.sample(range(cols), feature_count))
        tree = fit_tree([train_x[i] for i in indices], [train_y[i] for i in indices], depth=depth, feature_indices=features)
        forest.append(tree)
    return forest


def predict_forest(forest, rows):
    if not forest:
        return [0.0 for _ in rows]
    return [mean(predict_tree(tree, [row])[0] for tree in forest) for row in rows]


def threshold_metrics(labels, scores, threshold):
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "accuracy": accuracy,
    }


def roc_auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            if ranked[k][1] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def pr_curve(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives = sum(labels)
    if positives == 0:
        return [(0.0, 0.0, 1.0)]
    tp = 0
    fp = 0
    points = []
    last_score = None
    for score, label in pairs:
        if last_score is not None and score != last_score:
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / positives
            points.append((recall, precision, last_score))
        tp += 1 if label == 1 else 0
        fp += 1 if label == 0 else 0
        last_score = score
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / positives
    points.append((recall, precision, last_score if last_score is not None else 1.0))
    points.insert(0, (0.0, 1.0, 1.0))
    return points


def pr_auc(labels, scores):
    points = pr_curve(labels, scores)
    area = 0.0
    for left, right in zip(points, points[1:]):
        recall_delta = max(0.0, right[0] - left[0])
        area += recall_delta * right[1]
    return area


def threshold_grid(scores):
    values = sorted(set(scores))
    if not values:
        return [0.5]
    thresholds = [min(values) - 1e-9, max(values) + 1e-9]
    thresholds.extend(values)
    if len(thresholds) > 300:
        step = max(1, len(thresholds) // 300)
        thresholds = thresholds[::step]
    return sorted(set(thresholds))


def threshold_settings(labels, scores):
    rows = []
    for threshold in threshold_grid(scores):
        metrics = threshold_metrics(labels, scores, threshold)
        rows.append({"threshold": threshold, **metrics})
    balanced = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    high_recall_candidates = [row for row in rows if row["recall"] >= 0.80]
    if high_recall_candidates:
        high_recall = max(high_recall_candidates, key=lambda row: (row["precision"], row["f1"]))
    else:
        high_recall = max(rows, key=lambda row: (row["recall"], row["precision"], row["f1"]))
    high_precision_candidates = [row for row in rows if row["precision"] >= 0.50]
    if high_precision_candidates:
        high_precision = max(high_precision_candidates, key=lambda row: (row["recall"], row["f1"]))
    else:
        high_precision = max(rows, key=lambda row: (row["precision"], row["recall"], row["f1"]))
    return {
        "default_0.5": threshold_metrics(labels, scores, 0.5) | {"threshold": 0.5},
        "balanced_f1": balanced,
        "high_recall": high_recall,
        "high_precision": high_precision,
    }


def select_threshold_rule(train_rows, train_y, candidate_features):
    best = None
    for feature in candidate_features:
        values = [to_float(row.get(feature), 0.0) for row in train_rows]
        for direction in ["ge", "le"]:
            transformed = values if direction == "ge" else [-value for value in values]
            for threshold in threshold_grid(transformed):
                scores = [1.0 if value >= threshold else 0.0 for value in transformed]
                metrics = threshold_metrics(train_y, scores, 0.5)
                candidate = (metrics["f1"], metrics["recall"], metrics["precision"], feature, direction, threshold)
                if best is None or candidate > best:
                    best = candidate
    return best


def predict_threshold_rule(rows, rule):
    _, _, _, feature, direction, threshold = rule
    values = [to_float(row.get(feature), 0.0) for row in rows]
    if direction == "le":
        values = [-value for value in values]
    return [1.0 if value >= threshold else 0.0 for value in values], {
        "selected_feature": feature,
        "direction": direction,
        "threshold": threshold,
    }


def cross_validate(rows, names, model_name, folds, seed):
    labels = [is_unet_best(row) for row in rows]
    x = matrix(rows, names)
    fold_indices = stratified_folds(labels, folds, seed)
    scores = [0.0] * len(rows)
    metadata = []
    for fold, test_idx in enumerate(fold_indices):
        train_idx = [index for index in range(len(rows)) if index not in set(test_idx)]
        train_x = [x[index] for index in train_idx]
        train_y = [labels[index] for index in train_idx]
        test_x = [x[index] for index in test_idx]
        if model_name == "balanced_logistic_regression":
            model = fit_logistic_balanced(train_x, train_y)
            fold_scores = predict_logistic(model, test_x)
            metadata.append({"fold": fold})
        elif model_name == "balanced_shallow_tree":
            tree = fit_tree(train_x, train_y, depth=3)
            fold_scores = predict_tree(tree, test_x)
            metadata.append({"fold": fold})
        elif model_name == "balanced_random_forest":
            forest = fit_forest(train_x, train_y, seed + fold, trees=16, depth=2)
            fold_scores = predict_forest(forest, test_x)
            metadata.append({"fold": fold})
        elif model_name == "top_feature_threshold_rule":
            candidate_features = [feature for feature in THRESHOLD_FEATURES if feature in names]
            rule = select_threshold_rule([rows[index] for index in train_idx], train_y, candidate_features)
            fold_scores, rule_meta = predict_threshold_rule([rows[index] for index in test_idx], rule)
            metadata.append({"fold": fold, **rule_meta})
        else:
            raise ValueError(f"Unknown model: {model_name}")
        for index, score in zip(test_idx, fold_scores):
            scores[index] = max(0.0, min(1.0, score))
    return scores, metadata


def model_result_row(labels, scores, model, feature_group, features):
    default_metrics = threshold_metrics(labels, scores, 0.5)
    return {
        "model": model,
        "feature_group": feature_group,
        "features": features,
        "precision": default_metrics["precision"],
        "recall": default_metrics["recall"],
        "f1": default_metrics["f1"],
        "pr_auc": pr_auc(labels, scores),
        "roc_auc": roc_auc(labels, scores),
        "false_positive_rate": default_metrics["false_positive_rate"],
        "false_negative_rate": default_metrics["false_negative_rate"],
        "accuracy": default_metrics["accuracy"],
        "threshold": 0.5,
    }


def baseline_scores(labels, baseline, seed):
    rng = random.Random(seed)
    base_rate = mean(labels)
    if baseline == "always_non_unet":
        return [0.0 for _ in labels]
    if baseline == "random_at_base_rate":
        return [1.0 if rng.random() < base_rate else 0.0 for _ in labels]
    raise ValueError(f"Unknown baseline: {baseline}")


def confusion_rows(labels, scores, model, feature_group, threshold_name, threshold):
    metrics = threshold_metrics(labels, scores, threshold)
    return [
        {
            "model": model,
            "feature_group": feature_group,
            "threshold_name": threshold_name,
            "threshold": threshold,
            "true_label": 1,
            "predicted_label": 1,
            "count": metrics["tp"],
        },
        {
            "model": model,
            "feature_group": feature_group,
            "threshold_name": threshold_name,
            "threshold": threshold,
            "true_label": 0,
            "predicted_label": 1,
            "count": metrics["fp"],
        },
        {
            "model": model,
            "feature_group": feature_group,
            "threshold_name": threshold_name,
            "threshold": threshold,
            "true_label": 0,
            "predicted_label": 0,
            "count": metrics["tn"],
        },
        {
            "model": model,
            "feature_group": feature_group,
            "threshold_name": threshold_name,
            "threshold": threshold,
            "true_label": 1,
            "predicted_label": 0,
            "count": metrics["fn"],
        },
    ]


def feature_correlations(rows, names):
    labels = [is_unet_best(row) for row in rows]
    output = []
    for name in names:
        values = [to_float(row.get(name), 0.0) for row in rows]
        output.append(
            {
                "feature": name,
                "pearson_with_is_unet_best": pearson(values, labels),
                "spearman_with_is_unet_best": spearman(values, labels),
                "mean": mean(values),
                "std": std(values),
            }
        )
    return sorted(output, key=lambda row: abs(row["spearman_with_is_unet_best"]), reverse=True)


def posthoc_structure_rows(rows, scores, model_name):
    output = []
    for structure in sorted(set(row.get("structured_type", "unknown") for row in rows)):
        indices = [i for i, row in enumerate(rows) if row.get("structured_type", "unknown") == structure]
        labels = [is_unet_best(rows[i]) for i in indices]
        scoped_scores = [scores[i] for i in indices]
        metrics = threshold_metrics(labels, scoped_scores, 0.5)
        output.append(
            {
                "model": model_name,
                "structured_type": structure,
                "n": len(indices),
                "unet_positive_count": sum(labels),
                "unet_base_rate": mean(labels),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "pr_auc": pr_auc(labels, scoped_scores),
                "roc_auc": roc_auc(labels, scoped_scores),
                "false_positive_rate": metrics["false_positive_rate"],
                "false_negative_rate": metrics["false_negative_rate"],
            }
        )
    return output


def plot_pr_curve(path, labels, score_by_model):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, scores in score_by_model.items():
        points = pr_curve(labels, scores)
        ax.plot([p[0] for p in points], [p[1] for p in points], label=f"{model} AUC={pr_auc(labels, scores):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("U-Net Best Detection PR Curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_feature_importance(path, corr_rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    rows = corr_rows[:15]
    labels = [row["feature"] for row in rows]
    values = [abs(row["spearman_with_is_unet_best"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels[::-1], values[::-1])
    ax.set_xlabel("|Spearman with is_unet_best|")
    ax.set_title("Online Feature Correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def write_summary(path, rows, model_rows, baseline_rows, threshold_rows, corr_rows, oracle_rows, structure_rows):
    labels = [is_unet_best(row) for row in rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    base_rate = mean(labels)
    online_models = [row for row in model_rows if row["feature_group"] == "online"]
    oracle_models = [row for row in model_rows if row["feature_group"] == "oracle_only"]
    online_thresholds = [row for row in threshold_rows if row["feature_group"] == "online"]
    best_online = max(online_thresholds, key=lambda row: row["f1"]) if online_thresholds else {}
    best_baseline = max(baseline_rows, key=lambda row: row["f1"]) if baseline_rows else {}
    oracle_thresholds = [row for row in threshold_rows if row["feature_group"] == "oracle_only"]
    best_oracle = max(oracle_thresholds, key=lambda row: row["f1"]) if oracle_thresholds else {}
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Focused U-Net Trust-Signal Validation\n\n")
        file.write(
            "This analysis tests whether online-feasible early-search features can identify cases where U-Net tie-breaking should be trusted. "
            "It is diagnostic only and does not implement an adaptive A* algorithm.\n\n"
        )
        file.write("## Class Imbalance\n\n")
        file.write(f"- U-Net-best cases: {positives}\n")
        file.write(f"- non-U-Net-best cases: {negatives}\n")
        file.write(f"- U-Net base rate: {base_rate:.3f}\n\n")

        file.write("## Baselines\n\n")
        file.write("| baseline | precision | recall | F1 | PR-AUC | ROC-AUC | FPR | FNR |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in baseline_rows:
            file.write(
                f"| {row['model']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} "
                f"| {row['pr_auc']:.3f} | {row['roc_auc']:.3f} | {row['false_positive_rate']:.3f} "
                f"| {row['false_negative_rate']:.3f} |\n"
            )
        file.write("\n")

        file.write("## Online-Only Models\n\n")
        file.write("| model | precision | recall | F1 | PR-AUC | ROC-AUC | FPR | FNR |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in online_models:
            file.write(
                f"| {row['model']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} "
                f"| {row['pr_auc']:.3f} | {row['roc_auc']:.3f} | {row['false_positive_rate']:.3f} "
                f"| {row['false_negative_rate']:.3f} |\n"
            )
        file.write("\n")

        if oracle_models:
            file.write("## Oracle-Only Diagnostics\n\n")
            file.write("Oracle features are evaluated separately and are not mixed with online features.\n\n")
            file.write("| model | precision | recall | F1 | PR-AUC | ROC-AUC | FPR | FNR |\n")
            file.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in oracle_models:
                file.write(
                    f"| {row['model']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} "
                    f"| {row['pr_auc']:.3f} | {row['roc_auc']:.3f} | {row['false_positive_rate']:.3f} "
                    f"| {row['false_negative_rate']:.3f} |\n"
                )
            file.write("\n")

        file.write("## Strongest Online Correlations\n\n")
        file.write("| feature | Spearman | Pearson |\n|---|---:|---:|\n")
        for row in corr_rows[:12]:
            file.write(
                f"| {row['feature']} | {row['spearman_with_is_unet_best']:.3f} "
                f"| {row['pearson_with_is_unet_best']:.3f} |\n"
            )
        file.write("\n")

        file.write("## Threshold Tradeoffs\n\n")
        file.write("| model | setting | threshold | precision | recall | F1 | FPR | FNR |\n")
        file.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in threshold_rows:
            if row["feature_group"] != "online":
                continue
            file.write(
                f"| {row['model']} | {row['threshold_name']} | {row['threshold']:.3f} "
                f"| {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} "
                f"| {row['false_positive_rate']:.3f} | {row['false_negative_rate']:.3f} |\n"
            )
        file.write("\n")

        if structure_rows:
            file.write("## Post-Hoc Structure Reporting\n\n")
            file.write("Structure labels are used only here, not for training or prediction.\n\n")
            file.write("| structure | n | positives | base rate | precision | recall | F1 | PR-AUC |\n")
            file.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in structure_rows:
                file.write(
                    f"| {row['structured_type']} | {row['n']} | {row['unet_positive_count']} "
                    f"| {row['unet_base_rate']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} "
                    f"| {row['f1']:.3f} | {row['pr_auc']:.3f} |\n"
                )
            file.write("\n")

        file.write("## Critical Answers\n\n")
        file.write(
            f"- Best online U-Net detector by threshold-swept F1: {best_online.get('model', 'n/a')} "
            f"({best_online.get('threshold_name', 'n/a')}) with F1 "
            f"{best_online.get('f1', 0.0):.3f}, precision {best_online.get('precision', 0.0):.3f}, "
            f"recall {best_online.get('recall', 0.0):.3f}, PR-AUC {best_online.get('pr_auc', 0.0):.3f}.\n"
        )
        file.write(
            f"- Best baseline by F1: {best_baseline.get('model', 'n/a')} with F1 {best_baseline.get('f1', 0.0):.3f}.\n"
        )
        if best_oracle:
            file.write(
                f"- Best oracle-only diagnostic F1: {best_oracle.get('f1', 0.0):.3f}; this is separate from the online result.\n"
            )
        if (
            best_online.get("f1", 0.0) < 0.35
            or best_online.get("precision", 0.0) < 0.35
            or best_online.get("recall", 0.0) < 0.50
        ):
            file.write(
                "- Verdict: U-Net-positive detection is weak. U-Net trust-based adaptation is not currently justified because useful "
                "U-Net-best cases are rare and hard to identify without many false positives.\n"
            )
        else:
            file.write(
                "- Verdict: online features contain a usable but imperfect U-Net trust signal. The best threshold reaches moderate "
                "precision and recall, but this supports further diagnostic work rather than immediate deployment of adaptive "
                "neural-guided A*.\n"
            )
        file.write(
            "- Algorithm-design implication: current evidence supports adaptive choice among simple tie-breakers more strongly than "
            "active U-Net trust as a primary design direction.\n"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Focused validation of U-Net trust signals.")
    parser.add_argument("--online-features", default="outputs/trust_signal_validation/online_features.csv")
    parser.add_argument("--oracle-features", default="outputs/trust_signal_validation/oracle_diagnostic_features.csv")
    parser.add_argument("--output-dir", default="outputs/unet_trust_signal")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    online_rows = read_csv(os.path.join(project_root, args.online_features))
    oracle_rows = read_csv(os.path.join(project_root, args.oracle_features))
    if not online_rows:
        raise RuntimeError("online_features.csv is required.")

    labels = [is_unet_best(row) for row in online_rows]
    online_names = feature_names(online_rows, NON_FEATURE_COLUMNS)
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

    model_rows = []
    threshold_rows = []
    confusion = []
    scores_by_model = {}

    baseline_rows = []
    for baseline in ["always_non_unet", "random_at_base_rate"]:
        scores = baseline_scores(labels, baseline, args.seed)
        row = model_result_row(labels, scores, baseline, "baseline", 0)
        baseline_rows.append(row)
        for setting, metrics in threshold_settings(labels, scores).items():
            threshold_rows.append({"model": baseline, "feature_group": "baseline", "threshold_name": setting, **metrics})
        confusion.extend(confusion_rows(labels, scores, baseline, "baseline", "default_0.5", 0.5))

    for model_name in [
        "balanced_logistic_regression",
        "balanced_shallow_tree",
        "balanced_random_forest",
        "top_feature_threshold_rule",
    ]:
        scores, _ = cross_validate(online_rows, online_names, model_name, args.folds, args.seed)
        scores_by_model[model_name] = scores
        model_rows.append(model_result_row(labels, scores, model_name, "online", len(online_names)))
        for setting, metrics in threshold_settings(labels, scores).items():
            threshold_rows.append({"model": model_name, "feature_group": "online", "threshold_name": setting, **metrics})
            confusion.extend(confusion_rows(labels, scores, model_name, "online", setting, metrics["threshold"]))

    if oracle_rows and oracle_names:
        for model_name in ["balanced_logistic_regression", "balanced_shallow_tree", "balanced_random_forest"]:
            scores, _ = cross_validate(oracle_rows, oracle_names, model_name, args.folds, args.seed)
            model_rows.append(model_result_row(labels, scores, model_name, "oracle_only", len(oracle_names)))
            for setting, metrics in threshold_settings(labels, scores).items():
                threshold_rows.append(
                    {"model": model_name, "feature_group": "oracle_only", "threshold_name": setting, **metrics}
                )

    corr_rows = feature_correlations(online_rows, online_names)
    auc_lookup = {
        (row["model"], row["feature_group"]): {"pr_auc": row["pr_auc"], "roc_auc": row["roc_auc"]}
        for row in [*model_rows, *baseline_rows]
    }
    for row in threshold_rows:
        row.update(auc_lookup.get((row["model"], row["feature_group"]), {"pr_auc": 0.0, "roc_auc": 0.0}))
    structure_model = max([row for row in model_rows if row["feature_group"] == "online"], key=lambda row: row["f1"])[
        "model"
    ]
    structure_rows = posthoc_structure_rows(online_rows, scores_by_model[structure_model], structure_model)

    write_csv(os.path.join(output_dir, "unet_binary_model_results.csv"), model_rows)
    write_csv(os.path.join(output_dir, "unet_binary_baseline_results.csv"), baseline_rows)
    write_csv(os.path.join(output_dir, "unet_binary_confusion_matrices.csv"), confusion)
    write_csv(os.path.join(output_dir, "unet_threshold_results.csv"), threshold_rows)
    write_csv(os.path.join(output_dir, "unet_feature_correlations.csv"), corr_rows)
    write_csv(os.path.join(output_dir, "unet_posthoc_structure_results.csv"), structure_rows)
    plot_pr_curve(os.path.join(output_dir, "unet_pr_curves.png"), labels, scores_by_model)
    plot_feature_importance(os.path.join(output_dir, "unet_feature_importance.png"), corr_rows)
    write_summary(
        os.path.join(output_dir, "unet_trust_summary.md"),
        online_rows,
        model_rows,
        baseline_rows,
        threshold_rows,
        corr_rows,
        oracle_rows,
        structure_rows,
    )
    print(f"Saved focused U-Net trust-signal analysis to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
