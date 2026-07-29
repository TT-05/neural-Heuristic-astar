import argparse
import csv
import math
import os
from collections import Counter, defaultdict

import numpy as np


METHODS = ["large_g", "mlp", "unet"]
METADATA = {
    "benchmark",
    "map_id",
    "seed",
    "map_mode",
    "structured_type",
    "start",
    "goal",
    "best_method",
    "best_methods",
}


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


def to_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
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


def method_mean(row, suffix):
    return mean(to_float(row, f"{method}_{suffix}") for method in METHODS)


def method_std(row, suffix):
    return std(to_float(row, f"{method}_{suffix}") for method in METHODS)


def method_range(row, suffix):
    values = [to_float(row, f"{method}_{suffix}") for method in METHODS]
    return max(values) - min(values)


def load_behavior_rows(online_rows, oracle_rows):
    oracle_by_id = {row["map_id"]: row for row in oracle_rows if row.get("map_id")}
    rows = []
    for row in online_rows:
        oracle = oracle_by_id.get(row.get("map_id"), {})
        expanded = {method: to_float(row, f"{method}_expanded") for method in METHODS}
        competitor_mean = {
            method: mean(value for other, value in expanded.items() if other != method) for method in METHODS
        }
        behavior = {
            "map_id": row.get("map_id", ""),
            "benchmark": row.get("benchmark", ""),
            "structured_type": row.get("structured_type", ""),
            "best_method": row.get("best_method", ""),
            "large_g_expanded": expanded["large_g"],
            "mlp_expanded": expanded["mlp"],
            "unet_expanded": expanded["unet"],
            "mean_expanded": mean(expanded.values()),
            "best_expanded": min(expanded.values()),
            "expansion_efficiency": 1.0 / max(1.0, min(expanded.values())),
            "unet_improvement": competitor_mean["unet"] - expanded["unet"],
            "mlp_improvement": competitor_mean["mlp"] - expanded["mlp"],
            "large_g_improvement": competitor_mean["large_g"] - expanded["large_g"],
            "goal_progress_rate": method_mean(row, "early_goal_progress_rate"),
            "goal_progress_rate_variability": method_std(row, "early_goal_progress_rate"),
            "goal_progress_variance": method_mean(row, "early_progress_std"),
            "frontier_spread": method_mean(row, "early_frontier_bbox_area"),
            "frontier_spread_variability": method_std(row, "early_frontier_bbox_area"),
            "frontier_growth_rate": method_mean(row, "early_frontier_final_size")
            / max(1.0, method_mean(row, "early_expanded_count")),
            "expansion_direction_consistency": method_mean(row, "early_direction_consistency"),
            "tie_set_size": method_mean(row, "tie_set_mean_size"),
            "tie_set_density": method_mean(row, "tie_event_fraction"),
            "search_width": method_mean(row, "early_bbox_area"),
            "search_depth": method_mean(row, "early_expanded_count"),
            "expansion_persistence": method_mean(row, "early_direction_consistency")
            - method_mean(row, "early_progress_std"),
            "local_branching": method_mean(row, "early_frontier_mean_size")
            / max(1.0, method_mean(row, "early_expanded_count")),
            "heuristic_disagreement": mean(
                [
                    to_float(row, "large_g_unet_mlp_pairwise_disagreement_rate"),
                    to_float(row, "mlp_unet_mlp_pairwise_disagreement_rate"),
                    to_float(row, "unet_unet_mlp_pairwise_disagreement_rate"),
                    to_float(row, "large_g_unet_mlp_choice_disagreement_rate"),
                    to_float(row, "mlp_unet_mlp_choice_disagreement_rate"),
                    to_float(row, "unet_unet_mlp_choice_disagreement_rate"),
                ]
            ),
            "heuristic_variance": mean(
                [
                    method_mean(row, "unet_h_std"),
                    method_mean(row, "mlp_h_std"),
                    method_mean(row, "unet_h_range"),
                    method_mean(row, "mlp_h_range"),
                ]
            ),
            "unet_mlp_goal_progress_advantage": to_float(row, "unet_minus_mlp_early_goal_progress_rate"),
            "unet_large_g_goal_progress_advantage": to_float(row, "unet_minus_large_g_early_goal_progress_rate"),
            "unet_mlp_frontier_spread_gap": to_float(row, "unet_minus_mlp_early_spread"),
            "unet_large_g_frontier_spread_gap": to_float(row, "unet_minus_large_g_early_spread"),
            "unet_mlp_h_smoothness_gap": to_float(row, "unet_minus_mlp_h_smoothness"),
        }
        if oracle:
            behavior.update(
                {
                    "route_bias": mean(
                        to_float(oracle, f"{method}_oracle_early_route_concentration") for method in METHODS
                    ),
                    "off_path_exploration": mean(
                        to_float(oracle, f"{method}_oracle_early_off_path_ratio") for method in METHODS
                    ),
                    "distance_to_optimal_path": mean(
                        to_float(oracle, f"{method}_oracle_mean_distance_to_path") for method in METHODS
                    ),
                    "critical_overestimation": mean(
                        [
                            to_float(oracle, "mlp_oracle_critical_large_overestimate_rate"),
                            to_float(oracle, "unet_oracle_critical_large_overestimate_rate"),
                        ]
                    ),
                }
            )
        rows.append(behavior)
    return rows


def variable_metadata(variable):
    oracle = variable in {
        "route_bias",
        "off_path_exploration",
        "distance_to_optimal_path",
        "critical_overestimation",
    }
    online = not oracle
    controllable = variable in {
        "goal_progress_rate",
        "frontier_spread",
        "frontier_growth_rate",
        "expansion_direction_consistency",
        "tie_set_density",
        "search_width",
        "expansion_persistence",
        "local_branching",
        "heuristic_disagreement",
    }
    likely_descriptive = variable in {
        "route_bias",
        "off_path_exploration",
        "distance_to_optimal_path",
        "critical_overestimation",
        "heuristic_variance",
        "search_depth",
        "goal_progress_variance",
    }
    return {
        "online_measurable": int(online),
        "oracle_only": int(oracle),
        "plausibly_controllable": int(controllable),
        "likely_descriptive_only": int(likely_descriptive),
        "control_plausibility": "high" if controllable and online else ("diagnostic_only" if oracle else "medium"),
    }


def behavior_variables(rows):
    keys = sorted(set().union(*(row.keys() for row in rows)))
    return [
        key
        for key in keys
        if key not in METADATA
        and not key.endswith("_expanded")
        and key
        not in {
            "mean_expanded",
            "best_expanded",
            "expansion_efficiency",
            "unet_improvement",
            "mlp_improvement",
            "large_g_improvement",
        }
    ]


def target_values(rows):
    return {
        "expanded_nodes": [row["best_expanded"] for row in rows],
        "expansion_efficiency": [row["expansion_efficiency"] for row in rows],
        "unet_improvement": [row["unet_improvement"] for row in rows],
        "mlp_improvement": [row["mlp_improvement"] for row in rows],
        "large_g_improvement": [row["large_g_improvement"] for row in rows],
    }


def distribution(values):
    array = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def correlation_rows(rows, variables):
    targets = target_values(rows)
    output = []
    for variable in variables:
        xs = [row[variable] for row in rows]
        dist = distribution(xs)
        meta = variable_metadata(variable)
        for target, ys in targets.items():
            output.append(
                {
                    "variable": variable,
                    "target": target,
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                    **dist,
                    **meta,
                }
            )
    return output


def redundancy_rows(rows, variables):
    output = []
    for i, left in enumerate(variables):
        xs = [row[left] for row in rows]
        for right in variables[i + 1 :]:
            ys = [row[right] for row in rows]
            r = pearson(xs, ys)
            output.append(
                {
                    "variable_a": left,
                    "variable_b": right,
                    "pearson": r,
                    "abs_pearson": abs(r),
                    "highly_redundant": int(abs(r) >= 0.85),
                }
            )
    return sorted(output, key=lambda row: row["abs_pearson"], reverse=True)


def connected_components(edges, variables):
    parent = {variable: variable for variable in variables}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        root_l = find(left)
        root_r = find(right)
        if root_l != root_r:
            parent[root_r] = root_l

    for left, right in edges:
        union(left, right)
    groups = {}
    for variable in variables:
        groups.setdefault(find(variable), []).append(variable)
    return list(groups.values())


def pca_summary(rows, variables, max_components=6):
    matrix = np.array([[row[variable] for variable in variables] for row in rows], dtype=float)
    matrix = (matrix - matrix.mean(axis=0)) / np.where(matrix.std(axis=0) == 0, 1.0, matrix.std(axis=0))
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    variance = singular_values**2
    explained = variance / variance.sum() if variance.sum() else variance
    output = []
    for index in range(min(max_components, len(variables))):
        loadings = vt[index]
        top = sorted(zip(variables, loadings), key=lambda item: abs(item[1]), reverse=True)[:6]
        output.append(
            {
                "dimension": f"PC{index + 1}",
                "explained_variance_ratio": float(explained[index]),
                "top_variables": "; ".join(f"{name}:{weight:.3f}" for name, weight in top),
            }
        )
    return output


def cluster_rows(correlations, variables, redundancy):
    edges = [(row["variable_a"], row["variable_b"]) for row in redundancy if row["abs_pearson"] >= 0.75]
    components = connected_components(edges, variables)
    corr_lookup = defaultdict(float)
    for row in correlations:
        corr_lookup[row["variable"]] = max(corr_lookup[row["variable"]], abs(row["spearman"]))
    output = []
    for index, component in enumerate(sorted(components, key=lambda group: (-len(group), group[0])), start=1):
        representative = max(component, key=lambda variable: corr_lookup[variable])
        output.append(
            {
                "cluster_id": index,
                "size": len(component),
                "variables": "; ".join(component),
                "representative": representative,
            }
        )
    return output


def importance_rows(correlations, redundancy):
    by_variable = defaultdict(dict)
    for row in correlations:
        by_variable[row["variable"]][row["target"]] = abs(row["spearman"])
    redundant_count = Counter()
    for row in redundancy:
        if row["abs_pearson"] >= 0.85:
            redundant_count[row["variable_a"]] += 1
            redundant_count[row["variable_b"]] += 1
    output = []
    for variable, scores in by_variable.items():
        meta = variable_metadata(variable)
        max_target = max(scores, key=scores.get)
        max_corr = scores[max_target]
        control_bonus = 0.15 if meta["plausibly_controllable"] else 0.0
        online_bonus = 0.10 if meta["online_measurable"] else -0.10
        redundancy_penalty = min(0.25, redundant_count[variable] * 0.03)
        score = max_corr + control_bonus + online_bonus - redundancy_penalty
        output.append(
            {
                "variable": variable,
                "importance_score": score,
                "strongest_target": max_target,
                "strongest_abs_spearman": max_corr,
                "redundant_links_ge_085": redundant_count[variable],
                **meta,
            }
        )
    return sorted(output, key=lambda row: row["importance_score"], reverse=True)


def save_dependency_graph(path, importance, correlations):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    top = [row["variable"] for row in importance[:8]]
    corr = {
        (row["variable"], row["target"]): row["spearman"]
        for row in correlations
        if row["variable"] in top and row["target"] == "expansion_efficiency"
    }
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    neural = (0.1, 0.5)
    efficiency = (0.9, 0.5)
    ax.text(*neural, "neural guidance\n/ tie-break policy", ha="center", va="center", bbox=dict(boxstyle="round", fc="#e8eef8"))
    ax.text(*efficiency, "search efficiency", ha="center", va="center", bbox=dict(boxstyle="round", fc="#e8f6ea"))
    ys = np.linspace(0.88, 0.12, len(top))
    for y, variable in zip(ys, top):
        x = 0.5
        ax.text(x, y, variable, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round", fc="#fff4d6"))
        ax.add_patch(FancyArrowPatch((0.2, 0.5), (x - 0.09, y), arrowstyle="->", mutation_scale=12, alpha=0.55))
        ax.add_patch(FancyArrowPatch((x + 0.09, y), (0.8, 0.5), arrowstyle="->", mutation_scale=12, alpha=0.55))
        ax.text(0.68, y + 0.02, f"rho={corr.get((variable, 'expansion_efficiency'), 0.0):.2f}", fontsize=8)
    ax.set_title("Measured Behavior Dependency Graph")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_dimension_summary(path, pca_rows, cluster_rows_):
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Behavior Dimension Summary\n\n")
        file.write("## PCA Dimensions\n\n")
        file.write("| dimension | explained variance | top variables |\n|---|---:|---|\n")
        for row in pca_rows:
            file.write(
                f"| {row['dimension']} | {row['explained_variance_ratio']:.3f} | {row['top_variables']} |\n"
            )
        file.write("\n## Correlation Clusters\n\n")
        file.write("| cluster | size | representative | variables |\n|---:|---:|---|---|\n")
        for row in cluster_rows_:
            file.write(f"| {row['cluster_id']} | {row['size']} | {row['representative']} | {row['variables']} |\n")


def write_summary(path, variables, correlations, redundancy, importance, pca_rows):
    top_eff = sorted(
        [row for row in correlations if row["target"] == "expansion_efficiency"],
        key=lambda row: abs(row["spearman"]),
        reverse=True,
    )[:8]
    top_control = [row for row in importance if row["plausibly_controllable"] and row["online_measurable"]][:8]
    descriptive = [row for row in importance if row["likely_descriptive_only"]][:8]
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Unified Search-Behavior State Analysis\n\n")
        file.write(
            "This report builds a unified behavior representation from existing outputs. It does not design a new algorithm, "
            "modify A*, train models, or use map structure labels as prediction inputs.\n\n"
        )
        file.write(f"Analyzed {len(variables)} behavior variables.\n\n")
        file.write("## Strongest Correlations With Search Efficiency\n\n")
        file.write("| variable | Spearman | online | controllable | interpretation |\n|---|---:|---:|---:|---|\n")
        for row in top_eff:
            meta = variable_metadata(row["variable"])
            interp = "candidate control variable" if meta["plausibly_controllable"] and meta["online_measurable"] else "diagnostic/descriptive"
            file.write(
                f"| {row['variable']} | {row['spearman']:.3f} | {meta['online_measurable']} "
                f"| {meta['plausibly_controllable']} | {interp} |\n"
            )
        file.write("\n## Fundamental Candidate Variables\n\n")
        file.write("| variable | score | target | abs Spearman | redundancy links |\n|---|---:|---|---:|---:|\n")
        for row in top_control:
            file.write(
                f"| {row['variable']} | {row['importance_score']:.3f} | {row['strongest_target']} "
                f"| {row['strongest_abs_spearman']:.3f} | {row['redundant_links_ge_085']} |\n"
            )
        file.write("\n## Likely Consequences Or Diagnostics\n\n")
        file.write("| variable | strongest target | abs Spearman | reason |\n|---|---|---:|---|\n")
        for row in descriptive:
            file.write(
                f"| {row['variable']} | {row['strongest_target']} | {row['strongest_abs_spearman']:.3f} "
                "| oracle-only or hard to directly control |\n"
            )
        file.write("\n## Redundancy\n\n")
        file.write(
            f"Highly redundant pairs with |Pearson| >= 0.85: {sum(1 for row in redundancy if row['abs_pearson'] >= 0.85)}.\n\n"
        )
        file.write("Most redundant pairs:\n\n")
        for row in redundancy[:8]:
            file.write(f"- {row['variable_a']} / {row['variable_b']}: r={row['pearson']:.3f}\n")
        file.write("\n## Major Behavior Dimensions\n\n")
        for row in pca_rows[:4]:
            file.write(
                f"- {row['dimension']} explains {row['explained_variance_ratio']:.3f}: {row['top_variables']}.\n"
            )
        file.write("\n## Critical Answers\n\n")
        file.write(
            "- Fundamental rather than redundant candidates: early goal-progress behavior, search width/frontier spread, "
            "frontier growth/local branching, tie-set density, expansion persistence, and heuristic disagreement.\n"
        )
        file.write(
            "- Online measurable variables: all main behavior variables except route_bias, off_path_exploration, "
            "distance_to_optimal_path, and critical_overestimation.\n"
        )
        file.write(
            "- Plausibly controllable variables: progress rate, frontier spread/growth, direction consistency, tie-set density, "
            "search width, local branching, and disagreement-sensitive ordering.\n"
        )
        file.write(
            "- Likely consequences rather than causes: off-path exploration, route bias, distance to optimal path, critical "
            "overestimation, raw search_depth, and goal_progress_variance are useful diagnostics but are not clean direct controls.\n"
        )
        file.write(
            "- Strongest candidates for future search control: variables that are online measurable, non-oracle, correlated with "
            "efficiency, and not purely redundant: goal_progress_rate, frontier_spread/search_width, frontier_growth_rate, "
            "local_branching, tie_set_density, expansion_persistence, and heuristic_disagreement.\n\n"
        )
        file.write(
            "This supports the idea of a compact behavior-state foundation for future structure-agnostic neural search control, "
            "but it remains evidence synthesis rather than an algorithm proposal.\n"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze unified search-behavior state variables.")
    parser.add_argument("--online-features", default="outputs/trust_signal_validation/online_features.csv")
    parser.add_argument("--oracle-features", default="outputs/trust_signal_validation/oracle_diagnostic_features.csv")
    parser.add_argument("--output-dir", default="outputs/search_behavior_state")
    return parser.parse_args()


def main(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    online_rows = read_csv(os.path.join(project_root, args.online_features))
    oracle_rows = read_csv(os.path.join(project_root, args.oracle_features))
    if not online_rows:
        raise RuntimeError("online_features.csv is required.")

    behavior_rows = load_behavior_rows(online_rows, oracle_rows)
    variables = behavior_variables(behavior_rows)
    correlations = correlation_rows(behavior_rows, variables)
    redundancy = redundancy_rows(behavior_rows, variables)
    pca_rows = pca_summary(behavior_rows, variables)
    clusters = cluster_rows(correlations, variables, redundancy)
    importance = importance_rows(correlations, redundancy)

    write_csv(os.path.join(output_dir, "behavior_correlations.csv"), correlations)
    write_csv(os.path.join(output_dir, "behavior_redundancy.csv"), redundancy)
    write_csv(os.path.join(output_dir, "behavior_clusters.csv"), clusters)
    write_csv(os.path.join(output_dir, "behavior_importance.csv"), importance)
    write_dimension_summary(os.path.join(output_dir, "behavior_dimension_summary.md"), pca_rows, clusters)
    write_summary(
        os.path.join(output_dir, "search_behavior_summary.md"),
        variables,
        correlations,
        redundancy,
        importance,
        pca_rows,
    )
    save_dependency_graph(os.path.join(output_dir, "behavior_dependency_graph.png"), importance, correlations)
    print(f"Saved search-behavior state analysis to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
