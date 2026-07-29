import argparse
import csv
import heapq
import math
import os

from bfs_label import compute_distance_to_goal
from gen_map import gen_map
from model import load_mlp_heuristic, load_unet_heuristic, make_mlp_table_heuristic, make_unet_heuristic
from structured_maps import generate_structured_map


STRUCTURED_TYPES = ["maze_like", "bottleneck", "large_block", "narrow_corridor"]
SECONDARY_METHODS = ["mlp", "unet"]
RESULT_METHODS = [
    "manhattan_mlp_tiebreak",
    "manhattan_unet_tiebreak",
    "manhattan_true_distance_tiebreak",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(row, key, default=0.0):
    value = row.get(key, "")
    if value == "":
        return default
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    return float(value)


def to_int(row, key, default=0):
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


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


def map_key(row, source):
    return (
        source,
        row["seed"],
        row["map_size"],
        row["obstacle_rate"],
        row.get("map_mode", "random"),
        row.get("structured_type", "random"),
        row.get("start_goal_mode", "random"),
        row.get("start_row", ""),
        row.get("start_col", ""),
        row.get("goal_row", ""),
        row.get("goal_col", ""),
    )


def group_maps(rows, source):
    grouped = {}
    for row in rows:
        if row.get("heuristic") not in {"manhattan", *RESULT_METHODS}:
            continue
        row = dict(row)
        row.setdefault("map_mode", "random")
        row.setdefault("structured_type", "random")
        row["source"] = source
        grouped.setdefault(map_key(row, source), {})[row["heuristic"]] = row
    return [methods for methods in grouped.values() if "manhattan" in methods and not methods["manhattan"].get("skip_reason")]


def checkpoint_path(project_root, checkpoint, name):
    if checkpoint in {"compatible", "best", "latest"}:
        if name == "mlp":
            return os.path.join(project_root, "checkpoints", "mlp_heuristic.pt")
        suffix = {
            "compatible": "unet_heuristic.pt",
            "best": "unet_heuristic_best.pt",
            "latest": "unet_heuristic_latest.pt",
        }[checkpoint]
        return os.path.join(project_root, "checkpoints", suffix)
    return checkpoint


def build_grid(sample):
    seed = to_int(sample, "seed")
    map_size = to_int(sample, "map_size")
    obstacle_rate = to_float(sample, "obstacle_rate")
    map_mode = sample.get("map_mode", "random")
    structured_type = sample.get("structured_type", "random")
    if map_mode == "structured":
        grid = generate_structured_map(map_size, map_size, seed, obstacle_rate, structured_type)
    else:
        grid = gen_map(width=map_size, height=map_size, seed=seed, obstacle_rate=obstacle_rate)
    start = (to_int(sample, "start_row"), to_int(sample, "start_col"))
    goal = (to_int(sample, "goal_row"), to_int(sample, "goal_col"))
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0
    return grid, start, goal


def neighbors(grid, cell):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    r, c = cell
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            yield nr, nc


def manhattan(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def prediction_table(grid, goal, heuristic):
    table = {}
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == 0:
                table[(r, c)] = float(heuristic((r, c), goal))
    return table


def pairwise_ordering_accuracy(true_values, pred_values):
    total = 0
    correct = 0
    concordant = 0
    discordant = 0
    for i in range(len(true_values)):
        for j in range(i + 1, len(true_values)):
            true_diff = true_values[i] - true_values[j]
            if true_diff == 0:
                continue
            pred_diff = pred_values[i] - pred_values[j]
            total += 1
            product = true_diff * pred_diff
            if product > 0:
                correct += 1
                concordant += 1
            elif product < 0:
                discordant += 1
    if total == 0:
        return 0.0, 0.0
    return correct / total, (concordant - discordant) / total


def top1_accuracy(true_values, pred_values):
    if not true_values:
        return 0.0
    min_true = min(true_values)
    min_pred = min(pred_values)
    true_best = {i for i, value in enumerate(true_values) if value == min_true}
    pred_best = {i for i, value in enumerate(pred_values) if value == min_pred}
    return 1.0 if true_best & pred_best else 0.0


def tie_set_metrics(nodes, true_table, secondary_table):
    true_values = [true_table[node] for node in nodes]
    secondary_values = [secondary_table[node] for node in nodes]
    pairwise, kendall = pairwise_ordering_accuracy(true_values, secondary_values)
    return {
        "pairwise_ordering_accuracy": pairwise,
        "spearman": spearman(true_values, secondary_values),
        "kendall_tau": kendall,
        "top1_accuracy": top1_accuracy(true_values, secondary_values),
    }


def simulate_manhattan_tie_sets(grid, start, goal, true_table, mlp_table, unet_table):
    open_set = []
    start_f = manhattan(start, goal)
    heapq.heappush(open_set, (start_f, 0, start))
    came_from = {}
    g_score = {start: 0}
    expanded = 0
    tie_rows = []

    while open_set:
        f_primary, current_g, current = heapq.heappop(open_set)
        if current_g > g_score.get(current, float("inf")):
            continue

        active_nodes = [current]
        for queued_f, queued_g, queued_node in open_set:
            if queued_f == f_primary and queued_g == g_score.get(queued_node, float("inf")):
                active_nodes.append(queued_node)

        if len(active_nodes) >= 2:
            for method, table in [("mlp", mlp_table), ("unet", unet_table)]:
                metrics = tie_set_metrics(active_nodes, true_table, table)
                tie_rows.append(
                    {
                        "row_type": "tie_set",
                        "expanded_step": expanded,
                        "f_primary": f_primary,
                        "tie_set_size": len(active_nodes),
                        "method": method,
                        **metrics,
                    }
                )

        expanded += 1
        if current == goal:
            break

        for neighbor in neighbors(grid, current):
            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(open_set, (tentative_g + manhattan(neighbor, goal), tentative_g, neighbor))

    return tie_rows, expanded


def result_values(methods):
    values = {}
    for method in RESULT_METHODS:
        row = methods.get(method)
        if not row:
            continue
        values[f"{method}_expanded"] = to_float(row, "expanded_nodes")
    return values


def analyze_map(methods, mlp_model, unet_model):
    sample = methods["manhattan"]
    grid, start, goal = build_grid(sample)
    distance_grid = compute_distance_to_goal(grid, goal)
    true_table = {}
    for r, row in enumerate(distance_grid):
        for c, value in enumerate(row):
            if value >= 0:
                true_table[(r, c)] = float(value)

    mlp_h = make_mlp_table_heuristic(mlp_model, grid, goal)
    unet_h = make_unet_heuristic(unet_model, grid, goal)
    mlp_table = prediction_table(grid, goal, mlp_h)
    unet_table = prediction_table(grid, goal, unet_h)
    tie_rows, expanded_steps = simulate_manhattan_tie_sets(grid, start, goal, true_table, mlp_table, unet_table)

    base = {
        "source": sample["source"],
        "seed": sample["seed"],
        "map_size": sample["map_size"],
        "obstacle_rate": sample["obstacle_rate"],
        "map_mode": sample.get("map_mode", "random"),
        "structured_type": sample.get("structured_type", "random"),
        "start_row": sample["start_row"],
        "start_col": sample["start_col"],
        "goal_row": sample["goal_row"],
        "goal_col": sample["goal_col"],
    }
    for row in tie_rows:
        row.update(base)

    map_rows = []
    for method in SECONDARY_METHODS:
        scoped = [row for row in tie_rows if row["method"] == method]
        map_row = {
            "row_type": "map_summary",
            "method": method,
            **base,
            "expanded_steps": expanded_steps,
            "tie_sets": len(scoped),
            "tie_step_fraction": len(scoped) / expanded_steps if expanded_steps else 0.0,
            "mean_tie_set_size": mean(row["tie_set_size"] for row in scoped),
            "mean_pairwise_ordering_accuracy": mean(row["pairwise_ordering_accuracy"] for row in scoped),
            "mean_spearman": mean(row["spearman"] for row in scoped),
            "mean_kendall_tau": mean(row["kendall_tau"] for row in scoped),
            "mean_top1_accuracy": mean(row["top1_accuracy"] for row in scoped),
            **result_values(methods),
        }
        if all(key in map_row for key in ["manhattan_unet_tiebreak_expanded", "manhattan_mlp_tiebreak_expanded"]):
            map_row["unet_minus_mlp_expanded"] = (
                map_row["manhattan_unet_tiebreak_expanded"] - map_row["manhattan_mlp_tiebreak_expanded"]
            )
        if "manhattan_true_distance_tiebreak_expanded" in map_row:
            if method == "unet":
                map_row["gap_to_true_distance_oracle"] = (
                    map_row["manhattan_unet_tiebreak_expanded"] - map_row["manhattan_true_distance_tiebreak_expanded"]
                )
            else:
                map_row["gap_to_true_distance_oracle"] = (
                    map_row["manhattan_mlp_tiebreak_expanded"] - map_row["manhattan_true_distance_tiebreak_expanded"]
                )
        map_rows.append(map_row)

    return tie_rows, map_rows


def rows_matching(rows, benchmark, structured_type=None):
    scoped = [row for row in rows if row.get("source") == benchmark]
    if structured_type:
        scoped = [row for row in scoped if row.get("structured_type") == structured_type]
    return scoped


def aggregate_rows(map_rows):
    output = []
    scopes = [("random", None), ("structured", None)] + [("structured", structured_type) for structured_type in STRUCTURED_TYPES]
    for benchmark, structured_type in scopes:
        for method in SECONDARY_METHODS:
            scoped = [row for row in rows_matching(map_rows, benchmark, structured_type) if row["method"] == method]
            output.append(
                {
                    "row_type": "aggregate",
                    "source": benchmark,
                    "structured_type": structured_type or "all",
                    "method": method,
                    "maps": len(scoped),
                    "tie_sets": sum(row["tie_sets"] for row in scoped),
                    "mean_tie_step_fraction": mean(row["tie_step_fraction"] for row in scoped),
                    "mean_tie_set_size": mean(row["mean_tie_set_size"] for row in scoped),
                    "mean_pairwise_ordering_accuracy": mean(row["mean_pairwise_ordering_accuracy"] for row in scoped),
                    "mean_spearman": mean(row["mean_spearman"] for row in scoped),
                    "mean_kendall_tau": mean(row["mean_kendall_tau"] for row in scoped),
                    "mean_top1_accuracy": mean(row["mean_top1_accuracy"] for row in scoped),
                    "mean_mlp_tiebreak_expanded": mean(row["manhattan_mlp_tiebreak_expanded"] for row in scoped),
                    "mean_unet_tiebreak_expanded": mean(row["manhattan_unet_tiebreak_expanded"] for row in scoped),
                    "mean_true_distance_tiebreak_expanded": mean(
                        row["manhattan_true_distance_tiebreak_expanded"] for row in scoped
                    ),
                }
            )
    return output


def correlation_rows(map_rows):
    output = []
    scopes = [("random", None), ("structured", None)] + [("structured", structured_type) for structured_type in STRUCTURED_TYPES]
    for benchmark, structured_type in scopes:
        scoped_maps = rows_matching(map_rows, benchmark, structured_type)
        by_key = {}
        for row in scoped_maps:
            key = (
                row["seed"],
                row["map_size"],
                row["obstacle_rate"],
                row["start_row"],
                row["start_col"],
                row["goal_row"],
                row["goal_col"],
            )
            by_key.setdefault(key, {})[row["method"]] = row

        paired = [methods for methods in by_key.values() if "mlp" in methods and "unet" in methods]
        for metric in ["mean_pairwise_ordering_accuracy", "mean_top1_accuracy", "mean_spearman", "mean_kendall_tau"]:
            xs = [methods["unet"][metric] - methods["mlp"][metric] for methods in paired]
            ys = [methods["unet"]["unet_minus_mlp_expanded"] for methods in paired]
            output.append(
                {
                    "row_type": "correlation",
                    "source": benchmark,
                    "structured_type": structured_type or "all",
                    "x": f"unet_minus_mlp_{metric}",
                    "y": "unet_minus_mlp_expanded",
                    "n": len(xs),
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                }
            )
        for method in SECONDARY_METHODS:
            method_rows = [row for row in scoped_maps if row["method"] == method]
            for metric in ["mean_pairwise_ordering_accuracy", "mean_top1_accuracy"]:
                xs = [row[metric] for row in method_rows]
                expanded_key = f"manhattan_{method}_tiebreak_expanded"
                ys = [row[expanded_key] for row in method_rows]
                output.append(
                    {
                        "row_type": "correlation",
                        "source": benchmark,
                        "structured_type": structured_type or "all",
                        "method": method,
                        "x": metric,
                        "y": expanded_key,
                        "n": len(xs),
                        "pearson": pearson(xs, ys),
                        "spearman": spearman(xs, ys),
                    }
                )
    return output


def save_plots(path, aggregate, correlations):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(path)), "matplotlib_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    structured = [row for row in aggregate if row["source"] == "structured" and row["structured_type"] in STRUCTURED_TYPES]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, metric in enumerate(["mean_pairwise_ordering_accuracy", "mean_top1_accuracy", "mean_tie_step_fraction"]):
        ax = axes[idx]
        labels = STRUCTURED_TYPES
        mlp = [next(row for row in structured if row["structured_type"] == label and row["method"] == "mlp")[metric] for label in labels]
        unet = [next(row for row in structured if row["structured_type"] == label and row["method"] == "unet")[metric] for label in labels]
        x = range(len(labels))
        ax.bar([value - 0.2 for value in x], mlp, width=0.4, label="MLP")
        ax.bar([value + 0.2 for value in x], unet, width=0.4, label="U-Net")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(metric)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_summary(path, aggregate, correlations):
    by_scope = {(row["source"], row["structured_type"], row["method"]): row for row in aggregate}
    corr_lookup = {(row["source"], row["structured_type"], row.get("x", "")): row for row in correlations}
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Tie-Set Ordering Analysis\n\n")
        file.write(
            "This analysis instruments Manhattan-primary A* locally and measures secondary ordering only inside active equal-f tie sets. "
            "It does not modify production A* or benchmark outputs.\n\n"
        )
        file.write("## Aggregate Ordering Quality\n\n")
        file.write("| Scope | Method | Tie sets | Tie-step fraction | Tie size | Pairwise | Spearman | Kendall | Top1 |\n")
        file.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for source, structured_type in [("random", "all"), ("structured", "all")] + [
            ("structured", item) for item in STRUCTURED_TYPES
        ]:
            for method in SECONDARY_METHODS:
                row = by_scope[(source, structured_type, method)]
                file.write(
                    f"| {source}/{structured_type} | {method} | {row['tie_sets']} | "
                    f"{row['mean_tie_step_fraction']:.3f} | {row['mean_tie_set_size']:.3f} | "
                    f"{row['mean_pairwise_ordering_accuracy']:.3f} | {row['mean_spearman']:.3f} | "
                    f"{row['mean_kendall_tau']:.3f} | {row['mean_top1_accuracy']:.3f} |\n"
                )
        file.write("\n")

        file.write("## Key Questions\n\n")
        random_mlp = by_scope[("random", "all", "mlp")]
        random_unet = by_scope[("random", "all", "unet")]
        structured_mlp = by_scope[("structured", "all", "mlp")]
        structured_unet = by_scope[("structured", "all", "unet")]
        file.write(
            f"Q1: In random maps, pairwise tie-set ordering is MLP={random_mlp['mean_pairwise_ordering_accuracy']:.3f}, "
            f"U-Net={random_unet['mean_pairwise_ordering_accuracy']:.3f}. In structured maps, it is "
            f"MLP={structured_mlp['mean_pairwise_ordering_accuracy']:.3f}, U-Net={structured_unet['mean_pairwise_ordering_accuracy']:.3f}.\n\n"
        )
        file.write("Q2/Q3 structure dependence:\n")
        for structured_type in STRUCTURED_TYPES:
            mlp = by_scope[("structured", structured_type, "mlp")]
            unet = by_scope[("structured", structured_type, "unet")]
            winner = "U-Net" if unet["mean_pairwise_ordering_accuracy"] > mlp["mean_pairwise_ordering_accuracy"] else "MLP"
            file.write(
                f"- {structured_type}: {winner} has higher pairwise ordering "
                f"(MLP={mlp['mean_pairwise_ordering_accuracy']:.3f}, U-Net={unet['mean_pairwise_ordering_accuracy']:.3f}).\n"
            )
        file.write("\n")

        corr = corr_lookup.get(("structured", "all", "unet_minus_mlp_mean_pairwise_ordering_accuracy"), {})
        file.write(
            f"Q4: U-Net-minus-MLP pairwise ordering vs U-Net-minus-MLP expanded nodes has Spearman "
            f"{float(corr.get('spearman', 0.0)):.3f} on structured maps. Negative values mean better relative ordering tends "
            "to reduce relative expansions.\n\n"
        )
        file.write(
            f"Q5: Tie-breaking matters often: active tie-set fractions are random MLP={random_mlp['mean_tie_step_fraction']:.3f}, "
            f"structured MLP={structured_mlp['mean_tie_step_fraction']:.3f}; mean tie-set sizes are random="
            f"{random_mlp['mean_tie_set_size']:.3f}, structured={structured_mlp['mean_tie_set_size']:.3f}.\n\n"
        )
        file.write(
            f"Q6: On structured maps, MLP tie-breaking remains "
            f"{structured_mlp['mean_mlp_tiebreak_expanded'] - structured_mlp['mean_true_distance_tiebreak_expanded']:.3f} "
            f"expanded nodes above true-distance oracle tie-breaking, while U-Net remains "
            f"{structured_unet['mean_unet_tiebreak_expanded'] - structured_unet['mean_true_distance_tiebreak_expanded']:.3f} "
            "above oracle. This analysis focuses on MLP/U-Net quality relative to true distance inside tie sets, not on claiming causality.\n\n"
        )
        file.write("## Interpretation\n\n")
        file.write(
            "Secondary heuristics have substantial opportunity to influence Manhattan-primary A* because equal-f tie sets are frequent. "
            "The structure-specific ordering comparison should be interpreted as evidence about secondary ordering quality only; "
            "it does not make U-Net admissible and does not introduce a new algorithm.\n"
        )


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "tie_set_ordering")
    os.makedirs(output_dir, exist_ok=True)

    random_groups = group_maps(read_csv(args.random_results), "random")
    structured_groups = group_maps(read_csv(args.structured_results), "structured")
    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint, "unet"))

    tie_rows = []
    map_rows = []
    for group in random_groups + structured_groups:
        group_ties, group_map_rows = analyze_map(group, mlp_model, unet_model)
        tie_rows.extend(group_ties)
        map_rows.extend(group_map_rows)

    aggregate = aggregate_rows(map_rows)
    correlations = correlation_rows(map_rows)
    all_rows = tie_rows + map_rows + aggregate + correlations
    write_csv(os.path.join(output_dir, "tie_set_ordering_statistics.csv"), all_rows)
    save_plots(os.path.join(output_dir, "tie_set_ordering_plots.png"), aggregate, correlations)
    write_summary(os.path.join(output_dir, "tie_set_ordering_summary.md"), aggregate, correlations)
    print(f"Saved tie-set ordering analysis outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze ordering quality within Manhattan-primary A* tie sets.")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_tiebreak_controls_100.csv")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--checkpoint", default="best")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
