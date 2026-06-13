import argparse
import csv
import math
import os
from collections import deque

from bfs_label import compute_distance_to_goal
from gen_map import gen_map
from model import load_mlp_heuristic, load_unet_heuristic, make_mlp_table_heuristic, make_unet_heuristic
from structured_maps import generate_structured_map


METHODS = ["dijkstra", "manhattan", "mlp_table", "unet"]
STRUCTURED_SCOPES = ["maze_like", "bottleneck", "large_block", "narrow_corridor"]


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


def group_result_rows(rows, source):
    grouped = {}
    for row in rows:
        if row.get("heuristic") not in METHODS:
            continue
        row = dict(row)
        row.setdefault("map_mode", "random")
        row.setdefault("structured_type", "random")
        row.setdefault("start_goal_mode", "random")
        row["source"] = source
        grouped.setdefault(map_key(row, source), {})[row["heuristic"]] = row
    return grouped


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
    return grid


def neighbors(grid, cell):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    r, c = cell
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            yield nr, nc


def reconstruct_shortest_path(grid, distance_grid, start, goal):
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


def free_cells(grid, distance_grid):
    cells = []
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == 0 and distance_grid[r][c] >= 0:
                cells.append((r, c))
    return cells


def articulation_points(grid, cells):
    cell_set = set(cells)
    discovery = {}
    low = {}
    parent = {}
    points = set()
    time = 0

    def dfs(cell):
        nonlocal time
        discovery[cell] = time
        low[cell] = time
        time += 1
        children = 0
        for nxt in neighbors(grid, cell):
            if nxt not in cell_set:
                continue
            if nxt not in discovery:
                parent[nxt] = cell
                children += 1
                dfs(nxt)
                low[cell] = min(low[cell], low[nxt])
                if cell not in parent and children > 1:
                    points.add(cell)
                if cell in parent and low[nxt] >= discovery[cell]:
                    points.add(cell)
            elif parent.get(cell) != nxt:
                low[cell] = min(low[cell], discovery[nxt])

    for cell in cells:
        if cell not in discovery:
            dfs(cell)
    return points


def cells_within_k(grid, seeds, k):
    if not seeds:
        return set()
    reached = set(seeds)
    queue = deque((cell, 0) for cell in seeds)
    while queue:
        cell, distance = queue.popleft()
        if distance >= k:
            continue
        for nxt in neighbors(grid, cell):
            if nxt in reached:
                continue
            reached.add(nxt)
            queue.append((nxt, distance + 1))
    return reached


def identify_route_critical_cells(grid, distance_grid, path, k):
    all_free = free_cells(grid, distance_grid)
    path_cells = set(path)
    path_neighborhood = cells_within_k(grid, path_cells, k)
    low_degree = {cell for cell in all_free if sum(1 for _ in neighbors(grid, cell)) <= 2}
    articulations = articulation_points(grid, all_free)
    structural = (low_degree | articulations) & path_neighborhood
    critical = path_cells | path_neighborhood | structural
    return {
        "all_free": set(all_free),
        "critical": critical,
        "path": path_cells,
        "path_neighborhood": path_neighborhood,
        "low_degree_near_path": low_degree & path_neighborhood,
        "articulation_near_path": articulations & path_neighborhood,
    }


def prediction_grid(grid, goal, heuristic):
    return [[0.0 if grid[r][c] == 1 else heuristic((r, c), goal) for c in range(len(grid[0]))] for r in range(len(grid))]


def ordering_accuracy(true_values, predicted_values):
    total = 0
    concordant = 0
    for i in range(len(true_values)):
        for j in range(i + 1, len(true_values)):
            true_diff = true_values[i] - true_values[j]
            if true_diff == 0:
                continue
            pred_diff = predicted_values[i] - predicted_values[j]
            total += 1
            if true_diff * pred_diff > 0:
                concordant += 1
    return concordant / total if total else 0.0


def subset_metrics(distance_grid, pred_grid, cells):
    true_values = []
    pred_values = []
    errors = []
    for r, c in sorted(cells):
        true_value = distance_grid[r][c]
        if true_value < 0:
            continue
        pred_value = pred_grid[r][c]
        true_values.append(float(true_value))
        pred_values.append(float(pred_value))
        errors.append(float(pred_value - true_value))
    if not errors:
        return {
            "cell_count": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "overestimate_rate": 0.0,
            "large_overestimate_rate": 0.0,
            "ordering_accuracy": 0.0,
            "spearman": 0.0,
        }
    return {
        "cell_count": len(errors),
        "mae": mean(abs(error) for error in errors),
        "rmse": math.sqrt(mean(error * error for error in errors)),
        "overestimate_rate": sum(1 for error in errors if error > 0.0) / len(errors),
        "large_overestimate_rate": sum(1 for error in errors if error > 3.0) / len(errors),
        "ordering_accuracy": ordering_accuracy(true_values, pred_values),
        "spearman": spearman(true_values, pred_values),
    }


def add_metric_prefix(row, prefix, metrics):
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def enrich_map_metrics(grouped, mlp_model, unet_model, k):
    output = []
    for methods in grouped.values():
        if not all(method in methods for method in METHODS):
            continue
        sample = methods["unet"]
        if sample.get("skip_reason", ""):
            continue
        start = (to_int(sample, "start_row"), to_int(sample, "start_col"))
        goal = (to_int(sample, "goal_row"), to_int(sample, "goal_col"))
        grid = build_grid(sample)
        distance_grid = compute_distance_to_goal(grid, goal)
        path = reconstruct_shortest_path(grid, distance_grid, start, goal)
        if not path:
            continue
        masks = identify_route_critical_cells(grid, distance_grid, path, k)
        unet_h = make_unet_heuristic(unet_model, grid, goal)
        mlp_h = make_mlp_table_heuristic(mlp_model, grid, goal)
        unet_grid = prediction_grid(grid, goal, unet_h)
        mlp_grid = prediction_grid(grid, goal, mlp_h)

        row = {
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
            "optimal_cost": to_float(methods["dijkstra"], "path_length"),
            "unet_path_length": to_float(methods["unet"], "path_length"),
            "mlp_path_length": to_float(methods["mlp_table"], "path_length"),
            "cost_gap": to_float(methods["unet"], "path_length") - to_float(methods["dijkstra"], "path_length"),
            "unet_minus_mlp_expanded": to_float(methods["unet"], "expanded_nodes") - to_float(methods["mlp_table"], "expanded_nodes"),
            "unet_wins": to_float(methods["unet"], "expanded_nodes") < to_float(methods["mlp_table"], "expanded_nodes"),
            "mlp_wins": to_float(methods["unet"], "expanded_nodes") > to_float(methods["mlp_table"], "expanded_nodes"),
            "unet_optimal": methods["unet"]["optimal"] == "True",
            "critical_k": k,
            "path_cell_count": len(masks["path"]),
            "critical_cell_count": len(masks["critical"]),
            "all_free_cell_count": len(masks["all_free"]),
            "low_degree_near_path_count": len(masks["low_degree_near_path"]),
            "articulation_near_path_count": len(masks["articulation_near_path"]),
        }
        for scope_name, cells in [("global", masks["all_free"]), ("critical", masks["critical"])]:
            unet_metrics = subset_metrics(distance_grid, unet_grid, cells)
            mlp_metrics = subset_metrics(distance_grid, mlp_grid, cells)
            add_metric_prefix(row, f"unet_{scope_name}", unet_metrics)
            add_metric_prefix(row, f"mlp_{scope_name}", mlp_metrics)
            for metric in ["mae", "rmse", "overestimate_rate", "large_overestimate_rate", "ordering_accuracy", "spearman"]:
                row[f"{scope_name}_unet_minus_mlp_{metric}"] = unet_metrics[metric] - mlp_metrics[metric]
        row["unet_critical_minus_global_overestimate_rate"] = row["unet_critical_overestimate_rate"] - row["unet_global_overestimate_rate"]
        row["unet_critical_minus_global_large_overestimate_rate"] = (
            row["unet_critical_large_overestimate_rate"] - row["unet_global_large_overestimate_rate"]
        )
        row["unet_critical_minus_global_mae"] = row["unet_critical_mae"] - row["unet_global_mae"]
        row["unet_critical_minus_global_ordering_accuracy"] = (
            row["unet_critical_ordering_accuracy"] - row["unet_global_ordering_accuracy"]
        )
        output.append(row)
    return output


def rows_matching(rows, scope):
    if scope == "all":
        return rows
    if scope == "unet_wins":
        return [row for row in rows if row["unet_wins"]]
    if scope == "mlp_wins":
        return [row for row in rows if row["mlp_wins"]]
    if scope == "unet_optimal":
        return [row for row in rows if row["unet_optimal"]]
    if scope == "unet_non_optimal":
        return [row for row in rows if not row["unet_optimal"] or row["cost_gap"] > 0]
    if scope in STRUCTURED_SCOPES:
        return [row for row in rows if row["structured_type"] == scope]
    return []


def summary_row(rows, scope):
    scoped = rows_matching(rows, scope)
    row = {
        "row_type": "summary",
        "scope": scope,
        "maps": len(scoped),
        "mean_unet_minus_mlp_expanded": mean(row["unet_minus_mlp_expanded"] for row in scoped),
        "mean_cost_gap": mean(row["cost_gap"] for row in scoped),
        "mean_critical_fraction": mean(row["critical_cell_count"] / row["all_free_cell_count"] for row in scoped if row["all_free_cell_count"]),
    }
    for prefix in ["unet_global", "unet_critical", "mlp_global", "mlp_critical"]:
        for metric in ["mae", "rmse", "overestimate_rate", "large_overestimate_rate", "ordering_accuracy", "spearman"]:
            row[f"mean_{prefix}_{metric}"] = mean(item[f"{prefix}_{metric}"] for item in scoped)
    for metric in ["mae", "rmse", "overestimate_rate", "large_overestimate_rate", "ordering_accuracy", "spearman"]:
        row[f"mean_global_unet_minus_mlp_{metric}"] = mean(item[f"global_unet_minus_mlp_{metric}"] for item in scoped)
        row[f"mean_critical_unet_minus_mlp_{metric}"] = mean(item[f"critical_unet_minus_mlp_{metric}"] for item in scoped)
    row["mean_unet_critical_minus_global_overestimate_rate"] = mean(
        item["unet_critical_minus_global_overestimate_rate"] for item in scoped
    )
    row["mean_unet_critical_minus_global_large_overestimate_rate"] = mean(
        item["unet_critical_minus_global_large_overestimate_rate"] for item in scoped
    )
    row["mean_unet_critical_minus_global_mae"] = mean(item["unet_critical_minus_global_mae"] for item in scoped)
    row["mean_unet_critical_minus_global_ordering_accuracy"] = mean(
        item["unet_critical_minus_global_ordering_accuracy"] for item in scoped
    )
    return row


def corr_row(rows, x_key, y_key, scope, metric_scope):
    scoped = rows_matching(rows, scope)
    pairs = [(row[x_key], row[y_key]) for row in scoped if row.get(x_key, "") != "" and row.get(y_key, "") != ""]
    xs = [float(pair[0]) for pair in pairs]
    ys = [float(pair[1]) for pair in pairs]
    return {
        "row_type": "predictive_power",
        "scope": scope,
        "metric_scope": metric_scope,
        "x": x_key,
        "y": y_key,
        "n": len(pairs),
        "pearson": pearson(xs, ys),
        "spearman": spearman(xs, ys),
        "abs_spearman": abs(spearman(xs, ys)),
    }


def predictive_rows(rows):
    output = []
    scopes = ["all", "unet_wins", "mlp_wins", "unet_optimal", "unet_non_optimal"] + STRUCTURED_SCOPES
    metric_bases = ["mae", "rmse", "overestimate_rate", "large_overestimate_rate", "ordering_accuracy", "spearman"]
    targets = ["unet_minus_mlp_expanded", "cost_gap"]
    for scope in scopes:
        for target in targets:
            for metric in metric_bases:
                output.append(corr_row(rows, f"unet_global_{metric}", target, scope, "global"))
                output.append(corr_row(rows, f"unet_critical_{metric}", target, scope, "critical"))
                output.append(corr_row(rows, f"global_unet_minus_mlp_{metric}", target, scope, "global_delta"))
                output.append(corr_row(rows, f"critical_unet_minus_mlp_{metric}", target, scope, "critical_delta"))
    return output


def best_predictive(rows, target, metric_scope):
    candidates = [row for row in predictive_rows(rows) if row["scope"] == "all" and row["y"] == target and row["metric_scope"] == metric_scope]
    return max(candidates, key=lambda row: row["abs_spearman"]) if candidates else {}


def setup_plots(output_dir):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(output_dir), "matplotlib_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def scatter(ax, rows, x_key, y_key, title):
    ax.scatter([row[x_key] for row in rows], [row[y_key] for row in rows], s=10, alpha=0.5)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)


def save_plots(rows, output_dir, plt):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    scatter(axes[0][0], rows, "unet_global_overestimate_rate", "cost_gap", "Global overestimate vs cost gap")
    scatter(axes[0][1], rows, "unet_critical_overestimate_rate", "cost_gap", "Critical overestimate vs cost gap")
    scatter(axes[0][2], rows, "unet_critical_minus_global_overestimate_rate", "cost_gap", "Critical-global overestimate vs cost gap")
    scatter(axes[1][0], rows, "global_unet_minus_mlp_ordering_accuracy", "unet_minus_mlp_expanded", "Global ordering delta vs expansion")
    scatter(axes[1][1], rows, "critical_unet_minus_mlp_ordering_accuracy", "unet_minus_mlp_expanded", "Critical ordering delta vs expansion")
    scatter(axes[1][2], rows, "unet_critical_mae", "unet_minus_mlp_expanded", "Critical MAE vs expansion")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "route_critical_plots.png"))
    plt.close(fig)


def write_summary(path, rows, stats):
    unet_wins = summary_row(rows, "unet_wins")
    mlp_wins = summary_row(rows, "mlp_wins")
    optimal = summary_row(rows, "unet_optimal")
    non_optimal = summary_row(rows, "unet_non_optimal")
    expansion_global = best_predictive(rows, "unet_minus_mlp_expanded", "global")
    expansion_critical = best_predictive(rows, "unet_minus_mlp_expanded", "critical")
    gap_global = best_predictive(rows, "cost_gap", "global")
    gap_critical = best_predictive(rows, "cost_gap", "critical")

    critical_ordering_win_gap = (
        unet_wins["mean_critical_unet_minus_mlp_ordering_accuracy"]
        - mlp_wins["mean_critical_unet_minus_mlp_ordering_accuracy"]
    )
    global_ordering_win_gap = (
        unet_wins["mean_global_unet_minus_mlp_ordering_accuracy"]
        - mlp_wins["mean_global_unet_minus_mlp_ordering_accuracy"]
    )
    non_opt_critical_over_gap = (
        non_optimal["mean_unet_critical_overestimate_rate"] - non_optimal["mean_unet_global_overestimate_rate"]
    )
    opt_critical_over_gap = optimal["mean_unet_critical_overestimate_rate"] - optimal["mean_unet_global_overestimate_rate"]

    with open(path, "w", encoding="utf-8") as file:
        file.write("# Route-Critical Cell Analysis\n\n")
        file.write("This analysis tests whether U-Net search behavior is better explained by prediction behavior on route-critical cells than by averages over all reachable free cells. It does not rerun training or modify model/search code.\n\n")
        file.write(f"Maps analyzed: {len(rows)}\n\n")
        file.write("## Route-Critical Definition\n\n")
        file.write("- Optimal shortest-path cells from BFS distance labels.\n")
        file.write("- Free cells within configurable path distance k.\n")
        file.write("- Low-degree narrow-passage cells and articulation points near that path neighborhood.\n\n")
        file.write("## Q1: U-Net Wins\n\n")
        file.write(
            f"U-Net-win maps show critical ordering advantage gap {critical_ordering_win_gap:.3f} "
            f"versus global ordering advantage gap {global_ordering_win_gap:.3f}.\n\n"
        )
        file.write("## Q2: U-Net Non-Optimality\n\n")
        file.write(
            f"On non-optimal U-Net maps, critical minus global overestimate rate is {non_opt_critical_over_gap:.3f}. "
            f"On optimal U-Net maps it is {opt_critical_over_gap:.3f}.\n\n"
        )
        file.write("## Q3: Predictive Power\n\n")
        file.write(
            f"For U-Net minus MLP expansion gap, best global Spearman is {expansion_global.get('spearman', 0.0):.3f} "
            f"({expansion_global.get('x', '')}); best critical Spearman is {expansion_critical.get('spearman', 0.0):.3f} "
            f"({expansion_critical.get('x', '')}).\n"
        )
        file.write(
            f"For U-Net optimality gap, best global Spearman is {gap_global.get('spearman', 0.0):.3f} "
            f"({gap_global.get('x', '')}); best critical Spearman is {gap_critical.get('spearman', 0.0):.3f} "
            f"({gap_critical.get('x', '')}).\n\n"
        )
        file.write("## Structured Subsets\n\n")
        for scope in STRUCTURED_SCOPES:
            row = next(item for item in stats if item.get("row_type") == "summary" and item.get("scope") == scope)
            file.write(
                f"- {scope}: maps={row['maps']}, expansion_gap={row['mean_unet_minus_mlp_expanded']:.3f}, "
                f"cost_gap={row['mean_cost_gap']:.3f}, critical_over={row['mean_unet_critical_overestimate_rate']:.3f}, "
                f"critical_order_delta={row['mean_critical_unet_minus_mlp_ordering_accuracy']:.3f}\n"
            )
        file.write("\n## Final Comparison\n\n")
        file.write("Compare predictive power of global metrics vs route-critical metrics using the predictive_power rows in route_critical_statistics.csv. Higher absolute Spearman means stronger monotonic explanatory power for the target search behavior.\n")


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "route_critical_analysis")
    os.makedirs(output_dir, exist_ok=True)
    plt = setup_plots(output_dir)

    grouped = {}
    grouped.update(group_result_rows(read_csv(args.random_results), "random_sg_100"))
    grouped.update(group_result_rows(read_csv(args.structured_results), "structured_controlled_100"))

    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint, "unet"))
    rows = enrich_map_metrics(grouped, mlp_model, unet_model, args.path_radius)

    scopes = ["all", "unet_wins", "mlp_wins", "unet_optimal", "unet_non_optimal"] + STRUCTURED_SCOPES
    map_rows = []
    for row in rows:
        map_row = dict(row)
        map_row["row_type"] = "map"
        map_row["scope"] = "map"
        map_rows.append(map_row)

    stats = [summary_row(rows, scope) for scope in scopes]
    stats.extend(predictive_rows(rows))
    write_csv(os.path.join(output_dir, "route_critical_statistics.csv"), map_rows + stats)
    save_plots(rows, output_dir, plt)
    write_summary(os.path.join(output_dir, "route_critical_summary.md"), rows, stats)

    print(f"Saved route-critical analysis outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze heuristic behavior on route-critical cells.")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_sg_100.csv")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_controlled_100.csv")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--path-radius", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
