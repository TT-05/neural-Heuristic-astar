import argparse
import csv
import os


METRICS = ["mae", "mse", "overestimate_rate"]
TARGETS = ["expanded_nodes", "runtime_seconds", "optimal"]


def read_rows(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def to_float(row, key):
    value = row.get(key, "")
    if value == "":
        return None
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    return float(value)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return ""
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = sum(value * value for value in dx) ** 0.5
    denom_y = sum(value * value for value in dy) ** 0.5
    if denom_x == 0 or denom_y == 0:
        return ""
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def grouped_by_heuristic(rows):
    groups = {}
    for row in rows:
        if row.get("optimal_cost") in ("", "-1"):
            continue
        groups.setdefault(row["heuristic"], []).append(row)
    return groups


def write_correlation_summary(rows, output_path):
    groups = grouped_by_heuristic(rows)
    summary_rows = []

    for heuristic, group_rows in sorted(groups.items()):
        obstacle_groups = {"all": group_rows}
        for row in group_rows:
            obstacle_groups.setdefault(row["obstacle_rate"], []).append(row)

        for obstacle_rate, scoped_rows in sorted(obstacle_groups.items(), key=lambda item: item[0]):
            for metric in METRICS:
                for target in TARGETS:
                    xs = []
                    ys = []
                    for row in scoped_rows:
                        x = to_float(row, metric)
                        y = to_float(row, target)
                        if x is None or y is None:
                            continue
                        xs.append(x)
                        ys.append(y)

                    summary_rows.append(
                        {
                            "heuristic": heuristic,
                            "obstacle_rate": obstacle_rate,
                            "metric": metric,
                            "target": target,
                            "n": len(xs),
                            "pearson": pearson(xs, ys),
                            "mean_metric": sum(xs) / len(xs) if xs else "",
                            "mean_target": sum(ys) / len(ys) if ys else "",
                        }
                    )

    fieldnames = ["heuristic", "obstacle_rate", "metric", "target", "n", "pearson", "mean_metric", "mean_target"]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def map_key(row):
    return (row["seed"], row["map_size"], row["obstacle_rate"])


def write_unet_mlp_comparison(rows, output_path):
    by_map = {}
    for row in rows:
        if row.get("optimal_cost") in ("", "-1"):
            continue
        by_map.setdefault(map_key(row), {})[row["heuristic"]] = row

    groups = {}
    for key, heuristics in by_map.items():
        if "unet" not in heuristics or "mlp_table" not in heuristics:
            continue
        obstacle_rate = key[2]
        groups.setdefault(obstacle_rate, []).append((heuristics["unet"], heuristics["mlp_table"]))

    output_rows = []
    for obstacle_rate, pairs in sorted(groups.items(), key=lambda item: float(item[0])):
        deltas = []
        unet_better = 0
        mlp_better = 0
        ties = 0
        for unet_row, mlp_row in pairs:
            delta = to_float(unet_row, "expanded_nodes") - to_float(mlp_row, "expanded_nodes")
            deltas.append(delta)
            if delta < 0:
                unet_better += 1
            elif delta > 0:
                mlp_better += 1
            else:
                ties += 1

        output_rows.append(
            {
                "obstacle_rate": obstacle_rate,
                "paired_maps": len(pairs),
                "unet_better_count": unet_better,
                "mlp_table_better_count": mlp_better,
                "tie_count": ties,
                "mean_unet_minus_mlp_expanded": sum(deltas) / len(deltas) if deltas else "",
            }
        )

    fieldnames = [
        "obstacle_rate",
        "paired_maps",
        "unet_better_count",
        "mlp_table_better_count",
        "tie_count",
        "mean_unet_minus_mlp_expanded",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def save_scatter_plots(rows, output_dir):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    matplotlib_cache = os.path.join(project_root, "outputs", "matplotlib_cache")
    os.makedirs(matplotlib_cache, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_cache)
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Skipping scatter plots: matplotlib is not installed.")
        return

    groups = grouped_by_heuristic(rows)
    for heuristic, group_rows in sorted(groups.items()):
        for metric in METRICS:
            for target in TARGETS:
                xs = []
                ys = []
                for row in group_rows:
                    x = to_float(row, metric)
                    y = to_float(row, target)
                    if x is None or y is None:
                        continue
                    xs.append(x)
                    ys.append(y)

                if len(xs) < 2:
                    continue

                plt.figure(figsize=(5, 4))
                plt.scatter(xs, ys, alpha=0.7)
                plt.xlabel(metric)
                plt.ylabel(target)
                plt.title(f"{heuristic}: {metric} vs {target}")
                plt.tight_layout()
                filename = f"scatter_{heuristic}_{metric}_vs_{target}.png"
                plt.savefig(os.path.join(output_dir, filename))
                plt.close()


def analyze_results(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = args.input
    if input_path is None:
        input_path = os.path.join(project_root, "outputs", "experiments", "results.csv")

    output_dir = args.output_dir
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(project_root, "outputs", "analysis", base_name)
    os.makedirs(output_dir, exist_ok=True)

    rows = read_rows(input_path)
    correlation_path = os.path.join(output_dir, "correlation_summary.csv")
    comparison_path = os.path.join(output_dir, "unet_vs_mlp_table.csv")
    write_correlation_summary(rows, correlation_path)
    write_unet_mlp_comparison(rows, comparison_path)
    if not args.no_plots:
        save_scatter_plots(rows, output_dir)

    print(f"Saved correlation summary to {correlation_path}")
    print(f"Saved U-Net vs MLP table comparison to {comparison_path}")
    if args.no_plots:
        print("Skipped scatter plots.")
    else:
        print(f"Saved scatter plots to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze search-vs-regression correlations from experiment CSV files.")
    parser.add_argument("--input", default=None, help="Path to results CSV.")
    parser.add_argument("--output-dir", default=None, help="Directory for summary tables and scatter plots.")
    parser.add_argument("--no-plots", action="store_true", help="Only write CSV summaries.")
    return parser.parse_args()


if __name__ == "__main__":
    analyze_results(parse_args())
