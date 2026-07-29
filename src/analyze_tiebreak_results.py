import argparse
import csv
import os


METHODS = ["manhattan", "mlp_table", "unet", "manhattan_unet_tiebreak"]


def read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def to_float(row, key, default=0.0):
    value = row.get(key, "")
    if value == "":
        return default
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    return float(value)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def map_key(row):
    return (
        row["seed"],
        row["map_size"],
        row["obstacle_rate"],
        row.get("map_mode", "random"),
        row.get("structured_type", "random"),
        row.get("start_goal_mode", ""),
        row.get("start_row", ""),
        row.get("start_col", ""),
        row.get("goal_row", ""),
        row.get("goal_col", ""),
    )


def group_maps(rows):
    groups = {}
    for row in rows:
        if row.get("heuristic") not in METHODS:
            continue
        groups.setdefault(map_key(row), {})[row["heuristic"]] = row
    return groups


def scoped_maps(rows, structured_type=None):
    maps = []
    for methods in group_maps(rows).values():
        if not all(method in methods for method in METHODS):
            continue
        sample = methods["manhattan_unet_tiebreak"]
        if sample.get("skip_reason", ""):
            continue
        if structured_type is not None and sample.get("structured_type") != structured_type:
            continue
        maps.append(methods)
    return maps


def summarize(rows, benchmark, structured_type=None):
    maps = scoped_maps(rows, structured_type)
    output = {
        "benchmark": benchmark,
        "structured_type": structured_type or "all",
        "maps": len(maps),
    }
    for method in METHODS:
        method_rows = [methods[method] for methods in maps]
        output[f"{method}_expanded"] = mean(to_float(row, "expanded_nodes") for row in method_rows)
        output[f"{method}_optimality"] = mean(1.0 if row["optimal"] == "True" else 0.0 for row in method_rows)
        output[f"{method}_cost_gap"] = mean(
            max(0.0, to_float(row, "path_length") - to_float(row, "optimal_cost")) for row in method_rows
        )
    tiebreak = output["manhattan_unet_tiebreak_expanded"]
    output["tiebreak_minus_manhattan_expanded"] = tiebreak - output["manhattan_expanded"]
    output["tiebreak_minus_mlp_table_expanded"] = tiebreak - output["mlp_table_expanded"]
    output["tiebreak_minus_unet_expanded"] = tiebreak - output["unet_expanded"]
    return output


def write_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, rows):
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Neural Tie-Breaking A* Summary\n\n")
        file.write(
            "This analysis compares Manhattan, MLP table, raw U-Net, and `manhattan_unet_tiebreak`. "
            "The tie-break variant uses Manhattan as the admissible primary f-value and U-Net only as a secondary ordering signal.\n\n"
        )
        for row in rows:
            title = row["benchmark"] if row["structured_type"] == "all" else f"{row['benchmark']} / {row['structured_type']}"
            file.write(f"## {title}\n\n")
            file.write(f"Maps: {row['maps']}\n\n")
            file.write("| Method | Mean expanded | Optimality rate | Mean cost gap |\n")
            file.write("|---|---:|---:|---:|\n")
            for method in METHODS:
                file.write(
                    f"| {method} | {row[f'{method}_expanded']:.3f} | "
                    f"{row[f'{method}_optimality']:.3f} | {row[f'{method}_cost_gap']:.3f} |\n"
                )
            file.write("\n")
            file.write(
                f"Tie-break minus Manhattan expanded: {row['tiebreak_minus_manhattan_expanded']:.3f}\n\n"
            )
            file.write(f"Tie-break minus MLP table expanded: {row['tiebreak_minus_mlp_table_expanded']:.3f}\n\n")
            file.write(f"Tie-break minus raw U-Net expanded: {row['tiebreak_minus_unet_expanded']:.3f}\n\n")

        file.write("## Interpretation Focus\n\n")
        file.write(
            "If `manhattan_unet_tiebreak` improves over Manhattan while preserving optimality, this supports the route-ordering "
            "hypothesis as an algorithmic mechanism. If gains are small, U-Net's useful guidance likely requires stronger influence "
            "than secondary tie-breaking among equal Manhattan f-values.\n"
        )
        aggregate_rows = [row for row in rows if row["structured_type"] == "all"]
        if aggregate_rows:
            improves_manhattan = all(row["tiebreak_minus_manhattan_expanded"] < 0 for row in aggregate_rows)
            preserves_optimality = all(row["manhattan_unet_tiebreak_optimality"] >= 1.0 for row in aggregate_rows)
            file.write("\n## Observed Result\n\n")
            if improves_manhattan and preserves_optimality:
                file.write(
                    "`manhattan_unet_tiebreak` improves over Manhattan on the evaluated aggregate benchmarks while preserving "
                    "optimality. This is direct algorithmic evidence that the U-Net field contains useful ordering information even "
                    "when it is restricted to secondary tie-breaking under an admissible Manhattan primary priority.\n"
                )
            else:
                file.write(
                    "`manhattan_unet_tiebreak` does not consistently improve over Manhattan while preserving optimality across the "
                    "evaluated aggregate benchmarks. This would suggest that U-Net guidance may need stronger influence than pure "
                    "tie-breaking.\n"
                )


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "tiebreak_analysis")
    os.makedirs(output_dir, exist_ok=True)

    random_rows = read_csv(args.random_results)
    structured_rows = read_csv(args.structured_results)
    summaries = []
    if random_rows:
        summaries.append(summarize(random_rows, "random"))
    if structured_rows:
        summaries.append(summarize(structured_rows, "structured"))
        for structured_type in ["maze_like", "bottleneck", "large_block", "narrow_corridor"]:
            summaries.append(summarize(structured_rows, "structured", structured_type))

    write_csv(os.path.join(output_dir, "tiebreak_statistics.csv"), summaries)
    write_summary(os.path.join(output_dir, "tiebreak_summary.md"), summaries)
    print(f"Saved tie-break analysis outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Neural Tie-Breaking A* results.")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_tiebreak_100.csv")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_100.csv")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
