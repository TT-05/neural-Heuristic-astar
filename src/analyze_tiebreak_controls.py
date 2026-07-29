import argparse
import csv
import os


METHODS = [
    "manhattan",
    "manhattan_counter_tiebreak",
    "manhattan_large_g_tiebreak",
    "manhattan_small_g_tiebreak",
    "manhattan_mlp_tiebreak",
    "manhattan_unet_tiebreak",
    "manhattan_true_distance_tiebreak",
]
REFERENCE_METHODS = ["manhattan", "manhattan_large_g_tiebreak", "manhattan_mlp_tiebreak", "manhattan_unet_tiebreak"]
STRUCTURED_TYPES = ["maze_like", "bottleneck", "large_block", "narrow_corridor"]


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
    grouped = {}
    for row in rows:
        if row.get("heuristic") not in METHODS:
            continue
        grouped.setdefault(map_key(row), {})[row["heuristic"]] = row
    return grouped


def complete_maps(rows, structured_type=None):
    output = []
    for methods in group_maps(rows).values():
        if not all(method in methods for method in METHODS):
            continue
        sample = methods["manhattan"]
        if sample.get("skip_reason", ""):
            continue
        if structured_type is not None and sample.get("structured_type") != structured_type:
            continue
        output.append(methods)
    return output


def method_stats(maps, method):
    rows = [methods[method] for methods in maps]
    return {
        "expanded": mean(to_float(row, "expanded_nodes") for row in rows),
        "optimality": mean(1.0 if row["optimal"] == "True" else 0.0 for row in rows),
        "cost_gap": mean(max(0.0, to_float(row, "path_length") - to_float(row, "optimal_cost")) for row in rows),
        "runtime": mean(to_float(row, "runtime_seconds") for row in rows),
    }


def summarize(rows, benchmark, structured_type=None):
    maps = complete_maps(rows, structured_type)
    output = {"benchmark": benchmark, "structured_type": structured_type or "all", "maps": len(maps)}
    stats = {method: method_stats(maps, method) for method in METHODS}
    for method, values in stats.items():
        for key, value in values.items():
            output[f"{method}_{key}"] = value

    unet_expanded = output["manhattan_unet_tiebreak_expanded"]
    for reference in REFERENCE_METHODS:
        output[f"unet_tiebreak_minus_{reference}_expanded"] = unet_expanded - output[f"{reference}_expanded"]
    for method in METHODS:
        output[f"{method}_minus_manhattan_expanded"] = output[f"{method}_expanded"] - output["manhattan_expanded"]
    return output


def write_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def yes_no(value):
    return "yes" if value else "no"


def write_report(path, rows):
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Tie-Breaking Control Analysis\n\n")
        file.write(
            "This report tests whether `manhattan_unet_tiebreak` gains come from U-Net secondary ordering rather than incidental "
            "heap tuple or generic tie-breaking behavior. All variants keep Manhattan as the primary admissible f-value.\n\n"
        )
        for row in rows:
            title = row["benchmark"] if row["structured_type"] == "all" else f"{row['benchmark']} / {row['structured_type']}"
            file.write(f"## {title}\n\n")
            file.write(f"Maps: {row['maps']}\n\n")
            file.write("| Method | Mean expanded | Optimality | Cost gap | Runtime |\n")
            file.write("|---|---:|---:|---:|---:|\n")
            for method in METHODS:
                file.write(
                    f"| {method} | {row[f'{method}_expanded']:.3f} | {row[f'{method}_optimality']:.3f} | "
                    f"{row[f'{method}_cost_gap']:.3f} | {row[f'{method}_runtime']:.6f} |\n"
                )
            file.write("\n")
            file.write(
                f"U-Net tie-break minus Manhattan: {row['unet_tiebreak_minus_manhattan_expanded']:.3f}; "
                f"minus large-g: {row['unet_tiebreak_minus_manhattan_large_g_tiebreak_expanded']:.3f}; "
                f"minus MLP tie-break: {row['unet_tiebreak_minus_manhattan_mlp_tiebreak_expanded']:.3f}; "
                f"minus oracle: {row['manhattan_unet_tiebreak_expanded'] - row['manhattan_true_distance_tiebreak_expanded']:.3f}.\n\n"
            )

        aggregate = {row["benchmark"]: row for row in rows if row["structured_type"] == "all"}
        file.write("## Key Questions\n\n")
        for benchmark, row in aggregate.items():
            beats_simple = (
                row["manhattan_unet_tiebreak_expanded"] < row["manhattan_counter_tiebreak_expanded"]
                and row["manhattan_unet_tiebreak_expanded"] < row["manhattan_large_g_tiebreak_expanded"]
                and row["manhattan_unet_tiebreak_expanded"] < row["manhattan_small_g_tiebreak_expanded"]
            )
            beats_mlp = row["manhattan_unet_tiebreak_expanded"] < row["manhattan_mlp_tiebreak_expanded"]
            preserves_optimality = all(row[f"{method}_optimality"] >= 1.0 for method in METHODS)
            oracle_gap = row["manhattan_unet_tiebreak_expanded"] - row["manhattan_true_distance_tiebreak_expanded"]
            file.write(f"### {benchmark}\n\n")
            file.write(f"- Q1 U-Net beats counter/g controls: {yes_no(beats_simple)}.\n")
            file.write(f"- Q2 U-Net beats MLP tie-break: {yes_no(beats_mlp)}.\n")
            file.write(f"- Q3 U-Net minus true-distance oracle expanded nodes: {oracle_gap:.3f}.\n")
            file.write(f"- Q5 optimality preserved across all controls: {yes_no(preserves_optimality)}.\n\n")

        structured_rows = [row for row in rows if row["benchmark"] == "structured" and row["structured_type"] != "all"]
        if structured_rows:
            file.write("### Q4 Structure Dependence\n\n")
            for row in structured_rows:
                file.write(
                    f"- {row['structured_type']}: U-Net tie-break minus MLP tie-break = "
                    f"{row['unet_tiebreak_minus_manhattan_mlp_tiebreak_expanded']:.3f}; "
                    f"U-Net tie-break minus oracle = "
                    f"{row['manhattan_unet_tiebreak_expanded'] - row['manhattan_true_distance_tiebreak_expanded']:.3f}.\n"
                )
            file.write(
                "\nNegative U-Net-minus-MLP values indicate structures where U-Net provides stronger secondary ordering than MLP. "
                "Positive values indicate structures where MLP tie-breaking is stronger.\n\n"
            )

        file.write("## Interpretation\n\n")
        file.write(
            "If U-Net tie-breaking beats counter and g-based controls, the improvement is not merely a heap implementation artifact. "
            "If it beats MLP tie-breaking, U-Net provides obstacle-aware secondary ordering beyond geometry-only learned distance. "
            "If the true-distance oracle remains better, useful tie-breaking headroom remains. These conclusions validate only "
            "secondary ordering under an admissible Manhattan primary priority; they do not make raw U-Net admissible and do not "
            "establish a final algorithm.\n"
        )


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "tiebreak_controls_analysis")
    os.makedirs(output_dir, exist_ok=True)

    random_rows = read_csv(args.random_results)
    structured_rows = read_csv(args.structured_results)
    summaries = []
    if random_rows:
        summaries.append(summarize(random_rows, "random"))
    if structured_rows:
        summaries.append(summarize(structured_rows, "structured"))
        for structured_type in STRUCTURED_TYPES:
            summaries.append(summarize(structured_rows, "structured", structured_type))

    write_csv(os.path.join(output_dir, "tiebreak_controls_statistics.csv"), summaries)
    write_report(os.path.join(output_dir, "tiebreak_controls_summary.md"), summaries)
    print(f"Saved tie-break control analysis outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze A* tie-breaking control experiments.")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_tiebreak_controls_100.csv")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
