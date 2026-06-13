import argparse
import csv
import os


METHODS = ["manhattan", "mlp_table", "unet"]


def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames):
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


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else ""


def map_key(row):
    return (
        row["seed"],
        row["map_size"],
        row["obstacle_rate"],
        row.get("map_mode", ""),
        row.get("structured_type", ""),
        row.get("start_goal_mode", ""),
        row.get("start_row", ""),
        row.get("start_col", ""),
        row.get("goal_row", ""),
        row.get("goal_col", ""),
    )


def group_by_map(rows):
    groups = {}
    for row in rows:
        if row.get("heuristic") not in METHODS:
            continue
        groups.setdefault(map_key(row), {})[row["heuristic"]] = row
    return groups


def summarize_by_structured_type(rows):
    groups = {}
    for row in rows:
        if row.get("heuristic") not in METHODS:
            continue
        key = (row.get("structured_type", "unknown"), row["heuristic"])
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for structured_type, heuristic in sorted(groups.keys()):
        group_rows = groups[(structured_type, heuristic)]
        solvable_rows = [row for row in group_rows if to_float(row, "optimal_cost", -1.0) >= 0]
        found_rows = [row for row in solvable_rows if row["path_found"] == "True"]
        summary_rows.append(
            {
                "structured_type": structured_type,
                "heuristic": heuristic,
                "runs": len(group_rows),
                "solvable_maps": len(solvable_rows),
                "mean_expanded_nodes": mean(to_float(row, "expanded_nodes") for row in solvable_rows),
                "mean_runtime_seconds": mean(to_float(row, "runtime_seconds") for row in solvable_rows),
                "mean_path_length": mean(to_float(row, "path_length") for row in found_rows),
                "optimality_rate": mean(1.0 if row["optimal"] == "True" else 0.0 for row in solvable_rows),
                "mean_overestimate_rate": mean(to_float(row, "overestimate_rate") for row in solvable_rows),
            }
        )
    return summary_rows


def pairwise_gaps(rows):
    grouped = group_by_map(rows)
    by_type = {}
    skip_counts = {}
    for methods in grouped.values():
        sample = next(iter(methods.values()))
        structured_type = sample.get("structured_type", "unknown")
        skip_reason = sample.get("skip_reason", "")
        if skip_reason:
            skip_counts[skip_reason] = skip_counts.get(skip_reason, 0) + 1
            continue
        if not all(method in methods for method in METHODS):
            continue
        by_type.setdefault(structured_type, []).append(methods)

    gap_rows = []
    for structured_type, maps in sorted(by_type.items()):
        unet_minus_mlp = []
        unet_minus_manhattan = []
        mlp_minus_manhattan = []
        unet_better_mlp = 0
        mlp_better_unet = 0
        ties = 0
        for methods in maps:
            manhattan = to_float(methods["manhattan"], "expanded_nodes")
            mlp = to_float(methods["mlp_table"], "expanded_nodes")
            unet = to_float(methods["unet"], "expanded_nodes")
            unet_minus_mlp.append(unet - mlp)
            unet_minus_manhattan.append(unet - manhattan)
            mlp_minus_manhattan.append(mlp - manhattan)
            if unet < mlp:
                unet_better_mlp += 1
            elif mlp < unet:
                mlp_better_unet += 1
            else:
                ties += 1
        gap_rows.append(
            {
                "structured_type": structured_type,
                "maps": len(maps),
                "mean_unet_minus_mlp_expanded": mean(unet_minus_mlp),
                "mean_unet_minus_manhattan_expanded": mean(unet_minus_manhattan),
                "mean_mlp_minus_manhattan_expanded": mean(mlp_minus_manhattan),
                "unet_better_than_mlp_count": unet_better_mlp,
                "mlp_better_than_unet_count": mlp_better_unet,
                "tie_count": ties,
            }
        )
    return gap_rows, skip_counts


def write_report(path, summary_rows, gap_rows, skip_counts):
    summary = {(row["structured_type"], row["heuristic"]): row for row in summary_rows}
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Controlled Structured Benchmark Findings\n\n")
        file.write(f"Skipped maps by reason: `{skip_counts}`\n\n")
        file.write("## Pairwise Gaps\n\n")
        for row in gap_rows:
            file.write(
                f"- {row['structured_type']}: maps={row['maps']}, "
                f"U-Net-MLP={float(row['mean_unet_minus_mlp_expanded']):.2f}, "
                f"U-Net-Manhattan={float(row['mean_unet_minus_manhattan_expanded']):.2f}, "
                f"MLP-Manhattan={float(row['mean_mlp_minus_manhattan_expanded']):.2f}, "
                f"U-Net better than MLP={row['unet_better_than_mlp_count']}, "
                f"MLP better than U-Net={row['mlp_better_than_unet_count']}, "
                f"ties={row['tie_count']}\n"
            )

        file.write("\n## Method Summaries\n\n")
        for structured_type in sorted({row["structured_type"] for row in summary_rows}):
            file.write(f"\n### {structured_type}\n")
            for method in METHODS:
                row = summary[(structured_type, method)]
                file.write(
                    f"- {method}: maps={row['solvable_maps']}, "
                    f"expanded={float(row['mean_expanded_nodes']):.2f}, "
                    f"runtime={row['mean_runtime_seconds']}, "
                    f"path_length={float(row['mean_path_length']):.2f}, "
                    f"optimality={row['optimality_rate']}, "
                    f"overestimate={row['mean_overestimate_rate']}\n"
                )


def analyze_structured_benchmark(args):
    rows = read_csv(args.results)
    output_dir = args.output_dir
    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "outputs", "structured_controlled", args.output_tag)
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = summarize_by_structured_type(rows)
    gap_rows, skip_counts = pairwise_gaps(rows)

    summary_path = os.path.join(output_dir, "structured_type_summary.csv")
    gaps_path = os.path.join(output_dir, "structured_type_pairwise_gaps.csv")
    report_path = os.path.join(output_dir, "findings_report.md")
    write_csv(summary_path, summary_rows, list(summary_rows[0].keys()) if summary_rows else [])
    write_csv(gaps_path, gap_rows, list(gap_rows[0].keys()) if gap_rows else [])
    write_report(report_path, summary_rows, gap_rows, skip_counts)

    print(f"Saved structured type summary to {summary_path}")
    print(f"Saved structured type pairwise gaps to {gaps_path}")
    print(f"Saved findings report to {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize controlled structured benchmark results.")
    parser.add_argument("--results", required=True, help="Experiment results CSV.")
    parser.add_argument("--output-tag", required=True, help="Output tag under outputs/structured_controlled/.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    analyze_structured_benchmark(parse_args())
