import argparse
import csv
import math
import os


GROUPS = [
    "maze_like_unet_wins",
    "maze_like_mlp_wins",
    "large_block_unet_nonoptimal",
    "narrow_corridor_unet_fails",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_map(path):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            values = []
            for token in line.split():
                if token == "##":
                    values.append(None)
                else:
                    values.append(float(token))
            if values:
                rows.append(values)
    return rows


def valid_values(*maps):
    values = []
    rows = len(maps[0])
    cols = len(maps[0][0]) if rows else 0
    for r in range(rows):
        for c in range(cols):
            cell_values = [grid[r][c] for grid in maps]
            if all(value is not None for value in cell_values):
                values.append(tuple(cell_values))
    return values


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def pearson(xs, ys):
    if len(xs) < 2:
        return 0.0
    mean_x = mean(xs)
    mean_y = mean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(value * value for value in dx))
    denom_y = math.sqrt(sum(value * value for value in dy))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def neighbor_pairs(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is None:
                continue
            for dr, dc in [(1, 0), (0, 1)]:
                nr = r + dr
                nc = c + dc
                if nr >= rows or nc >= cols:
                    continue
                if grid[nr][nc] is not None:
                    yield grid[r][c], grid[nr][nc]


def local_roughness(grid):
    diffs = [abs(a - b) for a, b in neighbor_pairs(grid)]
    return mean(diffs), max(diffs) if diffs else 0.0


def consistency_violation_rate(grid):
    pairs = list(neighbor_pairs(grid))
    if not pairs:
        return 0.0
    directed = []
    for a, b in pairs:
        directed.append(max(0.0, a - b - 1.0))
        directed.append(max(0.0, b - a - 1.0))
    return sum(1 for value in directed if value > 0.0) / len(directed)


def case_metrics(case_root, row):
    case_dir = os.path.join(case_root, row["output_dir"])
    true_map = parse_map(os.path.join(case_dir, "true_distance_map.txt"))
    unet_map = parse_map(os.path.join(case_dir, "unet_predicted_map.txt"))
    error_map = parse_map(os.path.join(case_dir, "unet_error_map.txt"))
    mlp_map = parse_map(os.path.join(case_dir, "mlp_predicted_map.txt"))

    triples = valid_values(true_map, unet_map, mlp_map)
    errors = [error for (error,) in valid_values(error_map)]
    true_values = [item[0] for item in triples]
    unet_values = [item[1] for item in triples]
    mlp_values = [item[2] for item in triples]

    over_errors = [error for error in errors if error > 0]
    under_errors = [error for error in errors if error < 0]
    large_over_errors = [error for error in errors if error > 3]
    large_under_errors = [error for error in errors if error < -3]
    near_goal = [abs(error) for error, true_value in zip(errors, true_values) if true_value <= 5]
    far_goal = [abs(error) for error, true_value in zip(errors, true_values) if true_value >= 20]
    unet_roughness, unet_max_jump = local_roughness(unet_map)
    mlp_roughness, mlp_max_jump = local_roughness(mlp_map)

    metrics = {
        **row,
        "case_dir": case_dir,
        "true_unet_corr": pearson(true_values, unet_values),
        "true_mlp_corr": pearson(true_values, mlp_values),
        "mean_error": mean(errors),
        "mean_positive_error": mean(over_errors),
        "mean_negative_error": mean(under_errors),
        "large_overestimate_rate": len(large_over_errors) / len(errors) if errors else 0.0,
        "large_underestimate_rate": len(large_under_errors) / len(errors) if errors else 0.0,
        "near_goal_abs_error": mean(near_goal),
        "far_goal_abs_error": mean(far_goal),
        "unet_roughness": unet_roughness,
        "unet_max_neighbor_jump": unet_max_jump,
        "mlp_roughness": mlp_roughness,
        "mlp_max_neighbor_jump": mlp_max_jump,
        "computed_consistency_violation_rate": consistency_violation_rate(unet_map),
    }
    return metrics


def group_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["group"], []).append(row)
    return grouped


def summarize_group(rows):
    return {
        "cases": len(rows),
        "mean_unet_minus_mlp_expanded": mean(float(row["rerun_unet_minus_mlp_expanded"]) for row in rows),
        "mean_unet_cost_gap": mean(float(row["rerun_unet_cost_gap"]) for row in rows),
        "mean_unet_overestimate_rate": mean(float(row["unet_overestimate_rate"]) for row in rows),
        "mean_large_overestimate_rate": mean(row["large_overestimate_rate"] for row in rows),
        "mean_true_unet_corr": mean(row["true_unet_corr"] for row in rows),
        "mean_true_mlp_corr": mean(row["true_mlp_corr"] for row in rows),
        "mean_unet_roughness": mean(row["unet_roughness"] for row in rows),
        "mean_mlp_roughness": mean(row["mlp_roughness"] for row in rows),
        "mean_consistency_violation_rate": mean(row["computed_consistency_violation_rate"] for row in rows),
        "mean_near_goal_abs_error": mean(row["near_goal_abs_error"] for row in rows),
        "mean_far_goal_abs_error": mean(row["far_goal_abs_error"] for row in rows),
    }


def representative_image_link(row, image_name):
    rel_path = os.path.relpath(os.path.join(row["case_dir"], image_name), os.path.dirname(row["case_dir"]))
    # Make the link relative to qualitative_analysis.md's directory.
    return os.path.join(row["output_dir"], image_name)


def write_markdown(path, rows):
    grouped = group_rows(rows)
    summaries = {group: summarize_group(group_rows) for group, group_rows in grouped.items()}

    group_interpretations = {
        "maze_like_unet_wins": {
            "typical": "U-Net often produces a useful global field on long maze-like routes and reduces exploration relative to both MLP and Manhattan.",
            "optimality": "Mostly optimal in this selected group, with one selected case showing a small cost gap.",
            "mechanism": "The selected successes are dominated by underestimation or mild overestimation rather than strong positive barriers; the field can still guide search through the maze layout.",
            "evidence": "U-Net has lower expanded nodes than MLP by construction, moderate or low large-overestimate rates, and path overlays show the search path following the viable route family.",
        },
        "maze_like_mlp_wins": {
            "typical": "U-Net remains optimal but expands much more than MLP in these maze-like cases.",
            "optimality": "No selected case has a positive U-Net cost gap.",
            "mechanism": "The U-Net field appears less aligned with the useful route ordering: overestimation is higher and the predicted field is rougher/noisier, causing extra exploration without changing final path cost.",
            "evidence": "The group has high U-Net overestimate rates and large U-Net-minus-MLP expanded gaps while cost gap remains zero.",
        },
        "large_block_unet_nonoptimal": {
            "typical": "U-Net can create harmful barriers around large blocks, producing both extra expansions and non-optimal paths.",
            "optimality": "Optimality fails by definition in this selected group.",
            "mechanism": "Positive error regions are large enough to make the true shortest route look expensive; this is consistent with artificial heuristic barriers near necessary detours/passages.",
            "evidence": "Selected cases show positive U-Net cost gaps, high overestimate rates, and large positive error regions in the U-Net error maps.",
        },
        "narrow_corridor_unet_fails": {
            "typical": "U-Net often overestimates corridor states and does not preserve a smooth corridor gradient.",
            "optimality": "Some selected cases remain optimal, but at least one selected case is non-optimal.",
            "mechanism": "High overestimation along or near narrow passages makes the corridor less attractive and increases unnecessary expansion; in some cases it changes the selected path.",
            "evidence": "The selected failures have very high overestimate rates, high local consistency violation/roughness, and U-Net expanded nodes at or above Manhattan.",
        },
    }

    with open(path, "w", encoding="utf-8") as file:
        file.write("# Qualitative Case-Study Analysis\n\n")
        file.write("This report summarizes the 20 selected controlled-structured cases. It uses only generated case-study outputs: `selected_cases.csv`, true distance maps, U-Net predictions, U-Net error maps, MLP predictions, and path overlays. It does not rerun training or modify search/model code.\n\n")

        file.write("## Concise Group Table\n\n")
        file.write("| Group | Typical U-Net behavior | Optimality fails? | Likely mechanism | Evidence from maps/errors |\n")
        file.write("|---|---|---|---|---|\n")
        for group in GROUPS:
            info = group_interpretations[group]
            file.write(
                f"| `{group}` | {info['typical']} | {info['optimality']} | "
                f"{info['mechanism']} | {info['evidence']} |\n"
            )

        file.write("\n## Numeric Group Diagnostics\n\n")
        file.write("| Group | Cases | U-Net-MLP expanded | U-Net cost gap | Overestimate rate | Large overestimate rate | true/U-Net corr | true/MLP corr | U-Net roughness | Consistency violation |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for group in GROUPS:
            summary = summaries[group]
            file.write(
                f"| `{group}` | {summary['cases']} | "
                f"{summary['mean_unet_minus_mlp_expanded']:.2f} | "
                f"{summary['mean_unet_cost_gap']:.2f} | "
                f"{summary['mean_unet_overestimate_rate']:.3f} | "
                f"{summary['mean_large_overestimate_rate']:.3f} | "
                f"{summary['mean_true_unet_corr']:.3f} | "
                f"{summary['mean_true_mlp_corr']:.3f} | "
                f"{summary['mean_unet_roughness']:.2f} | "
                f"{summary['mean_consistency_violation_rate']:.3f} |\n"
            )

        for group in GROUPS:
            file.write(f"\n## {group}\n\n")
            info = group_interpretations[group]
            file.write(f"{info['typical']} {info['mechanism']}\n\n")
            top = grouped[group][0]
            file.write("Representative image references:\n\n")
            file.write(f"- [path overlay]({representative_image_link(top, 'path_overlay.png')})\n")
            file.write(f"- [obstacle map]({representative_image_link(top, 'obstacle_map.png')})\n")
            file.write(f"- [true distance map]({representative_image_link(top, 'true_distance_map.png')})\n")
            file.write(f"- [U-Net predicted map]({representative_image_link(top, 'unet_predicted_map.png')})\n")
            file.write(f"- [U-Net error map]({representative_image_link(top, 'unet_error_map.png')})\n")
            file.write(f"- [MLP predicted map]({representative_image_link(top, 'mlp_predicted_map.png')})\n\n")

            file.write("Selected cases:\n\n")
            for row in grouped[group]:
                file.write(
                    f"- `{row['case_id']}`: U-Net-MLP expanded "
                    f"{float(row['rerun_unet_minus_mlp_expanded']):.0f}, "
                    f"cost gap {float(row['rerun_unet_cost_gap']):.0f}, "
                    f"overestimate {float(row['unet_overestimate_rate']):.3f}, "
                    f"large-overestimate {row['large_overestimate_rate']:.3f}, "
                    f"true/U-Net corr {row['true_unet_corr']:.3f}, "
                    f"roughness {row['unet_roughness']:.2f}\n"
                )

        file.write("\n## Interpretation Boundaries\n\n")
        file.write("These conclusions are limited to the selected top-k cases and should be treated as qualitative evidence, not as a full distributional claim. The evidence supports pattern hypotheses to inspect visually: U-Net success in maze-like cases often comes from useful global route bias, while failures in large-block and narrow-corridor cases are consistent with overestimated passages, rough fields, and artificial heuristic barriers.\n")


def summarize_case_studies(args):
    case_root = args.case_root
    selected_path = os.path.join(case_root, "selected_cases.csv")
    rows = read_csv(selected_path)
    enriched = [case_metrics(case_root, row) for row in rows]
    output_path = os.path.join(case_root, "qualitative_analysis.md")
    write_markdown(output_path, enriched)
    print(f"Saved qualitative analysis to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize generated representative case-study outputs.")
    parser.add_argument(
        "--case-root",
        default="outputs/case_studies/structured_controlled_100",
        help="Directory containing selected_cases.csv and case subfolders.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    summarize_case_studies(parse_args())
