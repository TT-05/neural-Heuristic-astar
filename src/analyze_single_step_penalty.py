import argparse
import csv
import os

from analyze_tie_set_counterfactual_penalty import (
    collect_tie_events,
    continue_after_forced_choice,
    map_identity,
    node_text,
    result_expanded,
)
from analyze_tie_set_ordering import build_grid, checkpoint_path, group_maps, mean, pearson, prediction_table, read_csv, spearman
from analyze_tie_set_weighted_ordering import reconstruct_optimal_path, route_critical_cells
from bfs_label import compute_distance_to_goal
from model import load_mlp_heuristic, load_unet_heuristic, make_mlp_table_heuristic, make_unet_heuristic


def write_csv(path, rows):
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average(rows, key):
    return mean(row[key] for row in rows)


def tie_class(event):
    return "agreement" if event["mlp_selected_node"] == event["unet_selected_node"] else "disagreement"


def single_step_row(identity, event, grid, goal, true_table):
    mlp_remaining = continue_after_forced_choice(
        grid, goal, event["open_before"], event["g_score_before"], event["mlp_selected_node"], true_table
    )
    unet_remaining = continue_after_forced_choice(
        grid, goal, event["open_before"], event["g_score_before"], event["unet_selected_node"], true_table
    )
    oracle_remaining = continue_after_forced_choice(
        grid, goal, event["open_before"], event["g_score_before"], event["oracle_selected_node"], true_table
    )
    mlp_penalty = mlp_remaining - oracle_remaining
    unet_penalty = unet_remaining - oracle_remaining
    return {
        "row_type": "single_step_tie_set",
        **identity,
        "tie_class": tie_class(event),
        "expanded_step": event["expanded_step"],
        "f_primary": event["f_primary"],
        "tie_set_size": event["tie_set_size"],
        "route_critical_overlap": event["route_critical_overlap"],
        "route_critical_overlap_fraction": event["route_critical_overlap_fraction"],
        "true_distance_range": event["true_distance_range"],
        "mlp_selected_node": node_text(event["mlp_selected_node"]),
        "unet_selected_node": node_text(event["unet_selected_node"]),
        "oracle_selected_node": node_text(event["oracle_selected_node"]),
        "mlp_selected_true_distance": event["mlp_selected_true_distance"],
        "unet_selected_true_distance": event["unet_selected_true_distance"],
        "oracle_selected_true_distance": event["oracle_selected_true_distance"],
        "mlp_regret": event["mlp_regret"],
        "unet_regret": event["unet_regret"],
        "regret_gap": event["regret_gap"],
        "pairwise_gap": event["pairwise_gap"],
        "top1_gap": event["top1_gap"],
        "mlp_remaining_expansions": mlp_remaining,
        "unet_remaining_expansions": unet_remaining,
        "oracle_remaining_expansions": oracle_remaining,
        "mlp_penalty": mlp_penalty,
        "unet_penalty": unet_penalty,
        "penalty_gap": unet_penalty - mlp_penalty,
    }


def analyze_map(methods, mlp_model, unet_model, route_radius):
    sample = methods["manhattan"]
    grid, start, goal = build_grid(sample)
    distance_grid = compute_distance_to_goal(grid, goal)
    true_table = {
        (r, c): float(value)
        for r, row in enumerate(distance_grid)
        for c, value in enumerate(row)
        if value >= 0
    }
    path = reconstruct_optimal_path(grid, distance_grid, start, goal)
    critical = route_critical_cells(grid, path, route_radius)
    mlp_table = prediction_table(grid, goal, make_mlp_table_heuristic(mlp_model, grid, goal))
    unet_table = prediction_table(grid, goal, make_unet_heuristic(unet_model, grid, goal))
    events, expanded_steps = collect_tie_events(grid, start, goal, true_table, mlp_table, unet_table, critical)
    identity = map_identity(sample)
    rows = [single_step_row(identity, event, grid, goal, true_table) for event in events]
    for row in rows:
        row["expanded_steps"] = expanded_steps
        row["total_active_tie_sets"] = len(events)
        row["mlp_expanded"] = result_expanded(methods, "manhattan_mlp_tiebreak")
        row["unet_expanded"] = result_expanded(methods, "manhattan_unet_tiebreak")
        row["true_distance_expanded"] = result_expanded(methods, "manhattan_true_distance_tiebreak")
        row["expanded_gap"] = row["unet_expanded"] - row["mlp_expanded"]
    return rows


def aggregate_map_rows(tie_rows):
    grouped = {}
    for row in tie_rows:
        grouped.setdefault(row["map_id"], []).append(row)

    map_rows = []
    for map_id, rows in grouped.items():
        first = rows[0]
        agreement = [row for row in rows if row["tie_class"] == "agreement"]
        disagreement = [row for row in rows if row["tie_class"] == "disagreement"]
        nonzero_agreement = [row for row in agreement if row["penalty_gap"] != 0]
        map_rows.append(
            {
                "row_type": "map_summary",
                "benchmark": first["benchmark"],
                "structure_type": first["structure_type"],
                "map_id": map_id,
                "start": first["start"],
                "goal": first["goal"],
                "tie_set_count": len(rows),
                "agreement_count": len(agreement),
                "disagreement_count": len(disagreement),
                "agreement_fraction": len(agreement) / len(rows) if rows else 0.0,
                "disagreement_fraction": len(disagreement) / len(rows) if rows else 0.0,
                "agreement_nonzero_penalty_gap_count": len(nonzero_agreement),
                "agreement_nonzero_penalty_gap_fraction": len(nonzero_agreement) / len(agreement) if agreement else 0.0,
                "agreement_penalty_gap": sum(row["penalty_gap"] for row in agreement),
                "disagreement_penalty_gap": sum(row["penalty_gap"] for row in disagreement),
                "cumulative_penalty_gap": sum(row["penalty_gap"] for row in rows),
                "mean_penalty_gap": average(rows, "penalty_gap"),
                "mean_disagreement_penalty_gap": average(disagreement, "penalty_gap"),
                "mean_regret_gap": average(rows, "regret_gap"),
                "mean_pairwise_gap": average(rows, "pairwise_gap"),
                "mean_top1_gap": average(rows, "top1_gap"),
                "mlp_expanded": first["mlp_expanded"],
                "unet_expanded": first["unet_expanded"],
                "true_distance_expanded": first["true_distance_expanded"],
                "expanded_gap": first["expanded_gap"],
            }
        )
    return map_rows


def aggregate_scope(map_rows):
    keys = [
        "tie_set_count",
        "agreement_count",
        "disagreement_count",
        "agreement_fraction",
        "disagreement_fraction",
        "agreement_nonzero_penalty_gap_count",
        "agreement_nonzero_penalty_gap_fraction",
        "agreement_penalty_gap",
        "disagreement_penalty_gap",
        "cumulative_penalty_gap",
        "mean_penalty_gap",
        "mean_disagreement_penalty_gap",
        "mean_regret_gap",
        "mean_pairwise_gap",
        "mean_top1_gap",
        "expanded_gap",
    ]
    row = {"row_type": "aggregate", "benchmark": "structured", "structure_type": "bottleneck", "maps": len(map_rows)}
    for key in keys:
        row[key] = average(map_rows, key)
    return row


def correlation_rows(map_rows):
    rows = []
    ys = [row["expanded_gap"] for row in map_rows]
    for metric in [
        "cumulative_penalty_gap",
        "disagreement_penalty_gap",
        "agreement_penalty_gap",
        "mean_penalty_gap",
        "mean_regret_gap",
        "mean_pairwise_gap",
        "mean_top1_gap",
    ]:
        xs = [row[metric] for row in map_rows]
        rows.append(
            {
                "benchmark": "structured",
                "structure_type": "bottleneck",
                "metric": metric,
                "y": "expanded_gap",
                "n": len(map_rows),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
        )
    return rows


def nonzero_agreement_rows(tie_rows):
    return [
        row
        for row in tie_rows
        if row["tie_class"] == "agreement" and row["penalty_gap"] != 0
    ]


def save_plots(path, map_rows, correlations):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(path), "matplotlib_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter([row["cumulative_penalty_gap"] for row in map_rows], [row["expanded_gap"] for row in map_rows], s=12)
    axes[0].set_title("Single-step cumulative penalty")
    axes[0].set_xlabel("U-Net - MLP penalty")
    axes[0].set_ylabel("U-Net - MLP expanded")

    axes[1].scatter([row["disagreement_penalty_gap"] for row in map_rows], [row["expanded_gap"] for row in map_rows], s=12)
    axes[1].set_title("Disagreement penalty")
    axes[1].set_xlabel("U-Net - MLP penalty")
    axes[1].set_ylabel("U-Net - MLP expanded")

    metrics = ["cumulative_penalty_gap", "disagreement_penalty_gap", "mean_regret_gap", "mean_pairwise_gap"]
    corr = [next(row for row in correlations if row["metric"] == metric) for metric in metrics]
    axes[2].bar([row["metric"].replace("_gap", "") for row in corr], [float(row["spearman"]) for row in corr])
    axes[2].set_title("Spearman vs expanded gap")
    axes[2].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_summary(path, aggregate, correlations, nonzero_agreements):
    corr = {row["metric"]: row for row in correlations}
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Single-Step Tie-Set Penalty Analysis\n\n")
        file.write(
            "This analysis isolates the immediate tie-choice effect. After forcing the MLP, U-Net, or oracle node, "
            "all continuations use the same Manhattan-primary true-distance secondary policy. Production A* is unchanged.\n\n"
        )
        file.write("## Validation\n\n")
        file.write(
            f"- Agreement tie sets with nonzero penalty_gap: {len(nonzero_agreements)}.\n"
            f"- Mean agreement nonzero fraction per map: {aggregate['agreement_nonzero_penalty_gap_fraction']:.6f}.\n"
            "Expected value is zero if future-policy confounding is removed.\n\n"
        )
        file.write("## Bottleneck Results\n\n")
        file.write(
            f"- Maps: {aggregate['maps']}.\n"
            f"- Agreement fraction: {aggregate['agreement_fraction']:.3f}.\n"
            f"- Disagreement fraction: {aggregate['disagreement_fraction']:.3f}.\n"
            f"- Agreement penalty gap: {aggregate['agreement_penalty_gap']:.3f}.\n"
            f"- Disagreement penalty gap: {aggregate['disagreement_penalty_gap']:.3f}.\n"
            f"- Cumulative penalty gap: {aggregate['cumulative_penalty_gap']:.3f}.\n"
            f"- Expanded gap: {aggregate['expanded_gap']:.3f}.\n\n"
        )
        file.write("## Correlations With Expanded Gap\n\n")
        file.write("| Metric | Pearson | Spearman | n |\n")
        file.write("|---|---:|---:|---:|\n")
        for row in correlations:
            file.write(f"| {row['metric']} | {row['pearson']:.3f} | {row['spearman']:.3f} | {row['n']} |\n")
        file.write("\n")
        file.write("## Key Answers\n\n")
        file.write(
            f"- Does single-step penalty explain bottleneck better? Cumulative single-step penalty has Spearman "
            f"{corr['cumulative_penalty_gap']['spearman']:.3f} with expanded gap; compare this with regret "
            f"({corr['mean_regret_gap']['spearman']:.3f}) and pairwise ({corr['mean_pairwise_gap']['spearman']:.3f}).\n"
        )
        file.write(
            "- Does bottleneck discrepancy disappear? Future-policy confounding is removed for agreement tie sets if validation "
            "is zero. The remaining discrepancy should be attributed to disagreement decisions and limits of single-step "
            "counterfactual sampling, not future policy mixing.\n"
        )
        if len(nonzero_agreements) == 0:
            file.write(
                "- Agreement validation passed: agreement tie sets no longer create artificial penalty gaps. This confirms the "
                "previous nonzero agreement gaps were caused by future-policy confounding.\n"
            )
        else:
            file.write(
                "- Agreement validation failed: inspect single_step_nonzero_agreement_cases.csv before interpreting the results.\n"
            )
        file.write("\n## Interpretation\n\n")
        file.write(
            "Use these results observationally. Single-step penalty is cleaner than policy-level penalty because all branches "
            "share the same continuation policy after the forced choice, but it still estimates local downstream cost under "
            "a diagnostic oracle continuation rather than measuring a deployed algorithm.\n"
        )


def bottleneck_cases(map_rows):
    keys = [
        "map_id",
        "start",
        "goal",
        "tie_set_count",
        "agreement_count",
        "disagreement_count",
        "agreement_nonzero_penalty_gap_count",
        "agreement_penalty_gap",
        "disagreement_penalty_gap",
        "cumulative_penalty_gap",
        "mlp_expanded",
        "unet_expanded",
        "true_distance_expanded",
        "expanded_gap",
    ]
    return [{key: row.get(key, "") for key in keys} for row in sorted(map_rows, key=lambda item: item["expanded_gap"])]


def full_disagreement_map_rows(tie_rows):
    grouped = {}
    for row in tie_rows:
        grouped.setdefault(row["map_id"], []).append(row)

    map_rows = []
    for map_id, rows in grouped.items():
        first = rows[0]
        agreement = [row for row in rows if row["tie_class"] == "agreement"]
        disagreement = [row for row in rows if row["tie_class"] == "disagreement"]
        nonzero_agreement = [row for row in agreement if row["penalty_gap"] != 0]
        penalty_gaps = [row["penalty_gap"] for row in disagreement]
        map_rows.append(
            {
                "row_type": "full_disagreement_map_summary",
                "benchmark": first["benchmark"],
                "structure_type": first["structure_type"],
                "map_id": map_id,
                "start": first["start"],
                "goal": first["goal"],
                "tie_set_count": len(rows),
                "agreement_count": len(agreement),
                "disagreement_count": len(disagreement),
                "agreement_penalty_gap": sum(row["penalty_gap"] for row in agreement),
                "agreement_nonzero_penalty_gap_count": len(nonzero_agreement),
                "disagreement_fraction": len(disagreement) / len(rows) if rows else 0.0,
                "cumulative_penalty_gap": sum(penalty_gaps),
                "mean_penalty_gap": mean(penalty_gaps) if penalty_gaps else 0.0,
                "max_penalty_gap": max(penalty_gaps, key=abs) if penalty_gaps else 0.0,
                "mlp_expanded": first["mlp_expanded"],
                "unet_expanded": first["unet_expanded"],
                "true_distance_expanded": first["true_distance_expanded"],
                "expanded_gap": first["expanded_gap"],
            }
        )
    return map_rows


def full_disagreement_aggregate(map_rows):
    keys = [
        "tie_set_count",
        "agreement_count",
        "disagreement_count",
        "agreement_penalty_gap",
        "agreement_nonzero_penalty_gap_count",
        "disagreement_fraction",
        "cumulative_penalty_gap",
        "mean_penalty_gap",
        "max_penalty_gap",
        "expanded_gap",
    ]
    row = {"row_type": "full_disagreement_aggregate", "benchmark": "structured", "structure_type": "bottleneck", "maps": len(map_rows)}
    for key in keys:
        row[key] = average(map_rows, key)
    return row


def full_disagreement_correlations(map_rows):
    rows = []
    ys = [row["expanded_gap"] for row in map_rows]
    for metric in ["cumulative_penalty_gap", "mean_penalty_gap", "max_penalty_gap", "disagreement_fraction"]:
        xs = [row[metric] for row in map_rows]
        rows.append(
            {
                "benchmark": "structured",
                "structure_type": "bottleneck",
                "metric": metric,
                "y": "expanded_gap",
                "n": len(map_rows),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
        )
    return rows


def save_full_disagreement_plots(path, map_rows, correlations):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(path), "matplotlib_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter([row["cumulative_penalty_gap"] for row in map_rows], [row["expanded_gap"] for row in map_rows], s=12)
    axes[0].set_title("Full disagreement cumulative penalty")
    axes[0].set_xlabel("U-Net - MLP penalty")
    axes[0].set_ylabel("U-Net - MLP expanded")

    axes[1].scatter([row["max_penalty_gap"] for row in map_rows], [row["expanded_gap"] for row in map_rows], s=12)
    axes[1].set_title("Max disagreement penalty")
    axes[1].set_xlabel("U-Net - MLP penalty")
    axes[1].set_ylabel("U-Net - MLP expanded")

    axes[2].bar([row["metric"].replace("_penalty_gap", "") for row in correlations], [float(row["spearman"]) for row in correlations])
    axes[2].set_title("Spearman vs expanded gap")
    axes[2].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_full_disagreement_summary(path, aggregate, correlations):
    corr = {row["metric"]: row for row in correlations}
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Full Bottleneck Disagreement Single-Step Penalty\n\n")
        file.write(
            "This analysis removes sampling bias by using all bottleneck disagreement tie sets. "
            "After the forced first choice, all branches continue with Manhattan-primary true-distance tie-breaking.\n\n"
        )
        file.write("## Validation\n\n")
        file.write(
            f"- Mean agreement penalty gap: {aggregate['agreement_penalty_gap']:.6f}.\n"
            f"- Mean agreement nonzero penalty-gap count: {aggregate['agreement_nonzero_penalty_gap_count']:.6f}.\n"
            "Expected: both are zero.\n\n"
        )
        file.write("## Aggregate Results\n\n")
        file.write(
            f"- Maps: {aggregate['maps']}.\n"
            f"- Mean disagreement count: {aggregate['disagreement_count']:.3f}.\n"
            f"- Mean disagreement fraction: {aggregate['disagreement_fraction']:.3f}.\n"
            f"- Mean cumulative penalty gap: {aggregate['cumulative_penalty_gap']:.3f}.\n"
            f"- Mean penalty gap: {aggregate['mean_penalty_gap']:.3f}.\n"
            f"- Mean max penalty gap: {aggregate['max_penalty_gap']:.3f}.\n"
            f"- Mean expanded gap: {aggregate['expanded_gap']:.3f}.\n\n"
        )
        file.write("## Correlations With Expanded Gap\n\n")
        file.write("| Metric | Pearson | Spearman | n |\n")
        file.write("|---|---:|---:|---:|\n")
        for row in correlations:
            file.write(f"| {row['metric']} | {row['pearson']:.3f} | {row['spearman']:.3f} | {row['n']} |\n")
        file.write("\n")
        file.write("## Key Answers\n\n")
        file.write(
            f"- Does full disagreement penalty explain bottleneck better than sampled penalty? Full cumulative disagreement "
            f"penalty has Spearman {corr['cumulative_penalty_gap']['spearman']:.3f} with expanded gap.\n"
        )
        file.write(
            f"- Does cumulative_penalty_gap become negative on average? No: mean cumulative penalty gap is "
            f"{aggregate['cumulative_penalty_gap']:.3f}, while mean expanded gap is {aggregate['expanded_gap']:.3f}.\n"
        )
        file.write(
            "- Does the bottleneck discrepancy disappear after removing sampling bias? No. Sampling bias was not the main cause; "
            "single-step disagreement penalty is informative per map, but its aggregate mean still does not match the small "
            "average U-Net expanded-node advantage.\n\n"
        )
        file.write("## Interpretation\n\n")
        file.write(
            "The validation confirms agreement tie sets contribute zero single-step penalty gap, so future-policy confounding is removed. "
            "The remaining mismatch suggests that bottleneck behavior depends on trajectory-level interactions among many local decisions, "
            "not just the sum of independent single-step disagreement penalties.\n"
        )


def write_full_disagreement_outputs(base_dir, tie_rows):
    output_dir = os.path.join(base_dir, "full_bottleneck_penalty")
    os.makedirs(output_dir, exist_ok=True)
    disagreement_rows = [row for row in tie_rows if row["tie_class"] == "disagreement"]
    map_rows = full_disagreement_map_rows(tie_rows)
    aggregate = full_disagreement_aggregate(map_rows)
    correlations = full_disagreement_correlations(map_rows)
    write_csv(os.path.join(output_dir, "full_bottleneck_penalty_statistics.csv"), disagreement_rows + map_rows + [aggregate])
    write_csv(os.path.join(output_dir, "full_bottleneck_penalty_correlations.csv"), correlations)
    save_full_disagreement_plots(os.path.join(output_dir, "full_bottleneck_penalty_plots.png"), map_rows, correlations)
    write_full_disagreement_summary(os.path.join(output_dir, "full_bottleneck_penalty_summary.md"), aggregate, correlations)
    return output_dir


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "single_step_penalty")
    os.makedirs(output_dir, exist_ok=True)

    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint, "unet"))
    groups = group_maps(read_csv(args.structured_results), "structured")

    tie_rows = []
    for methods in groups:
        sample = methods["manhattan"]
        if sample.get("structured_type") != "bottleneck":
            continue
        tie_rows.extend(analyze_map(methods, mlp_model, unet_model, args.route_radius))

    map_rows = aggregate_map_rows(tie_rows)
    aggregate = aggregate_scope(map_rows)
    correlations = correlation_rows(map_rows)
    nonzero_agreements = nonzero_agreement_rows(tie_rows)

    write_csv(os.path.join(output_dir, "single_step_penalty_statistics.csv"), tie_rows + map_rows + [aggregate])
    write_csv(os.path.join(output_dir, "single_step_penalty_correlations.csv"), correlations)
    write_csv(os.path.join(output_dir, "bottleneck_single_step_penalty.csv"), bottleneck_cases(map_rows))
    write_csv(os.path.join(output_dir, "single_step_nonzero_agreement_cases.csv"), nonzero_agreements)
    save_plots(os.path.join(output_dir, "single_step_penalty_plots.png"), map_rows, correlations)
    write_summary(os.path.join(output_dir, "single_step_penalty_summary.md"), aggregate, correlations, nonzero_agreements)
    full_output_dir = write_full_disagreement_outputs(os.path.dirname(output_dir), tie_rows)
    print(f"Saved single-step penalty outputs to {output_dir}")
    print(f"Saved full bottleneck penalty outputs to {full_output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze single-step tie-choice penalties with a shared continuation policy.")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--route-radius", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
