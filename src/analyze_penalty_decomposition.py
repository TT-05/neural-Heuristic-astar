import argparse
import csv
import os

from analyze_tie_set_counterfactual_penalty import (
    collect_tie_events,
    continue_after_forced_choice,
    counterfactual_row,
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


def classify_event(event):
    return "agreement" if event["mlp_selected_node"] == event["unet_selected_node"] else "disagreement"


def penalty_for_event(identity, event, grid, goal, mlp_table, unet_table, true_table):
    row = counterfactual_row(identity, event, grid, goal, mlp_table, unet_table, true_table)
    row["tie_class"] = classify_event(event)
    row["mlp_selected_node"] = node_text(event["mlp_selected_node"])
    row["unet_selected_node"] = node_text(event["unet_selected_node"])
    row["oracle_selected_node"] = node_text(event["oracle_selected_node"])
    return row


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

    tie_rows = []
    for event in events:
        row = penalty_for_event(identity, event, grid, goal, mlp_table, unet_table, true_table)
        row["expanded_steps"] = expanded_steps
        row["total_active_tie_sets"] = len(events)
        row["mlp_expanded"] = result_expanded(methods, "manhattan_mlp_tiebreak")
        row["unet_expanded"] = result_expanded(methods, "manhattan_unet_tiebreak")
        row["true_distance_expanded"] = result_expanded(methods, "manhattan_true_distance_tiebreak")
        row["expanded_gap"] = row["unet_expanded"] - row["mlp_expanded"]
        tie_rows.append(row)
    return tie_rows


def aggregate_map_rows(tie_rows):
    grouped = {}
    for row in tie_rows:
        grouped.setdefault(row["map_id"], []).append(row)

    map_rows = []
    for map_id, rows in grouped.items():
        first = rows[0]
        agreement = [row for row in rows if row["tie_class"] == "agreement"]
        disagreement = [row for row in rows if row["tie_class"] == "disagreement"]
        total_abs_gap = sum(abs(row["penalty_gap"]) for row in rows)
        agreement_abs_gap = sum(abs(row["penalty_gap"]) for row in agreement)
        disagreement_abs_gap = sum(abs(row["penalty_gap"]) for row in disagreement)
        agreement_penalty_gap = sum(row["penalty_gap"] for row in agreement)
        disagreement_penalty_gap = sum(row["penalty_gap"] for row in disagreement)
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
                "agreement_penalty_gap": agreement_penalty_gap,
                "disagreement_penalty_gap": disagreement_penalty_gap,
                "cumulative_penalty_gap": agreement_penalty_gap + disagreement_penalty_gap,
                "agreement_penalty_share": agreement_abs_gap / total_abs_gap if total_abs_gap else 0.0,
                "disagreement_penalty_share": disagreement_abs_gap / total_abs_gap if total_abs_gap else 0.0,
                "mean_agreement_penalty_gap": average(agreement, "penalty_gap"),
                "mean_disagreement_penalty_gap": average(disagreement, "penalty_gap"),
                "mean_penalty_gap": average(rows, "penalty_gap"),
                "mean_route_critical_overlap": average(rows, "route_critical_overlap"),
                "mean_tie_set_size": average(rows, "tie_set_size"),
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
        "agreement_penalty_gap",
        "disagreement_penalty_gap",
        "cumulative_penalty_gap",
        "agreement_penalty_share",
        "disagreement_penalty_share",
        "mean_agreement_penalty_gap",
        "mean_disagreement_penalty_gap",
        "mean_penalty_gap",
        "mean_route_critical_overlap",
        "mean_tie_set_size",
        "expanded_gap",
    ]
    row = {"row_type": "aggregate", "benchmark": "structured", "structure_type": "bottleneck", "maps": len(map_rows)}
    for key in keys:
        row[key] = average(map_rows, key)
    return row


def correlation_rows(map_rows):
    rows = []
    ys = [row["expanded_gap"] for row in map_rows]
    for metric in ["agreement_penalty_gap", "disagreement_penalty_gap", "cumulative_penalty_gap"]:
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


def bottleneck_cases(map_rows):
    keys = [
        "map_id",
        "start",
        "goal",
        "tie_set_count",
        "agreement_count",
        "disagreement_count",
        "agreement_fraction",
        "disagreement_fraction",
        "agreement_penalty_gap",
        "disagreement_penalty_gap",
        "cumulative_penalty_gap",
        "agreement_penalty_share",
        "disagreement_penalty_share",
        "mlp_expanded",
        "unet_expanded",
        "true_distance_expanded",
        "expanded_gap",
    ]
    return [{key: row.get(key, "") for key in keys} for row in sorted(map_rows, key=lambda item: item["expanded_gap"])]


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
    axes[0].scatter([row["disagreement_penalty_gap"] for row in map_rows], [row["expanded_gap"] for row in map_rows], s=12)
    axes[0].set_title("Disagreement penalty gap")
    axes[0].set_xlabel("U-Net - MLP penalty")
    axes[0].set_ylabel("U-Net - MLP expanded")

    axes[1].scatter([row["agreement_penalty_gap"] for row in map_rows], [row["expanded_gap"] for row in map_rows], s=12)
    axes[1].set_title("Agreement penalty gap")
    axes[1].set_xlabel("U-Net - MLP penalty")
    axes[1].set_ylabel("U-Net - MLP expanded")

    axes[2].bar([row["metric"].replace("_penalty_gap", "") for row in correlations], [float(row["spearman"]) for row in correlations])
    axes[2].set_title("Spearman vs expanded gap")
    axes[2].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def corr_lookup(correlations, metric):
    return next(row for row in correlations if row["metric"] == metric)


def write_summary(path, aggregate, correlations):
    agreement_corr = corr_lookup(correlations, "agreement_penalty_gap")
    disagreement_corr = corr_lookup(correlations, "disagreement_penalty_gap")
    cumulative_corr = corr_lookup(correlations, "cumulative_penalty_gap")
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Penalty Decomposition Analysis\n\n")
        file.write(
            "This analysis decomposes bottleneck counterfactual expansion penalties into agreement and disagreement tie sets. "
            "It is analysis-only and does not modify production A*, checkpoints, or benchmark outputs.\n\n"
        )
        file.write("## Aggregate Bottleneck Results\n\n")
        file.write(
            f"- Maps: {aggregate['maps']}.\n"
            f"- Mean active tie sets per map: {aggregate['tie_set_count']:.3f}.\n"
            f"- Agreement fraction: {aggregate['agreement_fraction']:.3f}.\n"
            f"- Disagreement fraction: {aggregate['disagreement_fraction']:.3f}.\n"
            f"- Agreement penalty gap: {aggregate['agreement_penalty_gap']:.3f}.\n"
            f"- Disagreement penalty gap: {aggregate['disagreement_penalty_gap']:.3f}.\n"
            f"- Cumulative penalty gap: {aggregate['cumulative_penalty_gap']:.3f}.\n"
            f"- Agreement penalty share: {aggregate['agreement_penalty_share']:.3f}.\n"
            f"- Disagreement penalty share: {aggregate['disagreement_penalty_share']:.3f}.\n"
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
            f"- Q1: Disagreement tie sets contribute {aggregate['disagreement_penalty_share']:.3f} of absolute penalty mass "
            f"and have Spearman {disagreement_corr['spearman']:.3f} with expanded gap.\n"
        )
        file.write(
            f"- Q2: Agreement tie sets contribute {aggregate['agreement_penalty_share']:.3f} of absolute penalty mass. "
            "Because MLP and U-Net select the same node there, agreement penalties mainly reflect shared downstream structure "
            "rather than a decision difference between the methods.\n"
        )
        file.write(
            "- Q3: U-Net gains are not explained by agreement tie sets. Decision differences must come from disagreement events, "
            "but the aggregate bottleneck behavior can still be influenced by shared difficult tie states.\n"
        )
        file.write(
            f"- Q4: Disagreement tie sets account for {aggregate['disagreement_penalty_share']:.3f} of total sampled absolute "
            "penalty mass.\n"
        )
        if abs(float(disagreement_corr["spearman"])) > abs(float(agreement_corr["spearman"])):
            file.write(
                "- Q5: Disagreement penalty explains more variance than agreement penalty, so future diagnostics should focus on "
                "when neural guidance changes the selected node, while still tracking agreement states as context.\n\n"
            )
        else:
            file.write(
                "- Q5: Agreement penalty explains comparable or greater variance, so focusing only on disagreement tie sets would "
                "introduce sampling bias. Future work should also improve overall ordering quality and shared tie-state handling.\n\n"
            )
        file.write("## Interpretation\n\n")
        file.write(
            "These are counterfactual diagnostics, not causal proof. Agreement tie sets cannot directly favor MLP or U-Net at the "
            "selection step because both choose the same node, but their penalties reveal where both methods face costly tied "
            "states. Disagreement tie sets are the direct locus of method-specific decision differences.\n"
        )


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "penalty_decomposition")
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
    write_csv(os.path.join(output_dir, "penalty_decomposition_statistics.csv"), tie_rows + map_rows + [aggregate])
    write_csv(os.path.join(output_dir, "bottleneck_penalty_decomposition.csv"), bottleneck_cases(map_rows))
    save_plots(os.path.join(output_dir, "penalty_decomposition_plots.png"), map_rows, correlations)
    write_summary(os.path.join(output_dir, "penalty_decomposition_summary.md"), aggregate, correlations)
    print(f"Saved penalty decomposition outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Decompose bottleneck tie-set penalties by agreement/disagreement class.")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--route-radius", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
