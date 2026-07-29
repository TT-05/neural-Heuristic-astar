import argparse
import csv
import os
import re


STRUCTURES = ["maze_like", "bottleneck", "large_block", "narrow_corridor"]


def read_text(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def read_csv_dicts(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def as_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def first_match(text, pattern, default=""):
    match = re.search(pattern, text, flags=re.S)
    return match.group(1).strip() if match else default


def aggregate_rows(rows):
    return [row for row in rows if row.get("row_type") == "aggregate"]


def correlation_rows(rows):
    return [row for row in rows if row.get("row_type") == "correlation"]


def cross_structure_aggregates(rows):
    result = {}
    for row in aggregate_rows(rows):
        structure = row.get("structured_type")
        if structure:
            result[structure] = {
                "maps": as_float(row.get("maps")),
                "expanded_gap": as_float(row.get("expanded_gap")),
                "off_path_count_gap": as_float(row.get("off_path_count_gap")),
                "off_path_fraction_gap": as_float(row.get("off_path_fraction_gap")),
                "mean_recovery_time_gap": as_float(row.get("mean_recovery_time_gap")),
                "frontier_compactness_gap": as_float(row.get("frontier_compactness_gap")),
            }
    return result


def cross_structure_correlations(rows):
    result = {}
    for row in correlation_rows(rows):
        structure = row.get("structured_type")
        metric = row.get("metric")
        if structure and metric:
            result.setdefault(structure, {})[metric] = {
                "pearson": as_float(row.get("pearson")),
                "spearman": as_float(row.get("spearman")),
            }
    return result


def large_block_extra(rows):
    aggregate = {}
    correlations = {}
    for row in aggregate_rows(rows):
        aggregate = {
            "mean_true_distance_gap": as_float(row.get("mean_true_distance_gap")),
            "true_distance_progress_slope_gap": as_float(row.get("true_distance_progress_slope_gap")),
            "mean_obstacle_boundary_distance_gap": as_float(row.get("mean_obstacle_boundary_distance_gap")),
            "near_boundary_fraction_gap": as_float(row.get("near_boundary_fraction_gap")),
            "boundary_following_run_gap": as_float(row.get("boundary_following_run_gap")),
        }
    for row in correlation_rows(rows):
        metric = row.get("metric")
        if metric:
            correlations[metric] = {
                "pearson": as_float(row.get("pearson")),
                "spearman": as_float(row.get("spearman")),
            }
    return aggregate, correlations


def maze_visual_evidence(rows):
    groups = {}
    correlations = {}
    for row in rows:
        if row.get("row_type") == "group_summary":
            groups[row.get("winner_group")] = {
                "n": as_float(row.get("n")),
                "expanded_gap": as_float(row.get("expanded_gap")),
                "off_path_fraction_gap": as_float(row.get("off_path_ge2_fraction_gap")),
                "frontier_compactness_gap": as_float(row.get("frontier_compactness_gap")),
                "disagreement_count": as_float(row.get("disagreement_count")),
            }
        if row.get("row_type") == "correlation":
            metric = row.get("metric")
            if metric:
                correlations[metric] = {
                    "pearson": as_float(row.get("pearson")),
                    "spearman": as_float(row.get("spearman")),
                }
    return groups, correlations


def route_bias_evidence(rows):
    groups = {}
    correlations = {}
    for row in rows:
        if row.get("row_type") == "group_summary":
            groups[row.get("winner_group")] = {
                "expanded_gap": as_float(row.get("expanded_gap")),
                "final_cumulative_off_path_gap": as_float(row.get("final_cumulative_off_path_gap")),
                "off_path_fraction_gap": as_float(row.get("off_path_fraction_gap")),
                "max_consecutive_on_path_gap": as_float(row.get("max_consecutive_on_path_gap")),
                "mean_recovery_time_gap": as_float(row.get("mean_recovery_time_gap")),
            }
        if row.get("row_type") == "correlation":
            metric = row.get("metric")
            if metric:
                correlations[metric] = {
                    "pearson": as_float(row.get("pearson")),
                    "spearman": as_float(row.get("spearman")),
                }
    return groups, correlations


def unet_effect(expanded_gap):
    if expanded_gap is None:
        return "unclear"
    if expanded_gap < -0.05:
        return "helps"
    if expanded_gap > 0.05:
        return "hurts"
    return "roughly neutral"


def strongest_metrics(structure, cross_corr, large_corr, maze_corr, route_corr):
    if structure == "maze_like":
        return [
            f"off-path count Spearman {fmt(cross_corr.get('off_path_count_gap', {}).get('spearman'))}",
            f"compactness Spearman {fmt(cross_corr.get('frontier_compactness_gap', {}).get('spearman'))}",
            f"case-study off-path fraction Spearman {fmt(maze_corr.get('off_path_ge2_fraction_gap', {}).get('spearman'))}",
            f"cumulative off-path Spearman {fmt(route_corr.get('final_cumulative_off_path_gap', {}).get('spearman'))}",
        ]
    if structure == "bottleneck":
        return [
            f"compactness Spearman {fmt(cross_corr.get('frontier_compactness_gap', {}).get('spearman'))}",
            f"off-path count Spearman {fmt(cross_corr.get('off_path_count_gap', {}).get('spearman'))}",
            f"recovery-time Spearman {fmt(cross_corr.get('mean_recovery_time_gap', {}).get('spearman'))}",
        ]
    if structure == "large_block":
        return [
            f"mean true-distance Spearman {fmt(large_corr.get('mean_true_distance_gap', {}).get('spearman'))}",
            f"progress-slope Spearman {fmt(large_corr.get('true_distance_progress_slope_gap', {}).get('spearman'))}",
            f"compactness Spearman {fmt(large_corr.get('frontier_compactness_gap', {}).get('spearman'))}",
            f"near-boundary fraction Spearman {fmt(large_corr.get('near_boundary_fraction_gap', {}).get('spearman'))}",
        ]
    return [
        f"compactness Spearman {fmt(cross_corr.get('frontier_compactness_gap', {}).get('spearman'))}",
        f"off-path count Spearman {fmt(cross_corr.get('off_path_count_gap', {}).get('spearman'))}",
        f"disagreement count Spearman {fmt(cross_corr.get('disagreement_count', {}).get('spearman'))}",
    ]


def structure_interpretation(structure, aggregate, cross_corr, large_corr, maze_corr, route_corr):
    effect = unet_effect(aggregate.get("expanded_gap"))
    if structure == "maze_like":
        return {
            "effect": effect,
            "dominant_mechanism": "persistent route bias suppresses wrong-branch exploration",
            "failure_mode": "when local ordering is worse, U-Net can still lose selected cases despite useful global route bias",
            "design_implication": "trust neural guidance more when spatial context identifies connected route families; monitor trajectory off-path growth",
            "metrics": strongest_metrics(structure, cross_corr, large_corr, maze_corr, route_corr),
        }
    if structure == "bottleneck":
        return {
            "effect": effect,
            "dominant_mechanism": "small U-Net advantage is linked more to compact, route-aware progression than broad off-path reduction",
            "failure_mode": "average ordering metrics can miss a few high-impact entrance or passage decisions",
            "design_implication": "trust neural guidance cautiously near bottlenecks; evaluate high-impact local decisions, not only mean ordering",
            "metrics": strongest_metrics(structure, cross_corr, large_corr, maze_corr, route_corr),
        }
    if structure == "large_block":
        return {
            "effect": effect,
            "dominant_mechanism": "weaker geometric progress around large obstacles and poorer obstacle-edge ordering",
            "failure_mode": "boundary attraction is weakly supported; the stronger signal is slower goal-directed progression around the block",
            "design_implication": "weaken or constrain neural guidance when the task is mainly clean geometric progress around a large obstacle",
            "metrics": strongest_metrics(structure, cross_corr, large_corr, maze_corr, route_corr),
        }
    return {
        "effect": effect,
        "dominant_mechanism": "U-Net increases off-path and less compact exploration in constrained corridor geometry",
        "failure_mode": "learned spatial bias can be harmful when narrow geometry leaves little room for global route preference",
        "design_implication": "weaken neural guidance in narrow corridors unless local passage reliability is verified",
        "metrics": strongest_metrics(structure, cross_corr, large_corr, maze_corr, route_corr),
    }


def build_summary_table(cross_aggs, cross_corrs, large_aggs, large_corrs, maze_corrs, route_corrs):
    table = []
    for structure in STRUCTURES:
        aggregate = dict(cross_aggs.get(structure, {}))
        if structure == "large_block":
            aggregate.update(large_aggs)
        interpretation = structure_interpretation(
            structure,
            aggregate,
            cross_corrs.get(structure, {}),
            large_corrs,
            maze_corrs,
            route_corrs,
        )
        table.append(
            {
                "structure_type": structure,
                "unet_effect": interpretation["effect"],
                "mean_expanded_gap_unet_minus_mlp": fmt(aggregate.get("expanded_gap")),
                "off_path_count_gap": fmt(aggregate.get("off_path_count_gap")),
                "off_path_fraction_gap": fmt(aggregate.get("off_path_fraction_gap")),
                "recovery_time_gap": fmt(aggregate.get("mean_recovery_time_gap")),
                "frontier_compactness_gap": fmt(aggregate.get("frontier_compactness_gap")),
                "dominant_mechanism": interpretation["dominant_mechanism"],
                "strongest_supporting_metrics": "; ".join(interpretation["metrics"]),
                "failure_mode": interpretation["failure_mode"],
                "algorithm_design_implication": interpretation["design_implication"],
            }
        )
    return table


def write_csv(path, rows):
    fieldnames = [
        "structure_type",
        "unet_effect",
        "mean_expanded_gap_unet_minus_mlp",
        "off_path_count_gap",
        "off_path_fraction_gap",
        "recovery_time_gap",
        "frontier_compactness_gap",
        "dominant_mechanism",
        "strongest_supporting_metrics",
        "failure_mode",
        "algorithm_design_implication",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(file, rows):
    file.write(
        "| Structure | U-Net effect | expanded gap | dominant mechanism | strongest supporting metrics | design implication |\n"
    )
    file.write("|---|---|---:|---|---|---|\n")
    for row in rows:
        file.write(
            f"| {row['structure_type']} | {row['unet_effect']} | {row['mean_expanded_gap_unet_minus_mlp']} "
            f"| {row['dominant_mechanism']} | {row['strongest_supporting_metrics']} "
            f"| {row['algorithm_design_implication']} |\n"
        )


def write_synthesis(path, rows, summaries):
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Mechanism Design Synthesis\n\n")
        file.write(
            "This report synthesizes the existing mechanism analyses into design principles for future A*-based learned-heuristic search. "
            "It is synthesis-only: it does not implement a new algorithm, modify A*, retrain models, or change benchmark outputs.\n\n"
        )

        file.write("## Evidence Sources\n\n")
        for label, summary_path in summaries.items():
            file.write(f"- {label}: `{summary_path}`\n")
        file.write("\n")

        file.write("## 1. Mechanisms By Structure Type\n\n")
        write_markdown_table(file, rows)
        file.write("\n")

        file.write("## 2. Structure-Specific Interpretation\n\n")
        for row in rows:
            file.write(f"### {row['structure_type']}\n\n")
            file.write(f"- U-Net effect: {row['unet_effect']} relative to MLP tie-breaking.\n")
            file.write(f"- Dominant mechanism: {row['dominant_mechanism']}.\n")
            file.write(f"- Supporting metrics: {row['strongest_supporting_metrics']}.\n")
            file.write(f"- Failure mode: {row['failure_mode']}.\n")
            file.write(f"- Design implication: {row['algorithm_design_implication']}.\n\n")

        file.write("## 3. Unified Mechanism Chain\n\n")
        file.write(
            "The analyses support the following chain:\n\n"
            "```text\n"
            "map structure\n"
            "-> neural guidance behavior\n"
            "-> route bias / geometric progression / off-path behavior\n"
            "-> expanded-node count\n"
            "```\n\n"
        )
        file.write(
            "In maze-like maps, U-Net's obstacle-aware field tends to create persistent route bias. That bias reduces wrong-branch "
            "exploration, shortens recovery after leaving the path neighborhood, and lowers expansions. In bottleneck maps, the "
            "same effect is weaker and concentrated around compact progression and high-impact local decisions. In large-block maps, "
            "U-Net does not mainly fail through broad off-path exploration or strong obstacle-boundary attraction; it more often makes "
            "less efficient goal-distance progress around the block. In narrow corridors, learned spatial bias is more likely to add "
            "unnecessary exploration, so MLP's simpler learned geometry can be cleaner.\n\n"
        )

        file.write("## 4. What To Trust\n\n")
        file.write(
            "Neural guidance should be trusted most when the map contains maze-like or branching obstacle structure where global spatial "
            "context can identify a better route family than local geometry alone. The useful signal is not just lower heuristic value "
            "error; it is trajectory-level ordering that keeps expansions near the solution route and suppresses wrong branches.\n\n"
        )
        file.write(
            "Neural guidance should be weakened or constrained when the structure is dominated by narrow corridors or large-block detours, "
            "where the learned field can disrupt clean geometric progress. In these structures, aggregate heuristic accuracy is not "
            "enough; the relevant question is whether the learned ordering improves route-critical and obstacle-edge decisions.\n\n"
        )

        file.write("## 5. Why Pure Learned Heuristic Values Are Insufficient\n\n")
        file.write(
            "Pure learned heuristic values are insufficient because the same field property can help in one structure and hurt in another. "
            "Global route bias reduces expansions in maze-like maps, but learned spatial bias can be harmful in narrow-corridor and "
            "large-block settings. Average value accuracy also misses where errors occur: route-critical cells, bottleneck entrances, "
            "corridor passages, and obstacle edges matter more than many ordinary free cells.\n\n"
        )
        file.write(
            "The evidence therefore favors search-aware use of learned information: preserve ordering and route bias when they reduce "
            "off-path exploration, but constrain or downweight learned guidance when it creates poor progression, artificial barriers, "
            "or structure-specific failures.\n\n"
        )

        file.write("## 6. Design Principles\n\n")
        file.write("- Preserve neural route bias when it produces persistent route following and lower cumulative off-path exploration.\n")
        file.write("- Use learned signals as ordering guidance, not merely as absolute distance estimates.\n")
        file.write("- Evaluate route-critical and trajectory-level behavior separately from global heuristic error.\n")
        file.write("- Weaken learned guidance in structures where it worsens compactness, off-path exploration, or goal-distance progression.\n")
        file.write("- Treat large-block and narrow-corridor failures as first-class design constraints, not aggregate noise.\n")
        file.write("- Avoid relying on a single learned heuristic policy across all map structures without diagnostics or safeguards.\n\n")

        file.write("## Final Synthesis\n\n")
        file.write(
            "Future A*-based learned-heuristic algorithms should preserve U-Net-like obstacle-aware route bias where it creates persistent "
            "trajectory guidance, but should avoid treating learned heuristic values as uniformly trustworthy. The design target should "
            "be conditional use of neural guidance: trust it when it improves route-level ordering and suppresses wrong branches, weaken "
            "it when it harms local passage reliability or geometric progress, and evaluate both behaviors with structure-aware metrics.\n"
        )


def write_design_principles(path, rows):
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Proposed Algorithm Design Principles\n\n")
        file.write(
            "These principles are derived from existing mechanism analyses only. They are not a proposed algorithm implementation.\n\n"
        )
        file.write("## Trust Neural Guidance When\n\n")
        file.write("- The structure has branching or maze-like obstacle connectivity.\n")
        file.write("- U-Net reduces cumulative off-path expansion over the trajectory.\n")
        file.write("- U-Net improves compactness without increasing route-critical or obstacle-edge mistakes.\n")
        file.write("- The learned field improves ordering among plausible route alternatives rather than merely lowering global MAE.\n\n")

        file.write("## Weaken Neural Guidance When\n\n")
        file.write("- The map resembles narrow corridors where learned spatial bias adds unnecessary exploration.\n")
        file.write("- The task is dominated by clean geometric progress around a large block.\n")
        file.write("- U-Net worsens true-distance progression, frontier compactness, or recovery behavior.\n")
        file.write("- Route-critical or passage-level diagnostics indicate high-cost local mistakes.\n\n")

        file.write("## Useful Signals Beyond Heuristic Value Accuracy\n\n")
        file.write("- Tie-set ordering quality among equal Manhattan-primary f-values.\n")
        file.write("- Persistent route bias measured by cumulative off-path exploration.\n")
        file.write("- Recovery time after leaving the optimal-path neighborhood.\n")
        file.write("- Frontier compactness and goal-distance progression.\n")
        file.write("- Route-critical behavior near bottlenecks, corridors, gaps, and obstacle edges.\n\n")

        file.write("## Design Risks To Avoid\n\n")
        file.write("- Assuming a learned heuristic is globally useful because it helps a subset of structures.\n")
        file.write("- Optimizing average heuristic value error while ignoring search trajectory effects.\n")
        file.write("- Preserving route bias without constraining local failures near passages or obstacle edges.\n")
        file.write("- Reporting aggregate expansion gains that hide large_block or narrow_corridor regressions.\n\n")

        file.write("## Concise Principle\n\n")
        file.write(
            "A future method should use learned heuristics as conditional search guidance: preserve obstacle-aware route ordering when it "
            "keeps A* on productive trajectories, and reduce the influence of learned values when they degrade passage reliability, "
            "frontier compactness, or goal-directed progression.\n"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Synthesize mechanism analyses into design principles.")
    parser.add_argument("--maze-visual-dir", default="outputs/maze_visual_mechanisms")
    parser.add_argument("--route-bias-dir", default="outputs/route_bias_mechanisms")
    parser.add_argument("--cross-structure-dir", default="outputs/cross_structure_route_bias")
    parser.add_argument("--large-block-dir", default="outputs/large_block_failure_mechanisms")
    parser.add_argument("--output-dir", default="outputs/mechanism_design_synthesis")
    return parser.parse_args()


def main(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    maze_dir = os.path.join(project_root, args.maze_visual_dir)
    route_dir = os.path.join(project_root, args.route_bias_dir)
    cross_dir = os.path.join(project_root, args.cross_structure_dir)
    large_dir = os.path.join(project_root, args.large_block_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    summary_paths = {
        "maze visual mechanisms": os.path.join(args.maze_visual_dir, "maze_visual_mechanism_summary.md"),
        "route-bias mechanisms": os.path.join(args.route_bias_dir, "route_bias_mechanism_summary.md"),
        "cross-structure route bias": os.path.join(args.cross_structure_dir, "cross_structure_route_bias_summary.md"),
        "large-block failure mechanisms": os.path.join(args.large_block_dir, "large_block_failure_summary.md"),
    }
    for relative_path in summary_paths.values():
        read_text(os.path.join(project_root, relative_path))

    cross_rows = read_csv_dicts(os.path.join(cross_dir, "cross_structure_route_bias_statistics.csv"))
    large_rows = read_csv_dicts(os.path.join(large_dir, "large_block_failure_statistics.csv"))
    maze_rows = read_csv_dicts(os.path.join(maze_dir, "maze_visual_mechanism_statistics.csv"))
    route_rows = read_csv_dicts(os.path.join(route_dir, "route_bias_mechanism_statistics.csv"))

    cross_aggs = cross_structure_aggregates(cross_rows)
    cross_corrs = cross_structure_correlations(cross_rows)
    large_aggs, large_corrs = large_block_extra(large_rows)
    _, maze_corrs = maze_visual_evidence(maze_rows)
    _, route_corrs = route_bias_evidence(route_rows)

    table = build_summary_table(cross_aggs, cross_corrs, large_aggs, large_corrs, maze_corrs, route_corrs)

    write_csv(os.path.join(output_dir, "mechanism_summary_table.csv"), table)
    write_synthesis(os.path.join(output_dir, "mechanism_design_synthesis.md"), table, summary_paths)
    write_design_principles(os.path.join(output_dir, "proposed_algorithm_design_principles.md"), table)

    print(f"Saved mechanism design synthesis to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
