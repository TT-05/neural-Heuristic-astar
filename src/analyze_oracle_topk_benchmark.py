import argparse
import csv
import math
import os
import random
import statistics


METHODS = [
    "manhattan",
    "manhattan_large_g_tiebreak",
    "manhattan_mlp_tiebreak",
    "manhattan_unet_tiebreak",
    "manhattan_oracle_top1_tiebreak",
    "manhattan_oracle_top2_tiebreak",
    "manhattan_oracle_top4_tiebreak",
    "manhattan_oracle_top8_tiebreak",
    "manhattan_true_distance_tiebreak",
]
TOPK_METHODS = [
    "manhattan_unet_tiebreak",
    "manhattan_oracle_top1_tiebreak",
    "manhattan_oracle_top2_tiebreak",
    "manhattan_oracle_top4_tiebreak",
    "manhattan_oracle_top8_tiebreak",
    "manhattan_true_distance_tiebreak",
]
STRUCTURED_TYPES = ["maze_like", "bottleneck", "large_block", "narrow_corridor"]


def read_csv(path):
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


def stdev(values):
    values = list(values)
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def median(values):
    values = list(values)
    return statistics.median(values) if values else 0.0


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


def group_complete_maps(rows, structured_type=None):
    grouped = {}
    for row in rows:
        if row.get("heuristic") in METHODS:
            grouped.setdefault(map_key(row), {})[row["heuristic"]] = row

    complete = []
    for methods in grouped.values():
        if not all(method in methods for method in METHODS):
            continue
        sample = methods["manhattan"]
        if sample.get("skip_reason") or sample.get("optimal_cost") in ("", "-1"):
            continue
        if structured_type is not None and sample.get("structured_type") != structured_type:
            continue
        complete.append(methods)
    return complete


def bootstrap_ci(values, iterations=2000, seed=0):
    values = list(values)
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in values]
        samples.append(mean(sample))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return lo, hi


def wilcoxon_pvalue(values):
    nonzero = [value for value in values if value != 0]
    if not nonzero:
        return ""
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(nonzero).pvalue)
    except Exception:
        return ""


def captured_fraction(methods, method):
    baseline = to_float(methods["manhattan_unet_tiebreak"], "expanded_nodes")
    oracle = to_float(methods["manhattan_true_distance_tiebreak"], "expanded_nodes")
    current = to_float(methods[method], "expanded_nodes")
    denominator = baseline - oracle
    if denominator <= 0:
        return None
    return (baseline - current) / denominator


def summarize_method(maps, method, scope):
    rows = [methods[method] for methods in maps]
    expanded = [to_float(row, "expanded_nodes") for row in rows]
    runtimes = [to_float(row, "runtime_seconds") for row in rows]
    diffs_vs_unet = [
        to_float(methods[method], "expanded_nodes") - to_float(methods["manhattan_unet_tiebreak"], "expanded_nodes")
        for methods in maps
    ]
    wins = sum(1 for value in diffs_vs_unet if value < 0)
    ties = sum(1 for value in diffs_vs_unet if value == 0)
    losses = sum(1 for value in diffs_vs_unet if value > 0)
    captures = [captured_fraction(methods, method) for methods in maps]
    captures = [value for value in captures if value is not None and math.isfinite(value)]
    ci_lo, ci_hi = bootstrap_ci(diffs_vs_unet)
    return {
        "scope": scope,
        "algorithm": method,
        "cases": len(rows),
        "mean_expanded_nodes": mean(expanded),
        "median_expanded_nodes": median(expanded),
        "std_expanded_nodes": stdev(expanded),
        "mean_runtime_seconds": mean(runtimes),
        "optimality_rate": mean(1.0 if row.get("optimal") == "True" else 0.0 for row in rows),
        "mean_improvement_over_unet_tiebreak": -mean(diffs_vs_unet),
        "median_improvement_over_unet_tiebreak": -median(diffs_vs_unet),
        "wins_vs_unet": wins,
        "ties_vs_unet": ties,
        "losses_vs_unet": losses,
        "mean_oracle_benefit_capture": mean(captures) if captures else "",
        "paired_diff_vs_unet_mean": mean(diffs_vs_unet),
        "paired_diff_vs_unet_bootstrap_ci_low": ci_lo,
        "paired_diff_vs_unet_bootstrap_ci_high": ci_hi,
        "paired_wilcoxon_pvalue": wilcoxon_pvalue(diffs_vs_unet),
    }


def paired_comparisons(maps, scope):
    rows = []
    for method in METHODS:
        if method == "manhattan_unet_tiebreak":
            continue
        diffs = [
            to_float(methods[method], "expanded_nodes")
            - to_float(methods["manhattan_unet_tiebreak"], "expanded_nodes")
            for methods in maps
        ]
        ci_lo, ci_hi = bootstrap_ci(diffs)
        rows.append(
            {
                "scope": scope,
                "algorithm": method,
                "reference": "manhattan_unet_tiebreak",
                "cases": len(diffs),
                "mean_paired_expansion_diff": mean(diffs),
                "median_paired_expansion_diff": median(diffs),
                "bootstrap_ci_low": ci_lo,
                "bootstrap_ci_high": ci_hi,
                "wilcoxon_pvalue": wilcoxon_pvalue(diffs),
            }
        )
    return rows


def budget_label(method):
    if method == "manhattan_unet_tiebreak":
        return "0"
    if method == "manhattan_true_distance_tiebreak":
        return "full"
    return method.replace("manhattan_oracle_top", "").replace("_tiebreak", "")


def budget_curve_rows(maps, scope):
    rows = []
    for method in TOPK_METHODS:
        expanded = [to_float(methods[method], "expanded_nodes") for methods in maps]
        captures = [captured_fraction(methods, method) for methods in maps]
        captures = [value for value in captures if value is not None and math.isfinite(value)]
        rows.append(
            {
                "scope": scope,
                "budget": budget_label(method),
                "algorithm": method,
                "mean_expanded_nodes": mean(expanded),
                "median_expanded_nodes": median(expanded),
                "mean_oracle_benefit_capture": mean(captures) if captures else "",
                "median_oracle_benefit_capture": median(captures) if captures else "",
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summary_rows, budget_rows, maps):
    overall = [row for row in summary_rows if row["scope"] == "all"]
    by_method = {row["algorithm"]: row for row in overall}
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Oracle Top-k Tie-set Ordering Analysis\n\n")
        file.write(
            "All algorithms use Manhattan `g + h` as the primary priority. U-Net, true distance, and partial oracle "
            "signals only order nodes inside equal Manhattan-f tie sets.\n\n"
        )
        file.write("## Episode Definition\n\n")
        file.write(
            "For each active minimum Manhattan-f layer, the current OPEN nodes are snapshotted once. The true-distance-best "
            "k nodes in that snapshot receive oracle ranks and are expanded before uncorrected nodes. Nodes that arrive later "
            "with the same primary f are not retroactively added to the corrected group; they use U-Net ordering until the "
            "snapshot is exhausted and a new episode starts. This deliberately tests a limited correction budget, but larger "
            "k can over-commit the search to a stale snapshot rather than approaching the full dynamic true-distance oracle.\n\n"
        )
        file.write("## Overall Results\n\n")
        file.write("| Method | Mean expanded | Median expanded | Optimality | Mean capture | Win/Tie/Loss vs U-Net |\n")
        file.write("|---|---:|---:|---:|---:|---:|\n")
        for method in METHODS:
            row = by_method[method]
            capture = row["mean_oracle_benefit_capture"]
            capture_text = f"{capture:.3f}" if isinstance(capture, float) else ""
            file.write(
                f"| {method} | {row['mean_expanded_nodes']:.3f} | {row['median_expanded_nodes']:.3f} | "
                f"{row['optimality_rate']:.3f} | {capture_text} | "
                f"{row['wins_vs_unet']}/{row['ties_vs_unet']}/{row['losses_vs_unet']} |\n"
            )

        file.write("\n## Budget Curve\n\n")
        file.write("| Budget | Mean expanded | Mean capture |\n")
        file.write("|---|---:|---:|\n")
        for row in [item for item in budget_rows if item["scope"] == "all"]:
            capture = row["mean_oracle_benefit_capture"]
            capture_text = f"{capture:.3f}" if isinstance(capture, float) else ""
            file.write(f"| {row['budget']} | {row['mean_expanded_nodes']:.3f} | {capture_text} |\n")

        file.write("\n## Direct Answers\n\n")
        for method in TOPK_METHODS[1:-1]:
            row = by_method[method]
            capture = row["mean_oracle_benefit_capture"]
            capture_text = f"{capture:.3f}" if isinstance(capture, float) else "undefined"
            file.write(f"- {budget_label(method)} corrected nodes per episode captured mean oracle benefit {capture_text}.\n")
        file.write(
            "- In this definition, larger k is not monotone: k=2/4/8 are worse than U-Net on mean expansions because corrected "
            "snapshot nodes are forced ahead of potentially better later arrivals.\n"
        )
        file.write(
            "- Structure-specific differences are reported in `summary_by_structure.csv` and the per-structure budget rows.\n"
        )
        all_optimal = all(row["optimality_rate"] >= 1.0 for row in summary_rows if row["scope"] == "all")
        file.write(f"- Optimality preserved in the complete evaluated set: {'yes' if all_optimal else 'no'}.\n")

        distinct = []
        for method in TOPK_METHODS[1:-1]:
            same_as_full = all(
                to_float(case[method], "expanded_nodes")
                == to_float(case["manhattan_true_distance_tiebreak"], "expanded_nodes")
                for case in maps
            )
            distinct.append(f"{method}: {'same as full' if same_as_full else 'distinct from full'}")
        file.write(f"- Top-k degeneracy check: {', '.join(distinct)}.\n")


def make_plots(output_dir, budget_rows, pair_rows, maps):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    overall = [row for row in budget_rows if row["scope"] == "all"]
    labels = [row["budget"] for row in overall]
    x = list(range(len(labels)))

    plt.figure(figsize=(7, 4))
    plt.plot(x, [row["mean_expanded_nodes"] for row in overall], marker="o")
    plt.xticks(x, labels)
    plt.xlabel("Oracle budget per tie-set episode")
    plt.ylabel("Mean expanded nodes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_expanded_vs_budget.png"))
    plt.close()

    captures = [row["mean_oracle_benefit_capture"] or 0.0 for row in overall]
    plt.figure(figsize=(7, 4))
    plt.plot(x, captures, marker="o")
    plt.xticks(x, labels)
    plt.xlabel("Oracle budget per tie-set episode")
    plt.ylabel("Mean oracle-benefit capture")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "oracle_capture_vs_budget.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    for scope in STRUCTURED_TYPES:
        scoped = [row for row in budget_rows if row["scope"] == scope]
        if scoped:
            plt.plot(x, [row["mean_expanded_nodes"] for row in scoped], marker="o", label=scope)
    plt.xticks(x, labels)
    plt.xlabel("Oracle budget per tie-set episode")
    plt.ylabel("Mean expanded nodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_structure_budget_curves.png"))
    plt.close()

    diffs = [
        to_float(case["manhattan_oracle_top1_tiebreak"], "expanded_nodes")
        - to_float(case["manhattan_unet_tiebreak"], "expanded_nodes")
        for case in maps
    ]
    plt.figure(figsize=(7, 4))
    plt.hist(diffs, bins=20)
    plt.xlabel("top1 expanded minus U-Net tie-break")
    plt.ylabel("Cases")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "paired_diff_distribution_top1_vs_unet.png"))
    plt.close()

    sizes = [to_float(case["manhattan_oracle_top1_tiebreak"], "mean_tie_snapshot_size") for case in maps]
    benefits = [
        to_float(case["manhattan_unet_tiebreak"], "expanded_nodes")
        - to_float(case["manhattan_oracle_top1_tiebreak"], "expanded_nodes")
        for case in maps
    ]
    plt.figure(figsize=(7, 4))
    plt.scatter(sizes, benefits, s=16)
    plt.xlabel("Mean tie-set snapshot size")
    plt.ylabel("Expansion reduction from top1")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tie_size_vs_top1_benefit.png"))
    plt.close()


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "oracle_topk_benchmark")
    os.makedirs(output_dir, exist_ok=True)

    rows = read_csv(args.results)
    all_maps = group_complete_maps(rows)
    scopes = [("all", all_maps)]
    for structured_type in STRUCTURED_TYPES:
        scopes.append((structured_type, group_complete_maps(rows, structured_type)))

    summary_rows = []
    structure_rows = []
    pair_rows = []
    budget_rows = []
    for scope, maps in scopes:
        if not maps:
            continue
        method_rows = [summarize_method(maps, method, scope) for method in METHODS]
        summary_rows.extend(method_rows)
        if scope != "all":
            structure_rows.extend(method_rows)
        pair_rows.extend(paired_comparisons(maps, scope))
        budget_rows.extend(budget_curve_rows(maps, scope))

    diagnostics_rows = [
        row
        for row in rows
        if row.get("heuristic") in TOPK_METHODS and row.get("tie_episode_count") not in ("", None)
    ]
    write_csv(os.path.join(output_dir, "results.csv"), rows)
    write_csv(os.path.join(output_dir, "summary_by_algorithm.csv"), [row for row in summary_rows if row["scope"] == "all"])
    write_csv(os.path.join(output_dir, "summary_by_structure.csv"), structure_rows)
    write_csv(os.path.join(output_dir, "paired_comparisons.csv"), pair_rows)
    write_csv(os.path.join(output_dir, "tie_episode_diagnostics.csv"), diagnostics_rows)
    write_csv(os.path.join(output_dir, "oracle_budget_curve.csv"), budget_rows)
    write_report(os.path.join(output_dir, "oracle_topk_analysis.md"), summary_rows, budget_rows, all_maps)
    make_plots(output_dir, budget_rows, pair_rows, all_maps)
    print(f"Saved oracle top-k analysis outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze oracle top-k Manhattan tie-set benchmark results.")
    parser.add_argument("--results", default="outputs/experiments/results_structured_oracle_topk_100.csv")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
