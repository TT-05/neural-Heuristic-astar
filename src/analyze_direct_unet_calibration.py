"""Diagnostic experiment: constant calibration of direct expanded U-Net A*."""

import argparse
import csv
import math
import os
import time
from collections import defaultdict

from analyze_direct_vs_tiebreak import checked_search, trace_search
from analyze_unet_structure_behavior import (
    STRUCTURES, benchmark_cases, canonical_optimal_path, error_metrics, free_cells,
    mean, median, rebuild_case,
)
from bfs_label import compute_distance_to_goal
from model import load_unet_heuristic, make_unet_heuristic


CALIBRATIONS = {
    "direct_unet": None,
    "calibrated_n1": 1.0,
    "calibrated_n5": 5.0,
    "calibrated_n10": 10.0,
    "calibrated_n20": 20.0,
    "oracle_calibrated": "oracle",
}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison(raw_trace, calibrated_trace):
    raw = [item["expanded"]["node"] for item in raw_trace["trace"]]
    calibrated = [item["expanded"]["node"] for item in calibrated_trace["trace"]]
    shared = min(len(raw), len(calibrated))
    identical = sum(a == b for a, b in zip(raw[:shared], calibrated[:shared]))
    divergence = next((index for index, (a, b) in enumerate(zip(raw, calibrated)) if a != b), None)
    changed = sum(a != b for a, b in zip(raw[:shared], calibrated[:shared])) + abs(len(raw) - len(calibrated))
    return {
        "identical_expansion_fraction": identical / max(len(raw), len(calibrated)) if max(len(raw), len(calibrated)) else 1.0,
        "first_divergence_step": divergence if divergence is not None else "",
        "changed_node_ordering_count": changed,
    }


def case_id(row, start, goal):
    return f"{row['analysis_structure']}_rate{row['obstacle_rate']}_seed{row['seed']}_s{start[0]}-{start[1]}_g{goal[0]}-{goal[1]}"


def evaluate_case(row, model):
    grid, start, goal = rebuild_case(row)
    distance = compute_distance_to_goal(grid, goal)
    optimal_cost = distance[start[0]][start[1]]
    if optimal_cost != int(row["optimal_cost"]):
        raise AssertionError("Benchmark map reconstruction mismatch")
    from analyze_critical_decisions import optimal_path_nodes
    from_start = compute_distance_to_goal(grid, start)
    optimal_nodes = optimal_path_nodes(from_start, distance, optimal_cost)
    unet_h = make_unet_heuristic(model, grid, goal)
    raw_table = {cell: float(unet_h(cell, goal)) for cell in free_cells(grid)}
    raw_max_over = max(raw_table[cell] - distance[cell[0]][cell[1]] for cell in raw_table if distance[cell[0]][cell[1]] >= 0)
    oracle_n = max(raw_max_over, 0.0)
    raw_metrics = error_metrics(raw_table, distance)
    traces, tables, results = {}, {}, []
    current_case_id = case_id(row, start, goal)
    manhattan_table = {cell: float(abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])) for cell in raw_table}
    manhattan_metrics = error_metrics(manhattan_table, distance)
    manhattan_trace = trace_search(grid, start, goal, manhattan_table, distance, optimal_nodes, "direct_unet")
    checked_result, manhattan_runtime = checked_search(grid, start, goal, manhattan_table, "manhattan", manhattan_trace)
    if checked_result["expanded"] != int(row["expanded_nodes"]):
        raise AssertionError("Rebuilt Manhattan expansion count differs from the source benchmark")
    results.append({
        "case_id": current_case_id, "structure_type": row["analysis_structure"], "seed": row["seed"], "obstacle_rate": row["obstacle_rate"],
        "algorithm": "manhattan", "calibration_n": "", "expanded_nodes": manhattan_trace["expanded"], "generated_nodes": manhattan_trace["generated"],
        "max_open_size": manhattan_trace["max_open"], "runtime_seconds": manhattan_runtime, "path_cost": manhattan_trace["cost"], "optimal_cost": optimal_cost,
        "optimal": True, "cost_gap": 0, "mean_h_error": manhattan_metrics["mae"], "raw_mean_h_error": raw_metrics["mae"],
        "max_overestimation": 0.0, "raw_max_overestimation": raw_max_over, "admissibility_violation_count": 0,
    })
    for algorithm, calibration in CALIBRATIONS.items():
        n = oracle_n if calibration == "oracle" else (calibration or 0.0)
        table = {cell: value - n for cell, value in raw_table.items()}
        trace = trace_search(grid, start, goal, table, distance, optimal_nodes, "direct_unet")
        _, runtime = checked_search(grid, start, goal, table, "direct_unet", trace)
        metrics = error_metrics(table, distance)
        max_over = max(table[cell] - distance[cell[0]][cell[1]] for cell in table if distance[cell[0]][cell[1]] >= 0)
        violations = sum(table[cell] > distance[cell[0]][cell[1]] + 1e-6 for cell in table if distance[cell[0]][cell[1]] >= 0)
        results.append({
            "case_id": current_case_id, "structure_type": row["analysis_structure"], "seed": row["seed"], "obstacle_rate": row["obstacle_rate"],
            "algorithm": algorithm, "calibration_n": n, "expanded_nodes": trace["expanded"], "generated_nodes": trace["generated"],
            "max_open_size": trace["max_open"], "runtime_seconds": runtime, "path_cost": trace["cost"], "optimal_cost": optimal_cost,
            "optimal": trace["cost"] == optimal_cost, "cost_gap": trace["cost"] - optimal_cost,
            "mean_h_error": metrics["mae"], "raw_mean_h_error": raw_metrics["mae"], "max_overestimation": max_over,
            "raw_max_overestimation": raw_max_over, "admissibility_violation_count": violations,
        })
        traces[algorithm], tables[algorithm] = trace, table
    ordering = []
    for algorithm in CALIBRATIONS:
        if algorithm == "direct_unet":
            continue
        ordering.append({"case_id": current_case_id, "structure_type": row["analysis_structure"], "algorithm": algorithm,
                         "calibration_n": next(item["calibration_n"] for item in results if item["algorithm"] == algorithm),
                         **comparison(traces["direct_unet"], traces[algorithm])})
    return {"results": results, "ordering": ordering, "grid": grid, "distance": distance,
            "path": canonical_optimal_path(grid, start, goal, distance), "tables": tables, "traces": traces}


def aggregate(results, fields):
    grouped = defaultdict(list)
    for row in results:
        grouped[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        def numeric(field):
            return mean(float(v[field]) for v in values if v[field] != "")
        row = {field: value for field, value in zip(fields, key)}
        row.update({
            "cases": len(values), "mean_expanded_nodes": mean(v["expanded_nodes"] for v in values),
            "median_expanded_nodes": median(v["expanded_nodes"] for v in values), "mean_generated_nodes": mean(v["generated_nodes"] for v in values),
            "mean_max_open_size": mean(v["max_open_size"] for v in values), "mean_runtime_seconds": mean(v["runtime_seconds"] for v in values),
            "mean_path_cost": mean(v["path_cost"] for v in values), "optimality_rate": mean(float(v["optimal"]) for v in values),
            "mean_cost_gap": numeric("cost_gap"), "mean_max_overestimation": numeric("max_overestimation"),
            "mean_h_error": numeric("mean_h_error"), "mean_admissibility_violations": numeric("admissibility_violation_count"),
        })
        output.append(row)
    return output


def draw_case(output_dir, label, record, algorithm):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 5, figsize=(20, 4))
    grid, path = record["grid"], record["path"]
    raw_trace, trace = record["traces"]["direct_unet"], record["traces"][algorithm]
    for axis, current, title in ((axes[0], raw_trace, "Direct U-Net trajectory"), (axes[1], trace, f"{algorithm} trajectory")):
        axis.imshow(grid, cmap="Greys", vmin=0, vmax=1)
        axis.plot([node[1] for node in path], [node[0] for node in path], color="#f2c94c", linewidth=2)
        axis.scatter([item["expanded"]["node"][1] for item in current["trace"]], [item["expanded"]["node"][0] for item in current["trace"]], s=5, c="#2b7bba", alpha=.55)
        axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
    raw_table, calibrated = record["tables"]["direct_unet"], record["tables"][algorithm]
    for axis, table, title in ((axes[2], raw_table, "Raw U-Net h"), (axes[3], calibrated, "Calibrated h")):
        axis.imshow([[table.get((r, c), math.nan) if grid[r][c] == 0 else math.nan for c in range(len(grid[0]))] for r in range(len(grid))], cmap="viridis")
        axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
    first_open = trace["trace"][0]["open_before"][:8]
    text = "OPEN at step 0\n" + "\n".join(f"{item['node']}: {item['priority']}" for item in first_open)
    axes[4].axis("off"); axes[4].text(.02, .98, text, va="top", family="monospace", fontsize=8); axes[4].set_title("Recorded OPEN priority")
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "representative_cases", f"{label}.png"), dpi=160)
    plt.close(figure)


def draw_no_fix_message(output_dir):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 2.5))
    axis.axis("off")
    axis.text(.5, .58, "No constant calibration fixed a non-optimal Direct U-Net path.", ha="center", va="center", fontsize=13)
    axis.text(.5, .32, "All fixed-n and oracle-global calibrations preserved the raw Direct expansion order.", ha="center", va="center", fontsize=10)
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "representative_cases", "calibration_no_optimality_fix.png"), dpi=160)
    plt.close(figure)


def run(args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output = os.path.join(root, args.output_dir)
    os.makedirs(os.path.join(output, "representative_cases"), exist_ok=True)
    for filename in ("calibration_claimed_fix.png", "calibration_no_optimality_fix.png", "direct_unet_nonoptimal.png", "constant_shift_no_behavior_change.png"):
        path = os.path.join(output, "representative_cases", filename)
        if os.path.exists(path):
            os.remove(path)
    cases = benchmark_cases(root, args.structured_results, args.random_results)
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    model = load_unet_heuristic(os.path.join(root, args.expanded_checkpoint))
    all_results, all_ordering, records = [], [], []
    for index, row in enumerate(cases, start=1):
        record = evaluate_case(row, model)
        all_results.extend(record["results"]); all_ordering.extend(record["ordering"]); records.append(record)
        if index % 100 == 0: print(f"evaluated {index}/{len(cases)}")
    summaries = aggregate(all_results, ["algorithm"])
    structure_summaries = aggregate(all_results, ["structure_type", "algorithm"])
    write_csv(os.path.join(output, "results.csv"), all_results)
    write_csv(os.path.join(output, "summary_by_algorithm.csv"), summaries)
    write_csv(os.path.join(output, "summary_by_structure.csv"), structure_summaries)
    write_csv(os.path.join(output, "ordering_change_analysis.csv"), all_ordering)
    failed = [record for record in records if next(row for row in record["results"] if row["algorithm"] == "direct_unet")["optimal"] is False]
    fixed = [record for record in failed if any(row["optimal"] for row in record["results"] if row["algorithm"] in CALIBRATIONS and row["algorithm"] != "direct_unet")]
    unchanged = [record for record in records if all(
        row["expanded_nodes"] == next(item["expanded_nodes"] for item in record["results"] if item["algorithm"] == "direct_unet")
        for row in record["results"] if row["algorithm"] in CALIBRATIONS
    )]
    if failed: draw_case(output, "direct_unet_nonoptimal", failed[0], "oracle_calibrated")
    if fixed: draw_case(output, "calibration_claimed_fix", fixed[0], "oracle_calibrated")
    elif failed: draw_no_fix_message(output)
    if unchanged: draw_case(output, "constant_shift_no_behavior_change", unchanged[0], "calibrated_n20")
    summary_lookup = {row["algorithm"]: row for row in summaries}
    ordering_by_algorithm = defaultdict(list)
    for row in all_ordering: ordering_by_algorithm[row["algorithm"]].append(row)
    lines = ["# Calibrated Direct U-Net A*", "", "## Overall", "", "| Algorithm | Mean expanded | Optimality | Mean cost gap | Mean max overestimation | Mean admissibility violations |", "|---|---:|---:|---:|---:|---:|"]
    for algorithm in ("manhattan", *CALIBRATIONS):
        row = summary_lookup[algorithm]
        lines.append(f"| {algorithm} | {row['mean_expanded_nodes']:.2f} | {row['optimality_rate']:.3f} | {row['mean_cost_gap']:.3f} | {row['mean_max_overestimation']:.3f} | {row['mean_admissibility_violations']:.2f} |")
    lines += ["", "## Ordering Change", ""]
    for algorithm, values in sorted(ordering_by_algorithm.items()):
        lines.append(f"- {algorithm}: identical expansion fraction {mean(v['identical_expansion_fraction'] for v in values):.6f}; changed-ordering count {mean(v['changed_node_ordering_count'] for v in values):.6f}; divergences {sum(v['first_divergence_step'] != '' for v in values)}/{len(values)}.")
    lines += ["", f"- Direct non-optimal cases: {len(failed)}; cases made optimal by any constant calibration: {len(fixed)}.", "", "## Interpretation", "", "Subtracting a constant `n` changes every Direct-A* f value by the same amount: `(g + h_unet - n)`. It therefore preserves the priority ordering exactly in exact arithmetic. An oracle global constant can remove admissibility violations but cannot, by itself, change the expansion sequence, selected path, or recover lost optimality. Any observed numerical divergence would be a floating-point artifact and is reported explicitly in `ordering_change_analysis.csv`.", "", "This is a diagnostic experiment, not a deployable calibration method.", ""]
    with open(os.path.join(output, "calibration_effect_analysis.md"), "w", encoding="utf-8") as handle: handle.write("\n".join(lines))
    print(f"completed {len(cases)} cases in {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate constant-calibrated direct expanded U-Net A*.")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_tiebreak_controls_100.csv")
    parser.add_argument("--expanded-checkpoint", default="outputs/expanded_dataset_training/unet_heuristic_expanded_best.pt")
    parser.add_argument("--output-dir", default="outputs/direct_unet_calibration_analysis")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
