"""Evaluate a consistency-projected expanded U-Net as a Direct A* heuristic."""

import argparse
import csv
import heapq
import math
import os
import time
from collections import defaultdict

from analyze_direct_vs_tiebreak import checked_search, trace_search
from analyze_unet_structure_behavior import (
    STRUCTURES, benchmark_cases, canonical_optimal_path, error_metrics, free_cells,
    mean, median, neighbors, rebuild_case,
)
from bfs_label import compute_distance_to_goal
from model import load_unet_heuristic, make_unet_heuristic


ALGORITHMS = ["manhattan", "unet_tiebreak", "direct_unet", "calibrated_n5", "calibrated_n10", "calibrated_n20", "projected_unet"]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def project_consistent_lower_envelope(grid, goal, raw_table):
    """Greatest 1-Lipschitz minorant of raw_table, anchored at h(goal)=0.

    Multi-source Dijkstra computes min_y(raw_h(y) + dist(x, y)); all sources are
    raw h values and the goal has source value zero. It can only lower h values.
    """
    projected = dict(raw_table)
    projected[goal] = 0.0
    heap = [(value, node) for node, value in projected.items()]
    heapq.heapify(heap)
    while heap:
        value, node = heapq.heappop(heap)
        if value != projected[node]:
            continue
        for neighbor in neighbors(grid, node):
            candidate = value + 1.0
            if candidate < projected[neighbor]:
                projected[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return projected


def consistency_violations(grid, table):
    count = 0
    maximum = 0.0
    for node, value in table.items():
        for neighbor in neighbors(grid, node):
            if node >= neighbor:
                continue
            excess = abs(value - table[neighbor]) - 1.0
            if excess > 1e-6:
                count += 1
                maximum = max(maximum, excess)
    return count, maximum


def table_metrics(grid, table, distance):
    metrics = error_metrics(table, distance)
    reachable = [cell for cell in table if distance[cell[0]][cell[1]] >= 0]
    max_over = max(table[cell] - distance[cell[0]][cell[1]] for cell in reachable)
    admissibility = sum(table[cell] > distance[cell[0]][cell[1]] + 1e-6 for cell in reachable)
    violations, max_excess = consistency_violations(grid, table)
    return {**metrics, "max_overestimation": max_over, "admissibility_violations": admissibility,
            "consistency_violations": violations, "max_consistency_excess": max_excess}


def sequence_comparison(raw_trace, projected_trace):
    raw = [item["expanded"]["node"] for item in raw_trace["trace"]]
    projected = [item["expanded"]["node"] for item in projected_trace["trace"]]
    shared = min(len(raw), len(projected))
    same = sum(a == b for a, b in zip(raw[:shared], projected[:shared]))
    divergence = next((index for index, (a, b) in enumerate(zip(raw, projected)) if a != b), None)
    changed = sum(a != b for a, b in zip(raw[:shared], projected[:shared])) + abs(len(raw) - len(projected))
    return {"identical_expansion_fraction": same / max(len(raw), len(projected)),
            "first_divergence_step": divergence if divergence is not None else "", "changed_node_ordering_count": changed}


def case_id(row, start, goal):
    return f"{row['analysis_structure']}_rate{row['obstacle_rate']}_seed{row['seed']}_s{start[0]}-{start[1]}_g{goal[0]}-{goal[1]}"


def evaluate_case(row, model):
    grid, start, goal = rebuild_case(row)
    distance = compute_distance_to_goal(grid, goal)
    optimal_cost = distance[start[0]][start[1]]
    if optimal_cost != int(row["optimal_cost"]):
        raise AssertionError("Benchmark map reconstruction mismatch")
    from analyze_critical_decisions import optimal_path_nodes
    optimal_nodes = optimal_path_nodes(compute_distance_to_goal(grid, start), distance, optimal_cost)
    raw_h = make_unet_heuristic(model, grid, goal)
    raw_table = {cell: float(raw_h(cell, goal)) for cell in free_cells(grid)}
    projected_table = project_consistent_lower_envelope(grid, goal, raw_table)
    raw_metrics, projected_metrics = table_metrics(grid, raw_table, distance), table_metrics(grid, projected_table, distance)
    modified = [raw_table[cell] - projected_table[cell] for cell in raw_table]
    projection = {"modified_cells": sum(delta > 1e-6 for delta in modified), "mean_modification": mean(modified),
                  "max_modification": max(modified), **{f"raw_{key}": value for key, value in raw_metrics.items()},
                  **{f"projected_{key}": value for key, value in projected_metrics.items()}}
    tables = {
        "direct_unet": raw_table,
        "calibrated_n5": {cell: value - 5.0 for cell, value in raw_table.items()},
        "calibrated_n10": {cell: value - 10.0 for cell, value in raw_table.items()},
        "calibrated_n20": {cell: value - 20.0 for cell, value in raw_table.items()},
        "projected_unet": projected_table,
        "manhattan": {cell: float(abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])) for cell in raw_table},
    }
    current_case_id = case_id(row, start, goal)
    traces, result_rows = {}, []
    for algorithm in ALGORITHMS:
        table = tables.get(algorithm, raw_table)
        mode = "unet_tiebreak" if algorithm == "unet_tiebreak" else ("manhattan" if algorithm == "manhattan" else "direct_unet")
        trace = trace_search(grid, start, goal, table, distance, optimal_nodes, mode)
        checked, runtime = checked_search(grid, start, goal, table, mode, trace)
        if algorithm == "manhattan" and checked["expanded"] != int(row["expanded_nodes"]):
            raise AssertionError("Rebuilt Manhattan count differs from source benchmark")
        metrics = projected_metrics if algorithm == "projected_unet" else raw_metrics
        result_rows.append({
            "case_id": current_case_id, "structure_type": row["analysis_structure"], "seed": row["seed"], "obstacle_rate": row["obstacle_rate"],
            "algorithm": algorithm, "expanded_nodes": trace["expanded"], "generated_nodes": trace["generated"], "max_open_size": trace["max_open"],
            "runtime_seconds": runtime, "path_cost": trace["cost"], "optimal_cost": optimal_cost, "optimal": trace["cost"] == optimal_cost,
            "cost_gap": trace["cost"] - optimal_cost, "mean_h_error": metrics["mae"], "max_overestimation": metrics["max_overestimation"],
            "admissibility_violations": metrics["admissibility_violations"], "consistency_violations": metrics["consistency_violations"],
        })
        traces[algorithm] = trace
    return {"results": result_rows, "projection": {"case_id": current_case_id, "structure_type": row["analysis_structure"], **projection},
            "comparison": {"case_id": current_case_id, "structure_type": row["analysis_structure"], **sequence_comparison(traces["direct_unet"], traces["projected_unet"])},
            "grid": grid, "distance": distance, "path": canonical_optimal_path(grid, start, goal, distance), "raw_table": raw_table,
            "projected_table": projected_table, "traces": traces}


def aggregate(rows, fields):
    groups = defaultdict(list)
    for row in rows: groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, values in sorted(groups.items()):
        row = {field: value for field, value in zip(fields, key)}
        row.update({"cases": len(values), "mean_expanded_nodes": mean(v["expanded_nodes"] for v in values),
                    "median_expanded_nodes": median(v["expanded_nodes"] for v in values), "mean_generated_nodes": mean(v["generated_nodes"] for v in values),
                    "mean_max_open_size": mean(v["max_open_size"] for v in values), "mean_runtime_seconds": mean(v["runtime_seconds"] for v in values),
                    "mean_path_cost": mean(v["path_cost"] for v in values), "optimality_rate": mean(float(v["optimal"]) for v in values),
                    "mean_cost_gap": mean(v["cost_gap"] for v in values)})
        output.append(row)
    return output


def draw_case(output_dir, label, record):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    grid, path = record["grid"], record["path"]
    for axis, algorithm, title in ((axes[0], "direct_unet", "Raw Direct trajectory"), (axes[1], "projected_unet", "Projected trajectory")):
        axis.imshow(grid, cmap="Greys", vmin=0, vmax=1)
        axis.plot([node[1] for node in path], [node[0] for node in path], color="#f2c94c", linewidth=2)
        trace = record["traces"][algorithm]["trace"]
        axis.scatter([item["expanded"]["node"][1] for item in trace], [item["expanded"]["node"][0] for item in trace], s=5, c="#2b7bba", alpha=.55)
        axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
    for axis, table, title in ((axes[2], record["raw_table"], "Raw U-Net h"), (axes[3], record["projected_table"], "Projected h")):
        axis.imshow([[table.get((r, c), math.nan) if grid[r][c] == 0 else math.nan for c in range(len(grid[0]))] for r in range(len(grid))], cmap="viridis")
        axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
    axes[4].imshow([[value if value >= 0 else math.nan for value in row] for row in record["distance"]], cmap="viridis")
    axes[4].set_title("True distance"); axes[4].set_xticks([]); axes[4].set_yticks([])
    snapshots = record["traces"]["projected_unet"]["trace"][:3]
    open_text = []
    for snapshot in snapshots:
        top = snapshot["open_before"][:3]
        open_text.append(f"step {snapshot['step']}: " + "; ".join(f"{item['node']} f={item['f']:.1f}" for item in top))
    axes[5].axis("off"); axes[5].text(.02, .98, "Projected OPEN evolution\n\n" + "\n\n".join(open_text), va="top", family="monospace", fontsize=8)
    axes[5].set_title("Recorded OPEN evolution")
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, "representative_cases", f"{label}.png"), dpi=160); plt.close(fig)


def run(args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output = os.path.join(root, args.output_dir)
    os.makedirs(os.path.join(output, "representative_cases"), exist_ok=True)
    cases = benchmark_cases(root, args.structured_results, args.random_results)
    if args.max_cases is not None: cases = cases[:args.max_cases]
    model = load_unet_heuristic(os.path.join(root, args.expanded_checkpoint))
    all_rows, projections, comparisons, records = [], [], [], []
    for index, row in enumerate(cases, start=1):
        record = evaluate_case(row, model)
        all_rows.extend(record["results"]); projections.append(record["projection"]); comparisons.append(record["comparison"]); records.append(record)
        if index % 100 == 0: print(f"evaluated {index}/{len(cases)}")
    summaries = aggregate(all_rows, ["algorithm"])
    structures = aggregate(all_rows, ["structure_type", "algorithm"])
    write_csv(os.path.join(output, "results.csv"), all_rows)
    write_csv(os.path.join(output, "summary_by_algorithm.csv"), summaries)
    write_csv(os.path.join(output, "summary_by_structure.csv"), structures)
    write_csv(os.path.join(output, "heuristic_projection_statistics.csv"), projections)
    write_csv(os.path.join(output, "consistency_analysis.csv"), comparisons)
    by_case = defaultdict(dict)
    for row in all_rows: by_case[row["case_id"]][row["algorithm"]] = row
    direct_failed_projected_optimal = [record for record in records if (lambda values: not values["direct_unet"]["optimal"] and values["projected_unet"]["optimal"])(by_case[record["results"][0]["case_id"]])]
    improved = [record for record in records if (lambda values: values["projected_unet"]["expanded_nodes"] < values["direct_unet"]["expanded_nodes"])(by_case[record["results"][0]["case_id"]])]
    hurt = [record for record in records if (lambda values: values["projected_unet"]["expanded_nodes"] > values["direct_unet"]["expanded_nodes"])(by_case[record["results"][0]["case_id"]])]
    if direct_failed_projected_optimal: draw_case(output, "direct_nonoptimal_projected_optimal", direct_failed_projected_optimal[0])
    if improved: draw_case(output, "projection_improves_expansion", max(improved, key=lambda record: by_case[record["results"][0]["case_id"]]["direct_unet"]["expanded_nodes"] - by_case[record["results"][0]["case_id"]]["projected_unet"]["expanded_nodes"]))
    if hurt: draw_case(output, "projection_hurts_efficiency", max(hurt, key=lambda record: by_case[record["results"][0]["case_id"]]["projected_unet"]["expanded_nodes"] - by_case[record["results"][0]["case_id"]]["direct_unet"]["expanded_nodes"]))
    lookup = {row["algorithm"]: row for row in summaries}
    lines = ["# Consistency-projected Direct U-Net A*", "", "The projection is a multi-source shortest-path relaxation that computes the largest 1-Lipschitz heuristic no greater than the raw U-Net output, with h(goal)=0.", "", "## Overall", "", "| Algorithm | Mean expanded | Optimality | Mean cost gap |", "|---|---:|---:|---:|"]
    for algorithm in ALGORITHMS:
        row = lookup[algorithm]; lines.append(f"| {algorithm} | {row['mean_expanded_nodes']:.2f} | {row['optimality_rate']:.3f} | {row['mean_cost_gap']:.3f} |")
    lines += ["", "## Projection and Trajectory", "", f"- Raw-to-projected mean modified cells: {mean(row['modified_cells'] for row in projections):.2f}; mean decrease: {mean(row['mean_modification'] for row in projections):.3f}.", f"- Projected consistency violations: mean {mean(row['projected_consistency_violations'] for row in projections):.3f}; projected admissibility violations: mean {mean(row['projected_admissibility_violations'] for row in projections):.3f}.", f"- Direct/projected expansion sequences diverged on {sum(row['first_divergence_step'] != '' for row in comparisons)}/{len(comparisons)} cases.", f"- Direct non-optimal but projected optimal cases: {len(direct_failed_projected_optimal)}.", "", "## Limits", "", "This evaluates one lower-envelope consistency projection. It is an offline diagnostic, not a final Neural A* algorithm or a proof that consistency projection is generally optimal for learned heuristics.", ""]
    with open(os.path.join(output, "report.md"), "w", encoding="utf-8") as handle: handle.write("\n".join(lines))
    print(f"completed {len(cases)} cases in {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate consistency-projected Direct expanded U-Net A*.")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_tiebreak_controls_100.csv")
    parser.add_argument("--expanded-checkpoint", default="outputs/expanded_dataset_training/unet_heuristic_expanded_best.pt")
    parser.add_argument("--output-dir", default="outputs/projected_unet_astar_analysis")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
