"""Compare expanded U-Net as a primary heuristic and a Manhattan tie-break signal."""

import argparse
import csv
import heapq
import math
import os
import time
from collections import Counter, defaultdict

from analyze_critical_decisions import critical_events_for_case, optimal_path_nodes, true_distance_table
from analyze_unet_structure_behavior import (
    STRUCTURES,
    benchmark_cases,
    canonical_optimal_path,
    error_metrics,
    free_cells,
    mean,
    median,
    neighbors,
    rebuild_case,
)
from astar import astar_search
from bfs_label import compute_distance_to_goal
from model import load_unet_heuristic, make_unet_heuristic, manhattan_heuristic


ALGORITHMS = ["manhattan", "unet_tiebreak", "direct_unet"]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trace_search(grid, start, goal, unet_table, distance_grid, optimal_nodes, mode):
    """Reproduce astar.py priority and collect trace-only diagnostics."""
    true_table = true_distance_table(distance_grid)
    heap, g_score, parents, trace = [], {start: 0}, {}, []
    generated, maximum_open, counter = 1, 1, 0

    def primary(node):
        return float(manhattan_heuristic(node, goal)) if mode != "direct_unet" else unet_table[node]

    def entry(node, node_g, priority, order=0):
        return {
            "node": node, "g": node_g, "manhattan_h": float(manhattan_heuristic(node, goal)),
            "unet_h": unet_table[node], "true_distance": true_table.get(node, math.inf),
            "f": priority[0], "on_optimal_path": node in optimal_nodes,
            "priority": list(priority), "insertion_order": order,
        }

    def snapshot():
        result = []
        for item in heap:
            if mode == "unet_tiebreak":
                score, secondary, order, node_g, node = item
                priority = (score, secondary, order, node_g, node[0], node[1])
            else:
                score, node_g, node = item
                priority = (score, node_g, node[0], node[1])
                order = node[0] * len(grid[0]) + node[1]
            if node_g == g_score.get(node, math.inf):
                result.append(entry(node, node_g, priority, order))
        return sorted(result, key=lambda item: tuple(item["priority"]))

    if mode == "unet_tiebreak":
        heapq.heappush(heap, (manhattan_heuristic(start, goal), unet_table[start], counter, 0, start))
    else:
        heapq.heappush(heap, (primary(start), 0, start))

    while heap:
        open_before = snapshot()
        if mode == "unet_tiebreak":
            score, _, order, current_g, current = heapq.heappop(heap)
            priority = (score, unet_table[current], order, current_g, current[0], current[1])
        else:
            score, current_g, current = heapq.heappop(heap)
            priority = (score, current_g, current[0], current[1])
        if current_g > g_score.get(current, math.inf):
            continue
        trace.append({"step": len(trace), "expanded": entry(current, current_g, priority, priority[-3] if mode == "unet_tiebreak" else 0), "open_before": open_before})
        if current == goal:
            path = [current]
            while current in parents:
                current = parents[current]
                path.append(current)
            path.reverse()
            return {"path": path, "cost": len(path) - 1, "expanded": len(trace), "generated": generated, "max_open": maximum_open, "trace": trace}
        for neighbor in neighbors(grid, current):
            tentative = current_g + 1
            if tentative < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative
                parents[neighbor] = current
                generated += 1
                if mode == "unet_tiebreak":
                    counter += 1
                    heapq.heappush(heap, (tentative + manhattan_heuristic(neighbor, goal), unet_table[neighbor], counter, tentative, neighbor))
                else:
                    heapq.heappush(heap, (tentative + primary(neighbor), tentative, neighbor))
        maximum_open = max(maximum_open, len(snapshot()))
    return {"path": [], "cost": -1, "expanded": len(trace), "generated": generated, "max_open": maximum_open, "trace": trace}


def checked_search(grid, start, goal, table, mode, trace):
    heuristic = lambda node, unused_goal: table[node]
    started = time.perf_counter()
    if mode == "manhattan":
        heuristic = manhattan_heuristic
        result = astar_search(grid, start, goal, heuristic)
    elif mode == "unet_tiebreak":
        result = astar_search(grid, start, goal, manhattan_heuristic, secondary_heuristic=heuristic)
    else:
        result = astar_search(grid, start, goal, heuristic)
    runtime = time.perf_counter() - started
    if (result["expanded"], result["cost"], result["path"]) != (trace["expanded"], trace["cost"], trace["path"]):
        raise AssertionError(f"Trace mismatch for {mode}")
    return result, runtime


def route_metrics(trace, optimal_nodes):
    off_path = [item["expanded"] for item in trace if item["expanded"]["node"] not in optimal_nodes]
    return len(off_path), mean(
        min(abs(item["node"][0] - node[0]) + abs(item["node"][1] - node[1]) for node in optimal_nodes)
        for item in off_path
    )


def classify_failure(trace_result, events, table, distance_grid):
    if trace_result["cost"] < 0:
        return "other"
    path_overestimate = max((table[node] - distance_grid[node[0]][node[1]] for node in trace_result["path"]), default=0.0)
    if path_overestimate > 1e-6:
        return "unet_overestimation"
    if any(event["error_type"] == "A_unet_ordering_error" for event in events):
        return "unet_ordering_error"
    if trace_result["cost"] >= 0:
        return "wrong_global_route_preference"
    return "other"


def evaluate_case(row, model):
    grid, start, goal = rebuild_case(row)
    distance = compute_distance_to_goal(grid, goal)
    optimal_cost = distance[start[0]][start[1]]
    if optimal_cost != int(row["optimal_cost"]):
        raise AssertionError("Benchmark map reconstruction mismatch")
    from_start = compute_distance_to_goal(grid, start)
    optimal_nodes = optimal_path_nodes(from_start, distance, optimal_cost)
    unet_h = make_unet_heuristic(model, grid, goal)
    table = {cell: float(unet_h(cell, goal)) for cell in free_cells(grid)}
    metrics = error_metrics(table, distance)
    max_over = max(table[cell] - distance[cell[0]][cell[1]] for cell in table if distance[cell[0]][cell[1]] >= 0)
    violations = sum(table[cell] > distance[cell[0]][cell[1]] + 1e-6 for cell in table if distance[cell[0]][cell[1]] >= 0)
    traces, rows = {}, []
    case_id = f"{row['analysis_structure']}_rate{row['obstacle_rate']}_seed{row['seed']}_s{start[0]}-{start[1]}_g{goal[0]}-{goal[1]}"
    for algorithm in ALGORITHMS:
        trace = trace_search(grid, start, goal, table, distance, optimal_nodes, algorithm)
        checked_result, runtime = checked_search(grid, start, goal, table, algorithm, trace)
        if algorithm == "manhattan" and checked_result["expanded"] != int(row["expanded_nodes"]):
            raise AssertionError("Rebuilt Manhattan expansion count differs from the source benchmark")
        traces[algorithm] = trace
        off_path, route_distance = route_metrics(trace["trace"], optimal_nodes)
        rows.append({
            "case_id": case_id, "structure_type": row["analysis_structure"], "seed": row["seed"], "obstacle_rate": row["obstacle_rate"],
            "algorithm": algorithm, "expanded_nodes": trace["expanded"], "generated_nodes": trace["generated"], "max_open_size": trace["max_open"],
            "runtime_seconds": runtime, "path_cost": trace["cost"], "optimal_cost": optimal_cost,
            "optimal": trace["cost"] == optimal_cost, "cost_gap": trace["cost"] - optimal_cost if trace["cost"] >= 0 else math.inf,
            "mean_h_error": metrics["mae"], "max_overestimation": max_over, "admissibility_violation_count": violations if algorithm == "direct_unet" else "",
            "off_path_expansions": off_path, "mean_off_path_distance": route_distance,
        })
    direct_case = {"case_id": case_id, "structured_type": row["analysis_structure"], "optimal_cost": optimal_cost}
    events = critical_events_for_case(direct_case, traces["direct_unet"], optimal_nodes, weak_margin=0.5)
    failure = classify_failure(traces["direct_unet"], events, table, distance) if traces["direct_unet"]["cost"] != optimal_cost else ""
    return {"rows": rows, "grid": grid, "distance": distance, "path": canonical_optimal_path(grid, start, goal, distance), "table": table,
            "traces": traces, "events": events, "failure_type": failure}


def summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["structure_type"])].append(row)
    output = []
    for (algorithm, structure), values in sorted(grouped.items()):
        output.append({"scope": "structure", "algorithm": algorithm, "structure_type": structure, "cases": len(values),
                       "mean_expanded_nodes": mean(v["expanded_nodes"] for v in values), "median_expanded_nodes": median(v["expanded_nodes"] for v in values),
                       "mean_generated_nodes": mean(v["generated_nodes"] for v in values), "mean_max_open_size": mean(v["max_open_size"] for v in values),
                       "mean_runtime_seconds": mean(v["runtime_seconds"] for v in values), "mean_path_cost": mean(v["path_cost"] for v in values),
                       "optimality_rate": mean(float(v["optimal"]) for v in values), "mean_cost_gap": mean(v["cost_gap"] for v in values)})
    overall = defaultdict(list)
    for row in rows:
        overall[row["algorithm"]].append(row)
    for algorithm, values in sorted(overall.items()):
        output.append({"scope": "overall", "algorithm": algorithm, "structure_type": "all", "cases": len(values),
                       "mean_expanded_nodes": mean(v["expanded_nodes"] for v in values), "median_expanded_nodes": median(v["expanded_nodes"] for v in values),
                       "mean_generated_nodes": mean(v["generated_nodes"] for v in values), "mean_max_open_size": mean(v["max_open_size"] for v in values),
                       "mean_runtime_seconds": mean(v["runtime_seconds"] for v in values), "mean_path_cost": mean(v["path_cost"] for v in values),
                       "optimality_rate": mean(float(v["optimal"]) for v in values), "mean_cost_gap": mean(v["cost_gap"] for v in values)})
    return output


def draw_case(output_dir, label, record):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    grid, path = record["grid"], record["path"]
    direct = record["traces"]["direct_unet"]["trace"]
    tie = record["traces"]["unet_tiebreak"]["trace"]
    for axis, trace, title in ((axes[0], direct, "Direct expansion trajectory"), (axes[1], tie, "Tie-break expansion trajectory")):
        axis.imshow(grid, cmap="Greys", vmin=0, vmax=1)
        axis.plot([node[1] for node in path], [node[0] for node in path], color="#f2c94c", linewidth=2)
        axis.scatter([item["expanded"]["node"][1] for item in trace], [item["expanded"]["node"][0] for item in trace], s=5, c="#2b7bba", alpha=.55)
        axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
    axes[2].imshow([[record["table"].get((r, c), math.nan) if grid[r][c] == 0 else math.nan for c in range(len(grid[0]))] for r in range(len(grid))], cmap="viridis")
    axes[2].set_title("Expanded U-Net h")
    axes[3].imshow([[value if value >= 0 else math.nan for value in row] for row in record["distance"]], cmap="viridis")
    axes[3].set_title("True distance")
    for axis in axes[2:]: axis.set_xticks([]); axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "representative_cases", f"{label}.png"), dpi=160)
    plt.close(figure)


def run(args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output = os.path.join(root, args.output_dir)
    os.makedirs(os.path.join(output, "representative_cases"), exist_ok=True)
    cases = benchmark_cases(root, args.structured_results, args.random_results)
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    model = load_unet_heuristic(os.path.join(root, args.expanded_checkpoint))
    all_rows, records, failures = [], [], []
    for index, row in enumerate(cases, start=1):
        record = evaluate_case(row, model)
        all_rows.extend(record["rows"])
        records.append(record)
        direct = next(item for item in record["rows"] if item["algorithm"] == "direct_unet")
        if record["failure_type"]:
            failures.append({"case_id": direct["case_id"], "structure_type": direct["structure_type"], "failure_type": record["failure_type"], "path_cost": direct["path_cost"], "optimal_cost": direct["optimal_cost"], "cost_gap": direct["cost_gap"], "max_overestimation": direct["max_overestimation"], "critical_events": len(record["events"])})
        if index % 100 == 0: print(f"evaluated {index}/{len(cases)}")
    by_case = defaultdict(dict)
    for row in all_rows: by_case[row["case_id"]][row["algorithm"]] = row
    improved = [record for record in records if next(row for row in record["rows"] if row["algorithm"] == "direct_unet")["expanded_nodes"] < next(row for row in record["rows"] if row["algorithm"] == "unet_tiebreak")["expanded_nodes"]]
    degraded = [record for record in records if next(row for row in record["rows"] if row["algorithm"] == "direct_unet")["expanded_nodes"] > next(row for row in record["rows"] if row["algorithm"] == "unet_tiebreak")["expanded_nodes"]]
    nonoptimal = [record for record in records if next(row for row in record["rows"] if row["algorithm"] == "direct_unet")["optimal"] is False]
    if improved: draw_case(output, "direct_better_than_tiebreak", min(improved, key=lambda record: next(row for row in record["rows"] if row["algorithm"] == "direct_unet")["expanded_nodes"] - next(row for row in record["rows"] if row["algorithm"] == "unet_tiebreak")["expanded_nodes"]))
    if degraded: draw_case(output, "tiebreak_better_than_direct", max(degraded, key=lambda record: next(row for row in record["rows"] if row["algorithm"] == "direct_unet")["expanded_nodes"] - next(row for row in record["rows"] if row["algorithm"] == "unet_tiebreak")["expanded_nodes"]))
    if nonoptimal: draw_case(output, "direct_nonoptimal", max(nonoptimal, key=lambda record: next(row for row in record["rows"] if row["algorithm"] == "direct_unet")["cost_gap"]))
    summary = summaries(all_rows)
    write_csv(os.path.join(output, "results.csv"), all_rows)
    write_csv(os.path.join(output, "summary_by_algorithm.csv"), [row for row in summary if row["scope"] == "overall"])
    write_csv(os.path.join(output, "summary_by_structure.csv"), [row for row in summary if row["scope"] == "structure"])
    write_csv(os.path.join(output, "failure_case_analysis.csv"), failures or [{"case_id": "", "structure_type": "", "failure_type": "", "path_cost": "", "optimal_cost": "", "cost_gap": "", "max_overestimation": "", "critical_events": ""}])
    lookup = {(row["structure_type"], row["algorithm"]): row for row in summary if row["scope"] == "structure"}
    lines = ["# Direct U-Net vs U-Net Tie-break", "", "All searches use the expanded U-Net checkpoint and the same 2,000 existing benchmark cases. Direct uses `(g + h_unet, g, node)`; tie-break uses `(g + Manhattan h, h_unet, insertion_counter, g, node)`.", "", "## Structure Results", "", "| Structure | Direct mean expanded | Tie-break mean expanded | Direct optimality | Tie-break optimality |", "|---|---:|---:|---:|---:|"]
    available_structures = [structure for structure in STRUCTURES if (structure, "direct_unet") in lookup]
    for structure in available_structures:
        direct, tie = lookup[(structure, "direct_unet")], lookup[(structure, "unet_tiebreak")]
        lines.append(f"| {structure} | {direct['mean_expanded_nodes']:.2f} | {tie['mean_expanded_nodes']:.2f} | {direct['optimality_rate']:.3f} | {tie['optimality_rate']:.3f} |")
    direct_all, tie_all = next(row for row in summary if row["scope"] == "overall" and row["algorithm"] == "direct_unet"), next(row for row in summary if row["scope"] == "overall" and row["algorithm"] == "unet_tiebreak")
    failures_by_type = Counter(row["failure_type"] for row in failures)
    lines += ["", "## Mechanism Observations", "", f"- Direct is lower-expansion than tie-break on {len(improved)}/{len(records)} cases and higher-expansion on {len(degraded)}/{len(records)} cases.", f"- Direct off-path expansions are lower than tie-break only where its learned primary ordering avoids nodes that Manhattan primary-f would otherwise expand; this is behavioral evidence, not a direct causal intervention.", f"- Direct has {len(failures)} non-optimal paths. Failure labels: {dict(failures_by_type)}.", f"- Overall mean expansions: Direct {direct_all['mean_expanded_nodes']:.2f}, tie-break {tie_all['mean_expanded_nodes']:.2f}; optimality: Direct {direct_all['optimality_rate']:.3f}, tie-break {tie_all['optimality_rate']:.3f}.", "", "## Limits", "", "The failure labels are diagnostic categories based on observed path error, U-Net overestimation, and critical-decision traces. They do not establish causality or make Direct U-Net a final algorithm.", ""]
    with open(os.path.join(output, "mechanism_analysis.md"), "w", encoding="utf-8") as handle: handle.write("\n".join(lines))
    print(f"completed {len(cases)} cases in {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare direct expanded U-Net A* and U-Net tie-break A*.")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_tiebreak_controls_100.csv")
    parser.add_argument("--expanded-checkpoint", default="outputs/expanded_dataset_training/unet_heuristic_expanded_best.pt")
    parser.add_argument("--output-dir", default="outputs/direct_vs_tiebreak_analysis")
    parser.add_argument("--max-cases", type=int, default=None, help="Smoke-test limit; omit for the full benchmark.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
