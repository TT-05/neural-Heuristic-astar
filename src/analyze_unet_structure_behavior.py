"""Offline structural behavior comparison for old and expanded U-Net A*."""

import argparse
import csv
import heapq
import math
import os
import time
from collections import Counter, defaultdict

from analyze_critical_decisions import (
    canonical_optimal_path,
    critical_events_for_case,
    make_table,
    optimal_path_nodes,
    true_distance_table,
)
from astar import astar_search
from bfs_label import compute_distance_to_goal
from gen_map import gen_map
from model import load_unet_heuristic, make_unet_heuristic, manhattan_heuristic
from structured_maps import generate_structured_map


STRUCTURES = ["maze_like", "bottleneck", "large_block", "narrow_corridor", "open_random"]
ALGORITHMS = [
    "manhattan",
    "old_unet_tiebreak",
    "expanded_unet_tiebreak",
    "old_direct_unet",
    "expanded_direct_unet",
]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def median(values):
    values = sorted(values)
    if not values:
        return 0.0
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def pearson(xs, ys):
    if len(xs) < 2:
        return 0.0
    x_mean, y_mean = mean(xs), mean(ys)
    top = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    bottom = math.sqrt(sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys))
    return top / bottom if bottom else 0.0


def ranks(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for original, _ in ordered[index:end]:
            output[original] = rank
        index = end
    return output


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def case_key(row):
    return tuple(row[key] for key in ("seed", "map_size", "obstacle_rate", "map_mode", "structured_type", "start_row", "start_col", "goal_row", "goal_col"))


def benchmark_cases(project_root, structured_csv, random_csv):
    cases = {}
    for path in (structured_csv, random_csv):
        for row in read_rows(os.path.join(project_root, path)):
            if row["heuristic"] != "manhattan" or row["path_found"] != "True":
                continue
            structure = row["structured_type"] if row["map_mode"] == "structured" else "open_random"
            row = dict(row)
            row["analysis_structure"] = structure
            cases[case_key(row)] = row
    return list(cases.values())


def rebuild_case(row):
    size, seed, rate = int(row["map_size"]), int(row["seed"]), float(row["obstacle_rate"])
    if row["map_mode"] == "structured":
        grid = generate_structured_map(size, size, seed, rate, row["structured_type"])
    else:
        grid = gen_map(size, size, seed=seed, obstacle_rate=rate)
    start = (int(row["start_row"]), int(row["start_col"]))
    goal = (int(row["goal_row"]), int(row["goal_col"]))
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0
    return grid, start, goal


def free_cells(grid):
    return [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 0]


def neighbors(grid, node):
    r, c = node
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 0:
            yield nr, nc


def direct_trace(grid, start, goal, table, distance_grid, optimal_nodes):
    """Trace the same `(g + h_unet, g, node)` behavior used by astar_search."""
    true_table = true_distance_table(distance_grid)
    heap, g_score, came_from, trace = [], {start: 0}, {}, []
    heapq.heappush(heap, (table[start], 0, start))

    def open_snapshot():
        entries = []
        for f_score, node_g, node in heap:
            if node_g != g_score.get(node, math.inf):
                continue
            entries.append({
                "node": node, "g": node_g, "manhattan_h": float(manhattan_heuristic(node, goal)),
                "unet_h": table[node], "true_distance": true_table.get(node, math.inf), "f": f_score,
                "on_optimal_path": node in optimal_nodes, "priority": [f_score, node_g, node[0], node[1]],
                "insertion_order": node[0] * len(grid[0]) + node[1],
            })
        return sorted(entries, key=lambda entry: tuple(entry["priority"]))

    while heap:
        open_before = open_snapshot()
        f_score, current_g, current = heapq.heappop(heap)
        if current_g > g_score.get(current, math.inf):
            continue
        trace.append({"step": len(trace), "expanded": {
            "node": current, "g": current_g, "manhattan_h": float(manhattan_heuristic(current, goal)),
            "unet_h": table[current], "true_distance": true_table.get(current, math.inf), "f": f_score,
            "on_optimal_path": current in optimal_nodes,
        }, "open_before": open_before})
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return {"path": path, "cost": len(path) - 1, "expanded": len(trace), "trace": trace}
        for neighbor in neighbors(grid, current):
            tentative = current_g + 1
            if tentative < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                heapq.heappush(heap, (tentative + table[neighbor], tentative, neighbor))
    return {"path": [], "cost": -1, "expanded": len(trace), "trace": trace}


def run_search(grid, start, goal, heuristic, secondary=None):
    started = time.perf_counter()
    result = astar_search(grid, start, goal, heuristic, secondary_heuristic=secondary)
    return result, time.perf_counter() - started


def error_metrics(table, distance_grid):
    pairs = [(table[(r, c)], value) for r, row in enumerate(distance_grid) for c, value in enumerate(row) if value >= 0]
    pred, truth = zip(*pairs)
    return {
        "mae": mean(abs(a - b) for a, b in pairs),
        "mse": mean((a - b) ** 2 for a, b in pairs),
        "cell_spearman": spearman(pred, truth),
        "cell_pearson": pearson(pred, truth),
    }


def trace_ordering(trace):
    correct = total = 0
    for item in trace:
        entries = item["open_before"]
        for index, left in enumerate(entries):
            for right in entries[index + 1:]:
                if left["true_distance"] == right["true_distance"] or left["unet_h"] == right["unet_h"]:
                    continue
                total += 1
                correct += int((left["true_distance"] - right["true_distance"]) * (left["unet_h"] - right["unet_h"]) > 0)
    return correct, total


def trace_route_metrics(trace, optimal_nodes):
    off_path = [item["expanded"] for item in trace if item["expanded"]["node"] not in optimal_nodes]
    return len(off_path), mean(
        min(abs(entry["node"][0] - node[0]) + abs(entry["node"][1] - node[1]) for node in optimal_nodes)
        for entry in off_path
    )


def bottleneck_proxy(grid, path):
    if not path:
        return 0
    # Small local degree is a reproducible proxy; it is not a topological cut width.
    return min(len(list(neighbors(grid, node))) for node in path)


def evaluate_case(row, old_model, expanded_model):
    grid, start, goal = rebuild_case(row)
    distance_to_goal = compute_distance_to_goal(grid, goal)
    optimal_cost = distance_to_goal[start[0]][start[1]]
    if optimal_cost != int(row["optimal_cost"]):
        raise AssertionError(f"Rebuilt map changed optimal cost for seed {row['seed']}")
    distance_from_start = compute_distance_to_goal(grid, start)
    optimal_nodes = optimal_path_nodes(distance_from_start, distance_to_goal, optimal_cost)
    canonical_path = canonical_optimal_path(grid, start, goal, distance_to_goal)
    tables = {}
    for name, model in (("old", old_model), ("expanded", expanded_model)):
        tables[name] = make_table(grid, goal, make_unet_heuristic(model, grid, goal))
    outputs = {}
    outputs["manhattan"] = (*run_search(grid, start, goal, manhattan_heuristic), None)
    if outputs["manhattan"][0]["expanded"] != int(row["expanded_nodes"]):
        raise AssertionError(f"Rebuilt map changed Manhattan expansion count for seed {row['seed']}")
    for name in ("old", "expanded"):
        table = tables[name]
        heuristic = lambda node, unused_goal, values=table: values[node]
        outputs[f"{name}_unet_tiebreak"] = (*run_search(grid, start, goal, manhattan_heuristic, heuristic), None)
        trace_result = direct_trace(grid, start, goal, table, distance_to_goal, optimal_nodes)
        result, runtime = run_search(grid, start, goal, heuristic)
        if (result["expanded"], result["cost"], result["path"]) != (trace_result["expanded"], trace_result["cost"], trace_result["path"]):
            raise AssertionError("Direct U-Net trace diverged from astar_search")
        outputs[f"{name}_direct_unet"] = (result, runtime, trace_result)

    case_id = f"{row['analysis_structure']}_rate{row['obstacle_rate']}_seed{row['seed']}_s{start[0]}-{start[1]}_g{goal[0]}-{goal[1]}"
    common = {
        "case_id": case_id, "structure_type": row["analysis_structure"], "seed": int(row["seed"]),
        "obstacle_rate": float(row["obstacle_rate"]), "start": start, "goal": goal,
        "optimal_cost": optimal_cost, "manhattan_start_goal": manhattan_heuristic(start, goal),
        "detour_gap": optimal_cost - manhattan_heuristic(start, goal),
        "bottleneck_width_proxy": bottleneck_proxy(grid, canonical_path),
    }
    rows, diagnostics = [], {}
    for algorithm, (result, runtime, trace) in outputs.items():
        item = dict(common)
        item.update({"algorithm": algorithm, "expanded_nodes": result["expanded"], "path_cost": result["cost"],
                     "optimal": bool(result["path"]) and result["cost"] == optimal_cost, "runtime_seconds": runtime})
        rows.append(item)
        if trace is not None:
            model = algorithm.split("_", 1)[0]
            metrics = error_metrics(tables[model], distance_to_goal)
            pair_correct, pair_total = trace_ordering(trace["trace"])
            case_for_events = {"case_id": case_id, "structured_type": row["analysis_structure"], "optimal_cost": optimal_cost}
            events = critical_events_for_case(case_for_events, trace, optimal_nodes, weak_margin=0.5)
            off_path, route_distance = trace_route_metrics(trace["trace"], optimal_nodes)
            diagnostics[algorithm] = {
                **metrics, "pairwise_ordering_accuracy": pair_correct / pair_total if pair_total else 0.0,
                "pair_count": pair_total, "events": events, "off_path_expansions": off_path,
                "mean_off_path_distance": route_distance, "trace": trace, "table": tables[model],
            }
    return common, rows, diagnostics, grid, distance_to_goal, canonical_path


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["structure_type"], row["algorithm"])].append(row)
    output = []
    for (structure, algorithm), values in sorted(grouped.items()):
        output.append({
            "structure_type": structure, "algorithm": algorithm, "cases": len(values),
            "mean_expanded_nodes": mean(row["expanded_nodes"] for row in values),
            "median_expanded_nodes": median(row["expanded_nodes"] for row in values),
            "mean_path_cost": mean(row["path_cost"] for row in values),
            "optimality_rate": mean(float(row["optimal"]) for row in values),
            "mean_runtime_seconds": mean(row["runtime_seconds"] for row in values),
        })
    return output


def comparison_rows(rows):
    by_case = defaultdict(dict)
    for row in rows:
        by_case[row["case_id"]][row["algorithm"]] = row
    grouped = defaultdict(list)
    for values in by_case.values():
        for algorithm in ("old_direct_unet", "expanded_direct_unet"):
            grouped[(values[algorithm]["structure_type"], algorithm)].append(values)
    output = []
    for (structure, algorithm), values in sorted(grouped.items()):
        direct = [item[algorithm]["expanded_nodes"] for item in values]
        manhattan = [item["manhattan"]["expanded_nodes"] for item in values]
        tiebreak_algorithm = f"{algorithm.split('_', 1)[0]}_unet_tiebreak"
        tiebreak = [item[tiebreak_algorithm]["expanded_nodes"] for item in values]
        other = [item[("expanded_direct_unet" if algorithm == "old_direct_unet" else "old_direct_unet")]["expanded_nodes"] for item in values]
        output.append({"structure_type": structure, "model": algorithm.split("_")[0], "cases": len(values),
                       "direct_minus_manhattan_mean": mean(a - b for a, b in zip(direct, manhattan)),
                       "direct_minus_tiebreak_mean": mean(a - b for a, b in zip(direct, tiebreak)),
                       "direct_minus_other_model_mean": mean(a - b for a, b in zip(direct, other)),
                       "direct_wins_vs_manhattan": sum(a < b for a, b in zip(direct, manhattan)),
                       "direct_losses_vs_manhattan": sum(a > b for a, b in zip(direct, manhattan)),
                       "direct_wins_vs_tiebreak": sum(a < b for a, b in zip(direct, tiebreak)),
                       "direct_losses_vs_tiebreak": sum(a > b for a, b in zip(direct, tiebreak))})
    return output


def critical_rows(records):
    output = []
    for record in records:
        for algorithm, diag in record["diagnostics"].items():
            events = diag["events"]
            high = [event for event in events if event["recovery_cost"] >= 10]
            output.append({"case_id": record["common"]["case_id"], "structure_type": record["common"]["structure_type"],
                           "algorithm": algorithm, "critical_decisions": len(events), "wrong_critical_decisions": len(events),
                           "mean_recovery_cost": mean(event["recovery_cost"] for event in events),
                           "high_impact_failure_count": len(high), "off_path_expansions": diag["off_path_expansions"],
                           "pairwise_ordering_accuracy": diag["pairwise_ordering_accuracy"], "cell_spearman": diag["cell_spearman"],
                           "mae": diag["mae"], "mse": diag["mse"]})
    return output


def write_reports(output_dir, summaries, comparisons, critical, rows):
    summary_lookup = {(row["structure_type"], row["algorithm"]): row for row in summaries}
    comparison_lookup = {(row["structure_type"], row["model"]): row for row in comparisons}
    critical_group = defaultdict(list)
    for row in critical:
        critical_group[(row["structure_type"], row["algorithm"])].append(row)

    def detail(structure):
        lines = [f"# {structure} Direct U-Net Analysis", "", "## Search Results", ""]
        lines += ["| Algorithm | Mean expanded | Median expanded | Optimality |", "|---|---:|---:|---:|"]
        for algorithm in ALGORITHMS:
            row = summary_lookup[(structure, algorithm)]
            lines.append(f"| {algorithm} | {row['mean_expanded_nodes']:.2f} | {row['median_expanded_nodes']:.2f} | {row['optimality_rate']:.3f} |")
        lines += ["", "## Direct-Model Comparison", ""]
        for model in ("old", "expanded"):
            row = comparison_lookup[(structure, model)]
            lines.append(f"- {model}: direct minus Manhattan = {row['direct_minus_manhattan_mean']:.2f}; direct minus same-model tie-break = {row['direct_minus_tiebreak_mean']:.2f}.")
        if structure == "narrow_corridor":
            for algorithm in ("old_direct_unet", "expanded_direct_unet"):
                values = critical_group[(structure, algorithm)]
                direct_rows = [row for row in rows if row["structure_type"] == structure and row["algorithm"] == algorithm]
                lines.append(f"- {algorithm}: MAE {mean(v['mae'] for v in values):.3f}, MSE {mean(v['mse'] for v in values):.3f}, cell Spearman {mean(v['cell_spearman'] for v in values):.3f}, pairwise ordering {mean(v['pairwise_ordering_accuracy'] for v in values):.3f}, critical decisions/case {mean(v['critical_decisions'] for v in values):.2f}, mean recovery {mean(v['mean_recovery_cost'] for v in values):.2f}, high-impact failures/case {mean(v['high_impact_failure_count'] for v in values):.2f}, non-optimal cases {sum(not row['optimal'] for row in direct_rows)}.")
        if structure == "maze_like":
            for algorithm in ("old_direct_unet", "expanded_direct_unet"):
                values = critical_group[(structure, algorithm)]
                lines.append(f"- {algorithm}: off-path expansions/case {mean(v['off_path_expansions'] for v in values):.2f}; this is descriptive route-bias evidence, not a causal mechanism test.")
        if structure == "bottleneck":
            by_case = defaultdict(dict)
            for row in rows:
                if row["structure_type"] == structure:
                    by_case[row["case_id"]][row["algorithm"]] = row
            gaps = [values["old_direct_unet"]["expanded_nodes"] - values["manhattan"]["expanded_nodes"] for values in by_case.values()]
            detours = [values["old_direct_unet"]["detour_gap"] for values in by_case.values()]
            lines.append(f"- Old direct U-Net gap versus Manhattan has Pearson correlation {pearson(detours, gaps):.3f} with start-goal detour gap. This is an association, not evidence of a connectivity cause.")
            for algorithm in ("old_direct_unet", "expanded_direct_unet"):
                improved = [values for values in by_case.values() if values[algorithm]["expanded_nodes"] < values["manhattan"]["expanded_nodes"]]
                degraded = [values for values in by_case.values() if values[algorithm]["expanded_nodes"] > values["manhattan"]["expanded_nodes"]]
                def group_text(group):
                    return f"n={len(group)}, mean detour gap={mean(item[algorithm]['detour_gap'] for item in group):.2f}, mean local-width proxy={mean(item[algorithm]['bottleneck_width_proxy'] for item in group):.2f}"
                lines.append(f"- {algorithm}: improved cases {group_text(improved)}; degraded cases {group_text(degraded)}.")
        lines += ["", "## Interpretation", "", "Results are observational. Constant heuristic offsets preserve direct-A* node order, so calibration offsets alone cannot explain a change in expansion order under this implementation.", ""]
        return "\n".join(lines)

    for structure, filename in (("narrow_corridor", "narrow_corridor_analysis.md"), ("bottleneck", "bottleneck_analysis.md"), ("maze_like", "maze_like_analysis.md")):
        if (structure, "old_direct_unet") not in summary_lookup:
            continue
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as handle:
            handle.write(detail(structure))

    lines = ["# U-Net Structure Behavior Analysis", "", "## Scope", "", "All results use the existing 100-seed-per-rate controlled benchmark maps and start/goal pairs. Searches were rerun offline without changing A*, model weights, or training.", "", "## Main Results", ""]
    for structure in STRUCTURES:
        if (structure, "old") not in comparison_lookup:
            continue
        old = comparison_lookup[(structure, "old")]
        expanded = comparison_lookup[(structure, "expanded")]
        lines.append(f"- {structure}: old direct vs Manhattan {old['direct_minus_manhattan_mean']:.2f} expansions; expanded direct vs Manhattan {expanded['direct_minus_manhattan_mean']:.2f}; expanded minus old direct {expanded['direct_minus_other_model_mean']:.2f}.")
    lines += ["", "## Answers", "", "1. See the signed direct-minus-Manhattan values above and `old_vs_new_structure_comparison.csv`; a negative value indicates fewer expansions.", "2. Narrow-corridor changes are reported alongside MAE, ordering, and critical-decision measures. These establish correlated changes, not a causal training-data mechanism.", "3. Bottleneck detour-gap correlations are descriptive and cannot distinguish connectivity understanding from calibration by themselves.", "4. Direct U-Net removes Manhattan primary-f suppression, while tie-break retains it. The comparison separates behavior under those two uses but does not isolate representation from training-data effects.", ""]
    with open(os.path.join(output_dir, "final_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def draw_examples(output_dir, records):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    import matplotlib.pyplot as plt

    image_dir = os.path.join(output_dir, "representative_case_images")
    os.makedirs(image_dir, exist_ok=True)
    for structure in STRUCTURES:
        scoped = [record for record in records if record["common"]["structure_type"] == structure]
        if not scoped:
            continue
        ranked = sorted(scoped, key=lambda record: next(row["expanded_nodes"] for row in record["rows"] if row["algorithm"] == "expanded_direct_unet") - next(row["expanded_nodes"] for row in record["rows"] if row["algorithm"] == "manhattan"))
        for label, record in (("best", ranked[0]), ("worst", ranked[-1])):
            diag = record["diagnostics"]["expanded_direct_unet"]
            trace = diag["trace"]["trace"]
            events = diag["events"]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(record["grid"], cmap="Greys", vmin=0, vmax=1)
            path = record["path"]
            axes[0].plot([node[1] for node in path], [node[0] for node in path], color="#f2c94c", linewidth=2)
            axes[0].scatter([node["expanded"]["node"][1] for node in trace], [node["expanded"]["node"][0] for node in trace], s=5, c="#2b7bba", alpha=.55)
            for event in events[:10]:
                r, c = map(int, event["wrong_node"].split(",")); axes[0].scatter(c, r, c="#d62728", marker="x", s=22)
            axes[0].set_title(f"{structure} {label}: path/expansions")
            axes[1].imshow([[diag["table"].get((r, c), math.nan) if record["grid"][r][c] == 0 else math.nan for c in range(len(record["grid"][0]))] for r in range(len(record["grid"]))], cmap="viridis")
            axes[1].set_title("Expanded U-Net h")
            axes[2].imshow([[value if value >= 0 else math.nan for value in row] for row in record["distance"]], cmap="viridis")
            axes[2].set_title("True distance")
            for axis in axes:
                axis.set_xticks([]); axis.set_yticks([])
            fig.tight_layout()
            fig.savefig(os.path.join(image_dir, f"{structure}_{label}.png"), dpi=160)
            plt.close(fig)


def run(args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    old_path = os.path.join(root, "checkpoints", "unet_heuristic_best.pt")
    expanded_path = os.path.join(root, args.expanded_checkpoint)
    cases = benchmark_cases(root, args.structured_results, args.random_results)
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    old_model, expanded_model = load_unet_heuristic(old_path), load_unet_heuristic(expanded_path)
    all_rows, records = [], []
    for index, row in enumerate(cases, start=1):
        common, rows, diagnostics, grid, distance, path = evaluate_case(row, old_model, expanded_model)
        all_rows.extend(rows)
        records.append({"common": common, "rows": rows, "diagnostics": diagnostics, "grid": grid, "distance": distance, "path": path})
        if index % 100 == 0:
            print(f"evaluated {index}/{len(cases)} cases")
    summaries, comparisons, critical = aggregate(all_rows), comparison_rows(all_rows), critical_rows(records)
    write_csv(os.path.join(output_dir, "per_case_results.csv"), all_rows)
    write_csv(os.path.join(output_dir, "structure_summary.csv"), summaries)
    write_csv(os.path.join(output_dir, "old_vs_new_structure_comparison.csv"), comparisons)
    write_csv(os.path.join(output_dir, "critical_decision_comparison.csv"), critical)
    write_reports(output_dir, summaries, comparisons, critical, all_rows)
    draw_examples(output_dir, records)
    print(f"completed {len(cases)} benchmark cases in {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze direct U-Net A* behavior by map structure.")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_tiebreak_controls_100.csv")
    parser.add_argument("--expanded-checkpoint", default="outputs/expanded_dataset_training/unet_heuristic_expanded_best.pt")
    parser.add_argument("--output-dir", default="outputs/unet_structure_behavior_analysis")
    parser.add_argument("--max-cases", type=int, default=None, help="Smoke-test limit; omit for the full benchmark.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
