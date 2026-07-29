import argparse
import csv
import heapq
import json
import math
import os
from collections import Counter, defaultdict

from bfs_label import compute_distance_to_goal
from experiment import (
    DEFAULT_RANDOM_START_GOAL_RETRIES,
    MAP_SIZES,
    OBSTACLE_RATES,
    load_models,
    parse_seeds,
    select_start_goal,
)
from model import make_unet_heuristic
from structured_maps import generate_structured_map, parse_structured_types


STRUCTURED_TYPES = ["maze_like", "bottleneck", "large_block", "narrow_corridor"]
OUTPUT_DIR = "outputs/critical_decision_analysis"
WEAK_MARGIN_THRESHOLD = 0.5


def manhattan(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def node_text(node):
    return f"{node[0]},{node[1]}"


def parse_node(text):
    row, col = text.split(",", maxsplit=1)
    return int(row), int(col)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def median(values):
    values = sorted(values)
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    index = int(round((len(values) - 1) * pct))
    return values[index]


def write_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_case_grid(map_size, seed, obstacle_rate, structured_type):
    return generate_structured_map(
        width=map_size,
        height=map_size,
        seed=seed,
        obstacle_rate=obstacle_rate,
        structured_type=structured_type,
    )


def free_cells(grid):
    return [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 0]


def neighbors(grid, node):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    r, c = node
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            yield nr, nc


def make_table(grid, goal, heuristic):
    return {cell: float(heuristic(cell, goal)) for cell in free_cells(grid)}


def true_distance_table(distance_grid):
    return {(r, c): value for r, row in enumerate(distance_grid) for c, value in enumerate(row) if value >= 0}


def optimal_path_nodes(distance_from_start, distance_to_goal, optimal_cost):
    nodes = set()
    if optimal_cost < 0:
        return nodes
    for r, row in enumerate(distance_to_goal):
        for c, goal_distance in enumerate(row):
            start_distance = distance_from_start[r][c]
            if start_distance >= 0 and goal_distance >= 0 and start_distance + goal_distance == optimal_cost:
                nodes.add((r, c))
    return nodes


def canonical_optimal_path(grid, start, goal, distance_to_goal):
    if distance_to_goal[start[0]][start[1]] < 0:
        return []
    current = start
    path = [current]
    while current != goal:
        current_distance = distance_to_goal[current[0]][current[1]]
        candidates = [
            node for node in neighbors(grid, current) if distance_to_goal[node[0]][node[1]] == current_distance - 1
        ]
        if not candidates:
            break
        current = sorted(candidates)[0]
        path.append(current)
    return path


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def node_features(node, goal, g_score, unet_table, true_table, optimal_nodes):
    g = g_score[node]
    mh = float(manhattan(node, goal))
    uh = float(unet_table[node])
    td = true_table.get(node, math.inf)
    return {
        "node": node,
        "g": g,
        "manhattan_h": mh,
        "unet_h": uh,
        "true_distance": td,
        "f": g + mh,
        "on_optimal_path": node in optimal_nodes,
    }


def trace_unet_tiebreak_astar(grid, start, goal, unet_table, distance_to_goal, optimal_nodes):
    true_table = true_distance_table(distance_to_goal)
    open_heap = []
    g_score = {start: 0}
    came_from = {}
    counter = 0
    closed = set()
    trace = []

    heapq.heappush(open_heap, (manhattan(start, goal), unet_table[start], counter, 0, start))

    def valid_open_entries():
        entries = []
        for f_score, secondary, order, node_g, node in open_heap:
            if node_g != g_score.get(node, math.inf):
                continue
            features = node_features(node, goal, g_score, unet_table, true_table, optimal_nodes)
            features["priority"] = [f_score, secondary, order, node_g]
            features["insertion_order"] = order
            entries.append(features)
        entries.sort(key=lambda row: tuple(row["priority"] + [row["node"][0], row["node"][1]]))
        return entries

    while open_heap:
        open_before = valid_open_entries()
        if not open_before:
            break

        f_score, _, _, current_g, current = heapq.heappop(open_heap)
        if current_g > g_score.get(current, math.inf):
            continue

        current_features = node_features(current, goal, g_score, unet_table, true_table, optimal_nodes)
        current_features["f"] = f_score
        step = len(trace)
        closed.add(current)
        trace.append(
            {
                "step": step,
                "expanded": current_features,
                "open_before": open_before,
            }
        )

        if current == goal:
            path = reconstruct_path(came_from, current)
            return {
                "path": path,
                "cost": len(path) - 1,
                "expanded": len(trace),
                "trace": trace,
            }

        for neighbor in neighbors(grid, current):
            if neighbor in closed:
                continue
            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                counter += 1
                neighbor_f = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_heap, (neighbor_f, unet_table[neighbor], counter, tentative_g, neighbor))

    return {
        "path": [],
        "cost": -1,
        "expanded": len(trace),
        "trace": trace,
    }


def choose_best_optimal_candidate(open_before):
    candidates = [entry for entry in open_before if entry["on_optimal_path"]]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda entry: (
            entry["true_distance"],
            entry["f"],
            entry["unet_h"],
            -entry["g"],
            entry["insertion_order"],
        ),
    )


def classify_event(wrong, correct, weak_margin):
    unet_margin = correct["unet_h"] - wrong["unet_h"]
    if correct["true_distance"] < wrong["true_distance"] and wrong["unet_h"] < correct["unet_h"]:
        return "A_unet_ordering_error"
    if correct["unet_h"] < wrong["unet_h"] and correct["f"] > wrong["f"]:
        return "B_manhattan_f_limitation"
    if correct["unet_h"] <= wrong["unet_h"] and abs(unet_margin) <= weak_margin:
        return "C_weak_unet_separation"
    return "other"


def event_recovery(trace, event_index, optimal_nodes):
    off_path = 0
    recovery = None
    for later in trace[event_index + 1 :]:
        if later["expanded"]["on_optimal_path"]:
            recovery = later["step"] - trace[event_index]["step"]
            break
        off_path += 1
    if recovery is None:
        recovery = len(trace) - trace[event_index]["step"] - 1
    return recovery, off_path


def critical_events_for_case(case, trace_result, optimal_nodes, weak_margin):
    events = []
    trace = trace_result["trace"]
    for index, item in enumerate(trace):
        wrong = item["expanded"]
        if wrong["on_optimal_path"]:
            continue
        correct = choose_best_optimal_candidate(item["open_before"])
        if correct is None:
            continue

        recovery_cost, off_path_after = event_recovery(trace, index, optimal_nodes)
        error_type = classify_event(wrong, correct, weak_margin)
        events.append(
            {
                "case_id": case["case_id"],
                "structure_type": case["structured_type"],
                "step": item["step"],
                "wrong_node": node_text(wrong["node"]),
                "correct_node": node_text(correct["node"]),
                "error_type": error_type,
                "wrong_true_distance": wrong["true_distance"],
                "correct_true_distance": correct["true_distance"],
                "wrong_unet_h": wrong["unet_h"],
                "correct_unet_h": correct["unet_h"],
                "unet_margin": correct["unet_h"] - wrong["unet_h"],
                "recovery_cost": recovery_cost,
                "off_path_expansions_after_event": off_path_after,
                "wrong_g": wrong["g"],
                "correct_g": correct["g"],
                "wrong_manhattan_h": wrong["manhattan_h"],
                "correct_manhattan_h": correct["manhattan_h"],
                "wrong_f": wrong["f"],
                "correct_f": correct["f"],
                "total_expanded_nodes": trace_result["expanded"],
                "optimal_cost": case["optimal_cost"],
            }
        )
    return events


def trace_row(case, item):
    expanded = item["expanded"]
    open_snapshot = [
        {
            "node": node_text(entry["node"]),
            "g": entry["g"],
            "manhattan_h": entry["manhattan_h"],
            "unet_h": entry["unet_h"],
            "true_distance": entry["true_distance"],
            "f": entry["f"],
            "on_optimal_path": entry["on_optimal_path"],
        }
        for entry in item["open_before"]
    ]
    return {
        "case_id": case["case_id"],
        "structure_type": case["structured_type"],
        "step": item["step"],
        "expanded_node": node_text(expanded["node"]),
        "g": expanded["g"],
        "manhattan_h": expanded["manhattan_h"],
        "unet_h": expanded["unet_h"],
        "true_distance": expanded["true_distance"],
        "on_optimal_path": expanded["on_optimal_path"],
        "open_before": open_snapshot,
    }


def case_id_for(structured_type, obstacle_rate, seed, start, goal):
    return (
        f"{structured_type}_rate{obstacle_rate}_seed{seed}"
        f"_s{start[0]}-{start[1]}_g{goal[0]}-{goal[1]}"
    )


def collect_cases(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _, unet_model = load_models(project_root, args.checkpoint)
    cases = []
    skips = Counter()

    for map_size in MAP_SIZES:
        for obstacle_rate in OBSTACLE_RATES:
            for structured_type in parse_structured_types(args.structured_types):
                for seed in parse_seeds(args.seeds):
                    grid = generate_case_grid(map_size, seed, obstacle_rate, structured_type)
                    start, goal, distance_to_goal, optimal_cost, skip_reason = select_start_goal(
                        grid,
                        seed,
                        map_size,
                        obstacle_rate,
                        args.start_goal_mode,
                        args.random_start_goal_retries,
                    )
                    if skip_reason:
                        skips[skip_reason] += 1
                        continue

                    distance_from_start = compute_distance_to_goal(grid, start)
                    optimal_nodes = optimal_path_nodes(distance_from_start, distance_to_goal, optimal_cost)
                    unet_h = make_unet_heuristic(unet_model, grid, goal)
                    unet_table = make_table(grid, goal, unet_h)
                    trace_result = trace_unet_tiebreak_astar(grid, start, goal, unet_table, distance_to_goal, optimal_nodes)
                    if trace_result["cost"] != optimal_cost:
                        raise AssertionError(
                            f"U-Net tie-break path cost {trace_result['cost']} != optimal cost {optimal_cost} "
                            f"for {structured_type} seed {seed}"
                        )

                    case = {
                        "case_id": case_id_for(structured_type, obstacle_rate, seed, start, goal),
                        "structured_type": structured_type,
                        "seed": seed,
                        "map_size": map_size,
                        "obstacle_rate": obstacle_rate,
                        "start": start,
                        "goal": goal,
                        "optimal_cost": optimal_cost,
                        "grid": grid,
                        "optimal_nodes": optimal_nodes,
                        "canonical_path": canonical_optimal_path(grid, start, goal, distance_to_goal),
                        "trace_result": trace_result,
                    }
                    case["events"] = critical_events_for_case(
                        case, trace_result, optimal_nodes, args.weak_margin_threshold
                    )
                    cases.append(case)
    return cases, skips


def write_traces_jsonl(path, cases):
    with open(path, "w", encoding="utf-8") as file:
        for case in cases:
            for item in case["trace_result"]["trace"]:
                file.write(json.dumps(trace_row(case, item), sort_keys=True) + "\n")


def summarize(cases, events, skips):
    total_expansions = sum(case["trace_result"]["expanded"] for case in cases)
    total_extra_expansions = sum(max(0, case["trace_result"]["expanded"] - (case["optimal_cost"] + 1)) for case in cases)
    event_steps_by_case = defaultdict(set)
    for event in events:
        start = int(event["step"]) + 1
        stop = start + int(event["off_path_expansions_after_event"])
        event_steps_by_case[event["case_id"]].update(range(start, stop))
    unique_impacted_steps = sum(len(steps) for steps in event_steps_by_case.values())
    overlapping_event_impact = sum(int(event["off_path_expansions_after_event"]) for event in events)
    by_type = Counter(event["error_type"] for event in events)
    by_structure = defaultdict(list)
    for event in events:
        by_structure[event["structure_type"]].append(event)

    lines = []
    lines.append("# Critical Decision Analysis")
    lines.append("")
    lines.append("This analysis traces `manhattan_unet_tiebreak` offline. It does not modify A*, model weights, or training.")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Cases analyzed: {len(cases)}")
    lines.append(f"- Total expanded nodes: {total_expansions}")
    lines.append(f"- Extra expansions above an ideal path-only expansion lower bound: {total_extra_expansions}")
    lines.append(f"- Critical decisions: {len(events)}")
    pct_total = 100.0 * unique_impacted_steps / total_expansions if total_expansions else 0.0
    pct_extra = 100.0 * unique_impacted_steps / total_extra_expansions if total_extra_expansions else 0.0
    lines.append(f"- Unique off-path expansion steps after critical decisions: {unique_impacted_steps}")
    lines.append(f"- Percentage of total expansions covered by critical-decision aftermath: {pct_total:.2f}%")
    lines.append(f"- Percentage of extra expansions covered by critical-decision aftermath: {pct_extra:.2f}%")
    lines.append(f"- Overlapping per-event impact sum: {overlapping_event_impact}")
    lines.append(f"- Average recovery cost: {mean(float(event['recovery_cost']) for event in events):.3f}")
    lines.append(f"- Median recovery cost: {median(float(event['recovery_cost']) for event in events):.3f}")
    lines.append("- Validation: every traced U-Net tie-break path matched the reverse-BFS optimal path cost.")
    if skips:
        lines.append(f"- Skipped cases: {dict(skips)}")
    lines.append("")
    lines.append("## Distribution By Error Type")
    lines.append("")
    lines.append("| Error type | Count | Share | Mean recovery |")
    lines.append("|---|---:|---:|---:|")
    for error_type, count in by_type.most_common():
        scoped = [event for event in events if event["error_type"] == error_type]
        share = count / len(events) if events else 0.0
        lines.append(f"| {error_type} | {count} | {share:.3f} | {mean(float(e['recovery_cost']) for e in scoped):.3f} |")
    lines.append("")
    lines.append("## Results By Structure Type")
    lines.append("")
    lines.append("| Structure | Cases | Critical decisions | Mean decisions/case | Mean recovery |")
    lines.append("|---|---:|---:|---:|---:|")
    case_counts = Counter(case["structured_type"] for case in cases)
    for structure in sorted(case_counts):
        scoped = by_structure.get(structure, [])
        lines.append(
            f"| {structure} | {case_counts[structure]} | {len(scoped)} | "
            f"{len(scoped) / case_counts[structure]:.3f} | "
            f"{mean(float(e['recovery_cost']) for e in scoped):.3f} |"
        )
    lines.append("")
    lines.append("## Final Analysis Questions")
    lines.append("")
    if events:
        by_case = Counter(event["case_id"] for event in events)
        top_10_cases = sum(count for _, count in by_case.most_common(max(1, len(by_case) // 10)))
        lines.append(
            f"1. Critical decisions are partially concentrated: the highest-event decile of affected cases contains "
            f"{top_10_cases} of {len(events)} events."
        )
        lines.append(
            f"2. Using off-path expansions after each event as an approximate impact measure, critical decisions account for "
            f"{pct_total:.2f}% of total expansions after deduplicating overlapping event aftermath windows, and "
            f"{pct_extra:.2f}% of extra expansions above the path-only lower bound."
        )
        dominant = by_type.most_common(1)[0][0]
        lines.append(f"3. The most common classified cause is `{dominant}`.")
        structure_counts = Counter(event["structure_type"] for event in events)
        dominant_structure = structure_counts.most_common(1)[0][0]
        lines.append(f"4. Critical decisions are most frequent in `{dominant_structure}` in absolute count.")
    else:
        lines.append("No critical decisions were detected under the current definition.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Per-event impact estimates can overlap when several critical decisions occur before recovery.")
    lines.append("- `on_optimal_path` means membership in any shortest path, not only one rendered path.")
    lines.append("- Class C uses a fixed U-Net margin threshold and should be treated as sensitivity-dependent.")
    lines.append("")
    return "\n".join(lines) + "\n"


def plot_grid(ax, case, event):
    grid = case["grid"]
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    image = [[1.0 if grid[r][c] else 0.0 for c in range(cols)] for r in range(rows)]
    ax.imshow(image, cmap="Greys", vmin=0, vmax=1)

    if case["canonical_path"]:
        ys = [node[0] for node in case["canonical_path"]]
        xs = [node[1] for node in case["canonical_path"]]
        ax.plot(xs, ys, color="#f2c94c", linewidth=2)

    limit = min(len(case["trace_result"]["trace"]), int(event["step"]) + int(event["recovery_cost"]) + 1)
    trajectory = [item["expanded"]["node"] for item in case["trace_result"]["trace"][:limit]]
    if trajectory:
        ax.scatter([node[1] for node in trajectory], [node[0] for node in trajectory], s=8, c="#2b7bba", alpha=0.55)

    wrong = parse_node(event["wrong_node"])
    correct = parse_node(event["correct_node"])
    ax.scatter([wrong[1]], [wrong[0]], s=50, c="#d62728", marker="x", linewidths=2)
    ax.scatter([correct[1]], [correct[0]], s=50, marker="o", facecolors="none", edgecolors="#2ca02c", linewidths=2)
    ax.scatter([case["start"][1]], [case["start"][0]], s=35, c="#111111", marker="s")
    ax.scatter([case["goal"][1]], [case["goal"][0]], s=35, c="#9467bd", marker="*")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{case['structured_type']} step {event['step']} cost {event['recovery_cost']}", fontsize=8)


def create_plots(output_dir, cases, events):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    event_counts = Counter(event["case_id"] for event in events)
    case_expanded = {case["case_id"]: case["trace_result"]["expanded"] for case in cases}
    xs = [event_counts.get(case_id, 0) for case_id in case_expanded]
    ys = [expanded for expanded in case_expanded.values()]
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, s=16)
    plt.xlabel("Critical decisions per map")
    plt.ylabel("Total expanded nodes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "critical_decisions_vs_expanded.png"))
    plt.close()

    recovery = [float(event["recovery_cost"]) for event in events]
    plt.figure(figsize=(6, 4))
    plt.hist(recovery, bins=30)
    plt.xlabel("Recovery cost")
    plt.ylabel("Critical decisions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recovery_cost_distribution.png"))
    plt.close()

    margins = [float(event["unet_margin"]) for event in events]
    plt.figure(figsize=(6, 4))
    plt.scatter(margins, recovery, s=14, alpha=0.7)
    plt.xlabel("U-Net margin: correct h - wrong h")
    plt.ylabel("Recovery cost")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "unet_margin_vs_recovery_cost.png"))
    plt.close()

    top_events = sorted(events, key=lambda event: int(event["off_path_expansions_after_event"]), reverse=True)[:20]
    case_by_id = {case["case_id"]: case for case in cases}
    if top_events:
        figure_dir = os.path.join(output_dir, "top_impact_cases")
        os.makedirs(figure_dir, exist_ok=True)
        cols = 4
        rows = math.ceil(len(top_events) / cols)
        plt.figure(figsize=(cols * 3, rows * 3))
        for index, event in enumerate(top_events, start=1):
            ax = plt.subplot(rows, cols, index)
            plot_grid(ax, case_by_id[event["case_id"]], event)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top20_highest_impact_cases.png"))
        plt.close()

        for index, event in enumerate(top_events, start=1):
            plt.figure(figsize=(4, 4))
            ax = plt.gca()
            plot_grid(ax, case_by_id[event["case_id"]], event)
            plt.tight_layout()
            plt.savefig(os.path.join(figure_dir, f"top_case_{index:02d}.png"))
            plt.close()


def run(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    cases, skips = collect_cases(args)
    events = []
    for case in cases:
        events.extend(case["events"])

    event_rows = [
        {
            "case_id": event["case_id"],
            "structure_type": event["structure_type"],
            "step": event["step"],
            "wrong_node": event["wrong_node"],
            "correct_node": event["correct_node"],
            "error_type": event["error_type"],
            "wrong_true_distance": event["wrong_true_distance"],
            "correct_true_distance": event["correct_true_distance"],
            "wrong_unet_h": event["wrong_unet_h"],
            "correct_unet_h": event["correct_unet_h"],
            "unet_margin": event["unet_margin"],
            "recovery_cost": event["recovery_cost"],
            "off_path_expansions_after_event": event["off_path_expansions_after_event"],
        }
        for event in events
    ]
    write_csv(os.path.join(output_dir, "critical_decisions.csv"), event_rows)
    write_traces_jsonl(os.path.join(output_dir, "expansion_traces.jsonl"), cases)
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as file:
        file.write(summarize(cases, event_rows, skips))
    create_plots(output_dir, cases, event_rows)

    print(f"Saved critical decision analysis outputs to {output_dir}")
    print(f"Cases: {len(cases)}")
    print(f"Critical decisions: {len(event_rows)}")
    print(f"Skipped: {dict(skips)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline critical decision analysis for U-Net guided Manhattan A*.")
    parser.add_argument("--checkpoint", default="compatible")
    parser.add_argument("--seeds", default="0:100")
    parser.add_argument("--structured-types", default="all")
    parser.add_argument("--start-goal-mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--random-start-goal-retries", type=int, default=DEFAULT_RANDOM_START_GOAL_RETRIES)
    parser.add_argument("--weak-margin-threshold", type=float, default=WEAK_MARGIN_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
