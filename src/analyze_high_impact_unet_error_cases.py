"""Offline case studies for high-impact U-Net ordering errors.

The analysis consumes the previous high-impact event table, then reconstructs
only the selected maps.  It deliberately does not alter A*, model weights, or
training; all measurements are diagnostic properties of recorded decisions.
"""

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict

from analyze_critical_decisions import (
    DEFAULT_RANDOM_START_GOAL_RETRIES,
    canonical_optimal_path,
    generate_case_grid,
    make_table,
    neighbors,
    optimal_path_nodes,
    parse_node,
    trace_unet_tiebreak_astar,
    true_distance_table,
)
from bfs_label import compute_distance_to_goal
from experiment import load_models
from model import make_unet_heuristic


SOURCE_EVENTS = "outputs/high_impact_decision_analysis/high_impact_events.csv"
OUTPUT_DIR = "outputs/high_impact_unet_error_analysis"
CASE_ID_PATTERN = re.compile(
    r"^(?P<structure>.+)_rate(?P<rate>[0-9.]+)_seed(?P<seed>\d+)"
    r"_s(?P<start_r>\d+)-(?P<start_c>\d+)_g(?P<goal_r>\d+)-(?P<goal_c>\d+)$"
)
STRUCTURES = ["maze_like", "bottleneck", "large_block", "narrow_corridor"]


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def correlation(xs, ys):
    if len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else 0.0


def read_selected_events(path, limit):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    filtered = [
        row
        for row in rows
        if row["threshold"] == "recovery_ge_20" and row["error_type"] == "A_unet_ordering_error"
    ]
    # The source table has one row per decision; a decision is the case-study unit.
    unique = {(row["case_id"], row["step"]): row for row in filtered}
    return sorted(
        unique.values(),
        key=lambda row: (-int(row["recovery_cost"]), -float(row["extra_expansions_after_event"]), row["case_id"], int(row["step"])),
    )[:limit]


def case_spec(case_id):
    match = CASE_ID_PATTERN.match(case_id)
    if not match:
        raise ValueError(f"Cannot parse case id: {case_id}")
    data = match.groupdict()
    return {
        "structure_type": data["structure"],
        "obstacle_rate": float(data["rate"]),
        "seed": int(data["seed"]),
        "start": (int(data["start_r"]), int(data["start_c"])),
        "goal": (int(data["goal_r"]), int(data["goal_c"])),
    }


def degree(grid, node):
    return sum(1 for _ in neighbors(grid, node))


def window_cells(grid, center, radius):
    r0, c0 = center
    cells = []
    for r in range(max(0, r0 - radius), min(len(grid), r0 + radius + 1)):
        for c in range(max(0, c0 - radius), min(len(grid[0]), c0 + radius + 1)):
            cells.append((r, c))
    return cells


def local_metrics(grid, wrong, unet_table, true_table, radius=5):
    cells = [cell for cell in window_cells(grid, wrong, radius) if grid[cell[0]][cell[1]] == 0 and cell in true_table]
    predicted = [unet_table[cell] for cell in cells]
    actual = [true_table[cell] for cell in cells]
    pair_total = pair_correct = 0
    for index, left in enumerate(cells):
        for right in cells[index + 1 :]:
            true_delta = true_table[left] - true_table[right]
            unet_delta = unet_table[left] - unet_table[right]
            if true_delta == 0 or unet_delta == 0:
                continue
            pair_total += 1
            pair_correct += int((true_delta < 0) == (unet_delta < 0))
    return {
        "local_mae": mean(abs(p - t) for p, t in zip(predicted, actual)),
        "local_correlation": correlation(predicted, actual),
        "local_ranking_accuracy": pair_correct / pair_total if pair_total else 0.0,
        "local_free_cells": len(cells),
    }


def encode_window(grid, center, radius, canonical_path, trajectory, wrong, correct, unet_table, true_table):
    path_set = set(canonical_path)
    trajectory_set = set(trajectory)
    rows = []
    for cell in window_cells(grid, center, radius):
        r, c = cell
        rows.append(
            {
                "row": r,
                "col": c,
                "obstacle": bool(grid[r][c]),
                "on_canonical_optimal_path": cell in path_set,
                "on_expansion_trajectory": cell in trajectory_set,
                "is_wrong_node": cell == wrong,
                "is_correct_candidate": cell == correct,
                "unet_h": unet_table.get(cell),
                "true_distance": true_table.get(cell),
            }
        )
    return rows


def enters_dead_end(grid, trajectory, start, goal):
    # A degree-one free cell away from endpoints is an unambiguous local dead end.
    return any(node not in {start, goal} and degree(grid, node) <= 1 for node in trajectory)


def is_bottleneck_or_corridor(grid, node):
    return degree(grid, node) <= 2


def near_obstacle_boundary(grid, node):
    r, c = node
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 1:
                return True
    return False


def failure_category(event, indicators, metrics):
    """Assign only high-confidence patterns; remaining rows are explicitly reviewable."""
    true_gap = float(event["wrong_true_distance"]) - float(event["correct_true_distance"])
    if indicators["wrong_enters_dead_end"]:
        return "1_dead_end_attraction", "recovery_trajectory_reaches_degree_one_cell"
    if true_gap >= 2 and (metrics["local_correlation"] <= 0.2 or metrics["local_ranking_accuracy"] <= 0.5):
        return "2_connectivity_misunderstanding", "large_true_gap_with_locally_misordered_distance_field"
    # The original event records f, so recover Manhattan h directly from the trace before calling this function.
    if indicators["wrong_manhattan_h"] < indicators["correct_manhattan_h"] and true_gap >= 2:
        return "3_local_geometric_bias", "wrong_direction_is_manhattan_closer_but_has_longer_true_route"
    return "4_other_manual_review", "no_high_confidence_automatic_pattern"


def reconstruct_event(event, unet_model):
    spec = case_spec(event["case_id"])
    grid = generate_case_grid(20, spec["seed"], spec["obstacle_rate"], spec["structure_type"])
    distance_to_goal = compute_distance_to_goal(grid, spec["goal"])
    distance_from_start = compute_distance_to_goal(grid, spec["start"])
    optimal_cost = distance_to_goal[spec["start"][0]][spec["start"][1]]
    if optimal_cost < 0:
        raise AssertionError(f"Selected case is unexpectedly unsolved: {event['case_id']}")
    optimal_nodes = optimal_path_nodes(distance_from_start, distance_to_goal, optimal_cost)
    unet_h = make_unet_heuristic(unet_model, grid, spec["goal"])
    unet_table = make_table(grid, spec["goal"], unet_h)
    trace_result = trace_unet_tiebreak_astar(grid, spec["start"], spec["goal"], unet_table, distance_to_goal, optimal_nodes)
    if trace_result["cost"] != optimal_cost:
        raise AssertionError(f"Trace lost optimality for {event['case_id']}")
    step = int(event["step"])
    if step >= len(trace_result["trace"]):
        raise AssertionError(f"Missing selected event step {step} for {event['case_id']}")
    trace_item = trace_result["trace"][step]
    wrong = trace_item["expanded"]["node"]
    correct = parse_node(event["correct_node"])
    if wrong != parse_node(event["wrong_node"]):
        raise AssertionError(f"Trace mismatch at selected event {event['case_id']} step {step}")
    true_table = true_distance_table(distance_to_goal)
    recovery = int(event["recovery_cost"])
    trajectory = [item["expanded"]["node"] for item in trace_result["trace"][max(0, step - 20) : min(len(trace_result["trace"]), step + recovery + 1)]]
    aftermath = [item["expanded"]["node"] for item in trace_result["trace"][step : min(len(trace_result["trace"]), step + recovery + 1)]]
    metrics = local_metrics(grid, wrong, unet_table, true_table)
    local5 = window_cells(grid, wrong, 5)
    obstacles = sum(grid[r][c] for r, c in local5)
    indicators = {
        "obstacle_count_radius5": obstacles,
        "obstacle_density": obstacles / len(local5) if local5 else 0.0,
        "reachable_neighbor_count": degree(grid, wrong),
        "wrong_enters_dead_end": enters_dead_end(grid, aftermath, spec["start"], spec["goal"]),
        "correct_enters_bottleneck_or_corridor": is_bottleneck_or_corridor(grid, correct),
        "near_obstacle_boundary": near_obstacle_boundary(grid, wrong),
        "wrong_manhattan_h": trace_item["expanded"]["manhattan_h"],
        "correct_manhattan_h": next(entry["manhattan_h"] for entry in trace_item["open_before"] if entry["node"] == correct),
    }
    category, category_basis = failure_category(event, indicators, metrics)
    return {
        "spec": spec,
        "grid": grid,
        "distance_to_goal": distance_to_goal,
        "true_table": true_table,
        "unet_table": unet_table,
        "canonical_path": canonical_optimal_path(grid, spec["start"], spec["goal"], distance_to_goal),
        "trace_result": trace_result,
        "wrong": wrong,
        "correct": correct,
        "trajectory": trajectory,
        "metrics": metrics,
        "indicators": indicators,
        "category": category,
        "category_basis": category_basis,
    }


def output_row(event, context, rank):
    metrics, indicators = context["metrics"], context["indicators"]
    return {
        "case_rank": rank,
        "case_id": event["case_id"],
        "structure_type": event["structure_type"],
        "step": event["step"],
        "recovery_cost": event["recovery_cost"],
        "extra_expansions": event["extra_expansions_after_event"],
        "wrong_node": event["wrong_node"],
        "correct_node": event["correct_node"],
        "wrong_g": context["trace_result"]["trace"][int(event["step"])]["expanded"]["g"],
        "correct_g": next(entry["g"] for entry in context["trace_result"]["trace"][int(event["step"])]["open_before"] if entry["node"] == context["correct"]),
        "wrong_manhattan_h": context["trace_result"]["trace"][int(event["step"])]["expanded"]["manhattan_h"],
        "correct_manhattan_h": next(entry["manhattan_h"] for entry in context["trace_result"]["trace"][int(event["step"])]["open_before"] if entry["node"] == context["correct"]),
        "wrong_f": event["wrong_f"],
        "correct_f": event["correct_f"],
        "wrong_true_distance": event["wrong_true_distance"],
        "correct_true_distance": event["correct_true_distance"],
        "wrong_unet_h": event["wrong_unet_h"],
        "correct_unet_h": event["correct_unet_h"],
        "unet_rank_error": float(event["correct_unet_h"]) - float(event["wrong_unet_h"]),
        **metrics,
        **indicators,
        "failure_category": context["category"],
        "category_basis": context["category_basis"],
    }


def save_context(path, event, context):
    payload = {
        "event": event,
        "start": context["spec"]["start"],
        "goal": context["spec"]["goal"],
        "wrong_node": context["wrong"],
        "correct_candidate": context["correct"],
        "metrics": context["metrics"],
        "connectivity_geometry_indicators": context["indicators"],
        "failure_category": context["category"],
        "category_basis": context["category_basis"],
        "canonical_optimal_path": context["canonical_path"],
        "trajectory_step_minus_20_to_recovery": context["trajectory"],
        "local_window_radius_3": encode_window(context["grid"], context["wrong"], 3, context["canonical_path"], context["trajectory"], context["wrong"], context["correct"], context["unet_table"], context["true_table"]),
        "local_window_radius_5": encode_window(context["grid"], context["wrong"], 5, context["canonical_path"], context["trajectory"], context["wrong"], context["correct"], context["unet_table"], context["true_table"]),
        "grid": context["grid"],
        "unet_h_grid": [[context["unet_table"].get((r, c)) for c in range(len(context["grid"][0]))] for r in range(len(context["grid"]))],
        "true_distance_grid": context["distance_to_goal"],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def overlay(ax, context, title, values=None, crop_radius=None):
    import numpy as np

    grid = context["grid"]
    wrong, correct = context["wrong"], context["correct"]
    if values is None:
        image = np.array([[1.0 if cell else 0.0 for cell in row] for row in grid])
        ax.imshow(image, cmap="Greys", vmin=0, vmax=1)
    else:
        image = np.array(values, dtype=float)
        image[np.array(grid, dtype=bool)] = np.nan
        ax.imshow(image, cmap="viridis")
    path = context["canonical_path"]
    ax.plot([node[1] for node in path], [node[0] for node in path], color="#f2c94c", linewidth=1.7)
    trajectory = context["trajectory"]
    ax.scatter([node[1] for node in trajectory], [node[0] for node in trajectory], s=9, c="#1f77b4", alpha=0.55)
    ax.scatter([wrong[1]], [wrong[0]], s=65, c="#d62728", marker="x", linewidths=2)
    ax.scatter([correct[1]], [correct[0]], s=50, marker="o", facecolors="none", edgecolors="#2ca02c", linewidths=2)
    ax.scatter([context["spec"]["start"][1]], [context["spec"]["start"][0]], s=30, c="#111111", marker="s")
    ax.scatter([context["spec"]["goal"][1]], [context["spec"]["goal"][0]], s=45, c="#9467bd", marker="*")
    if crop_radius:
        ax.set_xlim(wrong[1] - crop_radius - 0.5, wrong[1] + crop_radius + 0.5)
        ax.set_ylim(wrong[0] + crop_radius + 0.5, wrong[0] - crop_radius - 0.5)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def make_case_figures(output_dir, selected):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    import matplotlib.pyplot as plt

    image_dir = os.path.join(output_dir, "top20_case_images")
    os.makedirs(image_dir, exist_ok=True)
    for rank, event, context in selected[:20]:
        figure, axes = plt.subplots(2, 3, figsize=(10, 6.5))
        overlay(axes[0, 0], context, "Map, path, and trajectory")
        overlay(axes[0, 1], context, "Local map (r=3)", crop_radius=3)
        overlay(axes[0, 2], context, "Local map (r=5)", crop_radius=5)
        unet_grid = [[context["unet_table"].get((r, c), math.nan) for c in range(len(context["grid"][0]))] for r in range(len(context["grid"]))]
        overlay(axes[1, 0], context, "U-Net h", unet_grid)
        overlay(axes[1, 1], context, "True distance", context["distance_to_goal"])
        overlay(axes[1, 2], context, "Local U-Net h (r=5)", unet_grid, crop_radius=5)
        figure.suptitle(f"#{rank} {event['structure_type']} | recovery {event['recovery_cost']} | step {event['step']}", fontsize=10)
        figure.tight_layout()
        figure.savefig(os.path.join(image_dir, f"case_{rank:03d}.png"), dpi=160)
        plt.close(figure)


def make_summary_plots(output_dir, rows):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.hist([float(row["unet_rank_error"]) for row in rows], bins=24, color="#4c78a8")
    plt.xlabel("U-Net rank error: correct h - wrong h")
    plt.ylabel("Selected events")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "unet_rank_error_histogram.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter([float(row["local_mae"]) for row in rows], [float(row["recovery_cost"]) for row in rows], s=18, alpha=0.65)
    plt.xlabel("Local U-Net MAE (radius 5)")
    plt.ylabel("Recovery cost")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "local_unet_error_vs_recovery.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter([float(row["obstacle_density"]) for row in rows], [float(row["recovery_cost"]) for row in rows], s=18, alpha=0.65)
    plt.xlabel("Obstacle density (radius 5)")
    plt.ylabel("Recovery cost")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "obstacle_density_vs_recovery.png"), dpi=160)
    plt.close()

    counts = Counter(row["failure_category"] for row in rows)
    labels, values = zip(*sorted(counts.items()))
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, color=["#e45756", "#4c78a8", "#f58518", "#72b7b2"])
    plt.xticks(rotation=18, ha="right")
    plt.ylabel("Selected events")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "failure_category_distribution.png"), dpi=160)
    plt.close()


def write_summary(path, rows):
    categories = Counter(row["failure_category"] for row in rows)
    structures = defaultdict(list)
    for row in rows:
        structures[row["structure_type"]].append(row)
    lines = [
        "# High-impact U-Net Ordering Error Case Study",
        "",
        "This offline diagnostic selects the top 100 `recovery_ge_20` events classified as `A_unet_ordering_error`, ranked by recovery cost and then post-event extra expansions.",
        "A selected case is one decision event; multiple events may originate from the same map. No A*, model, or training behavior was modified.",
        "",
        "## Category Distribution",
        "",
        "Automatic categories are conservative: dead-end attraction requires a degree-one cell in the recovery trajectory; connectivity misunderstanding requires a true-distance gap of at least 2 plus local correlation <= 0.2 or pairwise ranking accuracy <= 0.5; local geometric bias requires the wrong node to have lower Manhattan h despite a true-distance gap of at least 2. All remaining events are left for manual review.",
        "",
        "| Category | Events | Share | Mean recovery | Mean local MAE | Mean obstacle density |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category, count in sorted(categories.items()):
        scoped = [row for row in rows if row["failure_category"] == category]
        lines.append(f"| {category} | {count} | {count / len(rows):.3f} | {mean(float(r['recovery_cost']) for r in scoped):.3f} | {mean(float(r['local_mae']) for r in scoped):.3f} | {mean(float(r['obstacle_density']) for r in scoped):.3f} |")
    lines.extend(["", "## Structure-specific Patterns", "", "| Structure | Events | Mean recovery | Mean rank error | Mean local MAE | Dominant category |", "|---|---:|---:|---:|---:|---|"])
    for structure in STRUCTURES:
        scoped = structures.get(structure, [])
        dominant = Counter(row["failure_category"] for row in scoped).most_common(1)
        label = dominant[0][0] if dominant else "n/a"
        lines.append(f"| {structure} | {len(scoped)} | {mean(float(r['recovery_cost']) for r in scoped):.3f} | {mean(float(r['unet_rank_error']) for r in scoped):.3f} | {mean(float(r['local_mae']) for r in scoped):.3f} | {label} |")
    lines.extend(["", "## Representative Cases", "", "| Rank | Case | Structure | Recovery | Category |", "|---:|---|---|---:|---|"])
    for row in rows[:10]:
        lines.append(f"| {row['case_rank']} | {row['case_id']} | {row['structure_type']} | {row['recovery_cost']} | {row['failure_category']} |")
    largest = categories.most_common(1)[0][0] if categories else "none"
    connectivity = categories.get("2_connectivity_misunderstanding", 0)
    dead_ends = categories.get("1_dead_end_attraction", 0)
    geometry = categories.get("3_local_geometric_bias", 0)
    lines.extend([
        "",
        "## Observed Answers",
        "",
        f"1. In this selected high-impact subset, the most common high-confidence pattern is `{largest}`. Rows labelled `4_other_manual_review` intentionally remain unforced classifications.",
        f"2. The automatic categories identify {connectivity} connectivity/corridor, {geometry} local-geometry, and {dead_ends} dead-end-attraction events. These labels summarize trace and local geometry indicators; they do not establish causes.",
        "3. The structure table reports whether the selected events concentrate in particular benchmark structures; it should be interpreted within this impact-ranked subset rather than as a population estimate.",
        "4. Local MAE, correlation, and pairwise ranking accuracy are reported per event. They distinguish local value error from a wrong relative ordering, but neither alone establishes why the network failed.",
        "5. Cases where true-distance separation is large while U-Net reverses the pair are consistent with missing global connectivity or detour-cost information. This is an observed representation gap hypothesis, not a causal conclusion.",
        "",
        "## Caveats",
        "",
        "- `recovery_cost` and post-event extra expansions are trace-derived impact proxies; overlapping critical events can share downstream expansions.",
        "- The top-100 selection intentionally over-represents severe events and may include several events from one map.",
        "- Category 4 is retained for manual review instead of forcing a speculative mechanism label.",
        "",
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_path = os.path.join(project_root, args.source_events)
    output_dir = os.path.join(project_root, args.output_dir)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Missing high-impact event source: {source_path}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "case_contexts"), exist_ok=True)
    events = read_selected_events(source_path, args.limit)
    if not events:
        raise RuntimeError("No recovery_ge_20 A_unet_ordering_error events were found.")
    _, unet_model = load_models(project_root, args.checkpoint)
    selected, rows = [], []
    for rank, event in enumerate(events, start=1):
        context = reconstruct_event(event, unet_model)
        rows.append(output_row(event, context, rank))
        selected.append((rank, event, context))
        save_context(os.path.join(output_dir, "case_contexts", f"case_{rank:03d}.json"), event, context)
    write_csv(os.path.join(output_dir, "high_impact_cases.csv"), rows)
    make_summary_plots(output_dir, rows)
    make_case_figures(output_dir, selected)
    write_summary(os.path.join(output_dir, "summary.md"), rows)
    print(f"Saved {len(rows)} high-impact U-Net ordering-error case studies to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline high-impact U-Net ordering error case studies.")
    parser.add_argument("--checkpoint", default="compatible")
    parser.add_argument("--source-events", default=SOURCE_EVENTS)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
