"""Phase 1: evaluate lazy local-patch U-Net scoring for Manhattan A* ties.

The production A* implementation remains unchanged.  This experiment uses the
same primary Manhattan-f ordering and secondary-key semantics, but delays a
patch query until a node first reaches the current minimum-f tie set.  That is
the earliest point at which the secondary value can affect an expansion.
"""

import argparse
import csv
import gc
import hashlib
import heapq
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as functional

from astar import astar_search
from model import grid_goal_tensor, load_unet_heuristic, make_unet_heuristic, manhattan_heuristic
from structured_maps import generate_structured_map


SIZES = (100, 500, 1000)
STRUCTURES = ("open_random", "maze_like", "bottleneck", "large_block", "narrow_corridor")
PATCH_SIZES = {100: (32, 64), 500: (32, 64, 128, 256), 1000: (32, 64, 128, 256)}


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def median(values):
    values = sorted(values)
    if not values:
        return 0.0
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grid_hash(grid):
    text = "".join("".join(str(cell) for cell in row) for row in grid)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def parse_coordinate(text):
    row, col = text.split(",")
    return int(row), int(col)


def load_cases(root, selection):
    path = os.path.join(root, "outputs/runtime_scaling/cases.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in manifest:
        size = int(row["map_size"])
        structure = row["structure_type"]
        if size in SIZES and structure in STRUCTURES:
            grouped[(size, structure)].append(row)
    cases = []
    for size in SIZES:
        for structure in STRUCTURES:
            candidates = sorted(grouped[(size, structure)], key=lambda row: int(row["optimal_cost"]))
            if not candidates:
                raise ValueError(f"No runtime-scaling case available for {size}/{structure}")
            if selection == "median":
                sample = candidates[len(candidates) // 2]
            elif selection == "lowest_cost":
                sample = candidates[0]
            else:
                raise ValueError(f"Unknown case selection: {selection}")
            grid = generate_structured_map(size, size, int(sample["seed"]), float(sample["obstacle_rate"]), structure)
            if grid_hash(grid) != sample["grid_sha256"]:
                raise AssertionError(f"Rebuilt grid does not match manifest: {sample['case_id']}")
            cases.append(
                {
                    **sample,
                    "map_size": size,
                    "optimal_cost": int(sample["optimal_cost"]),
                    "start_coord": parse_coordinate(sample["start"]),
                    "goal_coord": parse_coordinate(sample["goal"]),
                    "grid": grid,
                    "case_selection": selection,
                }
            )
    return cases


class PatchScorer:
    """Lazily calculates a fixed local secondary score once per node."""

    def __init__(self, model, grid, goal, patch_size):
        self.model = model
        self.grid = grid
        self.goal = goal
        self.patch_size = patch_size
        self.values = {}
        self.patch_extraction_seconds = 0.0
        self.unet_inference_seconds = 0.0
        self.query_count = 0
        self.radius = patch_size // 2
        self.device = next(model.parameters()).device
        obstacle_grid = torch.tensor(grid, dtype=torch.float32, device=self.device)
        # Each centered patch becomes a contiguous slice; out-of-map space is blocked.
        self.padded_obstacles = functional.pad(obstacle_grid, (self.radius, self.radius, self.radius, self.radius), value=1.0)
        self.model_input = torch.zeros((1, 2, patch_size, patch_size), dtype=torch.float32, device=self.device)

    def score(self, node):
        if node in self.values:
            return self.values[node]
        started = time.perf_counter()
        self.model_input[0, 0].copy_(self.padded_obstacles[node[0] : node[0] + self.patch_size, node[1] : node[1] + self.patch_size])
        top = node[0] - self.radius
        left = node[1] - self.radius
        goal_row = min(max(self.goal[0] - top, 0), self.patch_size - 1)
        goal_col = min(max(self.goal[1] - left, 0), self.patch_size - 1)
        self.patch_extraction_seconds += time.perf_counter() - started
        started = time.perf_counter()
        self.model_input[0, 1].zero_()
        self.model_input[0, 1, goal_row, goal_col] = 1.0
        with torch.inference_mode():
            prediction = self.model(self.model_input).squeeze(0)
            value = max(0.0, float(prediction[self.patch_size // 2, self.patch_size // 2]) * (2.0 * self.patch_size))
        self.unet_inference_seconds += time.perf_counter() - started
        self.values[node] = value
        self.query_count += 1
        return value


class FixedScorer:
    """Test-only scorer used to validate lazy tie semantics against astar.py."""

    def __init__(self, values):
        self.values = values
        self.patch_extraction_seconds = 0.0
        self.unet_inference_seconds = 0.0
        self.query_count = 0

    def score(self, node):
        self.query_count += int(node not in self.values.get("_queried", set()))
        self.values.setdefault("_queried", set()).add(node)
        return self.values[node]


def lazy_tiebreak_search(grid, start, goal, scorer):
    """A* with the existing (f_Manhattan, secondary, insertion, g, node) order.

    Secondary scores are deferred until their nodes are in the active minimum-f
    group.  Once calculated, the score is immutable, making the effective
    ordering equivalent to a static secondary heuristic while avoiding queries
    for nodes never relevant to an f tie.
    """
    started = time.perf_counter()
    rows = len(grid)
    cols = len(grid[0])
    open_heap = []
    insertion_counter = 0
    heapq.heappush(open_heap, (manhattan_heuristic(start, goal), insertion_counter, 0, start))
    g_score = {start: 0}
    came_from = {}
    expanded = 0
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    active_f = None
    active_raw = []
    active_secondary_heap = []
    secondary_values = {}

    def secondary_value(node):
        if node not in secondary_values:
            secondary_values[node] = scorer.score(node)
        return secondary_values[node]

    def activate_next_f_layer():
        """Move the next primary-f layer into an incremental secondary queue."""
        nonlocal active_f, active_raw, active_secondary_heap
        first = None
        while open_heap:
            candidate = heapq.heappop(open_heap)
            if candidate[2] == g_score.get(candidate[3]):
                first = candidate
                break
        if first is None:
            return False
        active_f = first[0]
        active_raw = [first]
        while open_heap and open_heap[0][0] == active_f:
            candidate = heapq.heappop(open_heap)
            if candidate[2] == g_score.get(candidate[3]):
                active_raw.append(candidate)
        active_secondary_heap = []
        if len(active_raw) > 1:
            for candidate in active_raw:
                _, insertion, candidate_g, node = candidate
                heapq.heappush(active_secondary_heap, (secondary_value(node), insertion, candidate_g, node))
            active_raw = []
        return True

    def add_to_active_layer(candidate):
        nonlocal active_raw
        if active_secondary_heap:
            _, insertion, candidate_g, node = candidate
            heapq.heappush(active_secondary_heap, (secondary_value(node), insertion, candidate_g, node))
            return
        active_raw.append(candidate)
        if len(active_raw) > 1:
            for raw_candidate in active_raw:
                _, insertion, candidate_g, node = raw_candidate
                heapq.heappush(active_secondary_heap, (secondary_value(node), insertion, candidate_g, node))
            active_raw = []

    while open_heap or active_raw or active_secondary_heap:
        if not active_raw and not active_secondary_heap:
            if not activate_next_f_layer():
                break
        if active_raw:
            # A sole primary-f candidate cannot be affected by h2.
            selected = active_raw.pop()
        else:
            selected = None
            while active_secondary_heap:
                _secondary, insertion, candidate_g, node = heapq.heappop(active_secondary_heap)
                if candidate_g == g_score.get(node):
                    selected = (active_f, insertion, candidate_g, node)
                    break
            if selected is None:
                active_f = None
                continue
        _, _, current_g, current = selected
        expanded += 1
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            total = time.perf_counter() - started
            return {"path": path, "cost": len(path) - 1, "expanded": expanded, "total_runtime_seconds": total}
        for delta_row, delta_col in directions:
            neighbor = (current[0] + delta_row, current[1] + delta_col)
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue
            if grid[neighbor[0]][neighbor[1]] == 1:
                continue
            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                insertion_counter += 1
                primary_f = tentative_g + manhattan_heuristic(neighbor, goal)
                candidate = (primary_f, insertion_counter, tentative_g, neighbor)
                if primary_f == active_f:
                    add_to_active_layer(candidate)
                else:
                    heapq.heappush(open_heap, candidate)
    total = time.perf_counter() - started
    return {"path": [], "cost": -1, "expanded": expanded, "total_runtime_seconds": total}


def full_map_table(model, grid, goal):
    started = time.perf_counter()
    heuristic = make_unet_heuristic(model, grid, goal)
    values = {
        (row, col): heuristic((row, col), goal)
        for row, cells in enumerate(grid)
        for col, value in enumerate(cells)
        if value == 0
    }
    return values, time.perf_counter() - started


def validate_lazy_semantics(grid, start, goal, table):
    expected = astar_search(grid, start, goal, manhattan_heuristic, secondary_heuristic=lambda node, _goal: table[node])
    observed = lazy_tiebreak_search(grid, start, goal, FixedScorer(dict(table)))
    if (expected["cost"], expected["expanded"], expected["path"]) != (observed["cost"], observed["expanded"], observed["path"]):
        raise AssertionError("Lazy tie ordering does not match astar.py for a fixed secondary table.")


def make_result(case, algorithm, patch_size, result, optimal_cost, patch_extraction, unet_inference, queries, full_map_inference=0.0):
    total = result["total_runtime_seconds"]
    path_found = result["cost"] >= 0
    return {
        "case_id": case["case_id"],
        "map_size": case["map_size"],
        "structure_type": case["structure_type"],
        "seed": case["seed"],
        "obstacle_rate": case["obstacle_rate"],
        "start": case["start"],
        "goal": case["goal"],
        "optimal_cost": optimal_cost,
        "case_selection": case["case_selection"],
        "algorithm": algorithm,
        "patch_size": patch_size,
        "goal_marker_mode": "clamped_goal_position",
        "path_found": path_found,
        "path_cost": result["cost"],
        "path_cost_gap": result["cost"] - optimal_cost if path_found else "",
        "optimal": path_found and result["cost"] == optimal_cost,
        "expanded_nodes": result["expanded"],
        "patch_query_count": queries,
        "patch_extraction_seconds": patch_extraction,
        "unet_inference_seconds": unet_inference,
        "full_map_inference_seconds": full_map_inference,
        "astar_search_seconds": max(0.0, total - patch_extraction - unet_inference - full_map_inference),
        "total_runtime_seconds": total,
    }


def evaluate_case(case, model, validate):
    grid = case["grid"]
    start = case["start_coord"]
    goal = case["goal_coord"]
    table, full_inference = full_map_table(model, grid, goal)
    if validate:
        validate_lazy_semantics(grid, start, goal, table)
    started = time.perf_counter()
    full_result = astar_search(grid, start, goal, manhattan_heuristic, secondary_heuristic=lambda node, _goal: table[node])
    full_result["total_runtime_seconds"] = full_inference + (time.perf_counter() - started)
    rows = [make_result(case, "full_map_unet_tiebreak", "full", full_result, case["optimal_cost"], 0.0, 0.0, 0, full_map_inference=full_inference)]
    del table
    gc.collect()
    for patch_size in PATCH_SIZES[case["map_size"]]:
        scorer = PatchScorer(model, grid, goal, patch_size)
        result = lazy_tiebreak_search(grid, start, goal, scorer)
        rows.append(make_result(case, f"lazy_patch_{patch_size}", patch_size, result, case["optimal_cost"], scorer.patch_extraction_seconds, scorer.unet_inference_seconds, scorer.query_count))
    return rows


def summarize(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        item = dict(zip(fields, key))
        item.update(
            {
                "cases": len(group),
                "mean_expanded_nodes": mean(float(row["expanded_nodes"]) for row in group),
                "median_expanded_nodes": median(float(row["expanded_nodes"]) for row in group),
                "optimality_rate": mean(float(row["optimal"]) for row in group),
                "mean_path_cost_gap": mean(float(row["path_cost_gap"]) for row in group if row["path_cost_gap"] != ""),
                "mean_patch_query_count": mean(float(row["patch_query_count"]) for row in group),
                "mean_patch_extraction_seconds": mean(float(row["patch_extraction_seconds"]) for row in group),
                "mean_unet_inference_seconds": mean(float(row["unet_inference_seconds"]) for row in group),
                "mean_astar_search_seconds": mean(float(row["astar_search_seconds"]) for row in group),
                "mean_total_runtime_seconds": mean(float(row["total_runtime_seconds"]) for row in group),
            }
        )
        output.append(item)
    return output


def find_summary(rows, size, patch_size):
    return next(row for row in rows if row["map_size"] == size and row["patch_size"] == patch_size)


def write_report(output_dir, by_map_size, selection):
    lines = [
        "# Phase 1 Lazy Patch U-Net",
        "",
        "This experiment keeps the trained U-Net, Manhattan A* primary ordering, and U-Net secondary tie-break semantics fixed. It replaces the full-map U-Net table with local U-Net patch scores only when a node first enters an active minimum-Manhattan-f tie set.",
        "",
        f"One deterministic `{selection}`-difficulty case from the existing runtime-scaling manifest is used per size/structure stratum. This is a Phase 1 feasibility sample, not a population-level estimate.",
        "",
        "Patch inputs are centered on the queried node. Outside-map cells are obstacles; the global goal marker is clamped to the patch boundary when outside the window, preserving direction in the original two-channel model input.",
        "",
        "| Size | Patch | Expanded | Optimality | Cost gap | Queries | Patch extract s | U-Net s | A* s | Total s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_map_size:
        lines.append(f"| {row['map_size']} | {row['patch_size']} | {row['mean_expanded_nodes']:.2f} | {row['optimality_rate']:.3f} | {row['mean_path_cost_gap']:.3f} | {row['mean_patch_query_count']:.1f} | {row['mean_patch_extraction_seconds']:.3f} | {row['mean_unet_inference_seconds']:.3f} | {row['mean_astar_search_seconds']:.3f} | {row['mean_total_runtime_seconds']:.3f} |")
    lines += ["", "## Main Questions", ""]
    for size in SIZES:
        full = find_summary(by_map_size, size, "full")
        for patch_size in PATCH_SIZES[size]:
            patch = find_summary(by_map_size, size, patch_size)
            lines.append(f"- {size}x{size}, patch {patch_size}: expansion change versus full map = {patch['mean_expanded_nodes'] - full['mean_expanded_nodes']:.2f}; total-runtime change = {patch['mean_total_runtime_seconds'] - full['mean_total_runtime_seconds']:.3f}s; optimality = {patch['optimality_rate']:.3f}.")
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    for size in SIZES:
        full = find_summary(by_map_size, size, "full")
        patch32 = find_summary(by_map_size, size, 32)
        expansion_change = (patch32["mean_expanded_nodes"] - full["mean_expanded_nodes"]) / max(1.0, full["mean_expanded_nodes"])
        runtime_change = (patch32["mean_total_runtime_seconds"] - full["mean_total_runtime_seconds"]) / max(1e-9, full["mean_total_runtime_seconds"])
        lines.append(f"- Patch 32 at {size}x{size}: expansion change {expansion_change:.1%}; total-runtime change {runtime_change:.1%} relative to full-map U-Net.")
    lines += [
        "- In this feasibility sample, all variants retained optimal paths. Patch 32 is the only tested patch scale that can be evaluated as a potentially cheaper replacement; larger patches perform a separate U-Net forward for many tie candidates and therefore accumulate cost.",
        "- The result does not show that global context is unnecessary in general. Patch 32 changes both receptive field and effective goal representation, while larger patch sizes change the input distribution seen by the fixed 20x20-trained model. The one-case-per-stratum sample is insufficient to separate those effects.",
        "",
        "A patch is useful only if its total runtime falls below the full-map baseline while retaining acceptable expansion and optimality behavior. Since the scorer has local obstacle context plus a projected goal direction but not a full global map, performance degradation is evidence for missing global context; this small Phase 1 sample does not establish causality.",
    ]
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 1 lazy patch U-Net tie-break evaluation.")
    parser.add_argument("--checkpoint", default="outputs/ranking_lambda_sweep/lambda_0.5_best.pt")
    parser.add_argument("--case-selection", choices=("median", "lowest_cost"), default="median")
    parser.add_argument("--output-dir", default="outputs/lazy_patch_experiment")
    parser.add_argument("--skip-semantics-validation", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def read_map_size_summary(path):
    numeric_fields = {
        "map_size": int,
        "cases": int,
        "mean_expanded_nodes": float,
        "median_expanded_nodes": float,
        "optimality_rate": float,
        "mean_path_cost_gap": float,
        "mean_patch_query_count": float,
        "mean_patch_extraction_seconds": float,
        "mean_unet_inference_seconds": float,
        "mean_astar_search_seconds": float,
        "mean_total_runtime_seconds": float,
    }
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field, converter in numeric_fields.items():
            row[field] = converter(row[field])
        if row["patch_size"] != "full":
            row["patch_size"] = int(row["patch_size"])
    return rows


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint = os.path.join(root, args.checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output_dir = os.path.join(root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if args.report_only:
        write_report(output_dir, read_map_size_summary(os.path.join(output_dir, "summary_by_map_size.csv")), args.case_selection)
        return
    model = load_unet_heuristic(checkpoint)
    torch.set_num_threads(4)
    # Warm the fixed model once outside all recorded search timings.
    warm_grid = [[0] * 32 for _ in range(32)]
    with torch.inference_mode():
        _ = model(grid_goal_tensor(warm_grid, (16, 16)).unsqueeze(0))
    cases = load_cases(root, args.case_selection)
    manifest_rows = [{key: value for key, value in case.items() if key not in {"grid", "start_coord", "goal_coord"}} for case in cases]
    write_csv(os.path.join(output_dir, "cases.csv"), manifest_rows)
    rows = []
    for index, case in enumerate(cases, 1):
        rows.extend(evaluate_case(case, model, validate=not args.skip_semantics_validation and index == 1))
        # Preserve completed long-running cases if a later case or summary fails.
        write_csv(os.path.join(output_dir, "results_partial.csv"), rows)
        print(f"Evaluated {index}/{len(cases)}: {case['case_id']}")
    by_patch = summarize(rows, ("patch_size",))
    by_map_size = summarize(rows, ("map_size", "patch_size"))
    by_structure = summarize(rows, ("map_size", "structure_type", "patch_size"))
    write_csv(os.path.join(output_dir, "results.csv"), rows)
    write_csv(os.path.join(output_dir, "summary_by_patch_size.csv"), by_patch)
    write_csv(os.path.join(output_dir, "summary_by_map_size.csv"), by_map_size)
    write_csv(os.path.join(output_dir, "summary_by_structure.csv"), by_structure)
    write_report(output_dir, by_map_size, args.case_selection)
    print(f"Saved {len(rows)} runs for {len(cases)} cases to {output_dir}")


if __name__ == "__main__":
    main()
