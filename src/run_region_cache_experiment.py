"""Phase 2: cache local U-Net patch predictions during lazy tie-breaking.

This reuses the exact Phase 1 case manifest and full/no-cache baselines.  The
production A* code remains untouched; primary Manhattan-f ordering and the
secondary tie-break role remain unchanged.
"""

import argparse
import csv
import gc
import hashlib
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as functional

from model import grid_goal_tensor, load_unet_heuristic, make_unet_heuristic
from run_lazy_patch_experiment import (
    PATCH_SIZES,
    SIZES,
    STRUCTURES,
    lazy_tiebreak_search,
    mean,
    median,
    parse_coordinate,
    validate_lazy_semantics,
    write_csv,
)
from structured_maps import generate_structured_map


PATCH_VARIANTS = (32, 64)


def grid_hash(grid):
    text = "".join("".join(str(cell) for cell in row) for row in grid)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def load_phase1_cases(root):
    path = os.path.join(root, "outputs/lazy_patch_experiment/cases.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Phase 1 cases.csv is required so Phase 2 uses the unchanged dataset.")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cases = []
    for row in rows:
        size = int(row["map_size"])
        grid = generate_structured_map(size, size, int(row["seed"]), float(row["obstacle_rate"]), row["structure_type"])
        if grid_hash(grid) != row["grid_sha256"]:
            raise AssertionError(f"Phase 1 grid hash mismatch: {row['case_id']}")
        cases.append(
            {
                **row,
                "map_size": size,
                "optimal_cost": int(row["optimal_cost"]),
                "start_coord": parse_coordinate(row["start"]),
                "goal_coord": parse_coordinate(row["goal"]),
                "grid": grid,
            }
        )
    expected = {(size, structure) for size in SIZES for structure in STRUCTURES}
    observed = {(case["map_size"], case["structure_type"]) for case in cases}
    if observed != expected:
        raise AssertionError("Phase 1 case manifest does not contain one case for every required size/structure stratum.")
    return cases


class RegionCacheScorer:
    """Caches a full predicted patch map, keyed by the region that produced it."""

    def __init__(self, model, grid, goal, patch_size):
        self.model = model
        self.grid = grid
        self.goal = goal
        self.patch_size = patch_size
        self.radius = patch_size // 2
        self.device = next(model.parameters()).device
        obstacle_grid = torch.tensor(grid, dtype=torch.float32, device=self.device)
        self.padded_obstacles = functional.pad(obstacle_grid, (self.radius, self.radius, self.radius, self.radius), value=1.0)
        self.model_input = torch.zeros((1, 2, patch_size, patch_size), dtype=torch.float32, device=self.device)
        self.regions = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.patch_extraction_seconds = 0.0
        self.unet_inference_seconds = 0.0

    def cached_value(self, node):
        # Most recent region wins when patches overlap, making reuse deterministic.
        for region in reversed(self.regions):
            local_row = node[0] - region["top"]
            local_col = node[1] - region["left"]
            if 0 <= local_row < self.patch_size and 0 <= local_col < self.patch_size:
                self.cache_hits += 1
                return float(region["prediction"][local_row, local_col])
        return None

    def score(self, node):
        cached = self.cached_value(node)
        if cached is not None:
            return cached
        self.cache_misses += 1
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
            prediction = self.model(self.model_input).squeeze(0) * (2.0 * self.patch_size)
        self.unet_inference_seconds += time.perf_counter() - started
        prediction = prediction.detach().cpu().clone()
        self.regions.append({"top": top, "left": left, "prediction": prediction})
        return float(prediction[self.radius, self.radius])

    @property
    def query_count(self):
        return self.cache_hits + self.cache_misses

    @property
    def hit_rate(self):
        return self.cache_hits / self.query_count if self.query_count else 0.0


def load_phase1_baselines(root, cases):
    path = os.path.join(root, "outputs/lazy_patch_experiment/results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Phase 1 results.csv is required for unchanged full-map and no-cache baselines.")
    case_ids = {case["case_id"] for case in cases}
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["case_id"] in case_ids and row["algorithm"] in {"full_map_unet_tiebreak", "lazy_patch_32"}]
    expected = len(cases) * 2
    if len(selected) != expected:
        raise AssertionError(f"Expected {expected} Phase 1 baseline rows, found {len(selected)}.")
    return selected


def result_row(case, patch_size, result, scorer):
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
        "optimal_cost": case["optimal_cost"],
        "case_selection": case["case_selection"],
        "algorithm": f"region_cache_patch_{patch_size}",
        "patch_size": patch_size,
        "goal_marker_mode": "clamped_goal_position",
        "path_found": path_found,
        "path_cost": result["cost"],
        "path_cost_gap": result["cost"] - case["optimal_cost"] if path_found else "",
        "optimal": path_found and result["cost"] == case["optimal_cost"],
        "expanded_nodes": result["expanded"],
        "tie_break_query_count": scorer.query_count,
        "unet_call_count": scorer.cache_misses,
        "cache_hit_count": scorer.cache_hits,
        "cache_hit_rate": scorer.hit_rate,
        "cached_region_count": len(scorer.regions),
        "patch_extraction_seconds": scorer.patch_extraction_seconds,
        "unet_inference_seconds": scorer.unet_inference_seconds,
        "full_map_inference_seconds": 0.0,
        "astar_search_seconds": max(0.0, total - scorer.patch_extraction_seconds - scorer.unet_inference_seconds),
        "total_runtime_seconds": total,
    }


def normalize_baseline_row(row):
    full = row["algorithm"] == "full_map_unet_tiebreak"
    queries = int(float(row["patch_query_count"]))
    return {
        "case_id": row["case_id"],
        "map_size": int(row["map_size"]),
        "structure_type": row["structure_type"],
        "seed": int(row["seed"]),
        "obstacle_rate": float(row["obstacle_rate"]),
        "start": row["start"],
        "goal": row["goal"],
        "optimal_cost": int(row["optimal_cost"]),
        "case_selection": row["case_selection"],
        "algorithm": "full_map_unet_tiebreak" if full else "lazy_patch_32_no_cache",
        "patch_size": "full" if full else 32,
        "goal_marker_mode": row["goal_marker_mode"],
        "path_found": row["path_found"] == "True",
        "path_cost": int(row["path_cost"]),
        "path_cost_gap": float(row["path_cost_gap"]) if row["path_cost_gap"] else "",
        "optimal": row["optimal"] == "True",
        "expanded_nodes": int(row["expanded_nodes"]),
        "tie_break_query_count": queries,
        "unet_call_count": queries,
        "cache_hit_count": 0,
        "cache_hit_rate": 0.0,
        "cached_region_count": 0,
        "patch_extraction_seconds": float(row["patch_extraction_seconds"]),
        "unet_inference_seconds": float(row["unet_inference_seconds"]),
        "full_map_inference_seconds": float(row["full_map_inference_seconds"]),
        "astar_search_seconds": float(row["astar_search_seconds"]),
        "total_runtime_seconds": float(row["total_runtime_seconds"]),
    }


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
                "mean_tie_break_query_count": mean(float(row["tie_break_query_count"]) for row in group),
                "mean_unet_call_count": mean(float(row["unet_call_count"]) for row in group),
                "mean_cache_hit_rate": mean(float(row["cache_hit_rate"]) for row in group),
                "mean_cached_region_count": mean(float(row["cached_region_count"]) for row in group),
                "mean_patch_extraction_seconds": mean(float(row["patch_extraction_seconds"]) for row in group),
                "mean_unet_inference_seconds": mean(float(row["unet_inference_seconds"]) for row in group),
                "mean_astar_search_seconds": mean(float(row["astar_search_seconds"]) for row in group),
                "mean_total_runtime_seconds": mean(float(row["total_runtime_seconds"]) for row in group),
            }
        )
        output.append(item)
    return output


def lookup(summary, size, algorithm):
    return next(row for row in summary if row["map_size"] == size and row["algorithm"] == algorithm)


def write_report(output_dir, summary, selection):
    lines = [
        "# Phase 2 Region-Cache Lazy Patch U-Net",
        "",
        "Phase 2 uses exactly the Phase 1 case manifest and checkpoint. Manhattan primary-f ordering and U-Net secondary tie-break behavior are unchanged. Patch predictions are cached as full local maps; on a cache hit, the most recently created region covering the queried node supplies its predicted value without another U-Net call.",
        "",
        f"The benchmark retains the Phase 1 `{selection}`-difficulty feasibility sample: one deterministic case per size/structure stratum.",
        "",
        "| Size | Algorithm | Expanded | Optimality | Calls | Hit rate | Regions | U-Net s | A* s | Total s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['map_size']} | {row['algorithm']} | {row['mean_expanded_nodes']:.2f} | {row['optimality_rate']:.3f} | {row['mean_unet_call_count']:.1f} | {row['mean_cache_hit_rate']:.3f} | {row['mean_cached_region_count']:.1f} | {row['mean_unet_inference_seconds']:.3f} | {row['mean_astar_search_seconds']:.3f} | {row['mean_total_runtime_seconds']:.3f} |")
    lines += ["", "## Main Questions", ""]
    for size in SIZES:
        full = lookup(summary, size, "full_map_unet_tiebreak")
        raw = lookup(summary, size, "lazy_patch_32_no_cache")
        for patch_size in PATCH_VARIANTS:
            cache = lookup(summary, size, f"region_cache_patch_{patch_size}")
            call_reduction = 1.0 - cache["mean_unet_call_count"] / max(1.0, raw["mean_unet_call_count"])
            lines.append(f"- {size}x{size}, cache patch {patch_size}: calls reduced {call_reduction:.1%} versus no-cache patch 32; hit rate {cache['mean_cache_hit_rate']:.1%}; total runtime {cache['mean_total_runtime_seconds']:.3f}s versus full-map {full['mean_total_runtime_seconds']:.3f}s.")
    lines += [
        "",
        "A cache hit reuses the entire previously predicted local h map. It changes only how secondary U-Net values are obtained; Manhattan remains the primary key, so optimality must be interpreted from the measured paths rather than assumed from cache behavior alone.",
        "The single-case-per-stratum feasibility design does not establish general performance. Compare `results.csv` and the optional structure-level summary before drawing structure-wide conclusions.",
    ]
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 2 cached lazy-patch U-Net evaluation.")
    parser.add_argument("--checkpoint", default="outputs/ranking_lambda_sweep/lambda_0.5_best.pt")
    parser.add_argument("--output-dir", default="outputs/region_cache_experiment")
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint = os.path.join(root, args.checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output_dir = os.path.join(root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    cases = load_phase1_cases(root)
    baseline_rows = [normalize_baseline_row(row) for row in load_phase1_baselines(root, cases)]
    write_csv(os.path.join(output_dir, "cases.csv"), [{key: value for key, value in case.items() if key not in {"grid", "start_coord", "goal_coord"}} for case in cases])
    model = load_unet_heuristic(checkpoint)
    torch.set_num_threads(4)
    with torch.inference_mode():
        _ = model(grid_goal_tensor([[0] * 32 for _ in range(32)], (16, 16)).unsqueeze(0))
    rows = list(baseline_rows)
    for index, case in enumerate(cases, 1):
        if index == 1:
            table_heuristic = make_unet_heuristic(model, case["grid"], case["goal_coord"])
            table = {(row, col): table_heuristic((row, col), case["goal_coord"]) for row, values in enumerate(case["grid"]) for col, value in enumerate(values) if value == 0}
            validate_lazy_semantics(case["grid"], case["start_coord"], case["goal_coord"], table)
            del table
            gc.collect()
        for patch_size in PATCH_VARIANTS:
            scorer = RegionCacheScorer(model, case["grid"], case["goal_coord"], patch_size)
            result = lazy_tiebreak_search(case["grid"], case["start_coord"], case["goal_coord"], scorer)
            rows.append(result_row(case, patch_size, result, scorer))
        write_csv(os.path.join(output_dir, "results_partial.csv"), rows)
        print(f"Evaluated {index}/{len(cases)}: {case['case_id']}")
    summary = summarize(rows, ("map_size", "algorithm"))
    structure_summary = summarize(rows, ("map_size", "structure_type", "algorithm"))
    write_csv(os.path.join(output_dir, "results.csv"), rows)
    write_csv(os.path.join(output_dir, "summary.csv"), summary)
    write_csv(os.path.join(output_dir, "summary_by_structure.csv"), structure_summary)
    write_report(output_dir, summary, cases[0]["case_selection"])
    print(f"Saved {len(rows)} runs for {len(cases)} cases to {output_dir}")


if __name__ == "__main__":
    main()
