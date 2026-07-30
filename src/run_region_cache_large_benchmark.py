"""Large-scale validation for cached local U-Net tie-breaking.

The experiment deliberately keeps Manhattan ``f`` as the primary A* key.  It
reuses the Phase 1 lazy tie-set evaluator, whose fixed-secondary behavior is
validated against ``astar.py``.  This file only adds benchmark orchestration,
timing, and resumable result persistence; production search code is unchanged.
"""

import argparse
import csv
import hashlib
import os
import random
import time
from collections import defaultdict

import torch

from astar import astar_search
from bfs_label import compute_distance_to_goal
from model import grid_goal_tensor, load_unet_heuristic, manhattan_heuristic
from run_lazy_patch_experiment import PatchScorer, lazy_tiebreak_search, mean, median, validate_lazy_semantics, write_csv
from run_region_cache_experiment import RegionCacheScorer
from structured_maps import generate_structured_map


DEFAULT_SIZES = (100, 500, 1000)
DEFAULT_STRUCTURES = ("open_random", "maze_like", "bottleneck", "large_block", "narrow_corridor")
OBSTACLE_RATES = (0.1, 0.2, 0.3, 0.4)
BASE_SEED = 20_260_728
FULL_ALGORITHM = "full_map_unet_tiebreak"
NO_CACHE_ALGORITHM = "lazy_patch_32_no_cache"
CACHE_32_ALGORITHM = "region_cache_patch_32"
CACHE_64_ALGORITHM = "region_cache_patch_64"
ALL_ALGORITHMS = (FULL_ALGORITHM, NO_CACHE_ALGORITHM, CACHE_32_ALGORITHM, CACHE_64_ALGORITHM)


def select_device(requested_device):
    """Return an explicitly requested inference device without a silent fallback."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested_device)


def cuda_inference_seconds(model, operation):
    """Time one CUDA model forward with CUDA events, not a host wall clock."""
    device = next(model.parameters()).device
    if device.type != "cuda":
        started = time.perf_counter()
        return operation(), time.perf_counter() - started

    # Synchronizing on both sides makes the event interval independent of
    # preceding asynchronous preparation and of later host-side consumption.
    torch.cuda.synchronize(device)
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    value = operation()
    finished.record()
    torch.cuda.synchronize(device)
    return value, started.elapsed_time(finished) / 1000.0


def hardware_info(device):
    is_cuda = device.type == "cuda"
    return {
        "device": str(device),
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if is_cuda else "",
        "cuda_version": torch.version.cuda or "",
        "pytorch_version": torch.__version__,
    }

def grid_hash(grid):
    text = "".join("".join(str(cell) for cell in row) for row in grid)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def parse_coordinate(text):
    row, col = text.split(",")
    return int(row), int(col)


def free_cells(grid):
    return [(row, col) for row, cells in enumerate(grid) for col, value in enumerate(cells) if value == 0]


def choose_solvable_pair(grid, seed):
    cells = free_cells(grid)
    rng = random.Random(seed * 1_000_003 + len(grid) * 1009)
    # One BFS defines a start's connected component. Sampling its goal from
    # that component guarantees solvability and avoids doing up to 200 full
    # reverse BFS passes on a disconnected large map.
    for _ in range(8):
        start = rng.choice(cells)
        distance_from_start = compute_distance_to_goal(grid, start)
        reachable = [
            (row, col)
            for row, values in enumerate(distance_from_start)
            for col, value in enumerate(values)
            if value > 0
        ]
        if reachable:
            goal = rng.choice(reachable)
            return start, goal, distance_from_start[goal[0]][goal[1]]
    return None, None, None


def load_or_create_cases(output_dir, cases_per_structure, sizes, structures, regenerate):
    path = os.path.join(output_dir, "cases.csv")
    expected_count = cases_per_structure * len(sizes) * len(structures)
    if os.path.exists(path) and not regenerate:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["map_size"] = int(row["map_size"])
            row["seed"] = int(row["seed"])
            row["obstacle_rate"] = float(row["obstacle_rate"])
            row["optimal_cost"] = int(row["optimal_cost"])
    else:
        rows = []
    if regenerate:
        rows = []

    existing_ids = {row["case_id"] for row in rows}
    if len(rows) > expected_count:
        raise ValueError(f"Existing {path} has more cases than this requested benchmark configuration.")
    # Generate deterministically from the first candidate each time. If a prior
    # process was interrupted, existing case IDs are skipped and every newly
    # accepted case is written immediately, so no completed map is lost.
    for size in sizes:
        for structure_index, structure in enumerate(structures):
            stratum_rows = [row for row in rows if row["map_size"] == size and row["structure_type"] == structure]
            if len(stratum_rows) > cases_per_structure:
                raise ValueError(f"Existing manifest has duplicate cases for {size}/{structure}.")
            accepted = len(stratum_rows)
            seed_base = BASE_SEED + size * 1_000_000 + structure_index * 100_000
            # Stored seeds encode the attempted-map offset. Resume after the
            # last completed candidate rather than recomputing its BFS label.
            attempt = max((row["seed"] - seed_base + 1 for row in stratum_rows), default=0)
            while accepted < cases_per_structure:
                obstacle_rate = OBSTACLE_RATES[accepted % len(OBSTACLE_RATES)]
                seed = seed_base + attempt
                grid = generate_structured_map(size, size, seed, obstacle_rate, structure)
                start, goal, optimal_cost = choose_solvable_pair(grid, seed)
                attempt += 1
                if start is None:
                    continue
                case = {
                    "case_id": f"size{size}_{structure}_seed{seed}",
                    "map_size": size,
                    "structure_type": structure,
                    "seed": seed,
                    "obstacle_rate": obstacle_rate,
                    "grid_sha256": grid_hash(grid),
                    "start": f"{start[0]},{start[1]}",
                    "goal": f"{goal[0]},{goal[1]}",
                    "optimal_cost": optimal_cost,
                }
                accepted += 1
                if case["case_id"] not in existing_ids:
                    rows.append(case)
                    existing_ids.add(case["case_id"])
                    write_csv(path, rows)
                    print(f"Generated {len(rows)}/{expected_count} cases: {case['case_id']}", flush=True)
    if len(rows) != expected_count:
        raise AssertionError(f"Generated {len(rows)} cases; expected {expected_count}.")
    return rows


def materialize_case(case):
    grid = generate_structured_map(
        case["map_size"], case["map_size"], case["seed"], case["obstacle_rate"], case["structure_type"]
    )
    if grid_hash(grid) != case["grid_sha256"]:
        raise AssertionError(f"Grid hash mismatch for {case['case_id']}")
    return grid, parse_coordinate(case["start"]), parse_coordinate(case["goal"])


class TimedRegionCacheScorer(RegionCacheScorer):
    """Region cache with O(1) lookup and Phase 2's most-recent-region rule."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_lookup_seconds = 0.0
        self._cell_cache = {}

    def cached_value(self, node):
        started = time.perf_counter()
        value = self._cell_cache.get(node)
        self.cache_lookup_seconds += time.perf_counter() - started
        if value is not None:
            self.cache_hits += 1
        return value

    def score(self, node):
        cached = self.cached_value(node)
        if cached is not None:
            return cached
        self.cache_misses += 1
        started = time.perf_counter()
        self.model_input[0, 0].copy_(
            self.padded_obstacles[node[0] : node[0] + self.patch_size, node[1] : node[1] + self.patch_size]
        )
        top = node[0] - self.radius
        left = node[1] - self.radius
        goal_row = min(max(self.goal[0] - top, 0), self.patch_size - 1)
        goal_col = min(max(self.goal[1] - left, 0), self.patch_size - 1)
        self.model_input[0, 1].zero_()
        self.model_input[0, 1, goal_row, goal_col] = 1.0
        self.patch_extraction_seconds += time.perf_counter() - started
        with torch.inference_mode():
            prediction, inference_seconds = cuda_inference_seconds(
                self.model,
                lambda: self.model(self.model_input).squeeze(0) * (2.0 * self.patch_size),
            )
        self.unet_inference_seconds += inference_seconds
        prediction = prediction.detach().cpu().clone()
        self.regions.append({"top": top, "left": left, "prediction": prediction})
        # Writing every in-map coordinate preserves "most recent region wins"
        # on overlap while eliminating repeated scans over all cached regions.
        started = time.perf_counter()
        rows, cols = len(self.grid), len(self.grid[0])
        for local_row in range(self.patch_size):
            global_row = top + local_row
            if not 0 <= global_row < rows:
                continue
            for local_col in range(self.patch_size):
                global_col = left + local_col
                if 0 <= local_col + left < cols:
                    self._cell_cache[(global_row, left + local_col)] = float(prediction[local_row, local_col])
        self.patch_extraction_seconds += time.perf_counter() - started
        return float(prediction[self.radius, self.radius])


class BatchedPatchScorer(PatchScorer):
    """No-cache patch scorer that batches independent active-tie queries.

    Every node retains its own centered patch prediction and no value is reused
    for another node. The only optimization is combining multiple patches from
    the same active f-layer into one U-Net forward call.
    """

    def __init__(self, *args, batch_size=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.unet_forward_count = 0

    def score_many(self, nodes):
        output = {}
        missing = []
        seen = set()
        for node in nodes:
            if node in self.values:
                output[node] = self.values[node]
            elif node not in seen:
                missing.append(node)
                seen.add(node)
        for offset in range(0, len(missing), self.batch_size):
            batch_nodes = missing[offset : offset + self.batch_size]
            started = time.perf_counter()
            batch = torch.zeros(
                (len(batch_nodes), 2, self.patch_size, self.patch_size),
                dtype=torch.float32,
                device=self.device,
            )
            for index, node in enumerate(batch_nodes):
                batch[index, 0].copy_(
                    self.padded_obstacles[node[0] : node[0] + self.patch_size, node[1] : node[1] + self.patch_size]
                )
                top = node[0] - self.radius
                left = node[1] - self.radius
                goal_row = min(max(self.goal[0] - top, 0), self.patch_size - 1)
                goal_col = min(max(self.goal[1] - left, 0), self.patch_size - 1)
                batch[index, 1, goal_row, goal_col] = 1.0
            self.patch_extraction_seconds += time.perf_counter() - started
            if self.device.type == "cuda":
                with torch.inference_mode():
                    predictions, inference_seconds = cuda_inference_seconds(
                        self.model,
                        lambda: self.model(batch) * (2.0 * self.patch_size),
                    )
                self.unet_inference_seconds += inference_seconds
                predictions = predictions.detach().cpu()
            else:
                started = time.perf_counter()
                with torch.inference_mode():
                    predictions = self.model(batch).detach().cpu() * (2.0 * self.patch_size)
                self.unet_inference_seconds += time.perf_counter() - started
            self.unet_forward_count += 1
            for index, node in enumerate(batch_nodes):
                value = max(0.0, float(predictions[index, self.radius, self.radius]))
                self.values[node] = value
                output[node] = value
            self.query_count += len(batch_nodes)
        return {node: output.get(node, self.values[node]) for node in nodes}

    def score(self, node):
        return self.score_many([node])[node]


def full_map_prediction(model, grid, goal):
    """Produce the same full-grid h values as ``make_unet_heuristic``.

    Tensor construction is vectorized here because it is part of the baseline
    inference overhead on 500x500 and 1000x1000 maps.  It does not change the
    model, normalization, or values supplied to the tie-break.
    """
    started = time.perf_counter()
    device = next(model.parameters()).device
    if device.type == "cuda":
        rows, cols = len(grid), len(grid[0])
        model_input = torch.zeros((1, 2, rows, cols), dtype=torch.float32, device=device)
        model_input[0, 0].copy_(torch.tensor(grid, dtype=torch.float32, device=device))
        model_input[0, 1, goal[0], goal[1]] = 1.0
        with torch.inference_mode():
            prediction, inference_seconds = cuda_inference_seconds(
                model,
                lambda: model(model_input).squeeze(0) * float(rows + cols),
            )
        return prediction.detach().cpu(), inference_seconds

    rows, cols = len(grid), len(grid[0])
    model_input = torch.zeros((1, 2, rows, cols), dtype=torch.float32, device=device)
    model_input[0, 0].copy_(torch.tensor(grid, dtype=torch.float32, device=device))
    model_input[0, 1, goal[0], goal[1]] = 1.0
    with torch.inference_mode():
        prediction = (model(model_input).squeeze(0) * float(rows + cols)).detach().cpu()
    return prediction, time.perf_counter() - started


def prediction_heuristic(prediction):
    def heuristic(node, _goal):
        return max(0.0, float(prediction[node[0], node[1]]))

    return heuristic


def base_row(case, algorithm):
    return {
        "case_id": case["case_id"],
        "map_size": case["map_size"],
        "structure_type": case["structure_type"],
        "seed": case["seed"],
        "obstacle_rate": case["obstacle_rate"],
        "grid_sha256": case["grid_sha256"],
        "start": case["start"],
        "goal": case["goal"],
        "optimal_cost": case["optimal_cost"],
        "algorithm": algorithm,
    }


def build_result(case, algorithm, result, optimal_cost, *, calls, scored_nodes, hits, regions, patch_extract, cache_lookup, unet_inference, full_inference):
    path_found = result["cost"] >= 0
    total = result["total_runtime_seconds"]
    row = base_row(case, algorithm)
    row.update(
        {
            "path_found": path_found,
            "path_cost": result["cost"],
            "path_cost_gap": result["cost"] - optimal_cost if path_found else "",
            "optimal": path_found and result["cost"] == optimal_cost,
            "expanded_nodes": result["expanded"],
            "unet_call_count": calls,
            "unet_scored_node_count": scored_nodes,
            "cache_hit_count": hits,
            "cache_hit_rate": hits / (hits + calls) if hits + calls else 0.0,
            "cached_region_count": regions,
            "patch_extraction_seconds": patch_extract,
            "cache_lookup_seconds": cache_lookup,
            "unet_inference_seconds": unet_inference,
            "full_map_inference_seconds": full_inference,
            "astar_search_seconds": max(0.0, total - patch_extract - cache_lookup - unet_inference - full_inference),
            "total_runtime_seconds": total,
        }
    )
    return row


def run_full_map(case, grid, start, goal, model):
    prediction, inference_seconds = full_map_prediction(model, grid, goal)
    started = time.perf_counter()
    search = astar_search(grid, start, goal, manhattan_heuristic, secondary_heuristic=prediction_heuristic(prediction))
    search["total_runtime_seconds"] = inference_seconds + (time.perf_counter() - started)
    return build_result(
        case,
        FULL_ALGORITHM,
        search,
        case["optimal_cost"],
        calls=1,
        scored_nodes=0,
        hits=0,
        regions=0,
        patch_extract=0.0,
        cache_lookup=0.0,
        unet_inference=0.0,
        full_inference=inference_seconds,
    )


def run_no_cache(case, grid, start, goal, model):
    scorer = BatchedPatchScorer(model, grid, goal, 32)
    search = lazy_tiebreak_search(grid, start, goal, scorer)
    return build_result(
        case,
        NO_CACHE_ALGORITHM,
        search,
        case["optimal_cost"],
        calls=scorer.unet_forward_count,
        scored_nodes=scorer.query_count,
        hits=0,
        regions=0,
        patch_extract=scorer.patch_extraction_seconds,
        cache_lookup=0.0,
        unet_inference=scorer.unet_inference_seconds,
        full_inference=0.0,
    )


def run_region_cache(case, grid, start, goal, model, patch_size):
    scorer = TimedRegionCacheScorer(model, grid, goal, patch_size)
    search = lazy_tiebreak_search(grid, start, goal, scorer)
    return build_result(
        case,
        f"region_cache_patch_{patch_size}",
        search,
        case["optimal_cost"],
        calls=scorer.cache_misses,
        scored_nodes=scorer.query_count,
        hits=scorer.cache_hits,
        regions=len(scorer.regions),
        patch_extract=scorer.patch_extraction_seconds,
        cache_lookup=scorer.cache_lookup_seconds,
        unet_inference=scorer.unet_inference_seconds,
        full_inference=0.0,
    )


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # New rows retain native numeric types during a process; resumed CSV rows do
    # not. Normalize the grouping and sorting keys before combining them.
    for row in rows:
        row["map_size"] = int(row["map_size"])
        row["seed"] = int(row["seed"])
        row.setdefault("unet_scored_node_count", row.get("unet_call_count", 0))
    return rows


def as_number(value):
    if value == "":
        return 0.0
    return float(value)


def as_bool(value):
    return value is True or value == "True" or value == "true" or value == "1"


def summarize(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    summary = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        item = dict(zip(fields, key))
        item.update(
            {
                "cases": len(group),
                "mean_expanded_nodes": mean(as_number(row["expanded_nodes"]) for row in group),
                "median_expanded_nodes": median(as_number(row["expanded_nodes"]) for row in group),
                "optimality_rate": mean(float(as_bool(row["optimal"])) for row in group),
                "mean_path_cost_gap": mean(as_number(row["path_cost_gap"]) for row in group),
                "mean_unet_call_count": mean(as_number(row["unet_call_count"]) for row in group),
                "mean_unet_scored_node_count": mean(as_number(row["unet_scored_node_count"]) for row in group),
                "mean_cache_hit_rate": mean(as_number(row["cache_hit_rate"]) for row in group),
                "mean_cached_region_count": mean(as_number(row["cached_region_count"]) for row in group),
                "mean_patch_extraction_seconds": mean(as_number(row["patch_extraction_seconds"]) for row in group),
                "mean_cache_lookup_seconds": mean(as_number(row["cache_lookup_seconds"]) for row in group),
                "mean_unet_inference_seconds": mean(as_number(row["unet_inference_seconds"]) for row in group),
                "mean_full_map_inference_seconds": mean(as_number(row["full_map_inference_seconds"]) for row in group),
                "mean_astar_search_seconds": mean(as_number(row["astar_search_seconds"]) for row in group),
                "mean_total_runtime_seconds": mean(as_number(row["total_runtime_seconds"]) for row in group),
            }
        )
        summary.append(item)
    return summary


def summary_lookup(rows, filters):
    return next(row for row in rows if all(row[field] == value for field, value in filters.items()))


def speedup_rows(rows, grouping):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in grouping)].append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        by_algorithm = {row["algorithm"]: row for row in group}
        full = by_algorithm.get(FULL_ALGORITHM)
        if full is None:
            continue
        for algorithm, row in by_algorithm.items():
            if algorithm == FULL_ALGORITHM:
                continue
            full_total = full["mean_total_runtime_seconds"]
            full_expanded = full["mean_expanded_nodes"]
            item = dict(zip(grouping, key))
            item.update(
                {
                    "algorithm": algorithm,
                    "full_map_mean_total_runtime_seconds": full_total,
                    "variant_mean_total_runtime_seconds": row["mean_total_runtime_seconds"],
                    "runtime_speedup_vs_full": full_total / max(row["mean_total_runtime_seconds"], 1e-12),
                    "runtime_reduction_fraction_vs_full": 1.0 - row["mean_total_runtime_seconds"] / max(full_total, 1e-12),
                    "full_map_mean_expanded_nodes": full_expanded,
                    "variant_mean_expanded_nodes": row["mean_expanded_nodes"],
                    "expansion_reduction_fraction_vs_full": 1.0 - row["mean_expanded_nodes"] / max(full_expanded, 1e-12),
                    "unet_forward_change_fraction_vs_full": row["mean_unet_call_count"] / max(full["mean_unet_call_count"], 1e-12) - 1.0,
                }
            )
            output.append(item)
    return output


def write_report(output_dir, by_size, by_structure, speedups, cases_per_structure, sizes, structures, algorithms):
    total_cases = cases_per_structure * len(sizes) * len(structures)
    total_runs = total_cases * len(algorithms)
    lines = [
        "# Large-Scale Region-Cache Lazy Patch U-Net Benchmark",
        "",
        f"This benchmark contains {total_cases} deterministic, solvable maps ({cases_per_structure} per size/structure stratum) and {total_runs} algorithm runs. It keeps Manhattan `f=g+h_manhattan` as the primary order; U-Net values are only secondary tie-break keys.",
        "",
        "The full-map baseline materializes a complete U-Net prediction once per map. Lazy patch variants score only nodes in active minimum-Manhattan-f tie sets. The no-cache variant may batch independent patches from one tie set, but never reuses a prediction between nodes; `Forwards` is the actual model-call count and `Scored nodes` is the number of distinct patch scores. Region-cache variants reuse the most recently computed local h map covering a query. All timings include search plus their stated neural preparation costs.",
        "",
        "## By Map Size",
        "",
        "| Size | Algorithm | Expanded | Optimality | Forwards | Scored nodes | Hit rate | Regions | Full map s | Patch s | Lookup s | A* s | Total s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_size:
        lines.append(
            f"| {row['map_size']} | {row['algorithm']} | {row['mean_expanded_nodes']:.2f} | {row['optimality_rate']:.3f} | "
            f"{row['mean_unet_call_count']:.1f} | {row['mean_unet_scored_node_count']:.1f} | {row['mean_cache_hit_rate']:.3f} | {row['mean_cached_region_count']:.1f} | "
            f"{row['mean_full_map_inference_seconds']:.4f} | {row['mean_unet_inference_seconds']:.4f} | "
            f"{row['mean_cache_lookup_seconds']:.4f} | {row['mean_astar_search_seconds']:.4f} | {row['mean_total_runtime_seconds']:.4f} |"
        )
    lines += ["", "## Answers", ""]
    for size in sizes:
        try:
            full = summary_lookup(by_size, {"map_size": size, "algorithm": FULL_ALGORITHM})
        except StopIteration:
            continue
        for algorithm in algorithms:
            if algorithm == FULL_ALGORITHM:
                continue
            variant = summary_lookup(by_size, {"map_size": size, "algorithm": algorithm})
            speedup = full["mean_total_runtime_seconds"] / max(variant["mean_total_runtime_seconds"], 1e-12)
            expansion = 1.0 - variant["mean_expanded_nodes"] / max(full["mean_expanded_nodes"], 1e-12)
            lines.append(
                f"- {size}x{size}, `{algorithm}`: total-runtime speedup {speedup:.2f}x versus full-map; "
                f"expansion change {expansion:+.1%}; optimality {variant['optimality_rate']:.1%}."
            )
    lines += [
        "",
        "`summary_by_structure.csv` and `speedup_analysis.csv` give the corresponding structure-level comparisons. Reported optimality is measured against reverse-BFS cost on each saved map; it is not inferred from the Manhattan primary key.",
        "",
        "## Interpretation Limits",
        "",
        "The cache changes which local patch prediction supplies an overlapping node's secondary value, so it can legitimately change tie-break trajectories and expansion counts. Results establish measured benchmark associations, not that cache reuse or local context alone causes a given outcome. The U-Net was trained on smaller inputs; patch and full-map inputs therefore remain out-of-distribution at these scales.",
    ]
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def validate_semantics(model):
    grid = [[0] * 20 for _ in range(20)]
    grid[6][4:16] = [1] * 12
    goal = (18, 18)
    prediction, _ = full_map_prediction(model, grid, goal)
    table = {(row, col): prediction_heuristic(prediction)((row, col), goal) for row, cells in enumerate(grid) for col, value in enumerate(cells) if value == 0}
    validate_lazy_semantics(grid, (1, 1), goal, table)
    serial = PatchScorer(model, grid, goal, 32)
    batched = BatchedPatchScorer(model, grid, goal, 32)
    serial_result = lazy_tiebreak_search(grid, (1, 1), goal, serial)
    batched_result = lazy_tiebreak_search(grid, (1, 1), goal, batched)
    if (serial_result["cost"], serial_result["expanded"], serial_result["path"]) != (
        batched_result["cost"], batched_result["expanded"], batched_result["path"]
    ):
        raise AssertionError("Batched no-cache patch scoring changed the validated serial patch search trace.")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate region-cache lazy patch U-Net on a large structured benchmark.")
    parser.add_argument("--checkpoint", default="outputs/ranking_lambda_sweep/lambda_0.5_best.pt")
    parser.add_argument("--output-dir", default="outputs/region_cache_large_benchmark")
    parser.add_argument("--cases-per-structure", type=int, default=100)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--structures", nargs="+", default=DEFAULT_STRUCTURES)
    parser.add_argument("--algorithms", nargs="+", choices=ALL_ALGORITHMS)
    parser.add_argument("--skip-patch-64", action="store_true")
    parser.add_argument("--regenerate-cases", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="U-Net inference device; default cpu preserves the existing benchmark path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cases_per_structure < 1:
        raise ValueError("--cases-per-structure must be positive")
    sizes = tuple(args.sizes)
    structures = tuple(args.structures)
    if any(size % 4 for size in sizes):
        raise ValueError("All map sizes must be divisible by four for the fixed U-Net.")
    if any(structure not in DEFAULT_STRUCTURES for structure in structures):
        raise ValueError(f"--structures must be chosen from {DEFAULT_STRUCTURES}")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = select_device(args.device)
    write_csv(os.path.join(output_dir, "hardware_info.csv"), [hardware_info(device)])
    default_algorithms = [FULL_ALGORITHM, NO_CACHE_ALGORITHM, CACHE_32_ALGORITHM]
    if not args.skip_patch_64:
        default_algorithms.append(CACHE_64_ALGORITHM)
    algorithms = args.algorithms or default_algorithms
    if args.report_only:
        rows = load_rows(os.path.join(output_dir, "results.csv"))
        if not rows:
            raise FileNotFoundError("results.csv is required for --report-only")
        by_size = summarize(rows, ("map_size", "algorithm"))
        by_structure = summarize(rows, ("map_size", "structure_type", "algorithm"))
        speedups = speedup_rows(by_structure, ("map_size", "structure_type"))
        write_csv(os.path.join(output_dir, "summary_by_size.csv"), by_size)
        write_csv(os.path.join(output_dir, "summary_by_structure.csv"), by_structure)
        write_csv(os.path.join(output_dir, "speedup_analysis.csv"), speedups)
        write_report(output_dir, by_size, by_structure, speedups, args.cases_per_structure, sizes, structures, algorithms)
        return

    checkpoint = os.path.join(root, args.checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    torch.set_num_threads(4)
    model = load_unet_heuristic(checkpoint, device=device)
    with torch.inference_mode():
        _ = model(grid_goal_tensor([[0] * 32 for _ in range(32)], (16, 16), device=device).unsqueeze(0))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    validate_semantics(model)
    cases = load_or_create_cases(output_dir, args.cases_per_structure, sizes, structures, args.regenerate_cases)
    partial_path = os.path.join(output_dir, "results_partial.csv")
    rows = load_rows(partial_path)
    expected_keys = {(case["case_id"], algorithm) for case in cases for algorithm in algorithms}
    valid_case_ids = {case["case_id"] for case in cases}
    rows = [row for row in rows if row["case_id"] in valid_case_ids and row["algorithm"] in ALL_ALGORITHMS]
    complete = {(row["case_id"], row["algorithm"]) for row in rows}
    total_runs = len(expected_keys)
    for case_index, case in enumerate(cases, 1):
        needed = [algorithm for algorithm in algorithms if (case["case_id"], algorithm) not in complete]
        if not needed:
            continue
        grid, start, goal = materialize_case(case)
        for algorithm in needed:
            if algorithm == FULL_ALGORITHM:
                row = run_full_map(case, grid, start, goal, model)
            elif algorithm == NO_CACHE_ALGORITHM:
                row = run_no_cache(case, grid, start, goal, model)
            elif algorithm == CACHE_32_ALGORITHM:
                row = run_region_cache(case, grid, start, goal, model, 32)
            else:
                row = run_region_cache(case, grid, start, goal, model, 64)
            rows.append(row)
            complete.add((case["case_id"], algorithm))
            if not as_bool(row["optimal"]):
                print(f"OPTIMALITY FAILURE: {case['case_id']} / {algorithm} cost={row['path_cost']} bfs={case['optimal_cost']}")
            write_csv(partial_path, rows)
        print(f"Completed map {case_index}/{len(cases)}; runs {len(complete)}/{total_runs}: {case['case_id']}")

    rows.sort(key=lambda row: (row["map_size"], row["structure_type"], row["seed"], algorithms.index(row["algorithm"])))
    by_size = summarize(rows, ("map_size", "algorithm"))
    by_structure = summarize(rows, ("map_size", "structure_type", "algorithm"))
    speedups = speedup_rows(by_structure, ("map_size", "structure_type"))
    write_csv(os.path.join(output_dir, "results.csv"), rows)
    write_csv(os.path.join(output_dir, "summary_by_size.csv"), by_size)
    write_csv(os.path.join(output_dir, "summary_by_structure.csv"), by_structure)
    write_csv(os.path.join(output_dir, "speedup_analysis.csv"), speedups)
    write_report(output_dir, by_size, by_structure, speedups, args.cases_per_structure, sizes, structures, algorithms)
    print(f"Saved {len(rows)} completed runs for {len(cases)} maps to {output_dir}")


if __name__ == "__main__":
    main()
