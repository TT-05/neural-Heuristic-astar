"""Structure-aware scaling benchmark using existing Manhattan-primary Neural A*."""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from astar import astar_search
from bfs_label import compute_distance_to_goal
from model import grid_goal_tensor, load_unet_heuristic, manhattan_heuristic
from run_region_cache_large_benchmark import (
    CACHE_32_ALGORITHM,
    CACHE_64_ALGORITHM,
    FULL_ALGORITHM,
    as_bool,
    build_result,
    hardware_info,
    run_full_map,
    run_region_cache,
    select_device,
)
from structured_maps import generate_structured_map

SIZES = (500, 1000, 1500, 2000)
STRUCTURES = ("open_random", "maze_like", "bottleneck", "large_block", "narrow_corridor")
RATES = (0.1, 0.2, 0.3, 0.4)
BASE_SEED = 20_280_801
MANHATTAN = "manhattan_astar"
ALGORITHMS = (MANHATTAN, FULL_ALGORITHM, CACHE_32_ALGORITHM, CACHE_64_ALGORITHM)
COMPLETED = "completed"
INFEASIBLE = "skipped_infeasible"


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def median(values):
    values = sorted(values)
    if not values:
        return 0.0
    index = len(values) // 2
    return values[index] if len(values) % 2 else (values[index - 1] + values[index]) / 2


def grid_hash(grid):
    digest = hashlib.sha256()
    for row in grid:
        digest.update(bytes(row))
    return digest.hexdigest()


def coordinate(text):
    row, col = text.split(",")
    return int(row), int(col)


def choose_pair(grid, seed):
    rng = random.Random(seed * 1_000_003 + len(grid))
    rows, cols = len(grid), len(grid[0])
    for _ in range(8):
        start = None
        for _ in range(1_000):
            candidate = rng.randrange(rows), rng.randrange(cols)
            if grid[candidate[0]][candidate[1]] == 0:
                start = candidate
                break
        if start is None:
            continue
        distances = compute_distance_to_goal(grid, start)
        goal = None
        reachable = 0
        for row, values in enumerate(distances):
            for col, distance in enumerate(values):
                if distance > 0:
                    reachable += 1
                    if rng.randrange(reachable) == 0:
                        goal = row, col
        if goal is not None:
            return start, goal, distances[goal[0]][goal[1]]
    return None, None, None


def parse_overrides(text):
    result = {}
    if not text:
        return result
    for item in text.split(","):
        size, count = map(int, item.split(":", 1))
        if size not in SIZES or count < 1:
            raise ValueError("samples-by-size must use supported positive SIZE:COUNT pairs")
        result[size] = count
    return result


def load_or_create_cases(output, default_count, overrides, sizes, structures, regenerate):
    path = output / "cases.csv"
    rows = [] if regenerate or not path.exists() else list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    for row in rows:
        row["map_size"] = int(row["map_size"])
        row["seed"] = int(row["seed"])
        row["obstacle_rate"] = float(row["obstacle_rate"])
        row["optimal_cost"] = int(row["optimal_cost"])
    for size in sizes:
        target = overrides.get(size, default_count)
        for structure_index, structure in enumerate(structures):
            chosen = [row for row in rows if row["map_size"] == size and row["structure_type"] == structure]
            seed_base = BASE_SEED + size * 1_000_000 + structure_index * 100_000
            attempt = max((row["seed"] - seed_base + 1 for row in chosen), default=0)
            while len(chosen) < target:
                seed = seed_base + attempt
                rate = RATES[len(chosen) % len(RATES)]
                attempt += 1
                grid = generate_structured_map(size, size, seed, rate, structure)
                start, goal, cost = choose_pair(grid, seed)
                if start is None:
                    continue
                case = {
                    "case_id": f"size{size}_{structure}_seed{seed}",
                    "map_size": size,
                    "structure_type": structure,
                    "seed": seed,
                    "obstacle_rate": rate,
                    "grid_sha256": grid_hash(grid),
                    "start": f"{start[0]},{start[1]}",
                    "goal": f"{goal[0]},{goal[1]}",
                    "optimal_cost": cost,
                    "target_cases_per_structure": target,
                }
                rows.append(case)
                chosen.append(case)
                write_csv(path, rows)
    return rows


def materialize(case):
    grid = generate_structured_map(
        case["map_size"], case["map_size"], case["seed"], case["obstacle_rate"], case["structure_type"]
    )
    if grid_hash(grid) != case["grid_sha256"]:
        raise AssertionError(f"Grid hash mismatch for {case['case_id']}")
    return grid, coordinate(case["start"]), coordinate(case["goal"])


def run_manhattan(case, grid, start, goal):
    started = time.perf_counter()
    result = astar_search(grid, start, goal, manhattan_heuristic)
    result["total_runtime_seconds"] = time.perf_counter() - started
    return build_result(
        case, MANHATTAN, result, case["optimal_cost"], calls=0, scored_nodes=0,
        hits=0, regions=0, patch_extract=0.0, cache_lookup=0.0,
        unet_inference=0.0, full_inference=0.0,
    )


def load_rows(path):
    path = Path(path)
    return list(csv.DictReader(path.open(newline="", encoding="utf-8"))) if path.exists() else []


def is_memory_error(error):
    return isinstance(error, MemoryError) or "out of memory" in str(error).lower()


def failure_row(case, algorithm, reason):
    row = {key: case[key] for key in (
        "case_id", "map_size", "structure_type", "seed", "obstacle_rate",
        "grid_sha256", "start", "goal", "optimal_cost",
    )}
    row.update({"algorithm": algorithm, "status": INFEASIBLE, "failure_reason": reason})
    for key in (
        "path_found", "path_cost", "path_cost_gap", "optimal", "expanded_nodes",
        "unet_call_count", "unet_scored_node_count", "cache_hit_count",
        "cache_hit_rate", "cached_region_count", "patch_extraction_seconds",
        "cache_lookup_seconds", "unet_inference_seconds", "full_map_inference_seconds",
        "astar_search_seconds", "total_runtime_seconds",
    ):
        row[key] = ""
    return row


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def summarize(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        valid = [row for row in group if row["status"] == COMPLETED]
        item = dict(zip(fields, key))
        item.update({
            "planned_cases": len(group),
            "completed_cases": len(valid),
            "skipped_cases": len(group) - len(valid),
            "mean_expanded_nodes": mean(number(row, "expanded_nodes") for row in valid),
            "median_expanded_nodes": median(number(row, "expanded_nodes") for row in valid),
            "optimality_rate": mean(float(as_bool(row["optimal"])) for row in valid),
            "mean_path_cost_gap": mean(number(row, "path_cost_gap") for row in valid),
            "mean_total_runtime_seconds": mean(number(row, "total_runtime_seconds") for row in valid),
            "mean_astar_search_seconds": mean(number(row, "astar_search_seconds") for row in valid),
            "mean_neural_inference_seconds": mean(
                number(row, "unet_inference_seconds") + number(row, "full_map_inference_seconds") for row in valid
            ),
            "mean_patch_extraction_seconds": mean(number(row, "patch_extraction_seconds") for row in valid),
            "mean_cache_lookup_seconds": mean(number(row, "cache_lookup_seconds") for row in valid),
            "mean_unet_call_count": mean(number(row, "unet_call_count") for row in valid),
            "mean_cache_hit_rate": mean(number(row, "cache_hit_rate") for row in valid),
        })
        output.append(item)
    return output


def compare_with_manhattan(rows, has_structure=False):
    indexed = {
        (row["map_size"], row.get("structure_type", ""), row["algorithm"]): row
        for row in rows
    }
    for row in rows:
        structure = row.get("structure_type", "") if has_structure else ""
        baseline = indexed.get((row["map_size"], structure, MANHATTAN))
        if baseline and baseline["completed_cases"] and row["completed_cases"]:
            row["expansion_reduction_vs_manhattan"] = (
                1 - row["mean_expanded_nodes"] / baseline["mean_expanded_nodes"]
            )
            row["runtime_multiplier_vs_manhattan"] = (
                row["mean_total_runtime_seconds"] / baseline["mean_total_runtime_seconds"]
            )
        else:
            row["expansion_reduction_vs_manhattan"] = ""
            row["runtime_multiplier_vs_manhattan"] = ""


def write_breakdown(path, by_structure):
    rows = []
    for row in by_structure:
        for component, key in (
            ("astar_search", "mean_astar_search_seconds"),
            ("neural_inference", "mean_neural_inference_seconds"),
            ("patch_extraction", "mean_patch_extraction_seconds"),
            ("cache_lookup", "mean_cache_lookup_seconds"),
        ):
            rows.append({
                "map_size": row["map_size"],
                "structure_type": row["structure_type"],
                "algorithm": row["algorithm"],
                "component": component,
                "completed_cases": row["completed_cases"],
                "mean_seconds": row[key],
                "fraction_of_total": row[key] / row["mean_total_runtime_seconds"]
                if row["mean_total_runtime_seconds"] else 0.0,
            })
    write_csv(path, rows)


def write_report(path, by_size, by_structure):
    lines = [
        "# Structure-Aware Neural A* Scaling Report",
        "",
        "Manhattan remains the primary A* key. GPU is used only for U-Net inference.",
        "",
        "## By size",
        "",
        "| Size | Method | Planned | Completed | Skipped | Expanded | Reduction vs Manhattan | Optimality | Total s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_size:
        reduction = row["expansion_reduction_vs_manhattan"]
        reduction_text = f"{reduction:.1%}" if reduction != "" else "n/a"
        lines.append(
            f"| {row['map_size']} | {row['algorithm']} | {row['planned_cases']} | "
            f"{row['completed_cases']} | {row['skipped_cases']} | "
            f"{row['mean_expanded_nodes']:.2f} | {reduction_text} | "
            f"{row['optimality_rate']:.1%} | {row['mean_total_runtime_seconds']:.4f} |"
        )
    lines.extend(["", "## Evidence-based answers", ""])
    for algorithm in ALGORITHMS[1:]:
        trend = [
            (row["map_size"], row["expansion_reduction_vs_manhattan"])
            for row in by_size
            if row["algorithm"] == algorithm and row["expansion_reduction_vs_manhattan"] != ""
        ]
        reductions = [value for _, value in trend]
        monotonic = "non-decreasing" if all(a <= b for a, b in zip(reductions, reductions[1:])) else "not monotonic"
        values = ", ".join(f"{size}: {reduction:.1%}" for size, reduction in trend)
        lines.append(f"- {algorithm} expansion reduction by size: {values} ({monotonic}).")
        candidates = [
            row for row in by_structure
            if row["algorithm"] == algorithm and row["expansion_reduction_vs_manhattan"] != ""
        ]
        if candidates:
            best = max(candidates, key=lambda row: row["expansion_reduction_vs_manhattan"])
            worst = min(candidates, key=lambda row: row["expansion_reduction_vs_manhattan"])
            lines.append(
                f"  Structure range: best {best['expansion_reduction_vs_manhattan']:.1%} "
                f"at {best['map_size']} {best['structure_type']}; worst "
                f"{worst['expansion_reduction_vs_manhattan']:.1%} at "
                f"{worst['map_size']} {worst['structure_type']}."
            )
    for size in sorted({int(row["map_size"]) for row in by_size}):
        candidates = [
            row for row in by_size
            if int(row["map_size"]) == size and row["algorithm"] != MANHATTAN and row["completed_cases"]
        ]
        if candidates:
            fastest = min(candidates, key=lambda row: row["mean_total_runtime_seconds"])
            lines.append(
                f"- Runtime competitiveness at {size}x{size}: fastest neural method is "
                f"{fastest['algorithm']} at {fastest['mean_total_runtime_seconds']:.4f}s, "
                f"{fastest['runtime_multiplier_vs_manhattan']:.2f}x Manhattan."
            )
            full = next((row for row in candidates if row["algorithm"] == FULL_ALGORITHM), None)
            regions = [row for row in candidates if row["algorithm"] in (CACHE_32_ALGORITHM, CACHE_64_ALGORITHM)]
            if full and regions:
                fastest_region = min(regions, key=lambda row: row["mean_total_runtime_seconds"])
                relation = "faster" if full["mean_total_runtime_seconds"] < fastest_region["mean_total_runtime_seconds"] else "slower"
                lines.append(
                    f"  Full-map is {relation} than the fastest Region-cache variant "
                    f"({full['mean_total_runtime_seconds']:.4f}s versus "
                    f"{fastest_region['mean_total_runtime_seconds']:.4f}s)."
                )
    for row in by_size:
        if row["algorithm"] in (CACHE_32_ALGORITHM, CACHE_64_ALGORITHM) and row["completed_cases"]:
            components = {
                "A star search": row["mean_astar_search_seconds"],
                "U Net inference": row["mean_neural_inference_seconds"],
                "patch extraction": row["mean_patch_extraction_seconds"],
                "cache lookup": row["mean_cache_lookup_seconds"],
            }
            bottleneck, seconds = max(components.items(), key=lambda item: item[1])
            lines.append(f"- {row['map_size']} {row['algorithm']}: largest measured component is {bottleneck} at {seconds:.4f}s.")
    lines.extend([
        "",
        "Rows marked skipped_infeasible are excluded from means and record a resource limit.",
        "No runtime superiority is claimed unless total runtime is lower than Manhattan on the same completed sample set.",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run structure-aware Neural A* scaling evaluation.")
    parser.add_argument("--checkpoint", default="outputs/ranking_lambda_sweep/lambda_0.5_best.pt")
    parser.add_argument("--output-dir", default="outputs/large_scale_scaling")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--cases-per-structure", type=int, default=50)
    parser.add_argument("--samples-by-size", help="Optional reductions such as 2000:10,5000:1")
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--structures", nargs="+", default=STRUCTURES)
    parser.add_argument("--regenerate-cases", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sizes, structures = tuple(args.sizes), tuple(args.structures)
    if args.cases_per_structure < 1 or any(size not in SIZES or size % 4 for size in sizes):
        raise ValueError("Use positive samples and declared, divisible-by-four sizes")
    if any(structure not in STRUCTURES for structure in structures):
        raise ValueError("Use declared structures only")
    root = Path(__file__).resolve().parents[1]
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    final, partial = output / "scaling_results.csv", output / "scaling_results_partial.csv"
    if args.report_only:
        rows = load_rows(final) or load_rows(partial)
        if not rows:
            raise FileNotFoundError("scaling_results.csv or scaling_results_partial.csv is required for --report-only")
    else:
        device = select_device(args.device)
        write_csv(output / "hardware_info.csv", [hardware_info(device)])
        model = load_unet_heuristic(root / args.checkpoint, device=device)
        with torch.inference_mode():
            model(grid_goal_tensor([[0] * 32 for _ in range(32)], (16, 16), device=device).unsqueeze(0))
        torch.cuda.synchronize(device)
        cases = load_or_create_cases(
            output, args.cases_per_structure, parse_overrides(args.samples_by_size),
            sizes, structures, args.regenerate_cases,
        )
        valid_ids = {case["case_id"] for case in cases}
        rows = [
            row for row in load_rows(partial)
            if row["case_id"] in valid_ids and row["algorithm"] in ALGORITHMS
        ]
        complete = {(row["case_id"], row["algorithm"]) for row in rows}
        unavailable = {
            (int(row["map_size"]), row["algorithm"])
            for row in rows if row["status"] == INFEASIBLE
        }
        for index, case in enumerate(cases, 1):
            needed = [name for name in ALGORITHMS if (case["case_id"], name) not in complete]
            if not needed:
                continue
            try:
                grid, start, goal = materialize(case)
            except (MemoryError, RuntimeError) as error:
                if not is_memory_error(error):
                    raise
                generated = [failure_row(case, name, f"map materialization: {error}") for name in needed]
            else:
                generated = []
                for name in needed:
                    if (case["map_size"], name) in unavailable:
                        generated.append(failure_row(case, name, "previous case at this size exceeded memory"))
                        continue
                    try:
                        if name == MANHATTAN:
                            row = run_manhattan(case, grid, start, goal)
                        elif name == FULL_ALGORITHM:
                            row = run_full_map(case, grid, start, goal, model)
                        else:
                            row = run_region_cache(case, grid, start, goal, model, 32 if name == CACHE_32_ALGORITHM else 64)
                        row["status"], row["failure_reason"] = COMPLETED, ""
                    except (MemoryError, RuntimeError) as error:
                        if not is_memory_error(error):
                            raise
                        torch.cuda.empty_cache()
                        unavailable.add((case["map_size"], name))
                        row = failure_row(case, name, str(error))
                    generated.append(row)
            rows.extend(generated)
            complete.update((row["case_id"], row["algorithm"]) for row in generated)
            write_csv(partial, rows)
            print(f"Completed map {index}/{len(cases)}", flush=True)
        order = {name: index for index, name in enumerate(ALGORITHMS)}
        rows.sort(key=lambda row: (
            int(row["map_size"]), row["structure_type"], int(row["seed"]), order[row["algorithm"]]
        ))
        write_csv(final, rows)
    by_size = summarize(rows, ("map_size", "algorithm"))
    by_structure = summarize(rows, ("map_size", "structure_type", "algorithm"))
    compare_with_manhattan(by_size)
    compare_with_manhattan(by_structure, has_structure=True)
    write_csv(output / "scaling_by_size.csv", by_size)
    write_csv(output / "scaling_by_structure.csv", by_structure)
    write_breakdown(output / "runtime_breakdown.csv", by_structure)
    write_report(output / "scaling_report.md", by_size, by_structure)
    print(f"Saved {len(rows)} method-case rows to {output}")


if __name__ == "__main__":
    main()

