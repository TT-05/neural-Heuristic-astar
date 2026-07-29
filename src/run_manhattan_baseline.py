"""CPU Manhattan A* baseline on the saved region-cache benchmark cases.

This runner deliberately consumes the existing case manifest and existing U-Net
result CSV.  It never loads a checkpoint or invokes a neural search variant.
The manifest is regenerated only in memory to verify its saved grid hash; each
Manhattan result is independently checked against a freshly computed
reverse-BFS distance before it is persisted.
"""

import argparse
import csv
import os
import time
from collections import defaultdict

from astar import astar_search
from bfs_label import compute_distance_to_goal
from model import manhattan_heuristic
from run_lazy_patch_experiment import mean, median, write_csv
from run_region_cache_large_benchmark import materialize_case


MANHATTAN_ALGORITHM = "manhattan_astar"
COMPARATORS = (
    "full_map_unet_tiebreak",
    "region_cache_patch_32",
    "region_cache_patch_64",
)
RESULT_FIELDS = (
    "case_id",
    "map_size",
    "structure_type",
    "seed",
    "obstacle_rate",
    "grid_sha256",
    "start",
    "goal",
    "optimal_cost",
    "algorithm",
    "path_found",
    "path_cost",
    "path_cost_gap",
    "optimal",
    "expanded_nodes",
    "unet_call_count",
    "unet_scored_node_count",
    "cache_hit_count",
    "cache_hit_rate",
    "cached_region_count",
    "patch_extraction_seconds",
    "cache_lookup_seconds",
    "unet_inference_seconds",
    "full_map_inference_seconds",
    "astar_search_seconds",
    "total_runtime_seconds",
)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_cases(path):
    cases = read_csv(path)
    required = {"case_id", "map_size", "structure_type", "seed", "obstacle_rate", "grid_sha256", "start", "goal", "optimal_cost"}
    if not cases or not required.issubset(cases[0]):
        raise ValueError(f"Invalid or empty case manifest: {path}")
    for case in cases:
        case["map_size"] = int(case["map_size"])
        case["seed"] = int(case["seed"])
        case["obstacle_rate"] = float(case["obstacle_rate"])
        case["optimal_cost"] = int(case["optimal_cost"])
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Case manifest contains duplicate case IDs.")
    return cases


def as_float(row, field):
    value = row.get(field, "")
    return 0.0 if value == "" else float(value)


def as_bool(value):
    return value is True or value in ("True", "true", "1", 1)


def validate_comparators(cases, rows):
    """Return existing comparator rows after proving they match this manifest."""
    cases_by_id = {case["case_id"]: case for case in cases}
    selected = defaultdict(dict)
    for row in rows:
        algorithm = row.get("algorithm")
        if algorithm not in COMPARATORS:
            continue
        case_id = row.get("case_id")
        if case_id not in cases_by_id:
            raise ValueError(f"Comparator result refers to unknown case: {case_id}")
        if algorithm in selected[case_id]:
            raise ValueError(f"Duplicate comparator row: {case_id} / {algorithm}")
        case = cases_by_id[case_id]
        for field in ("grid_sha256", "start", "goal"):
            if row.get(field) != str(case[field]):
                raise ValueError(f"Comparator {algorithm} does not match manifest for {case_id}: {field}")
        if int(row["optimal_cost"]) != case["optimal_cost"]:
            raise ValueError(f"Comparator {algorithm} has a different optimal cost for {case_id}")
        selected[case_id][algorithm] = row
    missing = [(case["case_id"], algorithm) for case in cases for algorithm in COMPARATORS if algorithm not in selected[case["case_id"]]]
    if missing:
        raise ValueError(f"Existing U-Net results are incomplete for this manifest; first missing pair: {missing[0]}")
    return selected


def result_row(case, search, measured_optimal_cost, elapsed):
    path_found = search["cost"] >= 0
    return {
        "case_id": case["case_id"],
        "map_size": case["map_size"],
        "structure_type": case["structure_type"],
        "seed": case["seed"],
        "obstacle_rate": case["obstacle_rate"],
        "grid_sha256": case["grid_sha256"],
        "start": case["start"],
        "goal": case["goal"],
        "optimal_cost": measured_optimal_cost,
        "algorithm": MANHATTAN_ALGORITHM,
        "path_found": path_found,
        "path_cost": search["cost"],
        "path_cost_gap": search["cost"] - measured_optimal_cost if path_found else "",
        "optimal": path_found and search["cost"] == measured_optimal_cost,
        "expanded_nodes": search["expanded"],
        "unet_call_count": 0,
        "unet_scored_node_count": 0,
        "cache_hit_count": 0,
        "cache_hit_rate": 0.0,
        "cached_region_count": 0,
        "patch_extraction_seconds": 0.0,
        "cache_lookup_seconds": 0.0,
        "unet_inference_seconds": 0.0,
        "full_map_inference_seconds": 0.0,
        "astar_search_seconds": elapsed,
        "total_runtime_seconds": elapsed,
    }


def load_partial(path, cases):
    if not os.path.exists(path):
        return []
    allowed = {case["case_id"] for case in cases}
    rows = read_csv(path)
    for row in rows:
        if row.get("algorithm") != MANHATTAN_ALGORITHM or row.get("case_id") not in allowed:
            raise ValueError(f"Invalid row in resumable baseline output: {row.get('case_id')}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("Resumable baseline output contains duplicate case IDs.")
    return rows


def aggregate(rows):
    return {
        "cases": len(rows),
        "mean_expanded_nodes": mean(as_float(row, "expanded_nodes") for row in rows),
        "median_expanded_nodes": median(as_float(row, "expanded_nodes") for row in rows),
        "optimality_rate": mean(float(as_bool(row["optimal"])) for row in rows),
        "mean_path_cost_gap": mean(as_float(row, "path_cost_gap") for row in rows),
        "mean_astar_search_seconds": mean(as_float(row, "astar_search_seconds") for row in rows),
        "mean_total_runtime_seconds": mean(as_float(row, "total_runtime_seconds") for row in rows),
    }


def build_summary(cases, manhattan_rows, comparator_rows):
    all_by_algorithm = {MANHATTAN_ALGORITHM: {row["case_id"]: row for row in manhattan_rows}}
    for algorithm in COMPARATORS:
        all_by_algorithm[algorithm] = {case_id: rows[algorithm] for case_id, rows in comparator_rows.items()}
    scopes = [("overall", None, None)]
    scopes += [("map_size", size, None) for size in sorted({case["map_size"] for case in cases})]
    scopes += [
        ("map_size_structure", size, structure)
        for size in sorted({case["map_size"] for case in cases})
        for structure in sorted({case["structure_type"] for case in cases})
    ]
    output = []
    for scope, size, structure in scopes:
        case_ids = {
            case["case_id"]
            for case in cases
            if (size is None or case["map_size"] == size) and (structure is None or case["structure_type"] == structure)
        }
        baseline = aggregate([all_by_algorithm[MANHATTAN_ALGORITHM][case_id] for case_id in case_ids])
        for algorithm in (MANHATTAN_ALGORITHM, *COMPARATORS):
            metrics = aggregate([all_by_algorithm[algorithm][case_id] for case_id in case_ids])
            expanded_saved = baseline["mean_expanded_nodes"] - metrics["mean_expanded_nodes"]
            total_overhead = metrics["mean_total_runtime_seconds"] - baseline["mean_total_runtime_seconds"]
            search_overhead = metrics["mean_astar_search_seconds"] - baseline["mean_astar_search_seconds"]
            output.append(
                {
                    "scope": scope,
                    "map_size": "" if size is None else size,
                    "structure_type": "" if structure is None else structure,
                    "algorithm": algorithm,
                    **metrics,
                    "mean_expanded_nodes_saved_vs_manhattan": expanded_saved,
                    "expansion_reduction_fraction_vs_manhattan": expanded_saved / max(baseline["mean_expanded_nodes"], 1e-12),
                    "total_runtime_overhead_seconds_vs_manhattan": total_overhead,
                    "total_runtime_overhead_fraction_vs_manhattan": total_overhead / max(baseline["mean_total_runtime_seconds"], 1e-12),
                    "total_runtime_multiplier_vs_manhattan": metrics["mean_total_runtime_seconds"] / max(baseline["mean_total_runtime_seconds"], 1e-12),
                    "astar_search_overhead_seconds_vs_manhattan": search_overhead,
                    "astar_search_multiplier_vs_manhattan": metrics["mean_astar_search_seconds"] / max(baseline["mean_astar_search_seconds"], 1e-12),
                }
            )
    return output


def find_summary(summary, scope, size, algorithm):
    return next(
        row
        for row in summary
        if row["scope"] == scope and row["map_size"] == ("" if size is None else size) and row["algorithm"] == algorithm
    )


def write_report(path, cases, summary, source_dir):
    lines = [
        "# CPU Manhattan A* Baseline Benchmark",
        "",
        f"This benchmark runs pure Manhattan A* on the {len(cases)} saved deterministic cases from `{source_dir}/cases.csv`. It does not regenerate the manifest, load a U-Net checkpoint, or rerun U-Net methods.",
        "",
        "Every reconstructed grid is hash-checked against the manifest. For each run, the stored optimal cost is revalidated by a fresh reverse-BFS distance-to-goal computation. Expanded nodes are the non-stale expansions reported by the existing `astar_search` implementation.",
        "",
        "## Aggregate Comparison",
        "",
        "| Algorithm | Expanded | Saved vs Manhattan | Reduction | Optimality | A* s | Total s | Total runtime vs Manhattan |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm in (MANHATTAN_ALGORITHM, *COMPARATORS):
        row = find_summary(summary, "overall", None, algorithm)
        lines.append(
            f"| {algorithm} | {row['mean_expanded_nodes']:.2f} | {row['mean_expanded_nodes_saved_vs_manhattan']:.2f} | "
            f"{row['expansion_reduction_fraction_vs_manhattan']:.1%} | {row['optimality_rate']:.1%} | "
            f"{row['mean_astar_search_seconds']:.4f} | {row['mean_total_runtime_seconds']:.4f} | "
            f"{row['total_runtime_multiplier_vs_manhattan']:.2f}x ({row['total_runtime_overhead_fraction_vs_manhattan']:+.1%}) |"
        )
    lines += [
        "",
        "## By Map Size",
        "",
        "| Size | Algorithm | Expanded | Saved vs Manhattan | Reduction | A* s | Total s | Runtime multiplier |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for size in sorted({case["map_size"] for case in cases}):
        for algorithm in (MANHATTAN_ALGORITHM, *COMPARATORS):
            row = find_summary(summary, "map_size", size, algorithm)
            lines.append(
                f"| {size} | {algorithm} | {row['mean_expanded_nodes']:.2f} | {row['mean_expanded_nodes_saved_vs_manhattan']:.2f} | "
                f"{row['expansion_reduction_fraction_vs_manhattan']:.1%} | {row['mean_astar_search_seconds']:.4f} | "
                f"{row['mean_total_runtime_seconds']:.4f} | {row['total_runtime_multiplier_vs_manhattan']:.2f}x |"
            )
    lines += [
        "",
        "`summary.csv` also provides the same comparison for every size × structure stratum.",
        "",
        "## Interpretation",
        "",
        "Expansion savings compare each existing Manhattan-primary U-Net tie-break trajectory against pure Manhattan A* on identical saved cases. Runtime overhead compares the recorded end-to-end CPU timings: neural preparation plus search for U-Net variants, versus search only for Manhattan A*. These CPU measurements do not establish GPU runtime ranking; GPU comparison needs equivalent CUDA execution and synchronized timing.",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Manhattan A* on saved region-cache benchmark cases only.")
    parser.add_argument("--source-dir", default="outputs/region_cache_large_benchmark")
    parser.add_argument("--output-dir", default="outputs/manhattan_baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(root, args.source_dir)
    output_dir = os.path.join(root, args.output_dir)
    cases = parse_cases(os.path.join(source_dir, "cases.csv"))
    comparator_rows = validate_comparators(cases, read_csv(os.path.join(source_dir, "results.csv")))
    os.makedirs(output_dir, exist_ok=True)
    partial_path = os.path.join(output_dir, "results_partial.csv")
    rows = load_partial(partial_path, cases)
    completed = {row["case_id"] for row in rows}
    for index, case in enumerate(cases, 1):
        if case["case_id"] in completed:
            continue
        grid, start, goal = materialize_case(case)
        distances = compute_distance_to_goal(grid, goal)
        measured_optimal_cost = distances[start[0]][start[1]]
        if measured_optimal_cost != case["optimal_cost"]:
            raise AssertionError(f"Reverse-BFS mismatch for {case['case_id']}: {measured_optimal_cost} != {case['optimal_cost']}")
        started = time.perf_counter()
        search = astar_search(grid, start, goal, manhattan_heuristic)
        elapsed = time.perf_counter() - started
        row = result_row(case, search, measured_optimal_cost, elapsed)
        if not row["optimal"]:
            raise AssertionError(f"Manhattan optimality failure for {case['case_id']}: {search['cost']} != {measured_optimal_cost}")
        rows.append(row)
        completed.add(case["case_id"])
        write_csv(partial_path, rows)
        print(f"Completed {index}/{len(cases)}: {case['case_id']}", flush=True)
    rows.sort(key=lambda row: (int(row["map_size"]), row["structure_type"], int(row["seed"])))
    write_csv(os.path.join(output_dir, "results.csv"), rows)
    summary = build_summary(cases, rows, comparator_rows)
    write_csv(os.path.join(output_dir, "summary.csv"), summary)
    write_report(os.path.join(output_dir, "report.md"), cases, summary, args.source_dir)
    print(f"Saved {len(rows)} Manhattan runs and comparison summary to {output_dir}")


if __name__ == "__main__":
    main()
