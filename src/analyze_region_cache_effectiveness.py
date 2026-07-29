"""Analyze search-aware computation from completed region-cache benchmark CSVs.

No search, inference, map generation, or model code is run here.  In
particular, the completed benchmark does not retain patch-center coordinates
or expanded-node traces.  Exact *unique in-map* patch coverage therefore
cannot be reconstructed post hoc; this script reports it as ``not_logged``
instead of substituting a proxy.  Counts that are present in the benchmark
CSV, such as U-Net calls, patch output cells, lazy tie-break queries, and cache
hits, remain exact observations.
"""

import argparse
import csv
import os
from collections import defaultdict


MANHATTAN = "manhattan_astar"
FULL_MAP = "full_map_unet_tiebreak"
CACHE_32 = "region_cache_patch_32"
CACHE_64 = "region_cache_patch_64"
ALGORITHMS = (MANHATTAN, FULL_MAP, CACHE_32, CACHE_64)
PATCH_SIZES = {CACHE_32: 32, CACHE_64: 64}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty analysis: {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric(row, field):
    value = row.get(field, "")
    return 0.0 if value == "" else float(value)


def normalise_case_fields(row):
    copy = dict(row)
    copy["map_size"] = int(copy["map_size"])
    copy["seed"] = int(copy["seed"])
    copy["optimal_cost"] = int(copy["optimal_cost"])
    return copy


def load_cases(path):
    cases = [normalise_case_fields(row) for row in read_csv(path)]
    if not cases:
        raise ValueError(f"Empty case manifest: {path}")
    if len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("Case manifest has duplicate case IDs.")
    return cases


def select_and_validate(cases, benchmark_rows, manhattan_rows):
    """Validate that all four result streams reference the same saved cases."""
    expected = {case["case_id"]: case for case in cases}
    selected = {algorithm: {} for algorithm in ALGORITHMS}
    for row in benchmark_rows + manhattan_rows:
        algorithm = row.get("algorithm")
        if algorithm not in selected:
            continue
        case_id = row.get("case_id")
        if case_id not in expected:
            raise ValueError(f"{algorithm} result has an unknown case ID: {case_id}")
        if case_id in selected[algorithm]:
            raise ValueError(f"Duplicate result for {algorithm} / {case_id}")
        case = expected[case_id]
        for field in ("grid_sha256", "start", "goal"):
            if row.get(field) != str(case[field]):
                raise ValueError(f"{algorithm} {case_id} does not match saved case field {field}")
        if int(row["optimal_cost"]) != case["optimal_cost"]:
            raise ValueError(f"{algorithm} {case_id} has a different optimal cost")
        selected[algorithm][case_id] = normalise_case_fields(row)
    for algorithm in ALGORITHMS:
        missing = [case_id for case_id in expected if case_id not in selected[algorithm]]
        if missing:
            raise ValueError(f"Missing {len(missing)} {algorithm} results; first is {missing[0]}")
    return selected


def scopes(cases):
    sizes = sorted({case["map_size"] for case in cases})
    structures = sorted({case["structure_type"] for case in cases})
    output = [("overall", None, None)]
    output.extend(("map_size", size, None) for size in sizes)
    output.extend(("map_size_structure", size, structure) for size in sizes for structure in structures)
    return output


def in_scope(case, size, structure):
    return (size is None or case["map_size"] == size) and (structure is None or case["structure_type"] == structure)


def scope_fields(scope, size, structure):
    return {
        "scope": scope,
        "map_size": "" if size is None else size,
        "structure_type": "" if structure is None else structure,
    }


def total_map_cells(cases):
    return sum(case["map_size"] ** 2 for case in cases)


def result_rows(cases, selected, algorithm):
    return [selected[algorithm][case["case_id"]] for case in cases]


def sum_field(rows, field):
    return sum(numeric(row, field) for row in rows)


def neural_seconds(rows, algorithm):
    field = "full_map_inference_seconds" if algorithm == FULL_MAP else "unet_inference_seconds"
    return sum_field(rows, field)


def call_count(rows, algorithm):
    if algorithm == MANHATTAN:
        return 0
    return sum_field(rows, "unet_call_count")


def tie_queries(rows, algorithm):
    # full-map A* did not persist individual secondary-key accesses.  Its zero
    # scored-node field means "not recorded", not zero accesses.
    if algorithm == FULL_MAP:
        return None
    return sum_field(rows, "unet_scored_node_count")


def coverage_rows(cases, selected):
    output = []
    for scope, size, structure in scopes(cases):
        scoped = [case for case in cases if in_scope(case, size, structure)]
        map_cells = total_map_cells(scoped)
        for algorithm in ALGORITHMS:
            rows = result_rows(scoped, selected, algorithm)
            calls = call_count(rows, algorithm)
            if algorithm == MANHATTAN:
                unique_cells, coverage, output_cells, status = 0, 0.0, 0, "no_unet"
            elif algorithm == FULL_MAP:
                unique_cells, coverage, output_cells, status = map_cells, 1.0, map_cells, "exact_full_map"
            else:
                patch = PATCH_SIZES[algorithm]
                output_cells = int(calls * patch * patch)
                unique_cells, coverage, status = "", "", "not_logged_patch_centers"
            output.append(
                {
                    **scope_fields(scope, size, structure),
                    "algorithm": algorithm,
                    "cases": len(scoped),
                    "total_map_cells": map_cells,
                    "total_predicted_patch_output_cells": output_cells,
                    "unique_in_map_cells_evaluated_by_unet": unique_cells,
                    "prediction_coverage_ratio": coverage,
                    "nominal_patch_output_cells_per_map_cell": output_cells / max(map_cells, 1),
                    "coverage_observability": status,
                }
            )
    return output


def overlap_rows(cases, selected):
    output = []
    for scope, size, structure in scopes(cases):
        scoped = [case for case in cases if in_scope(case, size, structure)]
        for algorithm in ALGORITHMS:
            rows = result_rows(scoped, selected, algorithm)
            expanded = sum_field(rows, "expanded_nodes")
            queries = tie_queries(rows, algorithm)
            if algorithm == FULL_MAP:
                # Full-map prediction covers every in-map expanded node.
                expanded_overlap, expanded_coverage, query_overlap, query_coverage, status = (
                    expanded,
                    1.0,
                    "",
                    "",
                    "expanded_overlap_exact_full_map; tie_queries_not_logged",
                )
            elif algorithm in PATCH_SIZES:
                # Every lazy tie-break query is served from one of the cached
                # patch outputs, whether it was a hit or a miss.  The set of
                # expanded nodes intersecting prediction regions was not saved.
                expanded_overlap, expanded_coverage = "", ""
                query_overlap, query_coverage = queries, 1.0 if queries else 0.0
                status = "tie_query_overlap_exact; expanded_overlap_not_logged"
            else:
                expanded_overlap = expanded_coverage = query_overlap = query_coverage = ""
                status = "no_unet"
            output.append(
                {
                    **scope_fields(scope, size, structure),
                    "algorithm": algorithm,
                    "cases": len(scoped),
                    "expanded_cells": expanded,
                    "tie_break_queried_cells": "" if queries is None else queries,
                    "predicted_expanded_cell_overlap": expanded_overlap,
                    "expanded_search_coverage_ratio": expanded_coverage,
                    "predicted_tie_query_cell_overlap": query_overlap,
                    "tie_query_search_coverage_ratio": query_coverage,
                    "tie_queries_per_expanded_cell": "" if queries is None else queries / max(expanded, 1),
                    "overlap_observability": status,
                }
            )
    return output


def cache_rows(cases, selected):
    output = []
    for scope, size, structure in scopes(cases):
        scoped = [case for case in cases if in_scope(case, size, structure)]
        for algorithm in ALGORITHMS:
            rows = result_rows(scoped, selected, algorithm)
            if algorithm in PATCH_SIZES:
                queries = sum_field(rows, "unet_scored_node_count")
                hits = sum_field(rows, "cache_hit_count")
                misses = sum_field(rows, "unet_call_count")
                if round(queries) != round(hits + misses):
                    raise AssertionError(f"Cache accounting mismatch in {algorithm} / {scope} / {size} / {structure}")
                status = "exact_aggregate_counts"
            else:
                queries = hits = misses = ""
                status = "not_a_region_cache_method"
            output.append(
                {
                    **scope_fields(scope, size, structure),
                    "algorithm": algorithm,
                    "cases": len(scoped),
                    "total_cache_queries": queries,
                    "cache_hits": hits,
                    "cache_misses": misses,
                    "cache_hit_rate_weighted": "" if queries == "" else hits / max(queries, 1),
                    "mean_per_case_cache_hit_rate": "" if algorithm not in PATCH_SIZES else sum_field(rows, "cache_hit_rate") / len(rows),
                    "cache_reuse_factor_queries_per_miss": "" if misses == "" else queries / max(misses, 1),
                    "cache_observability": status,
                }
            )
    return output


def cost_rows(cases, selected):
    output = []
    for scope, size, structure in scopes(cases):
        scoped = [case for case in cases if in_scope(case, size, structure)]
        map_cells = total_map_cells(scoped)
        for algorithm in ALGORITHMS:
            rows = result_rows(scoped, selected, algorithm)
            calls = call_count(rows, algorithm)
            seconds = neural_seconds(rows, algorithm)
            if algorithm == FULL_MAP:
                output_cells, unique_cells, redundancy, status = map_cells, map_cells, 0.0, "exact_full_map"
            elif algorithm in PATCH_SIZES:
                output_cells = int(calls * PATCH_SIZES[algorithm] ** 2)
                unique_cells = redundancy = ""
                status = "unique_patch_coverage_not_logged"
            else:
                output_cells, unique_cells, redundancy, status = 0, 0, 0.0, "no_unet"
            output.append(
                {
                    **scope_fields(scope, size, structure),
                    "algorithm": algorithm,
                    "cases": len(scoped),
                    "unet_inference_calls": calls,
                    "total_predicted_patch_output_cells": output_cells,
                    "unique_in_map_predicted_cells": unique_cells,
                    "redundant_prediction_ratio": redundancy,
                    "total_neural_inference_seconds": seconds,
                    "mean_neural_inference_seconds_per_call": "" if not calls else seconds / calls,
                    "neural_cost_observability": status,
                }
            )
    return output


def efficiency_rows(cases, selected):
    output = []
    for scope, size, structure in scopes(cases):
        scoped = [case for case in cases if in_scope(case, size, structure)]
        baseline = result_rows(scoped, selected, MANHATTAN)
        baseline_expanded = sum_field(baseline, "expanded_nodes")
        for algorithm in ALGORITHMS:
            rows = result_rows(scoped, selected, algorithm)
            expanded = sum_field(rows, "expanded_nodes")
            saved = baseline_expanded - expanded
            seconds = neural_seconds(rows, algorithm)
            calls = call_count(rows, algorithm)
            output.append(
                {
                    **scope_fields(scope, size, structure),
                    "algorithm": algorithm,
                    "cases": len(scoped),
                    "manhattan_expanded_cells": baseline_expanded,
                    "algorithm_expanded_cells": expanded,
                    "expansion_reduction_vs_manhattan": saved,
                    "expansion_reduction_fraction_vs_manhattan": saved / max(baseline_expanded, 1),
                    "total_neural_inference_seconds": seconds,
                    "saved_expansions_per_neural_second": "" if seconds == 0 else saved / seconds,
                    "unet_inference_calls": calls,
                    "saved_expansions_per_inference_call": "" if calls == 0 else saved / calls,
                }
            )
    return output


def find(rows, scope, size, algorithm):
    key_size = "" if size is None else size
    return next(row for row in rows if row["scope"] == scope and row["map_size"] == key_size and row["algorithm"] == algorithm)


def number(value, digits=2):
    return "not logged" if value == "" else f"{float(value):.{digits}f}"


def write_report(path, cases, coverage, overlap, cache, costs, efficiency, source_dir, manhattan_path):
    lines = [
        "# Region-Cache Lazy-Patch U-Net Effectiveness Analysis",
        "",
        f"This is a post-hoc analysis of `{source_dir}/results.csv` plus the paired Manhattan baseline `{manhattan_path}` on {len(cases)} identical saved cases. It does not rerun A*, U-Net inference, map generation, or training.",
        "",
        "## Aggregate Search-Aware Computation",
        "",
        "| Method | Total map cells | U-Net calls | Patch output cells | Exact unique in-map predicted cells | Nominal output cells / map cell | Tie-break queries | Query coverage by predictions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        cov = find(coverage, "overall", None, algorithm)
        ov = find(overlap, "overall", None, algorithm)
        cost = find(costs, "overall", None, algorithm)
        lines.append(
            f"| {algorithm} | {cov['total_map_cells']} | {number(cost['unet_inference_calls'], 0)} | "
            f"{number(cov['total_predicted_patch_output_cells'], 0)} | {number(cov['unique_in_map_cells_evaluated_by_unet'], 0)} | "
            f"{number(cov['nominal_patch_output_cells_per_map_cell'], 3)} | {number(ov['tie_break_queried_cells'], 0)} | "
            f"{number(ov['tie_query_search_coverage_ratio'], 1)} |"
        )
    lines += [
        "",
        "For region-cache variants, `queries = hits + misses` and the query-coverage ratio is therefore 1.0. This is aggregate cache accounting combined with the implemented lazy-scoring semantics: a cache miss creates the patch that serves the query, and a hit reads an earlier patch. It is not a spatial trace measurement of where queries or patch footprints occurred.",
        "",
        "## Cache Reuse",
        "",
        "| Method | Cache queries | Hits | Misses / U-Net calls | Weighted hit rate | Queries per miss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for algorithm in (CACHE_32, CACHE_64):
        row = find(cache, "overall", None, algorithm)
        lines.append(
            f"| {algorithm} | {number(row['total_cache_queries'], 0)} | {number(row['cache_hits'], 0)} | {number(row['cache_misses'], 0)} | "
            f"{number(float(row['cache_hit_rate_weighted']) * 100, 2)}% | {number(row['cache_reuse_factor_queries_per_miss'], 1)} |"
        )
    lines += [
        "",
        "The weighted hit rate pools all queries. The CSV also retains the mean of per-case hit rates in `cache_reuse_analysis.csv`; the two summaries answer different questions and should not be conflated.",
        "",
        "## Expansion Cost-Benefit",
        "",
        "| Method | Expansions saved vs Manhattan | Reduction | Neural inference s | Saved expansions / neural s | Saved expansions / call |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for algorithm in (FULL_MAP, CACHE_32, CACHE_64):
        row = find(efficiency, "overall", None, algorithm)
        lines.append(
            f"| {algorithm} | {number(row['expansion_reduction_vs_manhattan'], 0)} | "
            f"{number(float(row['expansion_reduction_fraction_vs_manhattan']) * 100, 1)}% | "
            f"{number(row['total_neural_inference_seconds'], 3)} | {number(row['saved_expansions_per_neural_second'], 1)} | "
            f"{number(row['saved_expansions_per_inference_call'], 1)} |"
        )
    lines += [
        "",
        "## Limits of the Completed Logs",
        "",
        "The benchmark records counts but not patch centers, the union of in-map patch footprints, or expanded-node traces. Consequently, exact unique predicted in-map cells, exact redundant-prediction ratio, and predicted∩expanded-node overlap are marked `not_logged` rather than estimated. `total_predicted_patch_output_cells` is exact but counts all model output positions, including overlapping patches and out-of-map padded positions. Recovering the unavailable quantities would require rerunning the search with additional trace logging, which this analysis intentionally does not do.",
        "",
        "## Conclusion",
        "",
        "The completed results support a limited query-level conclusion: the lazy implementation only requests a secondary score when an active Manhattan-f tie can use it, and the aggregate cache counts show that these requests were mostly served by reuse. Cache hit rates, calls, inference time, and expansion reductions are directly recorded. By contrast, the completed logs cannot establish an exact spatial concentration ratio, exact patch-overlap redundancy, or a measured predicted∩expanded-node overlap; those require coordinate-level traces. Runtime conclusions remain CPU-specific.",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc effectiveness analysis for the region-cache benchmark.")
    parser.add_argument("--benchmark-dir", default="outputs/region_cache_large_benchmark")
    parser.add_argument("--manhattan-results", default="outputs/manhattan_baseline/results.csv")
    parser.add_argument("--output-dir", default="outputs/region_cache_effectiveness")
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    benchmark_dir = os.path.join(root, args.benchmark_dir)
    manhattan_path = os.path.join(root, args.manhattan_results)
    output_dir = os.path.join(root, args.output_dir)
    cases = load_cases(os.path.join(benchmark_dir, "cases.csv"))
    selected = select_and_validate(
        cases,
        read_csv(os.path.join(benchmark_dir, "results.csv")),
        read_csv(manhattan_path),
    )
    coverage = coverage_rows(cases, selected)
    overlap = overlap_rows(cases, selected)
    cache = cache_rows(cases, selected)
    costs = cost_rows(cases, selected)
    efficiency = efficiency_rows(cases, selected)
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, "coverage_analysis.csv"), coverage)
    write_csv(os.path.join(output_dir, "search_overlap_analysis.csv"), overlap)
    write_csv(os.path.join(output_dir, "cache_reuse_analysis.csv"), cache)
    write_csv(os.path.join(output_dir, "neural_cost_analysis.csv"), costs)
    write_csv(os.path.join(output_dir, "expansion_efficiency.csv"), efficiency)
    write_report(
        os.path.join(output_dir, "effectiveness_report.md"),
        cases,
        coverage,
        overlap,
        cache,
        costs,
        efficiency,
        args.benchmark_dir,
        args.manhattan_results,
    )
    print(f"Wrote five CSV analyses and report to {output_dir}")


if __name__ == "__main__":
    main()
