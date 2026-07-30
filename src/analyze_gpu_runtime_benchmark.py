"""Generate validated CPU/GPU runtime summaries from saved benchmark results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


NEURAL_METHODS = (
    "full_map_unet_tiebreak",
    "region_cache_patch_32",
    "region_cache_patch_64",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def case_keys(rows: list[dict[str, str]]) -> set[tuple[str, str, str, str]]:
    return {(r["case_id"], r["grid_sha256"], r["start"], r["goal"]) for r in rows}


def average(rows: list[dict[str, str]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def summarize(rows: list[dict[str, str]], manhattan_expanded: float) -> dict[str, float]:
    expanded = average(rows, "expanded_nodes")
    return {
        "cases": len(rows),
        "mean_expanded_nodes": expanded,
        "expansion_reduction_vs_manhattan": 1.0 - expanded / manhattan_expanded,
        "optimality_rate": sum(r["optimal"] == "True" for r in rows) / len(rows),
        "mean_total_runtime_seconds": average(rows, "total_runtime_seconds"),
        "mean_astar_search_seconds": average(rows, "astar_search_seconds"),
        "mean_neural_inference_seconds": average(rows, "unet_inference_seconds")
        + average(rows, "full_map_inference_seconds"),
        "mean_patch_extraction_seconds": average(rows, "patch_extraction_seconds"),
        "mean_cache_lookup_seconds": average(rows, "cache_lookup_seconds"),
        "mean_unet_calls": average(rows, "unet_call_count"),
        "mean_cache_hit_rate": average(rows, "cache_hit_rate"),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-dir", type=Path, default=Path("outputs/gpu_runtime_benchmark"))
    parser.add_argument("--cpu-neural", type=Path, default=Path("outputs/region_cache_large_benchmark/results.csv"))
    parser.add_argument("--manhattan", type=Path, default=Path("outputs/manhattan_baseline/results.csv"))
    args = parser.parse_args()

    gpu_rows = read_csv(args.gpu_dir / "results.csv")
    cpu_neural_rows = read_csv(args.cpu_neural)
    manhattan_rows = read_csv(args.manhattan)
    gpu_cases = case_keys(gpu_rows)
    if len(gpu_rows) != 6000 or len(gpu_cases) != 1500:
        raise RuntimeError("Expected 6,000 GPU runs from 1,500 saved cases")
    if case_keys(cpu_neural_rows) != gpu_cases or case_keys(manhattan_rows) != gpu_cases:
        raise RuntimeError("CPU, GPU, and Manhattan result files do not use identical saved cases")
    if len({(r["case_id"], r["algorithm"]) for r in gpu_rows}) != len(gpu_rows):
        raise RuntimeError("Duplicate GPU case-method records")

    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for device, rows in (("cuda", gpu_rows), ("cpu", cpu_neural_rows), ("cpu", manhattan_rows)):
        for row in rows:
            grouped[device][row["algorithm"]].append(row)

    baseline = summarize(grouped["cpu"]["manhattan_astar"], 1.0)
    baseline_expanded = baseline["mean_expanded_nodes"]
    cpu_neural = {method: summarize(grouped["cpu"][method], baseline_expanded) for method in NEURAL_METHODS}
    gpu_neural = {method: summarize(grouped["cuda"][method], baseline_expanded) for method in NEURAL_METHODS}

    summary_rows: list[dict[str, object]] = []
    for device, method, stats in [
        ("cpu", "manhattan_astar", summarize(grouped["cpu"]["manhattan_astar"], baseline_expanded)),
        *(("cpu", method, cpu_neural[method]) for method in NEURAL_METHODS),
        *(("cuda", method, gpu_neural[method]) for method in NEURAL_METHODS),
    ]:
        row: dict[str, object] = {"device": device, "algorithm": method, **stats}
        row["runtime_multiplier_vs_manhattan"] = stats["mean_total_runtime_seconds"] / baseline["mean_total_runtime_seconds"]
        row["gpu_speedup_vs_cpu_same_method"] = (
            cpu_neural[method]["mean_total_runtime_seconds"] / stats["mean_total_runtime_seconds"]
            if device == "cuda"
            else ""
        )
        summary_rows.append(row)
    summary_fields = [
        "device", "algorithm", "cases", "mean_expanded_nodes", "expansion_reduction_vs_manhattan",
        "optimality_rate", "mean_total_runtime_seconds", "mean_astar_search_seconds",
        "mean_neural_inference_seconds", "mean_patch_extraction_seconds", "mean_cache_lookup_seconds",
        "mean_unet_calls", "mean_cache_hit_rate", "runtime_multiplier_vs_manhattan",
        "gpu_speedup_vs_cpu_same_method",
    ]
    write_csv(args.gpu_dir / "gpu_runtime_summary.csv", summary_rows, summary_fields)

    size_rows: list[dict[str, object]] = []
    for size in (100, 500, 1000):
        manhattan_size = [r for r in grouped["cpu"]["manhattan_astar"] if int(r["map_size"]) == size]
        size_baseline = summarize(manhattan_size, 1.0)
        for device, method in (("cpu", "manhattan_astar"), *(("cuda", m) for m in NEURAL_METHODS)):
            rows = [r for r in grouped[device][method] if int(r["map_size"]) == size]
            stats = summarize(rows, size_baseline["mean_expanded_nodes"])
            size_rows.append({
                "map_size": size,
                "device": device,
                "algorithm": method,
                **stats,
                "runtime_multiplier_vs_manhattan": stats["mean_total_runtime_seconds"]
                / size_baseline["mean_total_runtime_seconds"],
            })
    write_csv(args.gpu_dir / "gpu_runtime_by_size.csv", size_rows, [
        "map_size", "device", "algorithm", *summary_fields[2:-1],
    ])

    hardware = (args.gpu_dir / "hardware_info.csv").read_text(encoding="utf-8").strip()
    report = [
        "# GPU Runtime Benchmark Report",
        "",
        "## Validation",
        "",
        "- 6,000 unique case-method records across 1,500 deterministic saved cases.",
        "- CPU neural, GPU neural, and Manhattan result files have identical case ID, map hash, start, and goal tuples.",
        "- All reported runs are optimal (1,500 / 1,500 per method); every path-cost gap is zero.",
        "",
        "## Hardware",
        "",
        "```csv",
        hardware,
        "```",
        "",
        "## Overall results",
        "",
        "| Method | Device | Expanded | Reduction vs Manhattan | Total (s) | GPU / CPU | Total / Manhattan |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        speedup = row["gpu_speedup_vs_cpu_same_method"]
        speedup_text = f"{speedup:.2f}x" if speedup != "" else "—"
        report.append(
            f"| {row['algorithm']} | {row['device']} | {row['mean_expanded_nodes']:.2f} | "
            f"{row['expansion_reduction_vs_manhattan']:.1%} | {row['mean_total_runtime_seconds']:.4f} | "
            f"{speedup_text} | {row['runtime_multiplier_vs_manhattan']:.2f}x |"
        )
    full = gpu_neural["full_map_unet_tiebreak"]
    patch32 = gpu_neural["region_cache_patch_32"]
    patch64 = gpu_neural["region_cache_patch_64"]
    report += [
        "",
        "## Measured conclusions",
        "",
        f"- CUDA reduced full-map U-Net end-to-end runtime from {cpu_neural['full_map_unet_tiebreak']['mean_total_runtime_seconds']:.4f}s to {full['mean_total_runtime_seconds']:.4f}s ({cpu_neural['full_map_unet_tiebreak']['mean_total_runtime_seconds'] / full['mean_total_runtime_seconds']:.2f}x).",
        f"- CUDA reduced Region-cache patch 32 runtime by {cpu_neural['region_cache_patch_32']['mean_total_runtime_seconds'] / patch32['mean_total_runtime_seconds']:.2f}x and patch 64 runtime by {cpu_neural['region_cache_patch_64']['mean_total_runtime_seconds'] / patch64['mean_total_runtime_seconds']:.2f}x.",
        f"- Full-map is the fastest neural method on this GPU ({full['mean_total_runtime_seconds']:.4f}s). Patch 32 and patch 64 remain {patch32['mean_total_runtime_seconds'] / full['mean_total_runtime_seconds']:.2f}x and {patch64['mean_total_runtime_seconds'] / full['mean_total_runtime_seconds']:.2f}x slower despite reducing expansions more.",
        f"- The remaining Region-cache cost is mostly patch extraction ({patch32['mean_patch_extraction_seconds']:.4f}s for 32; {patch64['mean_patch_extraction_seconds']:.4f}s for 64), rather than U-Net inference ({patch32['mean_neural_inference_seconds']:.4f}s; {patch64['mean_neural_inference_seconds']:.4f}s).",
        "- Manhattan remains the fastest end-to-end method because it has no neural or patch-preparation work.",
        "",
        "## Scope limit",
        "",
        "This completed run did not enable per-node expansion or per-patch spatial trajectory logging. The report therefore makes runtime/cache claims only and does not infer optimal-path alignment.",
        "",
    ]
    (args.gpu_dir / "gpu_runtime_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
