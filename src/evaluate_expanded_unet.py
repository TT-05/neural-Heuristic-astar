"""Evaluate old and expanded-data U-Net checkpoints on the held-out dataset split."""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

from analyze_critical_decisions import (
    critical_events_for_case,
    make_table,
    optimal_path_nodes,
    trace_unet_tiebreak_astar,
)
from expanded_unet_dataset import MAP_TYPES
from model import load_unet_heuristic, make_unet_heuristic


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def load_data(archive_path):
    import numpy as np

    archive = np.load(archive_path)
    return archive["grids"], archive["goals"], archive["labels"], archive["reference_starts"]


def tie_top1_accuracy(trace):
    scored = 0
    correct = 0
    for item in trace:
        current = item["expanded"]
        candidates = [entry for entry in item["open_before"] if entry["f"] == current["f"]]
        if len(candidates) < 2:
            continue
        oracle_distance = min(entry["true_distance"] for entry in candidates)
        scored += 1
        correct += int(current["true_distance"] == oracle_distance)
    return correct, scored


def evaluate_model(name, model, records, grids, goals, labels, starts):
    rows = []
    for record in records:
        index = int(record["index"])
        grid = grids[index].tolist()
        goal = tuple(int(value) for value in goals[index])
        start = tuple(int(value) for value in starts[index])
        distance_to_goal = labels[index].tolist()
        distance_from_start = __import__("bfs_label").compute_distance_to_goal(grid, start)
        optimal_cost = int(distance_to_goal[start[0]][start[1]])
        optimal_nodes = optimal_path_nodes(distance_from_start, distance_to_goal, optimal_cost)
        heuristic = make_unet_heuristic(model, grid, goal)
        unet_table = make_table(grid, goal, heuristic)
        reachable = [(r, c) for r, row in enumerate(distance_to_goal) for c, value in enumerate(row) if value >= 0]
        predictions = [unet_table[cell] for cell in reachable]
        targets = [distance_to_goal[r][c] for r, c in reachable]
        mae = mean(abs(prediction - target) for prediction, target in zip(predictions, targets))
        mse = mean((prediction - target) ** 2 for prediction, target in zip(predictions, targets))
        trace_result = trace_unet_tiebreak_astar(grid, start, goal, unet_table, distance_to_goal, optimal_nodes)
        if trace_result["cost"] != optimal_cost:
            raise AssertionError(f"{name} lost optimality on test map {index}")
        case = {"case_id": f"test_{index}", "structured_type": record["map_type"], "optimal_cost": optimal_cost}
        events = critical_events_for_case(case, trace_result, optimal_nodes, weak_margin=0.5)
        high = [event for event in events if int(event["recovery_cost"]) >= 20]
        high_unet = [event for event in high if event["error_type"] == "A_unet_ordering_error"]
        tie_correct, tie_total = tie_top1_accuracy(trace_result["trace"])
        rows.append(
            {
                "model": name,
                "index": index,
                "map_type": record["map_type"],
                "obstacle_rate": float(record["obstacle_rate"]),
                "mae": mae,
                "mse": mse,
                "expanded_nodes": trace_result["expanded"],
                "optimal": True,
                "tie_top1_correct": tie_correct,
                "tie_top1_total": tie_total,
                "high_impact_events": len(high),
                "high_impact_unet_ordering_errors": len(high_unet),
            }
        )
    return rows


def grouped_rows(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, scoped in sorted(groups.items()):
        result = {field: value for field, value in zip(fields, key)}
        result.update(
            {
                "cases": len(scoped),
                "mae": mean(float(row["mae"]) for row in scoped),
                "mse": mean(float(row["mse"]) for row in scoped),
                "mean_expanded_nodes": mean(float(row["expanded_nodes"]) for row in scoped),
                "optimality_rate": mean(float(row["optimal"]) for row in scoped),
                "tie_set_top1_accuracy": sum(int(row["tie_top1_correct"]) for row in scoped) / max(1, sum(int(row["tie_top1_total"]) for row in scoped)),
                "mean_high_impact_events": mean(float(row["high_impact_events"]) for row in scoped),
                "mean_high_impact_unet_ordering_errors": mean(float(row["high_impact_unet_ordering_errors"]) for row in scoped),
            }
        )
        output.append(result)
    return output


def plot_history(output_dir, history):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [float(row["train_loss"]) for row in history], label="train loss")
    plt.plot(epochs, [float(row["val_loss"]) for row in history], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=160)
    plt.close()


def dataset_statistics(output_dir, manifest):
    counts = Counter((row["split"], row["map_type"], row["obstacle_rate"]) for row in manifest)
    starts = [int(row["reference_start_distance"]) for row in manifest]
    lines = [
        "# Expanded Dataset Statistics",
        "",
        "## Baseline Dataset Audit",
        "",
        "- Maps: 500 total, randomly split into 400 train / 100 validation.",
        "- Structure distribution: open_random only; no maze_like, bottleneck, large_block, or narrow_corridor maps.",
        "- Obstacle-rate distribution: 0.2 only.",
        "- Map size distribution: 20x20 only.",
        "- Goal distribution: one uniformly selected free-cell goal per map. No start is stored because the original U-Net supervises a full distance-to-goal map rather than individual start-goal samples.",
        "",
        "## Expanded Dataset",
        "",
        f"- Maps: {len(manifest)}",
        f"- Unique grid hashes: {len({row['grid_sha256'] for row in manifest})}",
        "- Map size: 20x20 for every map, matching the model and benchmark.",
        "- Map types: open_random, maze_like, bottleneck, large_block, narrow_corridor.",
        "- Obstacle-rate strata: 0.1, 0.2, 0.3, 0.4.",
        "- Split is deterministic and disjoint by generated grid: 4,000 train / 500 validation / 500 test.",
        f"- Reference start-goal distance: mean {mean(starts):.2f}, min {min(starts)}, max {max(starts)}. Reference pairs are metadata; each training example supervises the complete distance map.",
        "",
        "| Split | Map type | Obstacle rate | Maps |",
        "|---|---|---:|---:|",
    ]
    for key, count in sorted(counts.items()):
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {count} |")
    with open(os.path.join(output_dir, "dataset_statistics.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_report(output_dir, old_rows, new_rows, structures):
    overall = grouped_rows(old_rows + new_rows, ["model"])
    lookup = {row["model"]: row for row in overall}
    old, new = lookup["old"], lookup["expanded"]
    lines = [
        "# Expanded Dataset Training Report",
        "",
        "The U-Net architecture, masked MSE loss, Adam optimizer, batch size, and 50-epoch schedule were unchanged. Only training-data quantity and composition changed.",
        "",
        "## Held-out Test Comparison",
        "",
        "| Model | MAE | MSE | Mean expanded nodes | Tie-set top-1 accuracy | High-impact events/case | Optimality |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(f"| {row['model']} | {row['mae']:.3f} | {row['mse']:.3f} | {row['mean_expanded_nodes']:.3f} | {row['tie_set_top1_accuracy']:.3f} | {row['mean_high_impact_events']:.3f} | {row['optimality_rate']:.3f} |")
    lines.extend(["", "## Structure Results", "", "See `structure_performance.csv` and `search_performance_comparison.csv` for all structure-level values.", "", "## Answers", ""])
    lines.append(f"1. Generalization {'improved' if new['mae'] < old['mae'] else 'did not improve'} on the independent balanced test split: MAE {old['mae']:.3f} -> {new['mae']:.3f}.")
    best_structure = max(structures, key=lambda row: next(item['mae'] for item in structures if item['model'] == 'old' and item['map_type'] == row['map_type']) - next(item['mae'] for item in structures if item['model'] == 'expanded' and item['map_type'] == row['map_type']))["map_type"]
    lines.append(f"2. The largest absolute MAE improvement is in `{best_structure}`.")
    lines.append(f"3. Tie-set oracle-top1 agreement changed from {old['tie_set_top1_accuracy']:.3f} to {new['tie_set_top1_accuracy']:.3f}.")
    lines.append(f"4. Mean U-Net tie-break expansions changed from {old['mean_expanded_nodes']:.3f} to {new['mean_expanded_nodes']:.3f}; all test paths remained optimal.")
    lines.append("5. This experiment isolates data coverage. Remaining errors after improvement are evidence that model capacity and/or the MSE objective may still limit search-relevant ordering, but do not establish that conclusion alone.")
    with open(os.path.join(output_dir, "final_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output_dir)
    archive_path = os.path.join(output_dir, "dataset", "expanded_unet_dataset.npz")
    manifest_path = os.path.join(output_dir, "dataset", "dataset_manifest.csv")
    expanded_path = os.path.join(output_dir, "unet_heuristic_expanded_best.pt")
    old_path = os.path.join(project_root, "checkpoints", "unet_heuristic.pt")
    records = [row for row in read_manifest(manifest_path) if row["split"] == "test"]
    grids, goals, labels, starts = load_data(archive_path)
    old_rows = evaluate_model("old", load_unet_heuristic(old_path), records, grids, goals, labels, starts)
    new_rows = evaluate_model("expanded", load_unet_heuristic(expanded_path), records, grids, goals, labels, starts)
    all_rows = old_rows + new_rows
    write_csv(os.path.join(output_dir, "evaluation_comparison.csv"), all_rows)
    structure_rows = grouped_rows(all_rows, ["model", "map_type"])
    write_csv(os.path.join(output_dir, "structure_performance.csv"), structure_rows)
    search_rows = grouped_rows(all_rows, ["model", "map_type", "obstacle_rate"])
    write_csv(os.path.join(output_dir, "search_performance_comparison.csv"), search_rows)
    with open(os.path.join(output_dir, "training_history.csv"), newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    plot_history(output_dir, history)
    dataset_statistics(output_dir, read_manifest(manifest_path))
    write_report(output_dir, old_rows, new_rows, structure_rows)
    print(f"Evaluated {len(records)} held-out maps per model in {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare original and expanded-data U-Nets.")
    parser.add_argument("--output-dir", default="outputs/expanded_dataset_training")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
