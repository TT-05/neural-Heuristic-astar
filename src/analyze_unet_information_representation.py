import argparse
import csv
import math
import os
import random
from collections import defaultdict

import numpy as np

from analyze_tie_set_ordering import checkpoint_path, manhattan, mean, pearson, prediction_table, spearman
from bfs_label import compute_distance_to_goal
from gen_map import gen_map
from model import load_mlp_heuristic, load_unet_heuristic, make_mlp_table_heuristic, make_unet_heuristic
from structured_maps import STRUCTURED_TYPES, generate_structured_map


METHODS = ["manhattan", "mlp", "unet"]
GAP_BINS = [(4, 7), (8, 11), (12, 10**9)]
MANHATTAN_BINS = [(0, 6), (7, 12), (13, 10**9)]


def write_csv(path, rows):
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def bin_label(value, bins):
    for low, high in bins:
        if low <= value <= high:
            return f"{low}-{high if high < 10**9 else 'inf'}"
    return "out_of_range"


def free_cells(grid):
    return [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 0]


def reachable_cells(distance_grid):
    return [(r, c) for r, row in enumerate(distance_grid) for c, value in enumerate(row) if value >= 0]


def choose_goal(grid, rng, min_reachable=60):
    candidates = free_cells(grid)
    rng.shuffle(candidates)
    for goal in candidates:
        distance_grid = compute_distance_to_goal(grid, goal)
        reachable = reachable_cells(distance_grid)
        if len(reachable) >= min_reachable:
            return goal, distance_grid
    return None, None


def generate_map(map_size, seed, obstacle_rate, map_type):
    if map_type == "random":
        return gen_map(map_size, map_size, seed=seed, obstacle_rate=obstacle_rate)
    return generate_structured_map(map_size, map_size, seed, obstacle_rate, map_type)


def map_id(map_type, seed, obstacle_rate, goal):
    return f"{map_type}_rate{obstacle_rate}_seed{seed}_g{goal[0]}-{goal[1]}"


def prediction_tables(grid, goal, mlp_model, unet_model):
    mlp = prediction_table(grid, goal, make_mlp_table_heuristic(mlp_model, grid, goal))
    unet = prediction_table(grid, goal, make_unet_heuristic(unet_model, grid, goal))
    manhattan_table = {cell: float(manhattan(cell, goal)) for cell in free_cells(grid)}
    return {"manhattan": manhattan_table, "mlp": mlp, "unet": unet}


def connectivity_pairs_for_map(
    grid,
    goal,
    distance_grid,
    tables,
    rng,
    map_type,
    seed,
    obstacle_rate,
    manhattan_tolerance,
    true_gap_threshold,
    max_pairs,
):
    cells = reachable_cells(distance_grid)
    by_manhattan = defaultdict(list)
    for cell in cells:
        by_manhattan[manhattan(cell, goal)].append(cell)

    candidate_pairs = []
    manhattan_values = sorted(by_manhattan)
    for h_value in manhattan_values:
        left_cells = []
        for near_h in range(h_value - manhattan_tolerance, h_value + manhattan_tolerance + 1):
            left_cells.extend(by_manhattan.get(near_h, []))
        for a in by_manhattan[h_value]:
            for b in left_cells:
                if a == b:
                    continue
                true_gap = distance_grid[a[0]][a[1]] - distance_grid[b[0]][b[1]]
                if true_gap >= true_gap_threshold:
                    candidate_pairs.append((a, b, true_gap))
    rng.shuffle(candidate_pairs)
    rows = []
    identity = map_id(map_type, seed, obstacle_rate, goal)
    for pair_index, (a, b, true_gap) in enumerate(candidate_pairs[:max_pairs]):
        manhattan_gap = manhattan(a, goal) - manhattan(b, goal)
        row = {
            "map_id": identity,
            "map_type": map_type,
            "seed": seed,
            "obstacle_rate": obstacle_rate,
            "goal": f"{goal[0]},{goal[1]}",
            "pair_index": pair_index,
            "a": f"{a[0]},{a[1]}",
            "b": f"{b[0]},{b[1]}",
            "true_distance_a": distance_grid[a[0]][a[1]],
            "true_distance_b": distance_grid[b[0]][b[1]],
            "true_gap": true_gap,
            "manhattan_a": manhattan(a, goal),
            "manhattan_b": manhattan(b, goal),
            "manhattan_gap": manhattan_gap,
            "true_gap_bin": bin_label(true_gap, GAP_BINS),
            "manhattan_bin": bin_label(min(manhattan(a, goal), manhattan(b, goal)), MANHATTAN_BINS),
        }
        for method in METHODS:
            pred_gap = tables[method][a] - tables[method][b]
            row[f"{method}_h_a"] = tables[method][a]
            row[f"{method}_h_b"] = tables[method][b]
            row[f"{method}_pred_gap"] = pred_gap
            row[f"{method}_correct"] = int(pred_gap > 0)
        rows.append(row)
    return rows


def aggregate_connectivity(pair_rows):
    output = []
    scopes = [("all", "all", pair_rows)]
    for key in ["map_type", "true_gap_bin", "manhattan_bin"]:
        groups = defaultdict(list)
        for row in pair_rows:
            groups[row[key]].append(row)
        scopes.extend((key, value, rows) for value, rows in sorted(groups.items()))

    for scope_type, scope_value, rows in scopes:
        true_gaps = [float(row["true_gap"]) for row in rows]
        for method in METHODS:
            pred_gaps = [float(row[f"{method}_pred_gap"]) for row in rows]
            output.append(
                {
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "method": method,
                    "pairs": len(rows),
                    "connectivity_discrimination_accuracy": mean(float(row[f"{method}_correct"]) for row in rows),
                    "connectivity_margin_pearson": pearson(true_gaps, pred_gaps),
                    "connectivity_margin_spearman": spearman(true_gaps, pred_gaps),
                    "mean_true_gap": mean(true_gaps),
                    "mean_pred_gap": mean(pred_gaps),
                }
            )
    return output


def neighbors4(grid, cell):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    r, c = cell
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def perturb_candidates(grid, goal, rng):
    candidates = []
    for cell in free_cells(grid):
        if cell == goal:
            continue
        free_neighbors = sum(1 for n in neighbors4(grid, cell) if grid[n[0]][n[1]] == 0)
        if free_neighbors >= 2:
            candidates.append(("close", cell))
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            cell = (r, c)
            if value != 1:
                continue
            free_neighbors = sum(1 for n in neighbors4(grid, cell) if grid[n[0]][n[1]] == 0)
            if free_neighbors >= 2:
                candidates.append(("open", cell))
    rng.shuffle(candidates)
    return candidates


def apply_perturbation(grid, edit):
    kind, cell = edit
    new_grid = [list(row) for row in grid]
    new_grid[cell[0]][cell[1]] = 0 if kind == "open" else 1
    return new_grid


def perturbation_metrics(before_grid, after_grid, goal, before_tables, after_tables, true_threshold, pred_threshold):
    before_distance = compute_distance_to_goal(before_grid, goal)
    after_distance = compute_distance_to_goal(after_grid, goal)
    common = [
        cell
        for cell in free_cells(before_grid)
        if after_grid[cell[0]][cell[1]] == 0 and before_distance[cell[0]][cell[1]] >= 0 and after_distance[cell[0]][cell[1]] >= 0
    ]
    if not common:
        return None, []
    rows = []
    delta_true = []
    delta_by_method = {method: [] for method in METHODS}
    for cell in common:
        true_delta = after_distance[cell[0]][cell[1]] - before_distance[cell[0]][cell[1]]
        delta_true.append(true_delta)
        row = {
            "cell": f"{cell[0]},{cell[1]}",
            "true_before": before_distance[cell[0]][cell[1]],
            "true_after": after_distance[cell[0]][cell[1]],
            "delta_true": true_delta,
        }
        for method in METHODS:
            delta_pred = after_tables[method][cell] - before_tables[method][cell]
            delta_by_method[method].append(delta_pred)
            row[f"delta_{method}"] = delta_pred
        rows.append(row)

    affected_true = {i for i, value in enumerate(delta_true) if abs(value) > true_threshold}
    metrics = {"valid_common_cells": len(common), "true_affected_cells": len(affected_true)}
    true_abs = np.array([abs(value) for value in delta_true], dtype=float)
    for method in METHODS:
        pred_delta = delta_by_method[method]
        affected_pred = {i for i, value in enumerate(pred_delta) if abs(value) > pred_threshold}
        intersection = len(affected_true & affected_pred)
        union = len(affected_true | affected_pred)
        precision = intersection / len(affected_pred) if affected_pred else 0.0
        recall = intersection / len(affected_true) if affected_true else 0.0
        iou = intersection / union if union else 0.0
        pred_abs = np.array([abs(value) for value in pred_delta], dtype=float)
        if pred_abs.sum() > 0 and true_abs.sum() > 0:
            true_weighted_center = np.average(np.arange(len(true_abs)), weights=true_abs)
            pred_weighted_center = np.average(np.arange(len(pred_abs)), weights=pred_abs)
            locality = 1.0 / (1.0 + abs(true_weighted_center - pred_weighted_center) / max(1, len(common)))
        else:
            locality = 0.0
        metrics.update(
            {
                f"{method}_geometry_sensitivity_pearson": pearson(delta_true, pred_delta),
                f"{method}_geometry_sensitivity_spearman": spearman(delta_true, pred_delta),
                f"{method}_affected_iou": iou,
                f"{method}_affected_precision": precision,
                f"{method}_affected_recall": recall,
                f"{method}_perturbation_locality": locality,
                f"{method}_mean_abs_delta": mean(abs(value) for value in pred_delta),
            }
        )
    return metrics, rows


def find_perturbation_for_map(grid, goal, tables, mlp_model, unet_model, rng, true_threshold, pred_threshold):
    before_distance = compute_distance_to_goal(grid, goal)
    for edit in perturb_candidates(grid, goal, rng)[:80]:
        new_grid = apply_perturbation(grid, edit)
        if new_grid[goal[0]][goal[1]] == 1:
            continue
        after_distance = compute_distance_to_goal(new_grid, goal)
        if len(reachable_cells(after_distance)) < 40:
            continue
        after_tables = prediction_tables(new_grid, goal, mlp_model, unet_model)
        metrics, cell_rows = perturbation_metrics(grid, new_grid, goal, tables, after_tables, true_threshold, pred_threshold)
        if metrics and metrics["true_affected_cells"] >= 5:
            return edit, new_grid, after_tables, metrics, cell_rows
    return None, None, None, None, []


def aggregate_geometry(rows):
    output = []
    scopes = [("all", "all", rows)]
    groups = defaultdict(list)
    for row in rows:
        groups[row["map_type"]].append(row)
    scopes.extend(("map_type", value, scoped) for value, scoped in sorted(groups.items()))
    for scope_type, scope_value, scoped in scopes:
        for method in METHODS:
            output.append(
                {
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "method": method,
                    "perturbations": len(scoped),
                    "geometry_sensitivity_pearson": mean(row[f"{method}_geometry_sensitivity_pearson"] for row in scoped),
                    "geometry_sensitivity_spearman": mean(row[f"{method}_geometry_sensitivity_spearman"] for row in scoped),
                    "influence_region_iou": mean(row[f"{method}_affected_iou"] for row in scoped),
                    "influence_region_precision": mean(row[f"{method}_affected_precision"] for row in scoped),
                    "influence_region_recall": mean(row[f"{method}_affected_recall"] for row in scoped),
                    "perturbation_locality": mean(row[f"{method}_perturbation_locality"] for row in scoped),
                    "mean_abs_delta": mean(row[f"{method}_mean_abs_delta"] for row in scoped),
                }
            )
    return output


def grid_array(grid):
    return np.array(grid, dtype=float)


def distance_array(distance_grid):
    array = np.array(distance_grid, dtype=float)
    array[array < 0] = np.nan
    return array


def table_array(grid, table):
    array = np.full((len(grid), len(grid[0])), np.nan)
    for (r, c), value in table.items():
        array[r, c] = value
    return array


def save_connectivity_example(path, grid, goal, pair_row, true_grid, unet_table):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = tuple(int(x) for x in pair_row["a"].split(","))
    b = tuple(int(x) for x in pair_row["b"].split(","))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    axes[0].imshow(grid_array(grid), cmap="gray_r")
    axes[0].scatter([goal[1]], [goal[0]], c="gold", marker="*", s=120)
    axes[0].scatter([a[1]], [a[0]], c="red", marker="o", s=70, label="A worse")
    axes[0].scatter([b[1]], [b[0]], c="cyan", marker="o", s=70, label="B better")
    axes[0].set_title("Map and pair")
    axes[0].legend(fontsize=7)
    im1 = axes[1].imshow(distance_array(true_grid), cmap="viridis")
    axes[1].set_title("True distance")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(table_array(grid, unet_table), cmap="viridis")
    axes[2].set_title("U-Net h-map")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_perturbation_example(path, before_grid, after_grid, true_before, true_after, unet_before, unet_after):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta_true = distance_array(true_after) - distance_array(true_before)
    delta_unet = table_array(before_grid, unet_after) - table_array(before_grid, unet_before)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    panels = [
        ("Original map", grid_array(before_grid), "gray_r"),
        ("Perturbed map", grid_array(after_grid), "gray_r"),
        ("True before", distance_array(true_before), "viridis"),
        ("True after", distance_array(true_after), "viridis"),
        ("Delta true", delta_true, "coolwarm"),
        ("U-Net before", table_array(before_grid, unet_before), "viridis"),
        ("U-Net after", table_array(before_grid, unet_after), "viridis"),
        ("Delta U-Net", delta_unet, "coolwarm"),
    ]
    for ax, (title, array, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(array, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary(path, connectivity_stats, geometry_stats, pair_rows, perturb_rows):
    def stat(method, key, rows):
        row = next(item for item in rows if item["scope_type"] == "all" and item["method"] == method)
        return row[key]

    unet_conn = stat("unet", "connectivity_discrimination_accuracy", connectivity_stats)
    mlp_conn = stat("mlp", "connectivity_discrimination_accuracy", connectivity_stats)
    manhattan_conn = stat("manhattan", "connectivity_discrimination_accuracy", connectivity_stats)
    unet_geo = stat("unet", "geometry_sensitivity_spearman", geometry_stats)
    mlp_geo = stat("mlp", "geometry_sensitivity_spearman", geometry_stats)
    manhattan_geo = stat("manhattan", "geometry_sensitivity_spearman", geometry_stats)
    with open(path, "w", encoding="utf-8") as file:
        file.write("# U-Net Information Representation Validation\n\n")
        file.write(
            "This analysis tests the trained h-map directly. It does not use A* expanded nodes as proof, does not modify A*, "
            "and does not train or change model checkpoints.\n\n"
        )
        file.write("## Experiment Scale\n\n")
        file.write(f"- connectivity pairs: {len(pair_rows)}\n")
        file.write(f"- perturbations: {len(perturb_rows)}\n\n")
        file.write("## Connectivity Representation\n\n")
        file.write("| method | accuracy | Pearson margin | Spearman margin |\n|---|---:|---:|---:|\n")
        for method in METHODS:
            row = next(item for item in connectivity_stats if item["scope_type"] == "all" and item["method"] == method)
            file.write(
                f"| {method} | {row['connectivity_discrimination_accuracy']:.3f} | "
                f"{row['connectivity_margin_pearson']:.3f} | {row['connectivity_margin_spearman']:.3f} |\n"
            )
        file.write("\n")
        file.write("## Geometry Perturbation Sensitivity\n\n")
        file.write("| method | Spearman delta | Pearson delta | IoU | precision | recall | locality |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for method in METHODS:
            row = next(item for item in geometry_stats if item["scope_type"] == "all" and item["method"] == method)
            file.write(
                f"| {method} | {row['geometry_sensitivity_spearman']:.3f} | "
                f"{row['geometry_sensitivity_pearson']:.3f} | {row['influence_region_iou']:.3f} | "
                f"{row['influence_region_precision']:.3f} | {row['influence_region_recall']:.3f} | "
                f"{row['perturbation_locality']:.3f} |\n"
            )
        file.write("\n")
        file.write("## Required Answers\n\n")
        if unet_conn > max(manhattan_conn, mlp_conn):
            file.write(
                "1. Connectivity information: supported. U-Net has the strongest connectivity discrimination among tested methods.\n"
            )
        else:
            file.write(
                "1. Connectivity information: not strongly supported as uniquely U-Net. U-Net does not clearly exceed all baselines.\n"
            )
        if unet_geo > max(manhattan_geo, mlp_geo):
            file.write(
                "2. Geometry perturbation response: supported. U-Net delta h aligns best with true-distance delta under obstacle edits.\n"
            )
        else:
            file.write(
                "2. Geometry perturbation response: weak or mixed. U-Net does not clearly exceed all baselines on delta alignment.\n"
            )
        file.write(
            f"3. Baseline comparison: connectivity accuracy is Manhattan={manhattan_conn:.3f}, MLP={mlp_conn:.3f}, "
            f"U-Net={unet_conn:.3f}; perturbation Spearman is Manhattan={manhattan_geo:.3f}, MLP={mlp_geo:.3f}, "
            f"U-Net={unet_geo:.3f}.\n"
        )
        if unet_conn > manhattan_conn and unet_geo > manhattan_geo:
            file.write(
                "4. Environment information beyond value accuracy: supported relative to Manhattan, because U-Net reacts to obstacles "
                "and connectivity where Manhattan is structurally insensitive.\n"
            )
        else:
            file.write(
                "4. Environment information beyond value accuracy: only partially supported; evidence should be stated cautiously.\n"
            )
        file.write(
            "5. Geometry vs connectivity: these are not fully separable in this setup. Geometry perturbations matter primarily when "
            "they change connectivity and shortest-path distance fields.\n"
        )
        file.write(
            "6. Remaining unverified: this does not prove causality for search efficiency, does not prove generalization beyond the "
            "tested map distribution, and does not establish that the representation is sufficient for a new algorithm.\n"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Validate U-Net h-map geometry/connectivity information.")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--map-size", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--pairs-per-map", type=int, default=160)
    parser.add_argument("--max-perturbations", type=int, default=100)
    parser.add_argument("--manhattan-tolerance", type=int, default=1)
    parser.add_argument("--true-gap-threshold", type=int, default=5)
    parser.add_argument("--true-delta-threshold", type=float, default=2.0)
    parser.add_argument("--pred-delta-threshold", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs/unet_information_representation")
    return parser.parse_args()


def main(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output_dir)
    example_dir = os.path.join(output_dir, "example_visualizations")
    conn_example_dir = os.path.join(output_dir, "connectivity_examples")
    perturb_example_dir = os.path.join(output_dir, "perturbation_examples")
    for directory in [output_dir, example_dir, conn_example_dir, perturb_example_dir]:
        os.makedirs(directory, exist_ok=True)

    rng = random.Random(0)
    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint, "unet"))

    map_types = ["random"] + STRUCTURED_TYPES
    obstacle_rates = [0.1, 0.2, 0.3]
    connectivity_rows = []
    perturbation_rows = []
    saved_conn = 0
    saved_perturb = 0
    perturbation_limit_by_type = max(1, args.max_perturbations // len(map_types))
    perturbation_counts = {map_type: 0 for map_type in map_types}

    for map_type in map_types:
        for obstacle_rate in obstacle_rates:
            for seed in range(args.seeds):
                grid = generate_map(args.map_size, seed, obstacle_rate, map_type)
                goal, distance_grid = choose_goal(grid, rng)
                if goal is None:
                    continue
                tables = prediction_tables(grid, goal, mlp_model, unet_model)
                pair_rows = connectivity_pairs_for_map(
                    grid,
                    goal,
                    distance_grid,
                    tables,
                    rng,
                    map_type,
                    seed,
                    obstacle_rate,
                    args.manhattan_tolerance,
                    args.true_gap_threshold,
                    args.pairs_per_map,
                )
                connectivity_rows.extend(pair_rows)
                if pair_rows and saved_conn < 8:
                    conn_path = os.path.join(conn_example_dir, f"connectivity_example_{saved_conn:02d}.png")
                    save_connectivity_example(
                        conn_path,
                        grid,
                        goal,
                        pair_rows[0],
                        distance_grid,
                        tables["unet"],
                    )
                    save_connectivity_example(
                        os.path.join(example_dir, f"connectivity_example_{saved_conn:02d}.png"),
                        grid,
                        goal,
                        pair_rows[0],
                        distance_grid,
                        tables["unet"],
                    )
                    saved_conn += 1

                if perturbation_counts[map_type] < perturbation_limit_by_type:
                    edit, after_grid, after_tables, metrics, _ = find_perturbation_for_map(
                        grid,
                        goal,
                        tables,
                        mlp_model,
                        unet_model,
                        rng,
                        args.true_delta_threshold,
                        args.pred_delta_threshold,
                    )
                    if metrics:
                        identity = map_id(map_type, seed, obstacle_rate, goal)
                        row = {
                            "map_id": identity,
                            "map_type": map_type,
                            "seed": seed,
                            "obstacle_rate": obstacle_rate,
                            "goal": f"{goal[0]},{goal[1]}",
                            "edit_type": edit[0],
                            "edit_cell": f"{edit[1][0]},{edit[1][1]}",
                            **metrics,
                        }
                        perturbation_rows.append(row)
                        perturbation_counts[map_type] += 1
                        if saved_perturb < 8:
                            example_path = os.path.join(
                                perturb_example_dir, f"perturbation_example_{saved_perturb:02d}.png"
                            )
                            true_before = compute_distance_to_goal(grid, goal)
                            true_after = compute_distance_to_goal(after_grid, goal)
                            save_perturbation_example(
                                example_path,
                                grid,
                                after_grid,
                                true_before,
                                true_after,
                                tables["unet"],
                                after_tables["unet"],
                            )
                            save_perturbation_example(
                                os.path.join(example_dir, f"perturbation_example_{saved_perturb:02d}.png"),
                                grid,
                                after_grid,
                                true_before,
                                true_after,
                                tables["unet"],
                                after_tables["unet"],
                            )
                            saved_perturb += 1

    if not connectivity_rows:
        raise RuntimeError("No connectivity pairs found.")
    if not perturbation_rows:
        raise RuntimeError("No valid perturbations found.")

    connectivity_stats = aggregate_connectivity(connectivity_rows)
    geometry_stats = aggregate_geometry(perturbation_rows)
    write_csv(os.path.join(output_dir, "connectivity_pair_results.csv"), connectivity_rows)
    write_csv(os.path.join(output_dir, "connectivity_statistics.csv"), connectivity_stats)
    write_csv(os.path.join(output_dir, "geometry_perturbation_results.csv"), perturbation_rows)
    write_csv(os.path.join(output_dir, "geometry_sensitivity_statistics.csv"), geometry_stats)
    write_summary(
        os.path.join(output_dir, "information_representation_summary.md"),
        connectivity_stats,
        geometry_stats,
        connectivity_rows,
        perturbation_rows,
    )
    print(f"Saved U-Net information representation validation to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
