import argparse
import csv
import heapq
import json
import math
import os
from collections import Counter, defaultdict

from analyze_tie_set_ordering import checkpoint_path, manhattan, pearson, spearman
from astar import astar_search
from bfs_label import compute_distance_to_goal
from gen_map import gen_map
from model import load_mlp_heuristic, load_unet_heuristic, make_mlp_table_heuristic, make_unet_heuristic
from structured_maps import STRUCTURED_TYPES, generate_structured_map


METHOD_ROWS = {
    "manhattan": "manhattan",
    "mlp_tiebreak": "manhattan_mlp_tiebreak",
    "unet_tiebreak": "manhattan_unet_tiebreak",
    "oracle_tiebreak": "manhattan_true_distance_tiebreak",
}
METHOD_LABELS = {
    "manhattan": "Manhattan",
    "mlp_tiebreak": "MLP tie-break",
    "unet_tiebreak": "U-Net tie-break",
    "oracle_tiebreak": "True-distance tie-break",
}
UNET_VARIANT_LABELS = {
    "unet_tiebreak": "Manhattan + U-Net tie-break",
    "unet_direct": "Direct U-Net A*",
    "unet_calibrated_5": "U-Net calibrated n=5",
    "unet_calibrated_10": "U-Net calibrated n=10",
    "unet_calibrated_20": "U-Net calibrated n=20",
    "unet_calibrated_manhattan": "Manhattan-distance calibrated U-Net",
    "unet_calibrated_oracle": "Oracle calibrated U-Net",
}
UNET_VARIANT_KEYS = list(UNET_VARIANT_LABELS)
COMBINED_LOSS_MODEL_LABELS = {
    "mse": "MSE",
    "ranking": "Ranking",
    "ranking_adm": "Ranking + Admissibility",
    "ranking_adm_cons": "Ranking + Admissibility + Consistency",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def to_int(value):
    return int(float(value))


def to_float(value, default=0.0):
    if value in ("", None):
        return default
    return float(value)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def result_key(row):
    return (
        row["seed"],
        row["map_size"],
        row["obstacle_rate"],
        row.get("map_mode", "random"),
        row.get("structured_type", "random"),
        row.get("start_row", ""),
        row.get("start_col", ""),
        row.get("goal_row", ""),
        row.get("goal_col", ""),
    )


def group_results(paths):
    grouped = {}
    for source, path in paths:
        if not os.path.exists(path):
            continue
        for row in read_csv(path):
            heuristic = row.get("heuristic")
            if heuristic not in set(METHOD_ROWS.values()) | {"mlp_table", "unet"}:
                continue
            row = dict(row)
            row["source"] = source
            row.setdefault("map_mode", "random")
            row.setdefault("structured_type", "random")
            grouped.setdefault((source, *result_key(row)), {})[heuristic] = row
    return [methods for methods in grouped.values() if all(name in methods for name in METHOD_ROWS.values())]


def expanded(methods, method_key):
    return to_float(methods[METHOD_ROWS[method_key]]["expanded_nodes"])


def case_identity(methods):
    sample = methods[METHOD_ROWS["manhattan"]]
    return (
        f"{sample.get('map_mode', 'random')}_{sample.get('structured_type', 'random')}"
        f"_rate{sample['obstacle_rate']}_seed{sample['seed']}"
        f"_s{sample['start_row']}-{sample['start_col']}_g{sample['goal_row']}-{sample['goal_col']}"
    )


def select_cases(grouped, max_cases):
    selected = []
    seen = set()

    def add(items):
        for methods in items:
            ident = case_identity(methods)
            if ident not in seen:
                selected.append(methods)
                seen.add(ident)
            if len(selected) >= max_cases:
                return

    all_valid = [m for m in grouped if m[METHOD_ROWS["unet_tiebreak"]].get("path_found") == "True"]
    unet_wins = sorted(all_valid, key=lambda m: expanded(m, "unet_tiebreak") - expanded(m, "mlp_tiebreak"))
    mlp_wins = sorted(all_valid, key=lambda m: expanded(m, "unet_tiebreak") - expanded(m, "mlp_tiebreak"), reverse=True)
    add(unet_wins[: max(4, max_cases // 5)])
    add(mlp_wins[: max(4, max_cases // 5)])

    for structure in ["random", *STRUCTURED_TYPES]:
        scoped = [m for m in all_valid if m[METHOD_ROWS["manhattan"]].get("structured_type", "random") == structure]
        add(sorted(scoped, key=lambda m: abs(expanded(m, "unet_tiebreak") - expanded(m, "mlp_tiebreak")), reverse=True)[:4])

    add(all_valid)
    return selected[:max_cases]


def build_grid(sample):
    seed = to_int(sample["seed"])
    size = to_int(sample["map_size"])
    obstacle_rate = to_float(sample["obstacle_rate"])
    map_mode = sample.get("map_mode", "random")
    structured_type = sample.get("structured_type", "random")
    if map_mode == "structured":
        grid = generate_structured_map(size, size, seed, obstacle_rate, structured_type)
    else:
        grid = gen_map(size, size, seed=seed, obstacle_rate=obstacle_rate)
    start = (to_int(sample["start_row"]), to_int(sample["start_col"]))
    goal = (to_int(sample["goal_row"]), to_int(sample["goal_col"]))
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0
    return grid, start, goal


def free_cells(grid):
    return [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 0]


def neighbors(grid, cell):
    rows = len(grid)
    cols = len(grid[0])
    r, c = cell
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            yield nr, nc


def table_to_grid(grid, table):
    return [[None if grid[r][c] == 1 else float(table.get((r, c), 0.0)) for c in range(len(grid[0]))] for r in range(len(grid))]


def distance_grid_to_json(distance_grid):
    return [[None if value < 0 else value for value in row] for row in distance_grid]


def true_distance_table(distance_grid):
    return {(r, c): value for r, row in enumerate(distance_grid) for c, value in enumerate(row) if value >= 0}


def table_heuristic(table):
    return lambda current, unused_goal: float(table[current])


def calibrated_table(unet_table, calibration):
    return {cell: value - calibration for cell, value in unet_table.items()}


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def distance_to_path(path, cell):
    if not path:
        return None
    return min(abs(cell[0] - p[0]) + abs(cell[1] - p[1]) for p in path)


def traced_astar(grid, start, goal, primary_table, secondary_table=None, true_table=None, optimal_path=None):
    open_heap = []
    counter = 0
    g_score = {start: 0}
    came_from = {}
    expanded_entries = []
    snapshots = []
    closed = set()

    def secondary(node):
        return 0.0 if secondary_table is None else float(secondary_table.get(node, 0.0))

    path_set = set(optimal_path or [])

    def open_entries():
        entries = []
        for item in open_heap:
            if use_secondary:
                qf, qs, qc, qg, qnode = item
                priority = [qf, qs, qc, qg]
            else:
                qf, qg, qnode = item
                qs = 0.0
                priority = [qf, qg]
            if qg != g_score.get(qnode, float("inf")):
                continue
            entries.append(
                {
                    "node": [qnode[0], qnode[1]],
                    "g": qg,
                    "h_primary": primary_table[qnode],
                    "h_secondary": qs,
                    "f": qf,
                    "true_distance": true_table.get(qnode) if true_table else None,
                    "on_path": qnode in path_set,
                    "distance_to_path": distance_to_path(optimal_path, qnode),
                    "priority": priority,
                }
            )
        entries.sort(key=lambda row: tuple(row["priority"] + [row["node"][0], row["node"][1]]))
        for rank, row in enumerate(entries, start=1):
            row["rank"] = rank
        return entries

    use_secondary = secondary_table is not None
    if use_secondary:
        heapq.heappush(open_heap, (primary_table[start], secondary(start), counter, 0, start))
    else:
        heapq.heappush(open_heap, (primary_table[start], 0, start))

    while open_heap:
        if use_secondary:
            f_primary, current_secondary, current_counter, current_g, current = heapq.heappop(open_heap)
            current_priority = [f_primary, current_secondary, current_counter, current_g]
        else:
            f_primary, current_g, current = heapq.heappop(open_heap)
            current_priority = [f_primary, current_g]
        if current_g > g_score.get(current, float("inf")):
            continue

        active = []
        frontier = []
        for item in open_heap:
            if use_secondary:
                qf, qs, _, qg, qnode = item
            else:
                qf, qg, qnode = item
                qs = 0.0
            if qg == g_score.get(qnode, float("inf")):
                frontier.append([qnode[0], qnode[1]])
                if qf == f_primary:
                    active.append(
                        {
                            "node": [qnode[0], qnode[1]],
                            "g": qg,
                            "h_primary": primary_table[qnode],
                            "h_secondary": qs,
                            "f": qf,
                            "true_distance": true_table.get(qnode) if true_table else None,
                            "on_path": qnode in path_set,
                            "distance_to_path": distance_to_path(optimal_path, qnode),
                            "distance_to_goal": manhattan(qnode, goal),
                        }
                    )
        active.append(
            {
                "node": [current[0], current[1]],
                "g": current_g,
                "h_primary": primary_table[current],
                "h_secondary": secondary(current),
                "f": f_primary,
                "true_distance": true_table.get(current) if true_table else None,
                "on_path": current in path_set,
                "distance_to_path": distance_to_path(optimal_path, current),
                "distance_to_goal": manhattan(current, goal),
            }
        )

        step = len(expanded_entries)
        closed.add(current)
        entry = {
            "step": step,
            "node": [current[0], current[1]],
            "g": current_g,
            "h_primary": primary_table[current],
            "h_secondary": secondary(current),
            "f": f_primary,
            "true_distance": true_table.get(current) if true_table else None,
            "on_path": current in path_set,
            "distance_to_path": distance_to_path(optimal_path, current),
            "tie_set": active,
            "priority": current_priority,
        }
        expanded_entries.append(entry)

        if current == goal:
            current_open = open_entries()
            snapshots.append(
                {
                    "step": step,
                    "convention": "after_expansion",
                    "frontier": [row["node"] for row in current_open],
                    "open": current_open,
                    "closed": [[r, c] for r, c in closed],
                }
            )
            path = reconstruct_path(came_from, current)
            return {
                "path": [[r, c] for r, c in path],
                "cost": len(path) - 1,
                "expanded": len(expanded_entries),
                "expanded_nodes": expanded_entries,
                "snapshots": snapshots,
                "expansion_order": [[r, c] for r, c in [tuple(e["node"]) for e in expanded_entries]],
            }

        for nb in neighbors(grid, current):
            tentative_g = current_g + 1
            if tentative_g < g_score.get(nb, float("inf")):
                g_score[nb] = tentative_g
                came_from[nb] = current
                counter += 1
                f_score = tentative_g + primary_table[nb]
                if use_secondary:
                    heapq.heappush(open_heap, (f_score, secondary(nb), counter, tentative_g, nb))
                else:
                    heapq.heappush(open_heap, (f_score, tentative_g, nb))

        current_open = open_entries()
        snapshots.append(
            {
                "step": step,
                "convention": "after_expansion",
                "frontier": [row["node"] for row in current_open],
                "open": current_open,
                "closed": [[r, c] for r, c in closed],
            }
        )

    return {"path": [], "cost": -1, "expanded": len(expanded_entries), "expanded_nodes": expanded_entries, "snapshots": snapshots, "expansion_order": []}


def h_stats(table, true_table):
    cells = [cell for cell in table if cell in true_table]
    pred = [table[cell] for cell in cells]
    true = [true_table[cell] for cell in cells]
    errors = [p - t for p, t in zip(pred, true)]
    pair_total = 0
    pair_ok = 0
    for i in range(0, len(cells), max(1, len(cells) // 80)):
        for j in range(i + 1, len(cells), max(1, len(cells) // 80)):
            td = true[i] - true[j]
            pd = pred[i] - pred[j]
            if td == 0 or pd == 0:
                continue
            pair_total += 1
            if td * pd > 0:
                pair_ok += 1
    return {
        "mae": mean(abs(e) for e in errors),
        "spearman": spearman(true, pred),
        "ordering_accuracy": pair_ok / pair_total if pair_total else None,
    }


def heuristic_property_stats(table, true_table):
    """Per-case properties shown in the viewer, using the experiment definitions."""
    cells = [cell for cell in table if cell in true_table]
    overestimates = [max(0.0, table[cell] - true_table[cell]) for cell in cells]
    admissibility = sum(value > 1e-6 for value in overestimates)
    consistency = 0
    consistency_magnitude = 0.0
    for cell in cells:
        for neighbor in ((cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1)):
            if neighbor not in table:
                continue
            excess = table[cell] - 1.0 - table[neighbor]
            if excess > 1e-6:
                consistency += 1
                consistency_magnitude += excess
    return {
        "admissibility_violation_count": admissibility,
        "mean_overestimation_error": mean(overestimates),
        "consistency_violation_count": consistency,
        "consistency_violation_magnitude": consistency_magnitude,
    }


def disagreement_steps(unet_trace, other_trace, other_name, true_table, optimal_path):
    rows = []
    limit = min(len(unet_trace["expanded_nodes"]), len(other_trace["expanded_nodes"]))
    for step in range(limit):
        un = unet_trace["expanded_nodes"][step]
        ot = other_trace["expanded_nodes"][step]
        if un["node"] != ot["node"]:
            ucell = tuple(un["node"])
            ocell = tuple(ot["node"])
            rows.append(
                {
                    "step": step,
                    "comparator": other_name,
                    "unet_node": un["node"],
                    "comparator_node": ot["node"],
                    "unet_true_distance": true_table.get(ucell),
                    "comparator_true_distance": true_table.get(ocell),
                    "unet_distance_to_path": distance_to_path(optimal_path, ucell),
                    "comparator_distance_to_path": distance_to_path(optimal_path, ocell),
                    "candidate_tie_set": un.get("tie_set", []),
                }
            )
    return rows


def validate_trace_against_astar(grid, start, goal, primary_table, trace, secondary_table=None):
    """Check that a serialized viewer trace replays the repository A* result."""
    result = astar_search(
        grid,
        start,
        goal,
        table_heuristic(primary_table),
        secondary_heuristic=table_heuristic(secondary_table) if secondary_table is not None else None,
    )
    expected_path = [[row, col] for row, col in result["path"]]
    return {
        "astar_expanded": result["expanded"],
        "trace_expanded": trace["expanded"],
        "astar_cost": result["cost"],
        "trace_cost": trace["cost"],
        "expanded_matches": result["expanded"] == trace["expanded"],
        "cost_matches": result["cost"] == trace["cost"],
        "path_matches": expected_path == trace["path"],
    }


def generate_case(case_id, methods, mlp_model, unet_models):
    sample = methods[METHOD_ROWS["manhattan"]]
    grid, start, goal = build_grid(sample)
    distance_grid = compute_distance_to_goal(grid, goal)
    true_table = true_distance_table(distance_grid)
    optimal_cost = true_table.get(start)
    manhattan_table = {cell: float(manhattan(cell, goal)) for cell in free_cells(grid)}
    mlp_h = make_mlp_table_heuristic(mlp_model, grid, goal)
    mlp_table = {cell: float(mlp_h(cell, goal)) for cell in free_cells(grid)}
    unet_tables = {}
    for model_name, model in unet_models.items():
        heuristic = make_unet_heuristic(model, grid, goal)
        unet_tables[model_name] = {cell: float(heuristic(cell, goal)) for cell in free_cells(grid)}
    oracle_table = {cell: float(true_table.get(cell, 10**9)) for cell in free_cells(grid)}

    oracle_path_trace = traced_astar(grid, start, goal, manhattan_table, oracle_table, true_table, [])
    optimal_path = [tuple(cell) for cell in oracle_path_trace["path"]]

    traces = {
        "manhattan": traced_astar(grid, start, goal, manhattan_table, None, true_table, optimal_path),
        "mlp_tiebreak": traced_astar(grid, start, goal, manhattan_table, mlp_table, true_table, optimal_path),
        "oracle_tiebreak": traced_astar(grid, start, goal, manhattan_table, oracle_table, true_table, optimal_path),
    }
    baseline_trace_validation = {
        "manhattan": validate_trace_against_astar(grid, start, goal, manhattan_table, traces["manhattan"]),
        "mlp_tiebreak": validate_trace_against_astar(grid, start, goal, manhattan_table, traces["mlp_tiebreak"], mlp_table),
        "oracle_tiebreak": validate_trace_against_astar(grid, start, goal, manhattan_table, traces["oracle_tiebreak"], oracle_table),
    }
    for method, validation in baseline_trace_validation.items():
        if not all(validation[key] for key in ("expanded_matches", "cost_matches", "path_matches")):
            raise AssertionError(f"Viewer trace mismatch for {case_id}/{method}: {validation}")
    unet_variant_traces = {}
    unet_algorithm_tables = {}
    unet_algorithm_metadata = {}
    for model_name, unet_table in unet_tables.items():
        oracle_calibration = max(max(unet_table[cell] - true_table[cell] for cell in true_table), 0.0)
        fixed_calibrations = {
            "unet_calibrated_5": 5.0,
            "unet_calibrated_10": 10.0,
            "unet_calibrated_20": 20.0,
            "unet_calibrated_manhattan": float(manhattan(start, goal)),
            "unet_calibrated_oracle": oracle_calibration,
        }
        tables = {"unet_tiebreak": unet_table, "unet_direct": unet_table}
        tables.update({key: calibrated_table(unet_table, value) for key, value in fixed_calibrations.items()})
        unet_algorithm_tables[model_name] = tables
        unet_algorithm_metadata[model_name] = {
            "unet_tiebreak": {"algorithm_name": UNET_VARIANT_LABELS["unet_tiebreak"], "heuristic_type": "manhattan_primary_unet_secondary", "calibration_value": None, "calibration_kind": "none", "priority_tuple": "(g + Manhattan h, U-Net h, insertion_counter, g, node)"},
            "unet_direct": {"algorithm_name": UNET_VARIANT_LABELS["unet_direct"], "heuristic_type": "unet", "calibration_value": 0.0, "calibration_kind": "none", "priority_tuple": "(g + U-Net h, g, node)"},
        }
        for key, calibration in fixed_calibrations.items():
            kind = "oracle" if key == "unet_calibrated_oracle" else ("start_goal_manhattan" if key == "unet_calibrated_manhattan" else "fixed")
            unet_algorithm_metadata[model_name][key] = {
                "algorithm_name": UNET_VARIANT_LABELS[key],
                "heuristic_type": "unet_minus_calibration",
                "calibration_value": calibration,
                "calibration_kind": kind,
                "priority_tuple": "(g + (U-Net h - calibration), g, node)",
            }
        unet_variant_traces[model_name] = {}
        for key in UNET_VARIANT_KEYS:
            if key == "unet_tiebreak":
                trace = traced_astar(grid, start, goal, manhattan_table, tables[key], true_table, optimal_path)
            else:
                trace = traced_astar(grid, start, goal, tables[key], None, true_table, optimal_path)
            trace["metadata"] = unet_algorithm_metadata[model_name][key]
            unet_variant_traces[model_name][key] = trace

    default_model = next(iter(unet_models))
    # Backward-compatible aliases for existing viewer consumers.
    traces["unet_tiebreak"] = unet_variant_traces[default_model]["unet_tiebreak"]
    unet_traces = {model_name: variants["unet_tiebreak"] for model_name, variants in unet_variant_traces.items()}

    for trace in [*traces.values(), *(trace for variants in unet_variant_traces.values() for trace in variants.values())]:
        trace["optimal"] = trace["cost"] == optimal_cost

    disagreements_by_unet_model = {}
    disagreements_by_unet_variant = {}
    for model_name, unet_trace in unet_traces.items():
        disagreements = []
        disagreements.extend(disagreement_steps(unet_trace, traces["mlp_tiebreak"], "mlp_tiebreak", true_table, optimal_path))
        disagreements.extend(disagreement_steps(unet_trace, traces["manhattan"], "manhattan", true_table, optimal_path))
        disagreements_by_unet_model[model_name] = disagreements[:200]
        disagreements_by_unet_variant[model_name] = {}
        for variant, trace in unet_variant_traces[model_name].items():
            rows = []
            rows.extend(disagreement_steps(trace, traces["mlp_tiebreak"], "mlp_tiebreak", true_table, optimal_path))
            rows.extend(disagreement_steps(trace, traces["manhattan"], "manhattan", true_table, optimal_path))
            disagreements_by_unet_variant[model_name][variant] = rows[:200]

    trace_validation = {}
    for model_name, variants in unet_variant_traces.items():
        trace_validation[model_name] = {}
        for variant, trace in variants.items():
            primary_table = manhattan_table if variant == "unet_tiebreak" else unet_algorithm_tables[model_name][variant]
            secondary_table = unet_algorithm_tables[model_name][variant] if variant == "unet_tiebreak" else None
            validation = validate_trace_against_astar(grid, start, goal, primary_table, trace, secondary_table)
            if not all(validation[key] for key in ("expanded_matches", "cost_matches", "path_matches")):
                raise AssertionError(f"Viewer trace mismatch for {case_id}/{model_name}/{variant}: {validation}")
            trace_validation[model_name][variant] = validation

    unet_algorithm_grids = {model_name: {key: table_to_grid(grid, table) for key, table in tables.items()} for model_name, tables in unet_algorithm_tables.items()}
    unet_algorithm_errors = {
        model_name: {
            key: [[None if distance_grid[r][c] < 0 or grid[r][c] else table[(r, c)] - distance_grid[r][c] for c in range(len(grid[0]))] for r in range(len(grid))]
            for key, table in tables.items()
        }
        for model_name, tables in unet_algorithm_tables.items()
    }
    stats = {
        "manhattan": h_stats(manhattan_table, true_table),
        "mlp": h_stats(mlp_table, true_table),
        "unet_models": {model_name: h_stats(table, true_table) for model_name, table in unet_tables.items()},
    }
    return {
        "case_id": case_id,
        "map_id": case_identity(methods),
        "map_type": sample.get("structured_type", "random"),
        "map_mode": sample.get("map_mode", "random"),
        "seed": sample["seed"],
        "obstacle_rate": sample["obstacle_rate"],
        "grid": grid,
        "start": [start[0], start[1]],
        "goal": [goal[0], goal[1]],
        "optimal_cost": optimal_cost,
        "h_maps": {
            "manhattan": table_to_grid(grid, manhattan_table),
            "mlp": table_to_grid(grid, mlp_table),
            "unet": table_to_grid(grid, unet_tables[default_model]),
            "unet_models": {model_name: table_to_grid(grid, table) for model_name, table in unet_tables.items()},
            "true_distance": distance_grid_to_json(distance_grid),
            "unet_error": unet_algorithm_errors[default_model]["unet_tiebreak"],
            "unet_error_models": {model_name: errors["unet_tiebreak"] for model_name, errors in unet_algorithm_errors.items()},
            "unet_algorithm_models": unet_algorithm_grids,
            "unet_algorithm_errors": unet_algorithm_errors,
            "mlp_error": [[None if distance_grid[r][c] < 0 or grid[r][c] else mlp_table[(r, c)] - distance_grid[r][c] for c in range(len(grid[0]))] for r in range(len(grid))],
        },
        "traces": traces,
        "unet_traces": unet_traces,
        "unet_variant_traces": unet_variant_traces,
        "unet_algorithm_metadata": unet_algorithm_metadata,
        "unet_model_labels": COMBINED_LOSS_MODEL_LABELS,
        "unet_properties": {model_name: heuristic_property_stats(table, true_table) for model_name, table in unet_tables.items()},
        "disagreements": disagreements_by_unet_model[default_model],
        "disagreements_by_unet_model": disagreements_by_unet_model,
        "disagreements_by_unet_variant": disagreements_by_unet_variant,
        "trace_validation": trace_validation,
        "baseline_trace_validation": baseline_trace_validation,
        "h_stats": stats,
        "expansion_gaps": {
            "unet_minus_mlp": traces["unet_tiebreak"]["expanded"] - traces["mlp_tiebreak"]["expanded"],
            "unet_minus_manhattan": traces["unet_tiebreak"]["expanded"] - traces["manhattan"]["expanded"],
        },
        "expansion_gaps_by_unet_model": {
            model_name: {
                "unet_minus_mlp": trace["expanded"] - traces["mlp_tiebreak"]["expanded"],
                "unet_minus_manhattan": trace["expanded"] - traces["manhattan"]["expanded"],
            }
            for model_name, trace in unet_traces.items()
        },
        "unet_variant_expansion_gaps": {
            model_name: {
                key: {
                    "unet_minus_mlp": trace["expanded"] - traces["mlp_tiebreak"]["expanded"],
                    "unet_minus_manhattan": trace["expanded"] - traces["manhattan"]["expanded"],
                }
                for key, trace in variants.items()
            }
            for model_name, variants in unet_variant_traces.items()
        },
    }


CASE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#f6f7f9;color:#17202a}
header{padding:16px 22px;background:#16202a;color:white}
main{padding:16px 22px;display:grid;gap:14px}
.panel{background:white;border:1px solid #d9dee7;border-radius:6px;padding:12px}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center}
label{font-size:13px} select,input,button{font:inherit}
button{padding:5px 10px;border:1px solid #9aa6b2;background:#fff;border-radius:4px}
.gridwrap{display:grid;grid-template-columns:1fr 1fr;gap:12px}
canvas{background:#fff;border:1px solid #c7ced8;max-width:100%;image-rendering:pixelated}
table{border-collapse:collapse;width:100%;font-size:13px} th,td{border-bottom:1px solid #e2e6ec;padding:5px;text-align:left}
tr.clickable{cursor:pointer} tr.clickable:hover{background:#eef5ff}
tr.current-row{background:#fff1d6} tr.path-row{box-shadow:inset 3px 0 #ffe156} tr.selected-row{outline:2px solid #7b2cbf;outline-offset:-2px}
.listgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.listpanel h3{margin:0 0 6px}.listwrap{max-height:260px;overflow:auto;border:1px solid #e2e6ec}.listwrap th{position:sticky;top:0;background:#edf1f5;z-index:1}.note{font-size:12px;color:#506070;margin:0 0 8px}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px}.sw{display:inline-block;width:12px;height:12px;border:1px solid #555;vertical-align:-2px;margin-right:4px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px;font-size:13px}
</style>
</head>
<body>
<header><h1>__TITLE__</h1></header>
<main>
<section class="panel stats" id="summary"></section>
<section class="panel controls">
<label>Step <input id="step" type="range" min="0" max="1" value="0" style="width:260px"></label><span id="stepLabel"></span>
<button id="play">Play</button>
<label>Algorithm <select id="algorithm"></select></label>
<label>U-Net model <select id="unetModel"><option value="mse">MSE</option><option value="ranking">Ranking</option><option value="ranking_adm">Ranking + Admissibility</option><option value="ranking_adm_cons">Ranking + Admissibility + Consistency</option></select></label>
<span id="unetModelName" class="note">Current U-Net model: MSE</span>
<label>Mode <select id="mode"><option value="single">single algorithm</option><option value="side">side-by-side</option><option value="diff">difference view</option></select></label>
<label>Compare <select id="compare"><option value="mlp_tiebreak">MLP vs U-Net</option><option value="manhattan">Manhattan vs U-Net</option><option value="oracle_tiebreak">Oracle vs U-Net</option></select></label>
<label>Background <select id="background"><option value="none">none</option><option value="h">selected h-map</option><option value="true_distance">true distance</option><option value="error">selected error map</option><option value="expansion">expansion order</option></select></label>
<label>List algorithm <select id="listAlgorithm"></select></label>
<label><input type="checkbox" id="showValues"> show cell values</label>
<label>Value label mode <select id="valueMode"><option value="h">h value</option><option value="true_distance">true distance</option><option value="error">prediction error</option><option value="g">g value</option><option value="f">f value</option></select></label>
<label><input type="checkbox" id="showObstacles" checked> obstacles</label>
<label><input type="checkbox" id="showPath" checked> final path</label>
<label><input type="checkbox" id="showExpanded" checked> expanded</label>
<label><input type="checkbox" id="showCurrent" checked> current</label>
<label><input type="checkbox" id="showFrontier"> frontier</label>
<label><input type="checkbox" id="showSG" checked> start/goal</label>
</section>
<section class="panel">
<div class="legend"><span><i class="sw" style="background:#222"></i>obstacle</span><span><i class="sw" style="background:#2b7bba"></i>expanded</span><span><i class="sw" style="background:#ff9f1c"></i>current</span><span><i class="sw" style="background:#00bcd4"></i>frontier</span><span><i class="sw" style="background:#ffe156"></i>path</span><span><i class="sw" style="background:#7b2cbf"></i>U-Net only</span><span><i class="sw" style="background:#2a9d8f"></i>comparator only</span></div>
<div class="gridwrap"><canvas id="canvasA" width="520" height="520"></canvas><canvas id="canvasB" width="520" height="520"></canvas></div>
</section>
<section class="panel"><h2>Open / Closed Lists</h2><p class="note">Snapshots use the state immediately after completing expansion at the selected step. Rank follows the selected algorithm's recorded effective priority tuple.</p><div id="listPanels" class="listgrid"></div></section>
<section class="panel"><h2>Disagreement steps</h2><div id="disagreements"></div></section>
</main>
<script id="case-data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('case-data').textContent);
const methods={manhattan:'Manhattan A*',mlp_tiebreak:'MLP tie-break',unet_tiebreak:'Manhattan + U-Net tie-break',unet_direct:'Direct U-Net A*',unet_calibrated_5:'U-Net calibrated n=5',unet_calibrated_10:'U-Net calibrated n=10',unet_calibrated_20:'U-Net calibrated n=20',unet_calibrated_manhattan:'Manhattan-distance calibrated U-Net',unet_calibrated_oracle:'Oracle calibrated U-Net',oracle_tiebreak:'True-distance tie-break'};
const unetVariants=new Set(['unet_tiebreak','unet_direct','unet_calibrated_5','unet_calibrated_10','unet_calibrated_20','unet_calibrated_manhattan','unet_calibrated_oracle']);
const stepEl=document.getElementById('step'), algEl=document.getElementById('algorithm'), modeEl=document.getElementById('mode'), compareEl=document.getElementById('compare'), bgEl=document.getElementById('background'), listAlgEl=document.getElementById('listAlgorithm'), unetModelEl=document.getElementById('unetModel'), unetModelNameEl=document.getElementById('unetModelName');
const showValuesEl=document.getElementById('showValues'), valueModeEl=document.getElementById('valueMode');
let selectedDisagreement=null, selectedCell=null;
Object.entries(methods).forEach(([k,v])=>{let o=document.createElement('option');o.value=k;o.textContent=v;algEl.appendChild(o);let p=document.createElement('option');p.value=k;p.textContent=v;listAlgEl.appendChild(p)}); algEl.value='unet_direct'; listAlgEl.value='unet_direct';
let timer=null;
function maxStep(){return Math.max(0,activeTrace(algEl.value).expanded_nodes.length-1)}
function updateStepLimit(){stepEl.max=maxStep();stepEl.value=Math.min(Number(stepEl.value),Number(stepEl.max))}
updateStepLimit();
function clampStep(v){return Math.max(0,Math.min(Number(stepEl.max),v))}
function setStep(v){stepEl.value=clampStep(v);render()}
function cellKey(c){return c[0]+','+c[1]}
function activeUnetModel(){return unetModelEl.value}
function activeUnetLabel(){return (data.unet_model_labels||{})[activeUnetModel()]||activeUnetModel()}
function isUnetVariant(method){return unetVariants.has(method)}
function activeTrace(method){if(!isUnetVariant(method))return data.traces[method];return data.unet_variant_traces[activeUnetModel()][method]}
function allTraces(){let traces=[...Object.values(data.traces)];Object.values(data.unet_variant_traces||{}).forEach(variants=>Object.values(variants).forEach(trace=>traces.push(trace)));return traces}
function hGrid(method){if(isUnetVariant(method)){const grids=data.h_maps.unet_algorithm_models||{};return grids[activeUnetModel()][method]}if(method==='mlp_tiebreak')return data.h_maps.mlp;if(method==='oracle_tiebreak')return data.h_maps.true_distance;return data.h_maps.manhattan}
function errorGrid(method){if(isUnetVariant(method)){const grids=data.h_maps.unet_algorithm_errors||{};return grids[activeUnetModel()][method]}return method==='mlp_tiebreak'?data.h_maps.mlp_error:null}
function valGrid(name, method){if(name==='true_distance')return data.h_maps.true_distance;if(name==='error')return errorGrid(method);if(name==='h')return hGrid(method);if(name==='expansion'){let n=data.grid.length,g=Array.from({length:n},()=>Array(n).fill(null));activeTrace(method).expanded_nodes.forEach((e,i)=>{g[e.node[0]][e.node[1]]=i});return g}return null}
function traceValueGrid(method, step, field){
 const n=data.grid.length, g=Array.from({length:n},()=>Array(n).fill(null)), tr=activeTrace(method), upto=Math.min(step,tr.expanded_nodes.length-1);
 for(let i=0;i<=upto;i++){let e=tr.expanded_nodes[i], r=e.node[0], c=e.node[1]; g[r][c]=field==='f'?e.f:e.g}
 if(tr.expanded_nodes[upto]&&tr.expanded_nodes[upto].tie_set){tr.expanded_nodes[upto].tie_set.forEach(e=>{g[e.node[0]][e.node[1]]=field==='f'?e.f:e.g})}
 return g;
}
function labelGrid(method, step){
 const mode=valueModeEl.value;
 if(mode==='h')return hGrid(method);
 if(mode==='true_distance')return data.h_maps.true_distance;
 if(mode==='error')return errorGrid(method);
 if(mode==='g'||mode==='f')return traceValueGrid(method, step, mode);
 return null;
}
function formatLabel(v, mode, method){
 if(v==null||Number.isNaN(v))return '';
 if(!Number.isFinite(v))return '∞';
 if(mode==='h'&&(method==='mlp_tiebreak'||isUnetVariant(method)))return Number(v).toFixed(1);
 if(Math.abs(v-Math.round(v))<1e-6)return String(Math.round(v));
 return Number(v).toFixed(1);
}
function luminance(color){
 let m=String(color).match(/rgb\((\d+),(\d+),(\d+)\)/); if(!m)return 255;
 return 0.2126*Number(m[1])+0.7152*Number(m[2])+0.0722*Number(m[3]);
}
function heatColor(v,min,max){ if(v==null||Number.isNaN(v))return '#fff'; let t=(v-min)/(max-min||1); t=Math.max(0,Math.min(1,t)); let r=Math.round(255*t), b=Math.round(255*(1-t)); return `rgb(${r},${Math.round(230-120*t)},${b})` }
function drawValueLabels(ctx, method, step, bg, min, max, cs){
 if(!showValuesEl.checked)return;
 const n=data.grid.length, labels=labelGrid(method,step); if(!labels)return;
 const mode=valueModeEl.value; if(cs<14)return;
 ctx.textAlign='center';ctx.textBaseline='middle';ctx.font=Math.max(7,Math.min(11,cs*0.32))+'px system-ui';
 for(let r=0;r<n;r++)for(let c=0;c<n;c++){
   if(data.grid[r][c]===1)continue;
   let text=formatLabel(labels[r][c],mode,method); if(!text)continue;
   let fill=bg?heatColor(bg[r][c],min,max):'#fff'; ctx.fillStyle=luminance(fill)<130?'#fff':'#111';
   ctx.fillText(text,c*cs+cs/2,r*cs+cs/2);
 }
}
function draw(canvas, method, step, title){
 const ctx=canvas.getContext('2d'), n=data.grid.length, cs=canvas.width/n; ctx.clearRect(0,0,canvas.width,canvas.height);
 let bg=valGrid(bgEl.value,method); let vals=[]; if(bg)bg.flat().forEach(v=>{if(v!=null)vals.push(v)}); let min=Math.min(...vals,0), max=Math.max(...vals,1);
 for(let r=0;r<n;r++)for(let c=0;c<n;c++){ctx.fillStyle=bg?heatColor(bg[r][c],min,max):'#fff';ctx.fillRect(c*cs,r*cs,cs,cs); if(document.getElementById('showObstacles').checked&&data.grid[r][c]===1){ctx.fillStyle='#222';ctx.fillRect(c*cs,r*cs,cs,cs)}}
 const tr=activeTrace(method), upto=Math.min(step,tr.expanded_nodes.length-1);
 if(document.getElementById('showExpanded').checked){for(let i=0;i<=upto;i++){let nd=tr.expanded_nodes[i].node;ctx.fillStyle=`rgba(43,123,186,${0.18+0.58*i/(upto+1)})`;ctx.fillRect(nd[1]*cs,nd[0]*cs,cs,cs)}}
 if(document.getElementById('showFrontier').checked&&tr.snapshots[upto]){ctx.strokeStyle='#00bcd4';ctx.lineWidth=2;tr.snapshots[upto].frontier.forEach(nd=>ctx.strokeRect(nd[1]*cs+2,nd[0]*cs+2,cs-4,cs-4))}
 if(document.getElementById('showPath').checked){ctx.fillStyle='rgba(255,225,86,.75)';tr.path.forEach(nd=>ctx.fillRect(nd[1]*cs+cs*.25,nd[0]*cs+cs*.25,cs*.5,cs*.5))}
 if(document.getElementById('showCurrent').checked&&tr.expanded_nodes[upto]){let nd=tr.expanded_nodes[upto].node;ctx.fillStyle='#ff9f1c';ctx.fillRect(nd[1]*cs,nd[0]*cs,cs,cs)}
 if(selectedDisagreement&&(method===algEl.value||method===selectedDisagreement.comparator)){
   ctx.lineWidth=3;
   if(method===algEl.value&&selectedDisagreement.candidate_tie_set){ctx.strokeStyle='rgba(123,44,191,.65)';selectedDisagreement.candidate_tie_set.forEach(x=>ctx.strokeRect(x.node[1]*cs+3,x.node[0]*cs+3,cs-6,cs-6))}
   let nd=method===algEl.value?selectedDisagreement.unet_node:selectedDisagreement.comparator_node;
   ctx.strokeStyle=method===algEl.value?'#7b2cbf':'#2a9d8f';ctx.strokeRect(nd[1]*cs+1,nd[0]*cs+1,cs-2,cs-2);
 }
 if(selectedCell){ctx.strokeStyle='#e000ff';ctx.lineWidth=3;ctx.strokeRect(selectedCell[1]*cs+1,selectedCell[0]*cs+1,cs-2,cs-2)}
 if(document.getElementById('showSG').checked){let s=data.start,g=data.goal;ctx.fillStyle='#2ca25f';ctx.fillRect(s[1]*cs,s[0]*cs,cs,cs);ctx.fillStyle='#d62828';ctx.fillRect(g[1]*cs,g[0]*cs,cs,cs)}
 ctx.strokeStyle='#ccd2da';ctx.lineWidth=0.5;for(let i=0;i<=n;i++){ctx.beginPath();ctx.moveTo(i*cs,0);ctx.lineTo(i*cs,canvas.height);ctx.stroke();ctx.beginPath();ctx.moveTo(0,i*cs);ctx.lineTo(canvas.width,i*cs);ctx.stroke()}
 drawValueLabels(ctx,method,step,bg,min,max,cs);
 ctx.fillStyle='#111';ctx.font='14px system-ui';ctx.fillText(title,8,18);
}
function drawDiff(canvas, primary, comp, step){
 const ctx=canvas.getContext('2d'), n=data.grid.length, cs=canvas.width/n; ctx.clearRect(0,0,canvas.width,canvas.height);
 for(let r=0;r<n;r++)for(let c=0;c<n;c++){ctx.fillStyle=data.grid[r][c]?'#222':'#fff';ctx.fillRect(c*cs,r*cs,cs,cs)}
 const u=activeTrace(primary).expanded_nodes.slice(0,step+1).map(e=>cellKey(e.node)); const cset=activeTrace(comp).expanded_nodes.slice(0,step+1).map(e=>cellKey(e.node));
 const us=new Set(u), cs2=new Set(cset); [...new Set([...u,...cset])].forEach(k=>{let [r,c]=k.split(',').map(Number); ctx.fillStyle=us.has(k)&&cs2.has(k)?'#999':(us.has(k)?'#7b2cbf':'#2a9d8f'); ctx.fillRect(c*cs,r*cs,cs,cs)});
 activeTrace(primary).path.forEach(nd=>{ctx.fillStyle='rgba(255,225,86,.8)';ctx.fillRect(nd[1]*cs+cs*.25,nd[0]*cs+cs*.25,cs*.5,cs*.5)});
 if(selectedCell){ctx.strokeStyle='#e000ff';ctx.lineWidth=3;ctx.strokeRect(selectedCell[1]*cs+1,selectedCell[0]*cs+1,cs-2,cs-2)}
 ctx.fillStyle='#111';ctx.font='14px system-ui';ctx.fillText('Difference: '+methods[primary]+' vs '+methods[comp]+' (cell labels hidden)',8,18);
}
function fmt(v){ if(v==null||Number.isNaN(v))return 'N/A'; if(!Number.isFinite(v))return '∞'; return Math.abs(v-Math.round(v))<1e-6?String(Math.round(v)):Number(v).toFixed(2)}
function fmtPriority(row){return (row.priority||[]).map(fmt).join(', ')}
function rowClass(row,currentKey){let cls='clickable'; if(cellKey(row.node)===currentKey)cls+=' current-row'; if(row.on_path)cls+=' path-row'; if(selectedCell&&cellKey(row.node)===cellKey(selectedCell))cls+=' selected-row'; return cls}
function openTable(method,step){
 const tr=activeTrace(method), upto=Math.min(step,tr.snapshots.length-1), snap=tr.snapshots[upto]||{open:[]}, rows=snap.open||[];
 return `<div class="listpanel"><h3>Open List - ${methods[method]}</h3><div class="listwrap"><table><thead><tr><th>rank</th><th>coord</th><th>g</th><th>h1</th><th>h2</th><th>f</th><th>priority</th><th>true d</th></tr></thead><tbody>`+
 rows.map(r=>`<tr class="${rowClass(r,'')}" data-node="${cellKey(r.node)}"><td>${r.rank??'N/A'}</td><td>${r.node}</td><td>${fmt(r.g)}</td><td>${fmt(r.h_primary)}</td><td>${fmt(r.h_secondary)}</td><td>${fmt(r.f)}</td><td>${fmtPriority(r)}</td><td>${fmt(r.true_distance)}</td></tr>`).join('')+
 `</tbody></table></div></div>`;
}
function closedTable(method,step){
 const tr=activeTrace(method), upto=Math.min(step,tr.expanded_nodes.length-1), rows=tr.expanded_nodes.slice(0,upto+1), currentKey=rows[rows.length-1]?cellKey(rows[rows.length-1].node):'';
 return `<div class="listpanel"><h3>Closed List - ${methods[method]}</h3><div class="listwrap"><table><thead><tr><th>order</th><th>coord</th><th>g</th><th>h1</th><th>h2</th><th>f</th><th>priority</th><th>true d</th></tr></thead><tbody>`+
 rows.map(r=>`<tr class="${rowClass(r,currentKey)}" data-node="${cellKey(r.node)}"><td>${r.step}</td><td>${r.node}</td><td>${fmt(r.g)}</td><td>${fmt(r.h_primary)}</td><td>${fmt(r.h_secondary)}</td><td>${fmt(r.f)}</td><td>${fmtPriority(r)}</td><td>${fmt(r.true_distance)}</td></tr>`).join('')+
 `</tbody></table></div></div>`;
}
function listMethodsForView(mode,alg,comp){if(mode==='side')return [comp,alg];if(mode==='diff')return [listAlgEl.value];return [alg]}
function renderLists(){
 const step=Number(stepEl.value), mode=modeEl.value, alg=algEl.value, comp=compareEl.value, methodsToShow=listMethodsForView(mode,alg,comp), div=document.getElementById('listPanels');
 div.style.gridTemplateColumns=methodsToShow.length>1?'1fr 1fr':'1fr 1fr';
 div.innerHTML=methodsToShow.map(m=>openTable(m,step)+closedTable(m,step)).join('');
 div.querySelectorAll('tr.clickable').forEach(tr=>tr.onclick=()=>{selectedCell=tr.dataset.node.split(',').map(Number);render()});
}
function render(){
 const step=Number(stepEl.value), mode=modeEl.value, alg=algEl.value, comp=compareEl.value; document.getElementById('stepLabel').textContent=`${step}/${stepEl.max}`;
 unetModelNameEl.textContent='Current U-Net model: '+activeUnetLabel(); initSummary();
 listAlgEl.disabled=mode!=='diff';
 const a=document.getElementById('canvasA'), b=document.getElementById('canvasB'); b.style.display=mode==='side'?'block':'none';
 if(mode==='diff'){b.style.display='none';drawDiff(a,alg,comp,step)} else if(mode==='side'){draw(a,comp,step,methods[comp]);draw(b,alg,step,isUnetVariant(alg)?methods[alg]+': '+activeUnetLabel():methods[alg])} else {draw(a,alg,step,isUnetVariant(alg)?methods[alg]+': '+activeUnetLabel():methods[alg])}
 renderLists();
}
function initSummary(){
 const s=document.getElementById('summary'),alg=algEl.value,tr=activeTrace(alg),stats=(data.h_stats.unet_models||{})[activeUnetModel()]||{},props=(data.unet_properties||{})[activeUnetModel()]||{},direct=activeTrace('unet_direct'),gaps=isUnetVariant(alg)?data.unet_variant_expansion_gaps[activeUnetModel()][alg]:null,meta=tr.metadata||{};let rows=[['case',data.case_id],['map type',data.map_type],['start',data.start],['goal',data.goal],['optimal cost',data.optimal_cost],['selected algorithm',methods[alg]],['expanded nodes',tr.expanded],['path cost',tr.cost],['path optimal',tr.optimal],['priority tuple',meta.priority_tuple||'(g + Manhattan h, g, node)']];
 if(isUnetVariant(alg)){rows.push(['Current U-Net model',activeUnetLabel()],['Direct A* expanded nodes',direct.expanded],['Direct A* optimal',direct.optimal],['admissibility violations',props.admissibility_violation_count],['consistency violations',props.consistency_violation_count],['mean overestimation',fmt(props.mean_overestimation_error)],['calibration',meta.calibration_kind==='none'?'none':`${meta.calibration_kind}: ${fmt(meta.calibration_value)}`],['selected - MLP',gaps.unet_minus_mlp],['selected - Manhattan',gaps.unet_minus_manhattan],['U-Net MAE',stats.mae?.toFixed(3)],['U-Net Spearman',stats.spearman?.toFixed(3)])}
 s.innerHTML=rows.map(r=>`<div><b>${r[0]}</b><br>${r[1]}</div>`).join('');
}
function initDisagreements(){
 const div=document.getElementById('disagreements'),alg=algEl.value;let rows=isUnetVariant(alg)?(data.disagreements_by_unet_variant||{})[activeUnetModel()][alg]:[];
 if(!rows.length){div.innerHTML='<p class="note">No U-Net-versus-baseline disagreement rows apply to the selected algorithm.</p>';return}
 div.innerHTML='<table><thead><tr><th>step</th><th>comparator</th><th>U-Net node</th><th>comparator node</th><th>true d U/C</th><th>path dist U/C</th></tr></thead><tbody>'+rows.map((r,i)=>`<tr class="clickable" data-step="${r.step}"><td>${r.step}</td><td>${methods[r.comparator]}</td><td>${r.unet_node}</td><td>${r.comparator_node}</td><td>${r.unet_true_distance}/${r.comparator_true_distance}</td><td>${r.unet_distance_to_path}/${r.comparator_distance_to_path}</td></tr>`).join('')+'</tbody></table>';
 div.querySelectorAll('tr.clickable').forEach((tr,i)=>tr.onclick=()=>{selectedDisagreement=rows[i];selectedCell=null;stepEl.value=tr.dataset.step;compareEl.value=selectedDisagreement.comparator;render()});
}
document.querySelectorAll('input,select').forEach(el=>{if(el!==unetModelEl&&el!==algEl)el.addEventListener('input',render)});
function changeUnetModel(){selectedDisagreement=null;updateStepLimit();initDisagreements();render()}
function changeAlgorithm(){selectedDisagreement=null;updateStepLimit();initDisagreements();render()}
unetModelEl.addEventListener('input',changeUnetModel);
unetModelEl.addEventListener('change',changeUnetModel);
algEl.addEventListener('input',changeAlgorithm);
algEl.addEventListener('change',changeAlgorithm);
document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play'}else{timer=setInterval(()=>{stepEl.value=(Number(stepEl.value)+1)%(Number(stepEl.max)+1);render()},180);document.getElementById('play').textContent='Pause'}};
document.addEventListener('keydown',event=>{
 const tag=(event.target&&event.target.tagName||'').toLowerCase();
 if(['input','select','textarea','button'].includes(tag))return;
 if(event.key==='ArrowRight'){event.preventDefault();setStep(Number(stepEl.value)+1)}
 else if(event.key==='ArrowLeft'){event.preventDefault();setStep(Number(stepEl.value)-1)}
 else if(event.key==='Home'){event.preventDefault();setStep(0)}
 else if(event.key==='End'){event.preventDefault();setStep(Number(stepEl.max))}
 else if(event.key===' '){event.preventDefault();document.getElementById('play').click()}
});
initSummary();initDisagreements();render();
</script>
</body></html>"""


def write_case_html(path, data):
    html = CASE_HTML.replace("__TITLE__", f"A* Case {data['case_id']}").replace("__DATA__", json.dumps(data))
    with open(path, "w", encoding="utf-8") as file:
        file.write(html)


def write_index(path, cases):
    rows = []
    for case in cases:
        rows.append(
            f"<tr><td><a href='cases/{case['case_id']}.html'>{case['case_id']}</a></td>"
            f"<td>{case['map_type']}</td><td>{case['start']}</td><td>{case['goal']}</td>"
            f"<td>{case['traces']['manhattan']['expanded']}</td><td>{case['traces']['mlp_tiebreak']['expanded']}</td>"
            f"<td>{case['traces']['unet_tiebreak']['expanded']}</td><td>{case['expansion_gaps']['unet_minus_mlp']}</td>"
            f"<td>{case['expansion_gaps']['unet_minus_manhattan']}</td><td>{case['optimal_cost']}</td></tr>"
        )
    html = """<!doctype html><html><head><meta charset='utf-8'><title>Interactive A* Viewer</title>
<style>body{font-family:system-ui;margin:24px;background:#f6f7f9}table{border-collapse:collapse;background:white}th,td{padding:7px 10px;border-bottom:1px solid #ddd;text-align:left}th{cursor:pointer;background:#edf1f5}</style></head>
<body><h1>Interactive A* Case Viewer</h1><p>Click a case id to open the step-by-step viewer. Click table headers to sort.</p>
<table id='t'><thead><tr><th>case id</th><th>map type</th><th>start</th><th>goal</th><th>Manhattan</th><th>MLP</th><th>MSE U-Net tie-break</th><th>MSE-MLP</th><th>MSE-Manhattan</th><th>cost</th></tr></thead><tbody>""" + "\n".join(rows) + """</tbody></table>
<script>document.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{let tb=document.querySelector('tbody');[...tb.rows].sort((a,b)=>{let x=a.cells[i].innerText,y=b.cells[i].innerText;let nx=parseFloat(x),ny=parseFloat(y);return (isNaN(nx)||isNaN(ny))?x.localeCompare(y):nx-ny}).forEach(r=>tb.appendChild(r))});</script>
</body></html>"""
    with open(path, "w", encoding="utf-8") as file:
        file.write(html)


def write_summary(path, cases):
    counts = Counter(case["map_type"] for case in cases)
    strongest_unet = min(cases, key=lambda c: c["expansion_gaps"]["unet_minus_mlp"])
    strongest_mlp = max(cases, key=lambda c: c["expansion_gaps"]["unet_minus_mlp"])
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Interactive A* Viewer Generation Summary\n\n")
        file.write(f"- cases generated: {len(cases)}\n")
        file.write("- algorithms included: Manhattan, MLP tie-break, Manhattan + U-Net tie-break, direct U-Net A*, U-Net calibrated n=5/n=10/n=20, Manhattan-distance calibrated U-Net A*, oracle-calibrated U-Net A*, and true-distance tie-break\n")
        file.write("- data fields: grid, start, goal, path, cost, optimality, expansion order, per-step frontier snapshots, g/h/f values, effective priority tuples, h-maps and independent traces for mse/ranking/ranking_adm/ranking_adm_cons, per-model admissibility and consistency statistics, true distance, error maps, and disagreement steps\n\n")
        file.write("## Viewer Updates\n\n")
        file.write("- added optional cell-value labels, disabled by default to avoid clutter\n")
        file.write("- supported label modes: h value, true distance, prediction error, g value, f value\n")
        file.write("- h-value labels are algorithm-specific: Manhattan shows Manhattan h, MLP shows MLP h, U-Net shows U-Net h, and oracle shows true distance\n")
        file.write("- the U-Net model and algorithm dropdowns select the matching precomputed independent trace, heatmap, labels, Open/Closed lists, rankings, frontier, path, and disagreement steps\n")
        file.write("- side-by-side mode labels each panel with that panel's own selected method; difference view hides labels and states this in the canvas title\n\n")
        file.write("## Keyboard And List Controls\n\n")
        file.write("- keyboard controls added: Right Arrow increments the step by 1, Left Arrow decrements the step by 1, both clamped to the valid step range\n")
        file.write("- optional keys: Space toggles play/pause, Home jumps to step 0, End jumps to the final step\n")
        file.write("- keyboard stepping is ignored while focus is inside inputs, selects, textareas, or buttons\n")
        file.write("- open/closed snapshots use the state immediately after completing expansion at the selected step\n")
        file.write("- open lists are full snapshots recorded during local trace generation and display-sorted by the effective priority tuple; sorting is for display only\n")
        file.write("- closed lists are reconstructed from the recorded expansion order up to the selected step\n")
        file.write("- list columns: rank/order, coordinate, g, h_primary, h_secondary, f, exact recorded priority, true distance\n\n")
        file.write("## Map Type Distribution\n\n")
        for key, value in sorted(counts.items()):
            file.write(f"- {key}: {value}\n")
        file.write("\n## Interesting Cases\n\n")
        file.write(f"- strongest U-Net win: {strongest_unet['case_id']} gap={strongest_unet['expansion_gaps']['unet_minus_mlp']}\n")
        file.write(f"- strongest MLP win: {strongest_mlp['case_id']} gap={strongest_mlp['expansion_gaps']['unet_minus_mlp']}\n")
        for structure in ["maze_like", "bottleneck", "narrow_corridor", "large_block", "random"]:
            match = next((case for case in cases if case["map_type"] == structure), None)
            if match:
                file.write(f"- {structure}: {match['case_id']}\n")
        file.write("\n## Opening Instructions\n\nOpen `outputs/interactive_astar_viewer/index.html` in a browser, then click a case.\n\n")
        file.write("## Known Limitations\n\n")
        file.write("- Disagreement steps are based on same expansion index differences; exact shared tie-set equivalence is approximated by the traced U-Net tie set.\n")
        file.write("- This is an interpretability tool only and does not prove causality or define a new algorithm.\n")
        file.write("- Frontier snapshots are recorded for these small 20x20 maps; larger maps may need snapshot thinning.\n")
        file.write("- Cell-value labels are suppressed automatically when cells are too small; dense labels can still be visually cluttered on heatmap backgrounds.\n")
        file.write("- Full open-list snapshots increase JSON/HTML size; this is acceptable for the current 20x20 case-study viewer but may need incremental trace storage for larger maps.\n")


def write_direct_unet_validation_report(path, cases):
    total = len(cases) * (len(UNET_VARIANT_KEYS) * len(COMBINED_LOSS_MODEL_LABELS) + 3)
    validations = [
        validation
        for case in cases
        for model_validations in case["trace_validation"].values()
        for validation in model_validations.values()
    ]
    validations.extend(
        validation
        for case in cases
        for validation in case["baseline_trace_validation"].values()
    )
    matches = sum(
        all(validation[key] for key in ("expanded_matches", "cost_matches", "path_matches"))
        for validation in validations
    )
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Direct U-Net Viewer Trace Validation\n\n")
        file.write("## Supported Algorithms\n\n")
        file.write("- Manhattan A*\n- Manhattan + U-Net tie-break\n- Direct U-Net A*\n- U-Net calibrated n=5\n- U-Net calibrated n=10\n- U-Net calibrated n=20\n- Manhattan-distance calibrated U-Net A*\n- Oracle-calibrated U-Net A*\n\n")
        file.write("## Trace Generation\n\n")
        file.write(f"- cases generated: {len(cases)}\n")
        file.write("- U-Net models per case: mse, ranking, ranking_adm, ranking_adm_cons\n")
        file.write(f"- independently generated traces checked: {total}\n")
        file.write("- each trace stores expansion order, Open/Closed snapshots, current-node priority, path, cost, and expanded-node count\n\n")
        file.write("## Validation\n\n")
        file.write(f"- traces checked against `src/astar.py::astar_search`: {len(validations)}\n")
        file.write(f"- exact matches for expansion count, path cost, and final path: {matches}/{len(validations)}\n")
        file.write("- result: PASS\n" if matches == len(validations) else "- result: FAIL\n")


def write_open_list_ranking_report(path, cases):
    lines = [
        "# Open List Ranking Investigation",
        "",
        "## Root Cause",
        "",
        "The A* implementation uses the correct ascending secondary-h ordering. The original multi-model viewer mixed an old-model trace rank with an expanded-model displayed h2 value, which made rankings appear inconsistent.",
        "",
        "## Actual Priority",
        "",
        "Both `src/astar.py` and the viewer trace use the ascending lexicographic tuple `(f, secondary_h, insertion_counter, g, node)`. `secondary_h` is minimized; there is no negative sign and no intentional reverse ordering.",
        "",
        "## Fix",
        "",
        "The viewer now stores and selects a complete U-Net trace for each supported combined-loss model: mse, ranking, ranking_adm, and ranking_adm_cons. Switching models updates the actual precomputed Open/Closed snapshots, rankings, frontier, expansion order, path, and disagreement rows. It does not rerun search in the browser.",
        "",
        "## Impact",
        "",
        "- The viewer now accurately replays each selected combined-loss model's corresponding search behavior.",
        "- The A* implementation and previous experiment results were not changed.",
        "- Previous U-Net tie-break experiments use the same ascending secondary-h ordering and do not need reconsideration on this issue.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate interactive HTML A* expansion viewer.")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_tiebreak_controls_100.csv")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--combined-loss-dir", default="outputs/combined_loss_ablation")
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--output-dir", default="outputs/interactive_astar_viewer")
    return parser.parse_args()


def main(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output_dir)
    case_dir = os.path.join(output_dir, "cases")
    data_dir = os.path.join(output_dir, "data")
    summary_dir = os.path.join(output_dir, "summary")
    for directory in [output_dir, case_dir, data_dir, summary_dir]:
        os.makedirs(directory, exist_ok=True)

    grouped = group_results(
        [
            ("random", os.path.join(project_root, args.random_results)),
            ("structured", os.path.join(project_root, args.structured_results)),
        ]
    )
    selected = select_cases(grouped, args.max_cases)
    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    combined_loss_dir = args.combined_loss_dir
    if not os.path.isabs(combined_loss_dir):
        combined_loss_dir = os.path.join(project_root, combined_loss_dir)
    unet_models = {}
    for model_name in COMBINED_LOSS_MODEL_LABELS:
        checkpoint = os.path.join(combined_loss_dir, f"{model_name}_best.pt")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Missing combined-loss U-Net checkpoint: {checkpoint}")
        unet_models[model_name] = load_unet_heuristic(checkpoint)

    cases = []
    for index, methods in enumerate(selected, start=1):
        case_id = f"case_{index:03d}"
        data = generate_case(case_id, methods, mlp_model, unet_models)
        cases.append(data)
        with open(os.path.join(data_dir, f"{case_id}.json"), "w", encoding="utf-8") as file:
            json.dump(data, file)
        write_case_html(os.path.join(case_dir, f"{case_id}.html"), data)
        print(f"Generated {case_id}: {data['map_type']}")

    write_index(os.path.join(output_dir, "index.html"), cases)
    write_summary(os.path.join(summary_dir, "viewer_generation_summary.md"), cases)
    write_direct_unet_validation_report(os.path.join(summary_dir, "direct_unet_trace_validation.md"), cases)
    write_open_list_ranking_report(os.path.join(summary_dir, "open_list_ranking_investigation.md"), cases)
    print(f"Saved interactive viewer to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
