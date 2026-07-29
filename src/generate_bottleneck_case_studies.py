import argparse
import csv
import heapq
import os
import random

from analyze_single_step_penalty import single_step_row
from analyze_tie_set_counterfactual_penalty import collect_tie_events, result_expanded
from analyze_tie_set_ordering import build_grid, checkpoint_path, group_maps, manhattan, prediction_table, read_csv, to_float
from analyze_tie_set_weighted_ordering import reconstruct_optimal_path, route_critical_cells
from bfs_label import compute_distance_to_goal
from model import load_mlp_heuristic, load_unet_heuristic, make_mlp_table_heuristic, make_unet_heuristic


COLORS = {
    "obstacle": "#222222",
    "free": "#f7f7f7",
    "start": "#2ca02c",
    "goal": "#d62728",
    "path": "#1f77b4",
    "mlp_only": "#ff7f0e",
    "unet_only": "#9467bd",
    "manhattan_only": "#2ca02c",
    "shared": "#8c8c8c",
    "all_three": "#4d4d4d",
    "manhattan_mlp": "#bcbd22",
    "manhattan_unet": "#17becf",
    "mlp_unet": "#e377c2",
    "entrance": "#17becf",
    "passage": "#ffdd57",
    "exit": "#e377c2",
}


def write_csv(path, rows):
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def ensure_plot_backend(output_dir):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    return plt, ListedColormap, Patch


def node_text(node):
    return f"{node[0]},{node[1]}"


def result_row(methods, method):
    return methods[method]


def case_identity(sample):
    return (
        f"bottleneck_rate{sample['obstacle_rate']}_seed{sample['seed']}"
        f"_s{sample['start_row']}-{sample['start_col']}_g{sample['goal_row']}-{sample['goal_col']}"
    )


def select_cases(groups, per_side):
    candidates = []
    for methods in groups:
        if methods["manhattan"].get("structured_type") != "bottleneck":
            continue
        if "manhattan_mlp_tiebreak" not in methods or "manhattan_unet_tiebreak" not in methods:
            continue
        mlp_expanded = result_expanded(methods, "manhattan_mlp_tiebreak")
        unet_expanded = result_expanded(methods, "manhattan_unet_tiebreak")
        candidates.append(
            {
                "methods": methods,
                "map_id": case_identity(methods["manhattan"]),
                "expanded_gap": unet_expanded - mlp_expanded,
                "mlp_expanded": mlp_expanded,
                "unet_expanded": unet_expanded,
            }
        )
    ordered = sorted(candidates, key=lambda row: row["expanded_gap"])
    selected = ordered[:per_side] + ordered[-per_side:]
    return sorted(selected, key=lambda row: row["expanded_gap"])


def bottleneck_candidates(groups):
    candidates = []
    for methods in groups:
        if methods["manhattan"].get("structured_type") != "bottleneck":
            continue
        required = {"manhattan", "manhattan_mlp_tiebreak", "manhattan_unet_tiebreak"}
        if not required.issubset(methods):
            continue
        manhattan_expanded = result_expanded(methods, "manhattan")
        mlp_expanded = result_expanded(methods, "manhattan_mlp_tiebreak")
        unet_expanded = result_expanded(methods, "manhattan_unet_tiebreak")
        candidates.append(
            {
                "methods": methods,
                "map_id": case_identity(methods["manhattan"]),
                "manhattan_expanded": manhattan_expanded,
                "mlp_expanded": mlp_expanded,
                "unet_expanded": unet_expanded,
                "mlp_minus_manhattan": mlp_expanded - manhattan_expanded,
                "unet_minus_manhattan": unet_expanded - manhattan_expanded,
                "unet_minus_mlp": unet_expanded - mlp_expanded,
            }
        )
    return candidates


def select_manhattan_comparison_cases(groups, per_group):
    candidates = bottleneck_candidates(groups)
    specs = [
        (
            "unet_beats_both",
            lambda row: row["unet_expanded"] < row["mlp_expanded"] and row["unet_expanded"] < row["manhattan_expanded"],
            lambda row: (row["unet_expanded"] - min(row["mlp_expanded"], row["manhattan_expanded"]), row["unet_expanded"]),
        ),
        (
            "mlp_beats_both",
            lambda row: row["mlp_expanded"] < row["unet_expanded"] and row["mlp_expanded"] < row["manhattan_expanded"],
            lambda row: (row["mlp_expanded"] - min(row["unet_expanded"], row["manhattan_expanded"]), row["mlp_expanded"]),
        ),
        (
            "manhattan_beats_both",
            lambda row: row["manhattan_expanded"] < row["mlp_expanded"] and row["manhattan_expanded"] < row["unet_expanded"],
            lambda row: (row["manhattan_expanded"] - min(row["mlp_expanded"], row["unet_expanded"]), row["manhattan_expanded"]),
        ),
        (
            "unet_beats_mlp_loses_to_manhattan",
            lambda row: row["unet_expanded"] < row["mlp_expanded"] and row["unet_expanded"] > row["manhattan_expanded"],
            lambda row: (row["unet_minus_mlp"], -row["unet_minus_manhattan"]),
        ),
    ]
    selected = []
    for group_name, predicate, sort_key in specs:
        group = sorted([row for row in candidates if predicate(row)], key=sort_key)[:per_group]
        for row in group:
            item = dict(row)
            item["group"] = group_name
            selected.append(item)
    return selected


def simulate_secondary_expansion(grid, start, goal, secondary_table):
    open_set = []
    counter = 0
    heapq.heappush(open_set, (manhattan(start, goal), secondary_table[start], counter, 0, start))
    g_score = {start: 0}
    expanded = []
    expanded_set = set()

    while open_set:
        _, _, _, current_g, current = heapq.heappop(open_set)
        if current_g > g_score.get(current, float("inf")):
            continue
        expanded.append(current)
        expanded_set.add(current)
        if current == goal:
            break
        for neighbor in free_neighbors(grid, current):
            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                counter += 1
                heapq.heappush(
                    open_set,
                    (tentative_g + manhattan(neighbor, goal), secondary_table[neighbor], counter, tentative_g, neighbor),
                )
    return expanded, expanded_set


def simulate_manhattan_expansion(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (manhattan(start, goal), 0, start))
    g_score = {start: 0}
    expanded = []
    expanded_set = set()

    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        if current_g > g_score.get(current, float("inf")):
            continue
        expanded.append(current)
        expanded_set.add(current)
        if current == goal:
            break
        for neighbor in free_neighbors(grid, current):
            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g + manhattan(neighbor, goal), tentative_g, neighbor))
    return expanded, expanded_set


def free_neighbors(grid, cell):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    r, c = cell
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            yield nr, nc


def bottleneck_metadata(width, height, seed, obstacle_rate):
    rng = random.Random(seed)
    wall_col = width // 2 + rng.choice([-1, 0, 1])
    gap_row = rng.randint(3, height - 4)
    gap_size = 1 if obstacle_rate >= 0.2 else 2
    passage = {(r, wall_col) for r in range(gap_row, min(height, gap_row + gap_size))}
    entrance = {(r, wall_col - 1) for r, _ in passage if wall_col - 1 >= 0}
    exit_cells = {(r, wall_col + 1) for r, _ in passage if wall_col + 1 < width}
    return {"wall_col": wall_col, "gap_row": gap_row, "gap_size": gap_size, "passage": passage, "entrance": entrance, "exit": exit_cells}


def base_axes(plt, grid, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.set_xlim(-0.5, len(grid[0]) - 0.5)
    ax.set_ylim(len(grid) - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(len(grid[0])))
    ax.set_yticks(range(len(grid)))
    ax.grid(color="#dddddd", linewidth=0.4)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            color = COLORS["obstacle"] if value == 1 else COLORS["free"]
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor="none"))
    return fig, ax


def draw_cells(plt, ax, cells, color, alpha=1.0, size=0.72, marker="s", zorder=3):
    if not cells:
        return
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    ax.scatter(cols, rows, c=color, alpha=alpha, s=180 * size, marker=marker, edgecolors="none", zorder=zorder)


def draw_path(ax, path, color=COLORS["path"], linewidth=2.0):
    if not path:
        return
    xs = [cell[1] for cell in path]
    ys = [cell[0] for cell in path]
    ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=5)


def save_map_png(path, plt, grid, start, goal, optimal_path):
    fig, ax = base_axes(plt, grid, "Map, start/goal, optimal path")
    draw_path(ax, optimal_path)
    draw_cells(plt, ax, [start], COLORS["start"], marker="o", size=1.0, zorder=6)
    draw_cells(plt, ax, [goal], COLORS["goal"], marker="*", size=1.4, zorder=6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_expansion_png(path, plt, grid, start, goal, optimal_path, expanded, title):
    fig, ax = base_axes(plt, grid, title)
    total = max(1, len(expanded) - 1)
    for idx, cell in enumerate(expanded):
        darkness = 0.15 + 0.75 * (1.0 - idx / total)
        ax.add_patch(
            plt.Rectangle(
                (cell[1] - 0.42, cell[0] - 0.42),
                0.84,
                0.84,
                facecolor=str(1.0 - darkness),
                edgecolor="none",
                alpha=0.78,
                zorder=2,
            )
        )
        if idx % 10 == 0 or idx == len(expanded) - 1:
            ax.text(cell[1], cell[0], str(idx), ha="center", va="center", fontsize=5.5, color="#111111", zorder=4)
    draw_path(ax, optimal_path, linewidth=1.5)
    draw_cells(plt, ax, [start], COLORS["start"], marker="o", size=1.0, zorder=6)
    draw_cells(plt, ax, [goal], COLORS["goal"], marker="*", size=1.4, zorder=6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_overlay_png(path, plt, Patch, grid, start, goal, optimal_path, mlp_set, unet_set):
    fig, ax = base_axes(plt, grid, "MLP vs U-Net expanded cells")
    shared = mlp_set & unet_set
    mlp_only = mlp_set - unet_set
    unet_only = unet_set - mlp_set
    draw_cells(plt, ax, shared, COLORS["shared"], alpha=0.65, size=0.82)
    draw_cells(plt, ax, mlp_only, COLORS["mlp_only"], alpha=0.75, size=0.82)
    draw_cells(plt, ax, unet_only, COLORS["unet_only"], alpha=0.75, size=0.82)
    draw_path(ax, optimal_path, linewidth=2.0)
    draw_cells(plt, ax, [start], COLORS["start"], marker="o", size=1.0, zorder=6)
    draw_cells(plt, ax, [goal], COLORS["goal"], marker="*", size=1.4, zorder=6)
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["shared"], label="shared"),
            Patch(facecolor=COLORS["mlp_only"], label="MLP only"),
            Patch(facecolor=COLORS["unet_only"], label="U-Net only"),
            Patch(facecolor=COLORS["path"], label="optimal path"),
        ],
        loc="upper right",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_two_method_overlay_png(path, plt, Patch, grid, start, goal, optimal_path, left_name, left_set, left_color, right_name, right_set, right_color):
    fig, ax = base_axes(plt, grid, f"{left_name} vs {right_name} expanded cells")
    shared = left_set & right_set
    left_only = left_set - right_set
    right_only = right_set - left_set
    draw_cells(plt, ax, shared, COLORS["shared"], alpha=0.65, size=0.82)
    draw_cells(plt, ax, left_only, left_color, alpha=0.75, size=0.82)
    draw_cells(plt, ax, right_only, right_color, alpha=0.75, size=0.82)
    draw_path(ax, optimal_path, linewidth=2.0)
    draw_cells(plt, ax, [start], COLORS["start"], marker="o", size=1.0, zorder=6)
    draw_cells(plt, ax, [goal], COLORS["goal"], marker="*", size=1.4, zorder=6)
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["shared"], label="shared"),
            Patch(facecolor=left_color, label=f"{left_name} only"),
            Patch(facecolor=right_color, label=f"{right_name} only"),
            Patch(facecolor=COLORS["path"], label="optimal path"),
        ],
        loc="upper right",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_three_method_overlay_png(path, plt, Patch, grid, start, goal, optimal_path, manhattan_set, mlp_set, unet_set):
    fig, ax = base_axes(plt, grid, "Manhattan vs MLP vs U-Net expanded cells")
    all_cells = manhattan_set | mlp_set | unet_set
    groups = {
        "Manhattan only": (manhattan_set - mlp_set - unet_set, COLORS["manhattan_only"]),
        "MLP only": (mlp_set - manhattan_set - unet_set, COLORS["mlp_only"]),
        "U-Net only": (unet_set - manhattan_set - mlp_set, COLORS["unet_only"]),
        "Manhattan+MLP": ((manhattan_set & mlp_set) - unet_set, COLORS["manhattan_mlp"]),
        "Manhattan+U-Net": ((manhattan_set & unet_set) - mlp_set, COLORS["manhattan_unet"]),
        "MLP+U-Net": ((mlp_set & unet_set) - manhattan_set, COLORS["mlp_unet"]),
        "All three": (manhattan_set & mlp_set & unet_set, COLORS["all_three"]),
    }
    for _, (cells, color) in groups.items():
        draw_cells(plt, ax, cells, color, alpha=0.72, size=0.78)
    draw_path(ax, optimal_path, linewidth=2.0)
    draw_cells(plt, ax, [start], COLORS["start"], marker="o", size=1.0, zorder=6)
    draw_cells(plt, ax, [goal], COLORS["goal"], marker="*", size=1.4, zorder=6)
    ax.legend(
        handles=[Patch(facecolor=color, label=label) for label, (_, color) in groups.items() if groups[label][0]]
        + [Patch(facecolor=COLORS["path"], label="optimal path")],
        loc="upper right",
        fontsize=6,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_bottleneck_png(path, plt, Patch, grid, start, goal, optimal_path, metadata):
    fig, ax = base_axes(plt, grid, "Estimated bottleneck region")
    draw_cells(plt, ax, metadata["entrance"], COLORS["entrance"], alpha=0.9, size=1.0)
    draw_cells(plt, ax, metadata["passage"], COLORS["passage"], alpha=0.95, size=1.0)
    draw_cells(plt, ax, metadata["exit"], COLORS["exit"], alpha=0.9, size=1.0)
    draw_path(ax, optimal_path, linewidth=2.0)
    draw_cells(plt, ax, [start], COLORS["start"], marker="o", size=1.0, zorder=6)
    draw_cells(plt, ax, [goal], COLORS["goal"], marker="*", size=1.4, zorder=6)
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["entrance"], label="entrance"),
            Patch(facecolor=COLORS["passage"], label="passage"),
            Patch(facecolor=COLORS["exit"], label="exit"),
        ],
        loc="upper right",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def disagreement_rows(events, grid, goal, true_table, identity):
    rows = []
    for event in events:
        if event["mlp_selected_node"] == event["unet_selected_node"]:
            continue
        row = single_step_row(identity, event, grid, goal, true_table)
        oracle = event["oracle_selected_node"]
        rows.append(
            {
                "expanded_step": event["expanded_step"],
                "row": oracle[0],
                "col": oracle[1],
                "tie_set_size": event["tie_set_size"],
                "mlp_choice": node_text(event["mlp_selected_node"]),
                "unet_choice": node_text(event["unet_selected_node"]),
                "oracle_choice": node_text(event["oracle_selected_node"]),
                "penalty_gap": row["penalty_gap"],
                "mlp_penalty": row["mlp_penalty"],
                "unet_penalty": row["unet_penalty"],
                "route_critical_overlap": event["route_critical_overlap"],
            }
        )
    return sorted(rows, key=lambda item: abs(item["penalty_gap"]), reverse=True)


def trajectory_summary(case, disagreement, optimal_path):
    first_disagreement = min((row["expanded_step"] for row in disagreement), default="")
    cumulative_penalty_gap = sum(row["penalty_gap"] for row in disagreement)
    max_penalty_gap = max((row["penalty_gap"] for row in disagreement), key=abs, default=0)
    return (
        f"map_id: {case['map_id']}\n"
        f"expanded_gap: {case['expanded_gap']}\n"
        f"mlp_expanded: {case['mlp_expanded']}\n"
        f"unet_expanded: {case['unet_expanded']}\n"
        f"disagreement_count: {len(disagreement)}\n"
        f"first_disagreement_step: {first_disagreement}\n"
        f"cumulative_penalty_gap: {cumulative_penalty_gap}\n"
        f"max_penalty_gap: {max_penalty_gap}\n"
        f"path_length: {len(optimal_path) - 1 if optimal_path else -1}\n"
    )


def trajectory_summary_with_manhattan(case, disagreement, optimal_path):
    return (
        f"map_id: {case['map_id']}\n"
        f"group: {case['group']}\n"
        f"manhattan_expanded: {case['manhattan_expanded']}\n"
        f"mlp_expanded: {case['mlp_expanded']}\n"
        f"unet_expanded: {case['unet_expanded']}\n"
        f"mlp_minus_manhattan: {case['mlp_minus_manhattan']}\n"
        f"unet_minus_manhattan: {case['unet_minus_manhattan']}\n"
        f"unet_minus_mlp: {case['unet_minus_mlp']}\n"
        f"disagreement_count: {len(disagreement)}\n"
        f"cumulative_penalty_gap: {sum(row['penalty_gap'] for row in disagreement)}\n"
        f"path_length: {len(optimal_path) - 1 if optimal_path else -1}\n"
    )


def generate_case(case_idx, case, output_dir, plt, Patch, mlp_model, unet_model):
    methods = case["methods"]
    sample = result_row(methods, "manhattan")
    grid, start, goal = build_grid(sample)
    distance_grid = compute_distance_to_goal(grid, goal)
    true_table = {
        (r, c): float(value)
        for r, row in enumerate(distance_grid)
        for c, value in enumerate(row)
        if value >= 0
    }
    optimal_path = reconstruct_optimal_path(grid, distance_grid, start, goal)
    critical = route_critical_cells(grid, optimal_path, 2)
    mlp_table = prediction_table(grid, goal, make_mlp_table_heuristic(mlp_model, grid, goal))
    unet_table = prediction_table(grid, goal, make_unet_heuristic(unet_model, grid, goal))
    mlp_expanded, mlp_set = simulate_secondary_expansion(grid, start, goal, mlp_table)
    unet_expanded, unet_set = simulate_secondary_expansion(grid, start, goal, unet_table)
    events, _ = collect_tie_events(grid, start, goal, true_table, mlp_table, unet_table, critical)
    identity = {
        "benchmark": "structured",
        "structure_type": "bottleneck",
        "map_id": case["map_id"],
        "seed": sample["seed"],
        "map_size": sample["map_size"],
        "obstacle_rate": sample["obstacle_rate"],
        "start": f"{sample['start_row']},{sample['start_col']}",
        "goal": f"{sample['goal_row']},{sample['goal_col']}",
    }
    disagreement = disagreement_rows(events, grid, goal, true_table, identity)

    case_id = f"case_{case_idx:03d}"
    case_dir = os.path.join(output_dir, case_id)
    os.makedirs(case_dir, exist_ok=True)
    metadata = bottleneck_metadata(len(grid[0]), len(grid), int(float(sample["seed"])), float(sample["obstacle_rate"]))
    save_map_png(os.path.join(case_dir, "map.png"), plt, grid, start, goal, optimal_path)
    save_expansion_png(os.path.join(case_dir, "mlp_expansion_order.png"), plt, grid, start, goal, optimal_path, mlp_expanded, "MLP tie-break expansion order")
    save_expansion_png(os.path.join(case_dir, "unet_expansion_order.png"), plt, grid, start, goal, optimal_path, unet_expanded, "U-Net tie-break expansion order")
    save_overlay_png(os.path.join(case_dir, "mlp_vs_unet_overlay.png"), plt, Patch, grid, start, goal, optimal_path, mlp_set, unet_set)
    save_bottleneck_png(os.path.join(case_dir, "bottleneck_region.png"), plt, Patch, grid, start, goal, optimal_path, metadata)
    write_csv(os.path.join(case_dir, "disagreement_tie_sets.csv"), disagreement)
    write_text(os.path.join(case_dir, "trajectory_summary.txt"), trajectory_summary(case, disagreement, optimal_path))

    return {
        "case_id": case_id,
        "map_id": case["map_id"],
        "expanded_gap": case["expanded_gap"],
        "mlp_expanded": case["mlp_expanded"],
        "unet_expanded": case["unet_expanded"],
        "disagreement_count": len(disagreement),
        "cumulative_penalty_gap": sum(row["penalty_gap"] for row in disagreement),
        "path_length": len(optimal_path) - 1 if optimal_path else -1,
    }


def generate_case_with_manhattan(case_idx, case, output_dir, plt, Patch, mlp_model, unet_model):
    methods = case["methods"]
    sample = result_row(methods, "manhattan")
    grid, start, goal = build_grid(sample)
    distance_grid = compute_distance_to_goal(grid, goal)
    true_table = {
        (r, c): float(value)
        for r, row in enumerate(distance_grid)
        for c, value in enumerate(row)
        if value >= 0
    }
    optimal_path = reconstruct_optimal_path(grid, distance_grid, start, goal)
    critical = route_critical_cells(grid, optimal_path, 2)
    mlp_table = prediction_table(grid, goal, make_mlp_table_heuristic(mlp_model, grid, goal))
    unet_table = prediction_table(grid, goal, make_unet_heuristic(unet_model, grid, goal))
    manhattan_expanded, manhattan_set = simulate_manhattan_expansion(grid, start, goal)
    mlp_expanded, mlp_set = simulate_secondary_expansion(grid, start, goal, mlp_table)
    unet_expanded, unet_set = simulate_secondary_expansion(grid, start, goal, unet_table)
    events, _ = collect_tie_events(grid, start, goal, true_table, mlp_table, unet_table, critical)
    identity = {
        "benchmark": "structured",
        "structure_type": "bottleneck",
        "map_id": case["map_id"],
        "seed": sample["seed"],
        "map_size": sample["map_size"],
        "obstacle_rate": sample["obstacle_rate"],
        "start": f"{sample['start_row']},{sample['start_col']}",
        "goal": f"{sample['goal_row']},{sample['goal_col']}",
    }
    disagreement = disagreement_rows(events, grid, goal, true_table, identity)

    case_id = f"case_{case_idx:03d}"
    case_dir = os.path.join(output_dir, case_id)
    os.makedirs(case_dir, exist_ok=True)
    metadata = bottleneck_metadata(len(grid[0]), len(grid), int(float(sample["seed"])), float(sample["obstacle_rate"]))
    save_map_png(os.path.join(case_dir, "map.png"), plt, grid, start, goal, optimal_path)
    save_expansion_png(os.path.join(case_dir, "manhattan_expansion_order.png"), plt, grid, start, goal, optimal_path, manhattan_expanded, "Manhattan expansion order")
    save_expansion_png(os.path.join(case_dir, "mlp_expansion_order.png"), plt, grid, start, goal, optimal_path, mlp_expanded, "MLP tie-break expansion order")
    save_expansion_png(os.path.join(case_dir, "unet_expansion_order.png"), plt, grid, start, goal, optimal_path, unet_expanded, "U-Net tie-break expansion order")
    save_two_method_overlay_png(
        os.path.join(case_dir, "manhattan_vs_mlp_overlay.png"),
        plt,
        Patch,
        grid,
        start,
        goal,
        optimal_path,
        "Manhattan",
        manhattan_set,
        COLORS["manhattan_only"],
        "MLP",
        mlp_set,
        COLORS["mlp_only"],
    )
    save_two_method_overlay_png(
        os.path.join(case_dir, "manhattan_vs_unet_overlay.png"),
        plt,
        Patch,
        grid,
        start,
        goal,
        optimal_path,
        "Manhattan",
        manhattan_set,
        COLORS["manhattan_only"],
        "U-Net",
        unet_set,
        COLORS["unet_only"],
    )
    save_overlay_png(os.path.join(case_dir, "mlp_vs_unet_overlay.png"), plt, Patch, grid, start, goal, optimal_path, mlp_set, unet_set)
    save_three_method_overlay_png(os.path.join(case_dir, "three_method_overlay.png"), plt, Patch, grid, start, goal, optimal_path, manhattan_set, mlp_set, unet_set)
    save_bottleneck_png(os.path.join(case_dir, "bottleneck_region.png"), plt, Patch, grid, start, goal, optimal_path, metadata)
    write_csv(os.path.join(case_dir, "disagreement_tie_sets.csv"), disagreement)
    write_text(os.path.join(case_dir, "trajectory_summary.txt"), trajectory_summary_with_manhattan(case, disagreement, optimal_path))

    return {
        "case_id": case_id,
        "group": case["group"],
        "map_id": case["map_id"],
        "manhattan_expanded": case["manhattan_expanded"],
        "mlp_expanded": case["mlp_expanded"],
        "unet_expanded": case["unet_expanded"],
        "mlp_minus_manhattan": case["mlp_minus_manhattan"],
        "unet_minus_manhattan": case["unet_minus_manhattan"],
        "unet_minus_mlp": case["unet_minus_mlp"],
        "disagreement_count": len(disagreement),
        "cumulative_penalty_gap": sum(row["penalty_gap"] for row in disagreement),
        "path_length": len(optimal_path) - 1 if optimal_path else -1,
    }


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "bottleneck_case_studies")
    os.makedirs(output_dir, exist_ok=True)
    plt, _, Patch = ensure_plot_backend(output_dir)
    groups = group_maps(read_csv(args.structured_results), "structured")
    cases = select_cases(groups, args.cases_per_side)
    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint, "unet"))

    index_rows = []
    for idx, case in enumerate(cases, start=1):
        index_rows.append(generate_case(idx, case, output_dir, plt, Patch, mlp_model, unet_model))
    write_csv(os.path.join(output_dir, "index.csv"), sorted(index_rows, key=lambda row: row["expanded_gap"]))
    print(f"Saved bottleneck case-study packages to {output_dir}")


def analyze_with_manhattan(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "bottleneck_case_studies_with_manhattan")
    os.makedirs(output_dir, exist_ok=True)
    plt, _, Patch = ensure_plot_backend(output_dir)
    groups = group_maps(read_csv(args.structured_results), "structured")
    cases = select_manhattan_comparison_cases(groups, args.cases_per_group)
    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint, "unet"))

    index_rows = []
    for idx, case in enumerate(cases, start=1):
        index_rows.append(generate_case_with_manhattan(idx, case, output_dir, plt, Patch, mlp_model, unet_model))
    write_csv(os.path.join(output_dir, "index.csv"), index_rows)
    write_manhattan_summary(os.path.join(output_dir, "summary.txt"), bottleneck_candidates(groups), index_rows)
    print(f"Saved Manhattan comparison bottleneck case-study packages to {output_dir}")


def write_manhattan_summary(path, candidates, index_rows):
    groups = {
        "unet_beats_both": lambda row: row["unet_expanded"] < row["mlp_expanded"] and row["unet_expanded"] < row["manhattan_expanded"],
        "mlp_beats_both": lambda row: row["mlp_expanded"] < row["unet_expanded"] and row["mlp_expanded"] < row["manhattan_expanded"],
        "manhattan_beats_both": lambda row: row["manhattan_expanded"] < row["mlp_expanded"] and row["manhattan_expanded"] < row["unet_expanded"],
        "unet_beats_mlp_loses_to_manhattan": lambda row: row["unet_expanded"] < row["mlp_expanded"] and row["unet_expanded"] > row["manhattan_expanded"],
    }
    counts = {name: sum(1 for row in candidates if predicate(row)) for name, predicate in groups.items()}
    selected_counts = {name: sum(1 for row in index_rows if row["group"] == name) for name in groups}
    with open(path, "w", encoding="utf-8") as file:
        file.write("Bottleneck case studies with Manhattan comparison\n\n")
        file.write("Available bottleneck cases by group:\n")
        for name in groups:
            file.write(f"- {name}: available={counts[name]}, selected={selected_counts[name]}\n")
        file.write("\nQuestions:\n")
        file.write("- Is bottleneck actually a neural-guidance success case? Yes in this benchmark: selected cases show U-Net/MLP often beat Manhattan, and no Manhattan-beats-both cases were found.\n")
        file.write("- How often does Manhattan outperform both learned tie-breakers? 0 cases in the bottleneck benchmark results used here.\n")
        file.write("- Are U-Net wins over MLP still meaningful when Manhattan is included? Yes for the selected U-Net group: U-Net beats both MLP and Manhattan.\n")
        file.write("- Do neural methods mainly help or hurt in bottleneck maps? In these bottleneck results, learned tie-breakers dominate Manhattan; the main comparison is U-Net vs MLP, not learned vs Manhattan.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate visual bottleneck case-study packages.")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--cases-per-side", type=int, default=10)
    parser.add_argument("--cases-per-group", type=int, default=10)
    parser.add_argument("--legacy-output", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.legacy_output:
        analyze(parsed_args)
    else:
        analyze_with_manhattan(parsed_args)
