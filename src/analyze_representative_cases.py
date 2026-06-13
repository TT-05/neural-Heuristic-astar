import argparse
import csv
import os

from astar import astar_search
from bfs_label import compute_distance_to_goal
from experiment import build_heuristics, dijkstra_heuristic, load_models
from model import manhattan_heuristic
from structured_maps import generate_structured_map
from analyze_failure_patterns import (
    error_metrics,
    local_consistency_metrics,
    prediction_grid_from_heuristic,
    save_path_overlay,
    setup_matplotlib,
)


GROUPS = [
    "maze_like_unet_wins",
    "maze_like_mlp_wins",
    "large_block_unet_nonoptimal",
    "narrow_corridor_unet_fails",
]
METHODS = ["dijkstra", "manhattan", "mlp_table", "unet"]


def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(row, key, default=0.0):
    value = row.get(key, "")
    if value == "":
        return default
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    return float(value)


def to_int(row, key, default=0):
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def map_key(row):
    return (
        row["seed"],
        row["map_size"],
        row["obstacle_rate"],
        row.get("map_mode", ""),
        row.get("structured_type", ""),
        row.get("start_goal_mode", ""),
        row.get("start_row", ""),
        row.get("start_col", ""),
        row.get("goal_row", ""),
        row.get("goal_col", ""),
    )


def group_results(rows):
    grouped = {}
    for row in rows:
        if row.get("heuristic") not in METHODS:
            continue
        grouped.setdefault(map_key(row), {})[row["heuristic"]] = row
    return grouped


def case_id(methods):
    row = methods["unet"]
    return (
        f"{row['structured_type']}_rate{row['obstacle_rate']}_seed{row['seed']}"
        f"_s{row['start_row']}-{row['start_col']}_g{row['goal_row']}-{row['goal_col']}"
    )


def base_case_record(group_name, methods, score):
    row = methods["unet"]
    return {
        "group": group_name,
        "case_id": case_id(methods),
        "score": score,
        "seed": to_int(row, "seed"),
        "map_size": to_int(row, "map_size"),
        "obstacle_rate": to_float(row, "obstacle_rate"),
        "map_mode": row.get("map_mode", ""),
        "structured_type": row.get("structured_type", ""),
        "start": (to_int(row, "start_row"), to_int(row, "start_col")),
        "goal": (to_int(row, "goal_row"), to_int(row, "goal_col")),
        "recorded_dijkstra_cost": to_float(methods["dijkstra"], "path_length"),
        "recorded_manhattan_expanded": to_float(methods["manhattan"], "expanded_nodes"),
        "recorded_mlp_expanded": to_float(methods["mlp_table"], "expanded_nodes"),
        "recorded_unet_expanded": to_float(methods["unet"], "expanded_nodes"),
        "recorded_unet_cost": to_float(methods["unet"], "path_length"),
        "recorded_unet_mae": to_float(methods["unet"], "mae"),
        "recorded_unet_overestimate_rate": to_float(methods["unet"], "overestimate_rate"),
    }


def select_cases(grouped, top_k):
    selected = {group: [] for group in GROUPS}

    for methods in grouped.values():
        if not all(method in methods for method in METHODS):
            continue
        if methods["unet"].get("skip_reason", ""):
            continue

        structured_type = methods["unet"].get("structured_type", "")
        dijkstra_cost = to_float(methods["dijkstra"], "path_length")
        manhattan_expanded = to_float(methods["manhattan"], "expanded_nodes")
        mlp_expanded = to_float(methods["mlp_table"], "expanded_nodes")
        unet_expanded = to_float(methods["unet"], "expanded_nodes")
        unet_cost = to_float(methods["unet"], "path_length")

        if structured_type == "maze_like" and unet_expanded < mlp_expanded:
            selected["maze_like_unet_wins"].append(
                base_case_record("maze_like_unet_wins", methods, mlp_expanded - unet_expanded)
            )

        if structured_type == "maze_like" and mlp_expanded < unet_expanded:
            selected["maze_like_mlp_wins"].append(
                base_case_record("maze_like_mlp_wins", methods, unet_expanded - mlp_expanded)
            )

        if structured_type == "large_block" and unet_cost > dijkstra_cost:
            selected["large_block_unet_nonoptimal"].append(
                base_case_record("large_block_unet_nonoptimal", methods, unet_cost - dijkstra_cost)
            )

        expanded_failure = unet_expanded - manhattan_expanded
        cost_failure = unet_cost - dijkstra_cost
        if structured_type == "narrow_corridor" and (expanded_failure >= 0 or cost_failure > 0):
            selected["narrow_corridor_unet_fails"].append(
                base_case_record("narrow_corridor_unet_fails", methods, max(expanded_failure, cost_failure))
            )

    output = []
    for group_name, cases in selected.items():
        cases.sort(key=lambda row: row["score"], reverse=True)
        output.extend(cases[:top_k])
    return output


def ensure_matplotlib(project_root):
    plt, np = setup_matplotlib(project_root)
    if plt is None:
        raise RuntimeError("matplotlib is required to generate representative case visualizations.")
    return plt, np


def save_obstacle_map(path, grid, start, goal, plt, np):
    image = np.ones((len(grid), len(grid[0]), 3), dtype=float)
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == 1:
                image[r, c] = [0.05, 0.05, 0.05]
    plt.figure(figsize=(5, 5))
    plt.imshow(image, interpolation="nearest")
    plt.scatter([start[1]], [start[0]], c="#1f77b4", marker="o", s=45, label="start")
    plt.scatter([goal[1]], [goal[0]], c="#ff7f0e", marker="*", s=85, label="goal")
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def masked_values(values, valid_mask, np):
    return np.array(
        [[values[r][c] if valid_mask[r][c] else np.nan for c in range(len(values[0]))] for r in range(len(values))],
        dtype=float,
    )


def save_heatmap(path, values, title, plt, np, valid_mask=None, cmap="viridis"):
    if valid_mask is not None:
        data = masked_values(values, valid_mask, np)
    else:
        data = np.array(values, dtype=float)
    plt.figure(figsize=(5, 4))
    image = plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_text_grid(path, values, valid_mask=None, decimals=2):
    lines = []
    for r, row in enumerate(values):
        cells = []
        for c, value in enumerate(row):
            if valid_mask is not None and not valid_mask[r][c]:
                cells.append("  ##  ")
            elif decimals == 0:
                cells.append(f"{int(value):4d}")
            else:
                cells.append(f"{float(value):7.{decimals}f}")
        lines.append(" ".join(cells))
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def rerun_case(case, mlp_model, unet_model):
    grid = generate_structured_map(
        width=case["map_size"],
        height=case["map_size"],
        seed=case["seed"],
        obstacle_rate=case["obstacle_rate"],
        structured_type=case["structured_type"],
    )
    start = case["start"]
    goal = case["goal"]
    reproduction_notes = []
    if grid[start[0]][start[1]] != 0:
        reproduction_notes.append("recorded start is not free after regeneration")
    if grid[goal[0]][goal[1]] != 0:
        reproduction_notes.append("recorded goal is not free after regeneration")

    distance_grid = compute_distance_to_goal(grid, goal)
    if distance_grid[start[0]][start[1]] == -1:
        reproduction_notes.append("recorded start-goal pair is not solvable after regeneration")

    heuristics = dict(build_heuristics(METHODS, mlp_model, unet_model, grid, goal))
    results = {}
    for name in METHODS:
        results[name] = astar_search(grid, start, goal, heuristics[name])

    unet_grid = prediction_grid_from_heuristic(grid, goal, heuristics["unet"])
    mlp_grid = prediction_grid_from_heuristic(grid, goal, heuristics["mlp_table"])
    valid_mask = [[value >= 0 for value in row] for row in distance_grid]
    error_grid = []
    for r, row in enumerate(distance_grid):
        error_row = []
        for c, true_value in enumerate(row):
            error_row.append(unet_grid[r][c] - true_value if valid_mask[r][c] else 0.0)
        error_grid.append(error_row)

    unet_metrics = error_metrics(distance_grid, unet_grid)
    consistency_metrics = local_consistency_metrics(unet_grid, valid_mask)

    return {
        "grid": grid,
        "distance_grid": distance_grid,
        "valid_mask": valid_mask,
        "unet_grid": unet_grid,
        "mlp_grid": mlp_grid,
        "error_grid": error_grid,
        "results": results,
        "heuristics": heuristics,
        "unet_metrics": unet_metrics,
        "consistency_metrics": consistency_metrics,
        "reproduction_notes": reproduction_notes,
    }


def save_case_outputs(case, rerun, case_dir, plt, np):
    os.makedirs(case_dir, exist_ok=True)
    start = case["start"]
    goal = case["goal"]
    grid = rerun["grid"]
    results = rerun["results"]

    save_path_overlay(
        os.path.join(case_dir, "path_overlay.png"),
        grid,
        start,
        goal,
        results["dijkstra"]["path"],
        results["manhattan"]["path"],
        results["unet"]["path"],
        case["case_id"],
        plt,
        np,
    )
    save_obstacle_map(os.path.join(case_dir, "obstacle_map.png"), grid, start, goal, plt, np)
    save_heatmap(
        os.path.join(case_dir, "true_distance_map.png"),
        rerun["distance_grid"],
        "BFS true distance",
        plt,
        np,
        valid_mask=rerun["valid_mask"],
    )
    save_heatmap(
        os.path.join(case_dir, "unet_predicted_map.png"),
        rerun["unet_grid"],
        "U-Net predicted distance",
        plt,
        np,
        valid_mask=rerun["valid_mask"],
    )
    save_heatmap(
        os.path.join(case_dir, "unet_error_map.png"),
        rerun["error_grid"],
        "U-Net error: predicted - true",
        plt,
        np,
        valid_mask=rerun["valid_mask"],
        cmap="coolwarm",
    )
    save_heatmap(
        os.path.join(case_dir, "mlp_predicted_map.png"),
        rerun["mlp_grid"],
        "MLP table predicted distance",
        plt,
        np,
        valid_mask=rerun["valid_mask"],
    )
    save_text_grid(os.path.join(case_dir, "true_distance_map.txt"), rerun["distance_grid"], rerun["valid_mask"], decimals=0)
    save_text_grid(os.path.join(case_dir, "unet_predicted_map.txt"), rerun["unet_grid"], rerun["valid_mask"])
    save_text_grid(os.path.join(case_dir, "unet_error_map.txt"), rerun["error_grid"], rerun["valid_mask"])
    save_text_grid(os.path.join(case_dir, "mlp_predicted_map.txt"), rerun["mlp_grid"], rerun["valid_mask"])


def selected_case_row(case, rerun, relative_dir):
    results = rerun["results"]
    row = {
        "group": case["group"],
        "rank_score": case["score"],
        "case_id": case["case_id"],
        "seed": case["seed"],
        "map_size": case["map_size"],
        "obstacle_rate": case["obstacle_rate"],
        "structured_type": case["structured_type"],
        "start_row": case["start"][0],
        "start_col": case["start"][1],
        "goal_row": case["goal"][0],
        "goal_col": case["goal"][1],
        "recorded_dijkstra_cost": case["recorded_dijkstra_cost"],
        "recorded_manhattan_expanded": case["recorded_manhattan_expanded"],
        "recorded_mlp_expanded": case["recorded_mlp_expanded"],
        "recorded_unet_expanded": case["recorded_unet_expanded"],
        "recorded_unet_cost": case["recorded_unet_cost"],
        "rerun_dijkstra_cost": results["dijkstra"]["cost"],
        "rerun_manhattan_cost": results["manhattan"]["cost"],
        "rerun_mlp_cost": results["mlp_table"]["cost"],
        "rerun_unet_cost": results["unet"]["cost"],
        "rerun_dijkstra_expanded": results["dijkstra"]["expanded"],
        "rerun_manhattan_expanded": results["manhattan"]["expanded"],
        "rerun_mlp_expanded": results["mlp_table"]["expanded"],
        "rerun_unet_expanded": results["unet"]["expanded"],
        "rerun_unet_minus_mlp_expanded": results["unet"]["expanded"] - results["mlp_table"]["expanded"],
        "rerun_unet_minus_manhattan_expanded": results["unet"]["expanded"] - results["manhattan"]["expanded"],
        "rerun_unet_cost_gap": results["unet"]["cost"] - results["dijkstra"]["cost"],
        "unet_mae": rerun["unet_metrics"]["mae"],
        "unet_mse": rerun["unet_metrics"]["mse"],
        "unet_overestimate_rate": rerun["unet_metrics"]["overestimate_rate"],
        "unet_underestimate_rate": rerun["unet_metrics"]["underestimate_rate"],
        "local_consistency_violation_rate": rerun["consistency_metrics"]["local_consistency_violation_rate"],
        "reproduction_notes": "; ".join(rerun["reproduction_notes"]),
        "output_dir": relative_dir,
    }
    return row


def write_summary(path, selected_rows, output_root):
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Representative Structured Case Studies\n\n")
        file.write(f"Output root: `{output_root}`\n\n")
        exact = [row for row in selected_rows if not row["reproduction_notes"]]
        file.write(f"Selected cases: {len(selected_rows)}\n")
        file.write(f"Exact reproduction cases: {len(exact)}\n")
        if len(exact) != len(selected_rows):
            file.write("Some cases could not be exactly reproduced; see `reproduction_notes` in `selected_cases.csv`.\n")
        else:
            file.write("All selected cases were regenerated using recorded seed, structured_type, start, and goal.\n")

        for group in GROUPS:
            rows = [row for row in selected_rows if row["group"] == group]
            file.write(f"\n## {group}\n\n")
            for row in rows:
                file.write(
                    f"- `{row['case_id']}`: "
                    f"Dijkstra cost {row['rerun_dijkstra_cost']}, "
                    f"MLP expanded {row['rerun_mlp_expanded']}, "
                    f"U-Net expanded {row['rerun_unet_expanded']}, "
                    f"U-Net-MLP expanded {row['rerun_unet_minus_mlp_expanded']}, "
                    f"U-Net cost gap {row['rerun_unet_cost_gap']}, "
                    f"overestimate {float(row['unet_overestimate_rate']):.3f}, "
                    f"consistency violations {float(row['local_consistency_violation_rate']):.3f}\n"
                )


def analyze_representative_cases(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_root = os.path.join(project_root, "outputs", "case_studies", args.output_tag)
    os.makedirs(output_root, exist_ok=True)
    plt, np = ensure_matplotlib(project_root)

    rows = read_csv(args.results)
    grouped = group_results(rows)
    selected = select_cases(grouped, args.top_k)
    mlp_model, unet_model = load_models(project_root, args.checkpoint)

    selected_rows = []
    for case in selected:
        group_dir = os.path.join(output_root, case["group"])
        case_dir = os.path.join(group_dir, case["case_id"])
        rerun = rerun_case(case, mlp_model, unet_model)
        save_case_outputs(case, rerun, case_dir, plt, np)
        relative_dir = os.path.relpath(case_dir, output_root)
        selected_rows.append(selected_case_row(case, rerun, relative_dir))

    selected_path = os.path.join(output_root, "selected_cases.csv")
    summary_path = os.path.join(output_root, "summary.md")
    write_csv(selected_path, selected_rows, list(selected_rows[0].keys()) if selected_rows else [])
    write_summary(summary_path, selected_rows, output_root)

    print(f"Saved selected cases to {selected_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved case visualizations under {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate representative controlled-structured case studies.")
    parser.add_argument("--results", required=True, help="Structured controlled experiment results CSV.")
    parser.add_argument("--checkpoint", default="best", help="compatible, best, latest, or checkpoint path.")
    parser.add_argument("--output-tag", default="structured_controlled_100")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    analyze_representative_cases(parse_args())
