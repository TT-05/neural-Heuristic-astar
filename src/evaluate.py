import os
import time
from collections import deque

import torch

from astar import astar_search
from bfs_label import compute_distance_to_goal
from gen_map import gen_map
from model import (
    distance_normalizer_for_grid,
    grid_goal_tensor,
    load_mlp_heuristic,
    load_unet_heuristic,
    make_mlp_heuristic,
    make_unet_heuristic,
    manhattan_heuristic,
)


def shortest_path_length_bfs(grid, start, goal):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    queue = deque([(start, 0)])
    visited = {start}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        current, distance = queue.popleft()
        if current == goal:
            return distance

        for dr, dc in directions:
            nr = current[0] + dr
            nc = current[1] + dc
            neighbor = (nr, nc)

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if grid[nr][nc] == 1 or neighbor in visited:
                continue

            visited.add(neighbor)
            queue.append((neighbor, distance + 1))

    return -1


def run_astar_case(name, grid, start, goal, heuristic, optimal_cost, diagnostics=None):
    start_time = time.perf_counter()
    result = astar_search(grid, start, goal, heuristic, diagnostics=diagnostics)
    runtime = time.perf_counter() - start_time

    path = result["path"]
    cost = result["cost"]
    path_found = bool(path)
    optimal = path_found and cost == optimal_cost

    print(f"{name}:")
    print(f"  path_found: {path_found}")
    print(f"  path_length: {cost}")
    print(f"  expanded_nodes: {result['expanded']}")
    print(f"  runtime_seconds: {runtime:.6f}")
    print(f"  optimal_vs_bfs: {optimal}")

    if not path_found:
        print("  warning: A* did not find a path.")
    elif not optimal:
        print(f"  warning: path length differs from BFS optimal cost {optimal_cost}.")

    return result


def format_map(values, mask=None, decimals=1):
    lines = []
    for r, row in enumerate(values):
        cells = []
        for c, value in enumerate(row):
            if mask is not None and not mask[r][c]:
                cells.append("  ##  ")
            else:
                cells.append(f"{value:6.{decimals}f}")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def save_text(path, content):
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def valid_cells_from_distance_grid(distance_grid):
    return [[value >= 0 for value in row] for row in distance_grid]


def run_unet_prediction(model, grid, goal):
    model_input = grid_goal_tensor(grid, goal).unsqueeze(0)
    normalizer = distance_normalizer_for_grid(grid)
    with torch.no_grad():
        return (model(model_input).squeeze(0).cpu() * normalizer).tolist()


def compute_unet_error_metrics(true_grid, predicted_grid, valid_mask):
    errors = []
    true_values = []
    predicted_values = []

    for r, row in enumerate(true_grid):
        for c, true_value in enumerate(row):
            if not valid_mask[r][c]:
                continue
            predicted_value = predicted_grid[r][c]
            errors.append(predicted_value - true_value)
            true_values.append(true_value)
            predicted_values.append(predicted_value)

    if not errors:
        raise ValueError("No valid cells found for U-Net debug metrics.")

    abs_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    underestimates = [error for error in errors if error < 0]
    large_underestimates = [error for error in errors if error < -3]

    return {
        "valid_cells": len(errors),
        "mean_error": sum(errors) / len(errors),
        "mae": sum(abs_errors) / len(abs_errors),
        "mse": sum(squared_errors) / len(squared_errors),
        "pct_pred_less_than_true": 100.0 * len(underestimates) / len(errors),
        "pct_pred_less_than_true_minus_3": 100.0 * len(large_underestimates) / len(errors),
        "min_prediction": min(predicted_values),
        "max_prediction": max(predicted_values),
        "min_true_distance": min(true_values),
        "max_true_distance": max(true_values),
    }


def save_heatmap(path, values, title, valid_mask=None):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f"Skipping heatmap {path}: matplotlib is not installed.")
        return False

    heatmap = []
    for r, row in enumerate(values):
        heatmap_row = []
        for c, value in enumerate(row):
            if valid_mask is not None and not valid_mask[r][c]:
                heatmap_row.append(float("nan"))
            else:
                heatmap_row.append(value)
        heatmap.append(heatmap_row)

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return True


def print_unet_astar_diagnostics(result, heuristic, goal, diagnostics):
    print("U-Net final path h values:")
    for index, node in enumerate(result["path"]):
        h_value = heuristic(node, goal)
        print(f"  step={index:02d} node={node} h={h_value:.3f}")

    print("Sample expanded nodes with g, h, f:")
    for item in diagnostics.get("expanded_nodes", []):
        node = item["node"]
        print(f"  node={node} g={item['g']:.3f} h={item['h']:.3f} f={item['f']:.3f}")


def debug_unet_prediction(unet_model, grid, goal, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    true_grid = compute_distance_to_goal(grid, goal)
    predicted_grid = run_unet_prediction(unet_model, grid, goal)
    valid_mask = valid_cells_from_distance_grid(true_grid)
    error_grid = []

    for r, row in enumerate(true_grid):
        error_row = []
        for c, true_value in enumerate(row):
            if valid_mask[r][c]:
                error_row.append(predicted_grid[r][c] - true_value)
            else:
                error_row.append(0.0)
        error_grid.append(error_row)

    metrics = compute_unet_error_metrics(true_grid, predicted_grid, valid_mask)

    print("U-Net heuristic map debug metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    save_text(
        os.path.join(output_dir, "true_distance_map.txt"),
        format_map(true_grid, valid_mask, decimals=0) + "\n",
    )
    save_text(
        os.path.join(output_dir, "predicted_distance_map.txt"),
        format_map(predicted_grid, valid_mask, decimals=2) + "\n",
    )
    save_text(
        os.path.join(output_dir, "error_map_pred_minus_true.txt"),
        format_map(error_grid, valid_mask, decimals=2) + "\n",
    )
    save_text(
        os.path.join(output_dir, "metrics.txt"),
        "\n".join(f"{key}: {value}" for key, value in metrics.items()) + "\n",
    )

    save_heatmap(os.path.join(output_dir, "true_distance_heatmap.png"), true_grid, "BFS true distance", valid_mask)
    save_heatmap(
        os.path.join(output_dir, "predicted_distance_heatmap.png"),
        predicted_grid,
        "U-Net predicted distance",
        valid_mask,
    )
    save_heatmap(
        os.path.join(output_dir, "error_heatmap_pred_minus_true.png"),
        error_grid,
        "U-Net error: predicted - true",
        valid_mask,
    )

    print(f"Saved U-Net debug outputs to {output_dir}")
    return predicted_grid, metrics


def make_hybrid_heuristic(unet_heuristic):
    def heuristic(current, goal):
        return max(manhattan_heuristic(current, goal), unet_heuristic(current, goal))

    return heuristic


def print_unet_improvement_notes(metrics):
    print("U-Net improvement notes:")
    if metrics["mean_error"] < 0 or metrics["pct_pred_less_than_true_minus_3"] > 20.0:
        print("  The current U-Net appears to underestimate many valid cells.")
    else:
        print("  The current U-Net is not strongly biased toward underestimation on this map.")
    print("  Suggested next steps:")
    print("  - Train longer.")
    print("  - Increase the training dataset size.")
    print("  - Normalize distance labels during training and scale predictions back during inference.")
    print("  - Try L1Loss or SmoothL1Loss instead of only MSE.")
    print("  - Keep clamping U-Net predictions to non-negative values during inference.")
    print("  - Compare raw U-Net with hybrid h = max(Manhattan, U-Net prediction).")
    print("  - Add validation loss tracking to detect undertraining or overfitting.")


def evaluate():
    grid = gen_map(width=20, height=20, seed=7, obstacle_rate=0.2)
    start = (0, 0)
    goal = (19, 19)

    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0

    optimal_cost = shortest_path_length_bfs(grid, start, goal)
    if optimal_cost == -1:
        raise RuntimeError("Evaluation map has no path from start to goal. Change the seed.")

    print(f"BFS optimal path length: {optimal_cost}")
    manhattan_result = run_astar_case(
        "A* with Manhattan heuristic",
        grid,
        start,
        goal,
        manhattan_heuristic,
        optimal_cost,
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    mlp_checkpoint_path = os.path.join(project_root, "checkpoints", "mlp_heuristic.pt")
    if not os.path.exists(mlp_checkpoint_path):
        raise FileNotFoundError(
            f"Missing checkpoint: {mlp_checkpoint_path}. Run `python3 train.py` from src/ first."
        )

    mlp_model = load_mlp_heuristic(mlp_checkpoint_path)
    mlp_heuristic = make_mlp_heuristic(mlp_model)
    mlp_result = run_astar_case(
        "A* with learned MLP heuristic",
        grid,
        start,
        goal,
        mlp_heuristic,
        optimal_cost,
    )

    unet_result = None
    hybrid_result = None
    unet_checkpoint_path = os.path.join(project_root, "checkpoints", "unet_heuristic.pt")
    if os.path.exists(unet_checkpoint_path):
        unet_model = load_unet_heuristic(unet_checkpoint_path)
        output_dir = os.path.join(project_root, "outputs", "unet_debug")
        predicted_grid, metrics = debug_unet_prediction(unet_model, grid, goal, output_dir)

        unet_heuristic = make_unet_heuristic(unet_model, grid, goal)
        unet_diagnostics = {"max_expanded_logs": 30}
        unet_result = run_astar_case(
            "A* with learned U-Net heuristic",
            grid,
            start,
            goal,
            unet_heuristic,
            optimal_cost,
            diagnostics=unet_diagnostics,
        )
        print_unet_astar_diagnostics(unet_result, unet_heuristic, goal, unet_diagnostics)

        hybrid_heuristic = make_hybrid_heuristic(unet_heuristic)
        hybrid_result = run_astar_case(
            "A* with hybrid max(Manhattan, U-Net) heuristic",
            grid,
            start,
            goal,
            hybrid_heuristic,
            optimal_cost,
        )
        print_unet_improvement_notes(metrics)
    else:
        print(f"Skipping U-Net heuristic: missing checkpoint {unet_checkpoint_path}")
        print("Run `python3 train_unet.py` from src/ to train it.")

    print("Comparison:")
    print(f"  manhattan_cost: {manhattan_result['cost']}")
    print(f"  mlp_cost: {mlp_result['cost']}")
    print(f"  mlp_matches_manhattan_cost: {mlp_result['cost'] == manhattan_result['cost']}")
    if unet_result is not None:
        print(f"  unet_cost: {unet_result['cost']}")
        print(f"  unet_matches_manhattan_cost: {unet_result['cost'] == manhattan_result['cost']}")
    if hybrid_result is not None:
        print(f"  hybrid_cost: {hybrid_result['cost']}")
        print(f"  hybrid_matches_manhattan_cost: {hybrid_result['cost'] == manhattan_result['cost']}")


if __name__ == "__main__":
    evaluate()
