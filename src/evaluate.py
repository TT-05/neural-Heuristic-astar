import os
import time
from collections import deque

from astar import astar_search
from gen_map import gen_map
from model import (
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


def run_astar_case(name, grid, start, goal, heuristic, optimal_cost):
    start_time = time.perf_counter()
    result = astar_search(grid, start, goal, heuristic)
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
    unet_checkpoint_path = os.path.join(project_root, "checkpoints", "unet_heuristic.pt")
    if os.path.exists(unet_checkpoint_path):
        unet_model = load_unet_heuristic(unet_checkpoint_path)
        unet_heuristic = make_unet_heuristic(unet_model, grid, goal)
        unet_result = run_astar_case(
            "A* with learned U-Net heuristic",
            grid,
            start,
            goal,
            unet_heuristic,
            optimal_cost,
        )
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


if __name__ == "__main__":
    evaluate()
