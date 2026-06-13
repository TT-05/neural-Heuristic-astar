import argparse
import json
import os

from bfs_label import compute_distance_to_goal
from evaluate import debug_unet_prediction, save_text
from gen_map import gen_map
from model import load_unet_heuristic


CHECKPOINT_CHOICES = {
    "compatible": "unet_heuristic.pt",
    "best": "unet_heuristic_best.pt",
    "latest": "unet_heuristic_latest.pt",
}


def checkpoint_path(project_root, checkpoint):
    if checkpoint in CHECKPOINT_CHOICES:
        return os.path.join(project_root, "checkpoints", CHECKPOINT_CHOICES[checkpoint])
    return checkpoint


def load_case(case_dir):
    metadata_path = os.path.join(case_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return {
        "seed": int(metadata["seed"]),
        "map_size": int(metadata["map_size"]),
        "obstacle_rate": float(metadata["obstacle_rate"]),
        "start": tuple(metadata["start"]),
        "goal": tuple(metadata["goal"]),
        "category": metadata.get("category", os.path.basename(case_dir)),
    }


def build_grid(map_size, seed, obstacle_rate, start, goal):
    grid = gen_map(width=map_size, height=map_size, seed=seed, obstacle_rate=obstacle_rate)
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0
    return grid


def local_consistency_metrics(predicted_grid, valid_mask):
    total_edges = 0
    violations = 0
    magnitudes = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rows = len(predicted_grid)
    cols = len(predicted_grid[0]) if rows else 0

    for r in range(rows):
        for c in range(cols):
            if not valid_mask[r][c]:
                continue
            current_h = predicted_grid[r][c]
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if not valid_mask[nr][nc]:
                    continue
                total_edges += 1
                violation = current_h - predicted_grid[nr][nc] - 1.0
                if violation > 0:
                    violations += 1
                    magnitudes.append(violation)

    return {
        "directed_edges": total_edges,
        "local_consistency_violation_rate": violations / total_edges if total_edges else 0.0,
        "local_consistency_mean_violation": sum(magnitudes) / len(magnitudes) if magnitudes else 0.0,
        "local_consistency_max_violation": max(magnitudes) if magnitudes else 0.0,
    }


def valid_mask_from_distance_grid(distance_grid):
    return [[value >= 0 for value in row] for row in distance_grid]


def write_combined_metrics(output_dir, metrics, consistency_metrics):
    combined = {**metrics, **consistency_metrics}
    save_text(
        os.path.join(output_dir, "metrics_with_consistency.txt"),
        "\n".join(f"{key}: {value}" for key, value in combined.items()) + "\n",
    )


def analyze_case(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.case_dir:
        case = load_case(args.case_dir)
        seed = case["seed"]
        map_size = case["map_size"]
        obstacle_rate = case["obstacle_rate"]
        start = case["start"]
        goal = case["goal"]
        default_name = f"{case['category']}_rate{obstacle_rate}_seed{seed}"
    else:
        seed = args.seed
        map_size = args.map_size
        obstacle_rate = args.obstacle_rate
        start = tuple(args.start)
        goal = tuple(args.goal) if args.goal else (map_size - 1, map_size - 1)
        default_name = f"rate{obstacle_rate}_seed{seed}_size{map_size}"

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(project_root, "outputs", "heuristic_error", default_name)

    grid = build_grid(map_size, seed, obstacle_rate, start, goal)
    distance_grid = compute_distance_to_goal(grid, goal)
    if distance_grid[start[0]][start[1]] == -1:
        raise RuntimeError("Selected map is not solvable from start to goal.")

    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint))
    predicted_grid, metrics = debug_unet_prediction(unet_model, grid, goal, output_dir)
    valid_mask = valid_mask_from_distance_grid(distance_grid)
    consistency_metrics = local_consistency_metrics(predicted_grid, valid_mask)
    write_combined_metrics(output_dir, metrics, consistency_metrics)
    print("Local consistency metrics:")
    for key, value in consistency_metrics.items():
        print(f"  {key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze U-Net heuristic error on a selected map.")
    parser.add_argument("--case-dir", default=None, help="Path to a saved failure-case directory.")
    parser.add_argument("--checkpoint", default="best", help="compatible, best, latest, or a checkpoint path.")
    parser.add_argument("--map-size", type=int, default=20)
    parser.add_argument("--obstacle-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start", type=int, nargs=2, default=(0, 0))
    parser.add_argument("--goal", type=int, nargs=2, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    analyze_case(parse_args())
