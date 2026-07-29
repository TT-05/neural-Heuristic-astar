import argparse
import os

from analyze_single_step_penalty import single_step_row
from analyze_tie_set_counterfactual_penalty import collect_tie_events, result_expanded
from analyze_tie_set_ordering import build_grid, checkpoint_path, group_maps, prediction_table, read_csv
from analyze_tie_set_weighted_ordering import reconstruct_optimal_path, route_critical_cells
from bfs_label import compute_distance_to_goal
from generate_bottleneck_case_studies import (
    COLORS,
    ensure_plot_backend,
    node_text,
    save_expansion_png,
    save_map_png,
    save_three_method_overlay_png,
    simulate_manhattan_expansion,
    simulate_secondary_expansion,
    write_csv,
    write_text,
)
from model import load_mlp_heuristic, load_unet_heuristic, make_mlp_table_heuristic, make_unet_heuristic


def case_identity(sample):
    return (
        f"maze_like_rate{sample['obstacle_rate']}_seed{sample['seed']}"
        f"_s{sample['start_row']}-{sample['start_col']}_g{sample['goal_row']}-{sample['goal_col']}"
    )


def select_cases(groups, per_side):
    candidates = []
    for methods in groups:
        sample = methods["manhattan"]
        if sample.get("structured_type") != "maze_like":
            continue
        if "manhattan_mlp_tiebreak" not in methods or "manhattan_unet_tiebreak" not in methods:
            continue
        manhattan_expanded = result_expanded(methods, "manhattan")
        mlp_expanded = result_expanded(methods, "manhattan_mlp_tiebreak")
        unet_expanded = result_expanded(methods, "manhattan_unet_tiebreak")
        candidates.append(
            {
                "methods": methods,
                "map_id": case_identity(sample),
                "manhattan_expanded": manhattan_expanded,
                "mlp_expanded": mlp_expanded,
                "unet_expanded": unet_expanded,
                "expanded_gap": unet_expanded - mlp_expanded,
            }
        )
    ordered = sorted(candidates, key=lambda row: row["expanded_gap"])
    return ordered[:per_side] + ordered[-per_side:]


def disagreement_rows(events, grid, goal, true_table, identity):
    rows = []
    for event in events:
        if event["mlp_selected_node"] == event["unet_selected_node"]:
            continue
        row = single_step_row(identity, event, grid, goal, true_table)
        rows.append(
            {
                "expanded_step": event["expanded_step"],
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
    return sorted(rows, key=lambda item: item["expanded_step"])


def trajectory_summary(case, disagreement):
    first_disagreement = min((row["expanded_step"] for row in disagreement), default="")
    return (
        f"map_id: {case['map_id']}\n"
        f"manhattan_expanded: {case['manhattan_expanded']}\n"
        f"mlp_expanded: {case['mlp_expanded']}\n"
        f"unet_expanded: {case['unet_expanded']}\n"
        f"expanded_gap: {case['expanded_gap']}\n"
        f"disagreement_count: {len(disagreement)}\n"
        f"first_disagreement_step: {first_disagreement}\n"
    )


def generate_case(case_idx, case, output_dir, plt, Patch, mlp_model, unet_model):
    methods = case["methods"]
    sample = methods["manhattan"]
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
        "structure_type": "maze_like",
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
    save_map_png(os.path.join(case_dir, "map.png"), plt, grid, start, goal, optimal_path)
    save_expansion_png(os.path.join(case_dir, "manhattan_expansion_order.png"), plt, grid, start, goal, optimal_path, manhattan_expanded, "Manhattan expansion order")
    save_expansion_png(os.path.join(case_dir, "mlp_expansion_order.png"), plt, grid, start, goal, optimal_path, mlp_expanded, "MLP tie-break expansion order")
    save_expansion_png(os.path.join(case_dir, "unet_expansion_order.png"), plt, grid, start, goal, optimal_path, unet_expanded, "U-Net tie-break expansion order")
    save_three_method_overlay_png(os.path.join(case_dir, "three_method_overlay.png"), plt, Patch, grid, start, goal, optimal_path, manhattan_set, mlp_set, unet_set)
    write_csv(os.path.join(case_dir, "disagreement_tie_sets.csv"), disagreement)
    write_text(os.path.join(case_dir, "trajectory_summary.txt"), trajectory_summary(case, disagreement))

    return {
        "case_id": case_id,
        "map_id": case["map_id"],
        "manhattan_expanded": case["manhattan_expanded"],
        "mlp_expanded": case["mlp_expanded"],
        "unet_expanded": case["unet_expanded"],
        "expanded_gap": case["expanded_gap"],
        "disagreement_count": len(disagreement),
        "first_disagreement_step": min((row["expanded_step"] for row in disagreement), default=""),
    }


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "maze_case_studies")
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
    print(f"Saved maze case-study packages to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate visual maze-like case-study packages.")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--cases-per-side", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
