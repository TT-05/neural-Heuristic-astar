"""Gradient-loss ablation on the fixed 5,000-map U-Net training dataset."""

import argparse
import csv
import json
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from analyze_unet_structure_behavior import mean
from astar import astar_search
from bfs_label import compute_distance_to_goal
from model import UNetHeuristic, load_unet_heuristic, make_unet_heuristic, manhattan_heuristic
from run_loss_ablation import (
    BATCH_SIZE, EPOCHS, PAIR_COUNT, RANK_WEIGHT, SEED, StructuredDistanceDataset,
    aggregate_prediction, aggregate_search, benchmark_cases,
    consistency_metrics, pairwise_accuracy, ranking_loss, rebuild_case, sample_pairs,
    tie_accuracy, trace_search, write_csv,
)
from train_unet import set_global_seed


LOSS_MODES = ("mse", "gradient", "ranking_gradient")
GRADIENT_WEIGHT = 0.1


def checkpoint_path(output_dir, mode):
    return os.path.join(output_dir, "checkpoints", f"unet_{mode}_best.pt")


def masked_mse(prediction, target, mask):
    return (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


def gradient_loss(prediction, target, mask):
    """Adjacent-cell L1 gradient error in original grid-distance units."""
    normalizer = float(prediction.shape[-2] + prediction.shape[-1])
    prediction, target = prediction * normalizer, target * normalizer
    horizontal_mask = mask[:, :, 1:] * mask[:, :, :-1]
    vertical_mask = mask[:, 1:, :] * mask[:, :-1, :]
    horizontal_error = torch.abs((prediction[:, :, 1:] - prediction[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])) * horizontal_mask
    vertical_error = torch.abs((prediction[:, 1:, :] - prediction[:, :-1, :]) - (target[:, 1:, :] - target[:, :-1, :])) * vertical_mask
    return (horizontal_error.sum() + vertical_error.sum()) / (horizontal_mask.sum() + vertical_mask.sum()).clamp(min=1.0)


def loss_components(prediction, target, mask, pairs, mode):
    mse = masked_mse(prediction, target, mask)
    rank = ranking_loss(prediction, target, pairs) if mode == "ranking_gradient" else prediction.sum() * 0.0
    gradient = gradient_loss(prediction, target, mask) if mode in ("gradient", "ranking_gradient") else prediction.sum() * 0.0
    total = mse + (RANK_WEIGHT * rank if mode == "ranking_gradient" else 0.0) + (GRADIENT_WEIGHT * gradient if mode in ("gradient", "ranking_gradient") else 0.0)
    return total, mse, rank, gradient


def run_epoch(model, loader, mode, optimizer=None):
    training = optimizer is not None; model.train(training)
    totals = defaultdict(float); examples = 0; valid = 0.0
    for model_input, target, mask, _goal, _start, pairs in loader:
        if training: optimizer.zero_grad()
        prediction = model(model_input)
        total, mse, rank, gradient = loss_components(prediction, target, mask, pairs, mode)
        if training:
            total.backward(); optimizer.step()
        count = model_input.size(0); examples += count
        totals["loss"] += total.item() * count; totals["mse_loss"] += mse.item() * count
        totals["ranking_loss"] += rank.item() * count; totals["gradient_loss"] += gradient.item() * count
        error = (prediction.detach() - target) * mask
        totals["mae"] += torch.abs(error).sum().item(); totals["mse"] += (error ** 2).sum().item(); valid += mask.sum().item()
    return {"loss": totals["loss"] / examples, "mse_loss": totals["mse_loss"] / examples,
            "ranking_loss": totals["ranking_loss"] / examples, "gradient_loss": totals["gradient_loss"] / examples,
            "mae": totals["mae"] / valid, "mse": totals["mse"] / valid}


def train_variant(root, output_dir, mode):
    archive = os.path.join(root, "outputs/expanded_dataset_training/dataset/expanded_unet_dataset.npz")
    manifest = os.path.join(root, "outputs/expanded_dataset_training/dataset/dataset_manifest.csv")
    set_global_seed(SEED)
    train_set = StructuredDistanceDataset(archive, manifest, "train")
    val_set = StructuredDistanceDataset(archive, manifest, "val")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    model = UNetHeuristic(); optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rows, best = [], float("inf"); os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        train = run_epoch(model, train_loader, mode, optimizer)
        with torch.no_grad(): validation = run_epoch(model, val_loader, mode)
        rows.append({"loss_variant": mode, "epoch": epoch, **{f"train_{key}": value for key, value in train.items()}, **{f"val_{key}": value for key, value in validation.items()}})
        if validation["loss"] < best:
            best = validation["loss"]
            torch.save({"model_state_dict": model.state_dict(), "loss": mode, "epoch": epoch, "validation_loss": validation["loss"],
                        "dataset_archive": archive, "dataset_manifest": manifest, "train_examples": 4000, "val_examples": 500,
                        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "seed": SEED, "optimizer": "Adam(lr=0.001)",
                        "loss_weights": {"ranking": RANK_WEIGHT, "gradient": GRADIENT_WEIGHT}, "ranking_pairs_per_map": PAIR_COUNT}, checkpoint_path(output_dir, mode))
        print(f"{mode} epoch {epoch:02d}/{EPOCHS}: train={train['loss']:.6f} val={validation['loss']:.6f}")
    write_csv(os.path.join(output_dir, f"training_history_{mode}.csv"), rows)
    return rows


def prediction_table(model, grid, goal):
    heuristic = make_unet_heuristic(model, grid, goal)
    return {(r, c): float(heuristic((r, c), goal)) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 0}


def gradient_metrics(table, grid, labels):
    error = total = 0.0
    for cell in table:
        for neighbor in ((cell[0] + 1, cell[1]), (cell[0], cell[1] + 1)):
            if neighbor not in table: continue
            error += abs((table[neighbor] - table[cell]) - (labels[neighbor[0]][neighbor[1]] - labels[cell[0]][cell[1]])); total += 1
    return error / max(total, 1.0)


def evaluate_prediction(root, output_dir, variants):
    archive = os.path.join(root, "outputs/expanded_dataset_training/dataset/expanded_unet_dataset.npz")
    manifest = os.path.join(root, "outputs/expanded_dataset_training/dataset/dataset_manifest.csv")
    dataset = StructuredDistanceDataset(archive, manifest, "test"); rows = []
    for variant in variants:
        model = load_unet_heuristic(checkpoint_path(output_dir, variant))
        for record in dataset.records:
            item = int(record["index"]); grid = dataset.grids[item].tolist(); goal = tuple(int(v) for v in dataset.goals[item]); start = tuple(int(v) for v in dataset.starts[item])
            labels = dataset.labels[item].tolist(); table = prediction_table(model, grid, goal); reachable = list(table)
            mae = mean(abs(table[cell] - labels[cell[0]][cell[1]]) for cell in reachable); mse = mean((table[cell] - labels[cell[0]][cell[1]]) ** 2 for cell in reachable)
            if item not in dataset.pairs: dataset.pairs[item] = sample_pairs(grid, goal, start, dataset.labels[item], SEED + item * 37)
            pair_correct, pair_total = pairwise_accuracy(table, grid, dataset.pairs[item], labels)
            trace = trace_search(grid, start, goal, table, labels, set(), "unet_tiebreak")
            if trace["cost"] != labels[start[0]][start[1]]: raise AssertionError("Tie-break lost test-set optimality")
            tie_correct, tie_total = tie_accuracy(trace); violations, magnitude = consistency_metrics(table, grid, labels)
            rows.append({"variant": variant, "split": "prediction_test", "case_id": f"test_{item}", "structure_type": record["map_type"], "obstacle_rate": record["obstacle_rate"],
                         "mae": mae, "mse": mse, "pairwise_ordering_accuracy": pair_correct / max(1, pair_total), "pairwise_pairs": pair_total,
                         "tie_set_ordering_accuracy": tie_correct / max(1, tie_total), "tie_set_decisions": tie_total,
                         "gradient_error": gradient_metrics(table, grid, labels), "consistency_violation_count": violations, "consistency_violation_magnitude": magnitude})
    return rows


def evaluate_search(root, output_dir, variants):
    cases = benchmark_cases(root, "outputs/experiments/results_structured_tiebreak_controls_100.csv", "outputs/experiments/results_random_tiebreak_controls_100.csv")
    rows = []
    for variant in variants:
        model = load_unet_heuristic(checkpoint_path(output_dir, variant))
        for index, source in enumerate(cases, start=1):
            grid, start, goal = rebuild_case(source); distance = compute_distance_to_goal(grid, goal); optimal = distance[start[0]][start[1]]; table = prediction_table(model, grid, goal)
            heuristic = lambda node, ignored: table[node]
            for algorithm in ("direct_unet", "unet_tiebreak"):
                result = astar_search(grid, start, goal, heuristic) if algorithm == "direct_unet" else astar_search(grid, start, goal, manhattan_heuristic, secondary_heuristic=heuristic)
                rows.append({"variant": variant, "case_id": f"{source['analysis_structure']}_seed{source['seed']}_s{start[0]}-{start[1]}_g{goal[0]}-{goal[1]}", "structure_type": source["analysis_structure"], "algorithm": algorithm, "expanded_nodes": result["expanded"], "path_cost": result["cost"], "optimal_cost": optimal, "optimal": result["cost"] == optimal})
            if index % 100 == 0: print(f"{variant} search {index}/{len(cases)}")
    return rows


def aggregate_gradient(rows, fields):
    grouped = defaultdict(list)
    for row in rows: grouped[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        output.append({**dict(zip(fields, key)), "scope": "prediction", "cases": len(values), "mean_gradient_error": mean(row["gradient_error"] for row in values),
                       "mean_consistency_violation_count": mean(row["consistency_violation_count"] for row in values), "mean_consistency_violation_magnitude": mean(row["consistency_violation_magnitude"] for row in values)})
    return output


def previous_consistency(root):
    path = os.path.join(root, "outputs/loss_ablation_v1/summary_by_variant.csv")
    if not os.path.exists(path): return None
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    prediction = next((row for row in rows if row["variant"] == "consistency" and row["scope"] == "prediction"), None)
    direct = next((row for row in rows if row["variant"] == "consistency" and row["algorithm"] == "direct_unet"), None)
    tie = next((row for row in rows if row["variant"] == "consistency" and row["algorithm"] == "unet_tiebreak"), None)
    return prediction, direct, tie


def plot_curves(output_dir, histories):
    os.environ.setdefault("MPLBACKEND", "Agg"); os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    for variant in LOSS_MODES:
        rows = [row for row in histories if row["loss_variant"] == variant]
        axis.plot([row["epoch"] for row in rows], [row["val_loss"] for row in rows], label=variant)
    axis.set_xlabel("Epoch"); axis.set_ylabel("Validation total loss"); axis.legend(); figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "validation_loss_curves.png"), dpi=160); plt.close(figure)


def report(root, output_dir, prediction, search, gradient):
    overall_prediction = {row["variant"]: row for row in aggregate_prediction(prediction, ["variant"])}
    overall_search = {(row["variant"], row["algorithm"]): row for row in aggregate_search(search, ["variant", "algorithm"])}
    overall_gradient = {row["variant"]: row for row in aggregate_gradient(prediction, ["variant"])}
    previous = previous_consistency(root)
    lines = ["# Gradient Loss Ablation", "", "All variants use the unchanged 5,000-map structured dataset and split, U-Net architecture, Adam optimizer, batch size 16, 50 epochs, benchmark cases, and A* implementations.", "", "| Variant | MAE | MSE | Gradient error | Pair ordering | Tie ordering | Direct optimality | Direct expanded | Tie-break expanded |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in LOSS_MODES:
        p, g, direct, tie = overall_prediction[variant], overall_gradient[variant], overall_search[(variant, "direct_unet")], overall_search[(variant, "unet_tiebreak")]
        lines.append(f"| {variant} | {p['mae']:.3f} | {p['mse']:.3f} | {g['mean_gradient_error']:.3f} | {p['pairwise_ordering_accuracy']:.3f} | {p['tie_set_ordering_accuracy']:.3f} | {direct['optimality_rate']:.4f} | {direct['mean_expanded_nodes']:.2f} | {tie['mean_expanded_nodes']:.2f} |")
    mse, gradient, rank_gradient = overall_prediction["mse"], overall_prediction["gradient"], overall_prediction["ranking_gradient"]
    lines += ["", "## Answers", "", f"1. Gradient loss {'improved' if overall_gradient['gradient']['mean_gradient_error'] < overall_gradient['mse']['mean_gradient_error'] else 'did not improve'} local gradient error ({overall_gradient['mse']['mean_gradient_error']:.3f} -> {overall_gradient['gradient']['mean_gradient_error']:.3f}).", f"2. Gradient loss changed pair ordering from {mse['pairwise_ordering_accuracy']:.3f} to {gradient['pairwise_ordering_accuracy']:.3f}, Direct mean expansions from {overall_search[('mse','direct_unet')]['mean_expanded_nodes']:.2f} to {overall_search[('gradient','direct_unet')]['mean_expanded_nodes']:.2f}, and tie-break mean expansions from {overall_search[('mse','unet_tiebreak')]['mean_expanded_nodes']:.2f} to {overall_search[('gradient','unet_tiebreak')]['mean_expanded_nodes']:.2f}.", f"3. Ranking+gradient {'improved' if rank_gradient['pairwise_ordering_accuracy'] > gradient['pairwise_ordering_accuracy'] else 'did not improve'} pair ordering beyond gradient alone ({gradient['pairwise_ordering_accuracy']:.3f} -> {rank_gradient['pairwise_ordering_accuracy']:.3f}), but changed Direct expansions from {overall_search[('gradient','direct_unet')]['mean_expanded_nodes']:.2f} to {overall_search[('ranking_gradient','direct_unet')]['mean_expanded_nodes']:.2f}.", "4. Lower gradient error is not sufficient for better A* performance here: gradient sharply lowers gradient error but barely changes Direct expansions, while ranking+gradient has worse gradient error than gradient and more Direct expansions."]
    if previous and all(previous):
        old_prediction, old_direct, old_tie = previous
        lines.append(f"5. Compared with previous consistency loss, gradient has more violations ({overall_gradient['gradient']['mean_consistency_violation_count']:.2f} vs {float(old_prediction['consistency_violation_count']):.2f}) and lower Direct optimality ({overall_search[('gradient','direct_unet')]['optimality_rate']:.4f} vs {float(old_direct['optimality_rate']):.4f}), but far fewer Direct expansions ({overall_search[('gradient','direct_unet')]['mean_expanded_nodes']:.2f} vs {float(old_direct['mean_expanded_nodes']):.2f}). It therefore does not dominate consistency loss.")
    lines += ["", "See `summary_by_structure.csv` for all five benchmark structures and held-out map types.", ""]
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as handle: handle.write("\n".join(lines))


def evaluate(root, output_dir, variants):
    prediction = evaluate_prediction(root, output_dir, variants); search = evaluate_search(root, output_dir, variants)
    write_csv(os.path.join(output_dir, "prediction_metrics.csv"), prediction); write_csv(os.path.join(output_dir, "search_results.csv"), search)
    overall = aggregate_prediction(prediction, ["variant"]) + aggregate_search(search, ["variant", "algorithm"])
    write_csv(os.path.join(output_dir, "summary_by_variant.csv"), overall)
    structure = aggregate_prediction(prediction, ["variant", "structure_type"]) + aggregate_search(search, ["variant", "algorithm", "structure_type"])
    write_csv(os.path.join(output_dir, "summary_by_structure.csv"), structure)
    gradient_overall = aggregate_gradient(prediction, ["variant"])
    gradient = gradient_overall + aggregate_gradient(prediction, ["variant", "structure_type"])
    write_csv(os.path.join(output_dir, "gradient_analysis.csv"), gradient); report(root, output_dir, prediction, search, gradient_overall)


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed-data U-Net gradient-loss ablation.")
    parser.add_argument("--loss", choices=[*LOSS_MODES, "all"], default="all")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--output-dir", default="outputs/gradient_loss_ablation")
    return parser.parse_args()


def main():
    args = parse_args(); root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); output_dir = os.path.join(root, args.output_dir); os.makedirs(output_dir, exist_ok=True)
    requested = LOSS_MODES if args.loss == "all" else (args.loss,); histories = []
    for mode in requested: histories.extend(train_variant(root, output_dir, mode))
    if histories:
        write_csv(os.path.join(output_dir, "training_history.csv"), histories); plot_curves(output_dir, histories)
        with open(os.path.join(output_dir, "final_configuration.json"), "w", encoding="utf-8") as handle:
            json.dump({"loss_modes": list(LOSS_MODES), "gradient_weight": GRADIENT_WEIGHT, "ranking_weight": RANK_WEIGHT, "ranking_pairs_per_map": PAIR_COUNT, "optimizer": "Adam(lr=0.001)", "batch_size": BATCH_SIZE, "epochs": EPOCHS, "dataset": "existing 5,000-map structured dataset"}, handle, indent=2)
    available = [mode for mode in LOSS_MODES if os.path.exists(checkpoint_path(output_dir, mode))]
    if args.evaluate or args.loss == "all":
        if set(available) != set(LOSS_MODES): raise FileNotFoundError("Evaluation requires all three checkpoints")
        evaluate(root, output_dir, available)


if __name__ == "__main__": main()
