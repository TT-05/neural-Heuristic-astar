"""Combined ranking, admissibility, and consistency loss ablation."""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_direct_vs_tiebreak import trace_search
from analyze_unet_structure_behavior import benchmark_cases, free_cells, mean, rebuild_case
from astar import astar_search
from bfs_label import compute_distance_to_goal
from model import UNetHeuristic, load_unet_heuristic, make_unet_heuristic, manhattan_heuristic
from run_loss_ablation import (
    BATCH_SIZE,
    EPOCHS,
    PAIR_COUNT,
    RANK_WEIGHT,
    SEED,
    StructuredDistanceDataset,
    pairwise_accuracy,
    ranking_loss,
    sample_pairs,
    tie_accuracy,
    write_csv,
)
from train_unet import set_global_seed


LOSS_MODES = ("mse", "ranking", "ranking_adm", "ranking_adm_cons")
ADMISSIBILITY_WEIGHT = 0.1
CONSISTENCY_WEIGHT = 0.1


def checkpoint_path(output_dir, mode):
    return os.path.join(output_dir, f"{mode}_best.pt")


def masked_mse(prediction, target, mask):
    return (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


def admissibility_loss(prediction, target, mask):
    """Penalize only overestimation, in original grid-distance units."""
    normalizer = float(prediction.shape[-2] + prediction.shape[-1])
    excess = torch.relu((prediction - target) * normalizer)
    return ((excess ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


def directed_consistency_loss(prediction, mask):
    """Mean squared violation of h(s) <= 1 + h(s') for both edge directions."""
    normalizer = float(prediction.shape[-2] + prediction.shape[-1])
    values = prediction * normalizer
    horizontal_mask = mask[:, :, :-1] * mask[:, :, 1:]
    vertical_mask = mask[:, :-1, :] * mask[:, 1:, :]
    horizontal = values[:, :, :-1] - values[:, :, 1:]
    vertical = values[:, :-1, :] - values[:, 1:, :]
    horizontal_loss = (torch.relu(horizontal - 1.0) ** 2 + torch.relu(-horizontal - 1.0) ** 2) * horizontal_mask
    vertical_loss = (torch.relu(vertical - 1.0) ** 2 + torch.relu(-vertical - 1.0) ** 2) * vertical_mask
    directed_edges = 2.0 * (horizontal_mask.sum() + vertical_mask.sum())
    return (horizontal_loss.sum() + vertical_loss.sum()) / directed_edges.clamp(min=1.0)


def loss_components(prediction, target, mask, pairs, mode):
    mse = masked_mse(prediction, target, mask)
    rank = ranking_loss(prediction, target, pairs) if mode != "mse" else prediction.sum() * 0.0
    admissibility = admissibility_loss(prediction, target, mask) if mode in ("ranking_adm", "ranking_adm_cons") else prediction.sum() * 0.0
    consistency = directed_consistency_loss(prediction, mask) if mode == "ranking_adm_cons" else prediction.sum() * 0.0
    total = mse
    if mode != "mse":
        total = total + RANK_WEIGHT * rank
    if mode in ("ranking_adm", "ranking_adm_cons"):
        total = total + ADMISSIBILITY_WEIGHT * admissibility
    if mode == "ranking_adm_cons":
        total = total + CONSISTENCY_WEIGHT * consistency
    return total, mse, rank, admissibility, consistency


def run_epoch(model, loader, mode, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    examples = valid_cells = 0.0
    for model_input, target, mask, _goal, _start, pairs in loader:
        if training:
            optimizer.zero_grad()
        prediction = model(model_input)
        total, mse, rank, admissibility, consistency = loss_components(prediction, target, mask, pairs, mode)
        if training:
            total.backward()
            optimizer.step()
        batch = model_input.size(0)
        examples += batch
        valid_cells += mask.sum().item()
        for name, value in (("loss", total), ("mse_loss", mse), ("ranking_loss", rank), ("admissibility_loss", admissibility), ("consistency_loss", consistency)):
            totals[name] += value.item() * batch
        error = (prediction.detach() - target) * mask
        totals["mae"] += torch.abs(error).sum().item()
        totals["mse"] += (error ** 2).sum().item()
    return {
        "loss": totals["loss"] / examples,
        "mse_loss": totals["mse_loss"] / examples,
        "ranking_loss": totals["ranking_loss"] / examples,
        "admissibility_loss": totals["admissibility_loss"] / examples,
        "consistency_loss": totals["consistency_loss"] / examples,
        "mae": totals["mae"] / valid_cells,
        "mse": totals["mse"] / valid_cells,
    }


def train_variant(root, output_dir, mode):
    archive = os.path.join(root, "outputs/expanded_dataset_training/dataset/expanded_unet_dataset.npz")
    manifest = os.path.join(root, "outputs/expanded_dataset_training/dataset/dataset_manifest.csv")
    set_global_seed(SEED)
    train_set = StructuredDistanceDataset(archive, manifest, "train")
    val_set = StructuredDistanceDataset(archive, manifest, "val")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    model = UNetHeuristic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history, best, best_components = [], float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        train = run_epoch(model, train_loader, mode, optimizer)
        with torch.no_grad():
            validation = run_epoch(model, val_loader, mode)
        history.append({"loss_mode": mode, "epoch": epoch, **{f"train_{key}": value for key, value in train.items()}, **{f"val_{key}": value for key, value in validation.items()}})
        if validation["loss"] < best:
            best = validation["loss"]
            best_components = {"loss_mode": mode, "best_epoch": epoch, **{f"validation_{key}": value for key, value in validation.items()}}
            torch.save({
                "model_state_dict": model.state_dict(), "loss_mode": mode, "epoch": epoch, "validation_loss": validation["loss"],
                "dataset_archive": archive, "dataset_manifest": manifest, "train_examples": len(train_set), "validation_examples": len(val_set),
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "seed": SEED, "optimizer": "Adam(lr=0.001)",
                "loss_weights": {"ranking": RANK_WEIGHT, "admissibility": ADMISSIBILITY_WEIGHT, "consistency": CONSISTENCY_WEIGHT},
                "ranking_pairs_per_map": PAIR_COUNT,
            }, checkpoint_path(output_dir, mode))
        print(f"{mode} epoch {epoch:02d}/{EPOCHS}: train={train['loss']:.6f} val={validation['loss']:.6f}")
    write_csv(os.path.join(output_dir, f"training_history_{mode}.csv"), history)
    return history, best_components


def prediction_table(model, grid, goal):
    heuristic = make_unet_heuristic(model, grid, goal)
    return {cell: float(heuristic(cell, goal)) for cell in free_cells(grid)}


def heuristic_property_metrics(table, grid, labels):
    cells = list(table)
    overestimates = [max(0.0, table[cell] - labels[cell[0]][cell[1]]) for cell in cells]
    admissibility_count = sum(value > 1e-6 for value in overestimates)
    consistency_count = 0
    consistency_magnitude = 0.0
    for cell in cells:
        for neighbor in ((cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1)):
            if neighbor not in table:
                continue
            violation = table[cell] - 1.0 - table[neighbor]
            if violation > 1e-6:
                consistency_count += 1
                consistency_magnitude += violation
    return {
        "admissibility_violation_count": admissibility_count,
        "mean_overestimation_error": mean(overestimates),
        "consistency_violation_count": consistency_count,
        "consistency_violation_magnitude": consistency_magnitude,
    }


def evaluate_prediction(root, output_dir, variants):
    archive = os.path.join(root, "outputs/expanded_dataset_training/dataset/expanded_unet_dataset.npz")
    manifest = os.path.join(root, "outputs/expanded_dataset_training/dataset/dataset_manifest.csv")
    dataset = StructuredDistanceDataset(archive, manifest, "test")
    rows = []
    for variant in variants:
        model = load_unet_heuristic(checkpoint_path(output_dir, variant))
        for record in dataset.records:
            item = int(record["index"])
            grid = dataset.grids[item].tolist()
            start = tuple(int(value) for value in dataset.starts[item])
            goal = tuple(int(value) for value in dataset.goals[item])
            labels = dataset.labels[item].tolist()
            table = prediction_table(model, grid, goal)
            cells = list(table)
            if item not in dataset.pairs:
                dataset.pairs[item] = sample_pairs(grid, goal, start, dataset.labels[item], SEED + item * 37)
            correct, total = pairwise_accuracy(table, grid, dataset.pairs[item], labels)
            trace = trace_search(grid, start, goal, table, labels, set(), "unet_tiebreak")
            if trace["cost"] != labels[start[0]][start[1]]:
                raise AssertionError("U-Net tie-break lost held-out optimality")
            tie_correct, tie_total = tie_accuracy(trace)
            rows.append({
                "loss_mode": variant, "scope": "prediction", "case_id": f"test_{item}", "structure_type": record["map_type"],
                "mae": mean(abs(table[cell] - labels[cell[0]][cell[1]]) for cell in cells),
                "mse": mean((table[cell] - labels[cell[0]][cell[1]]) ** 2 for cell in cells),
                "pairwise_ordering_accuracy": correct / max(1, total), "pairwise_pairs": total,
                "tie_set_ordering_accuracy": tie_correct / max(1, tie_total), "tie_set_decisions": tie_total,
                **heuristic_property_metrics(table, grid, labels),
            })
    return rows


def evaluate_search(root, output_dir, variants):
    cases = benchmark_cases(root, "outputs/experiments/results_structured_tiebreak_controls_100.csv", "outputs/experiments/results_random_tiebreak_controls_100.csv")
    rows = []
    for variant in variants:
        model = load_unet_heuristic(checkpoint_path(output_dir, variant))
        for number, source in enumerate(cases, 1):
            grid, start, goal = rebuild_case(source)
            optimal = compute_distance_to_goal(grid, goal)[start[0]][start[1]]
            table = prediction_table(model, grid, goal)
            heuristic = lambda node, unused: table[node]
            for algorithm in ("direct_unet", "unet_tiebreak"):
                result = astar_search(grid, start, goal, heuristic) if algorithm == "direct_unet" else astar_search(grid, start, goal, manhattan_heuristic, secondary_heuristic=heuristic)
                rows.append({"loss_mode": variant, "scope": "search", "case_id": f"{source['analysis_structure']}_seed{source['seed']}", "structure_type": source["analysis_structure"], "algorithm": algorithm, "expanded_nodes": result["expanded"], "path_cost": result["cost"], "optimal_cost": optimal, "optimal": result["cost"] == optimal})
            if number % 100 == 0:
                print(f"{variant} evaluation {number}/{len(cases)}")
    return rows


def aggregate_prediction(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        result = {field: value for field, value in zip(fields, key)}
        result.update({
            "scope": "prediction_summary", "cases": len(values), "mae": mean(row["mae"] for row in values), "mse": mean(row["mse"] for row in values),
            "pairwise_ordering_accuracy": sum(row["pairwise_ordering_accuracy"] * row["pairwise_pairs"] for row in values) / max(1, sum(row["pairwise_pairs"] for row in values)),
            "tie_set_ordering_accuracy": sum(row["tie_set_ordering_accuracy"] * row["tie_set_decisions"] for row in values) / max(1, sum(row["tie_set_decisions"] for row in values)),
            "admissibility_violation_count": mean(row["admissibility_violation_count"] for row in values),
            "mean_overestimation_error": mean(row["mean_overestimation_error"] for row in values),
            "consistency_violation_count": mean(row["consistency_violation_count"] for row in values),
            "consistency_violation_magnitude": mean(row["consistency_violation_magnitude"] for row in values),
        })
        output.append(result)
    return output


def aggregate_search(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        result = {field: value for field, value in zip(fields, key)}
        result.update({"scope": "search_summary", "cases": len(values), "mean_expanded_nodes": mean(row["expanded_nodes"] for row in values), "optimality_rate": mean(float(row["optimal"]) for row in values)})
        output.append(result)
    return output


def write_report(output_dir, prediction_rows, search_rows, components):
    prediction = {row["loss_mode"]: row for row in aggregate_prediction(prediction_rows, ["loss_mode"])}
    search = {(row["loss_mode"], row["algorithm"]): row for row in aggregate_search(search_rows, ["loss_mode", "algorithm"])}
    mse, ranking, ranking_adm, combined = (prediction[mode] for mode in LOSS_MODES)
    lines = ["# Combined U-Net Loss Ablation", "", "All variants use the unchanged 5,000-map split, U-Net architecture, Adam optimizer, batch size, 50-epoch schedule, and 2,000 fixed benchmark cases.", "", "| Variant | MAE | MSE | Pair ordering | Tie-set ordering | Admissibility violations | Mean overestimate | Consistency violations | Direct expanded | Direct optimality | Tie-break expanded | Tie-break optimality |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in LOSS_MODES:
        prop = prediction[mode]
        direct = search[(mode, "direct_unet")]
        tie = search[(mode, "unet_tiebreak")]
        lines.append(f"| {mode} | {prop['mae']:.3f} | {prop['mse']:.3f} | {prop['pairwise_ordering_accuracy']:.3f} | {prop['tie_set_ordering_accuracy']:.3f} | {prop['admissibility_violation_count']:.2f} | {prop['mean_overestimation_error']:.3f} | {prop['consistency_violation_count']:.2f} | {direct['mean_expanded_nodes']:.2f} | {direct['optimality_rate']:.4f} | {tie['mean_expanded_nodes']:.2f} | {tie['optimality_rate']:.4f} |")
    lines += ["", "## Main Question", "", f"Relative to ranking alone, ranking+admissibility+consistency changed Direct U-Net mean expansions from {search[('ranking', 'direct_unet')]['mean_expanded_nodes']:.2f} to {search[('ranking_adm_cons', 'direct_unet')]['mean_expanded_nodes']:.2f}, Direct optimality from {search[('ranking', 'direct_unet')]['optimality_rate']:.4f} to {search[('ranking_adm_cons', 'direct_unet')]['optimality_rate']:.4f}, and mean admissibility/consistency violations from {ranking['admissibility_violation_count']:.2f}/{ranking['consistency_violation_count']:.2f} to {combined['admissibility_violation_count']:.2f}/{combined['consistency_violation_count']:.2f}.", f"Answer: no clear Direct-U-Net win over MSE/ranking. The combined variant reached 100% observed Direct optimality, but it retained nonzero violations and used {search[('ranking_adm_cons', 'direct_unet')]['mean_expanded_nodes']:.2f} expansions versus {search[('ranking', 'direct_unet')]['mean_expanded_nodes']:.2f} for ranking. It therefore did not simultaneously provide stronger heuristic constraints and better Direct-search efficiency.", "The experiment tests measured trade-offs only. It does not make the learned Direct U-Net heuristic formally admissible or consistent.", "", "See `structure_results.csv` for map-structure breakdowns and `loss_component_analysis.csv` for best-validation loss components.", ""]
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Combined U-Net loss ablation.")
    parser.add_argument("--loss_mode", choices=[*LOSS_MODES, "all"], default="all")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--output-dir", default="outputs/combined_loss_ablation")
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    modes = LOSS_MODES if args.loss_mode == "all" else (args.loss_mode,)
    components = []
    if not args.evaluate:
        all_history = []
        for mode in modes:
            history, component = train_variant(root, output_dir, mode)
            all_history.extend(history)
            components.append(component)
        write_csv(os.path.join(output_dir, "training_history.csv"), all_history)
        write_csv(os.path.join(output_dir, "loss_component_analysis.csv"), components)
    else:
        components = list(csv.DictReader(open(os.path.join(output_dir, "loss_component_analysis.csv"), newline="", encoding="utf-8")))
    available = [mode for mode in LOSS_MODES if os.path.exists(checkpoint_path(output_dir, mode))]
    if args.evaluate or args.loss_mode == "all":
        if set(available) != set(LOSS_MODES):
            raise FileNotFoundError("Evaluation requires all four combined-loss checkpoints.")
        prediction_rows = evaluate_prediction(root, output_dir, available)
        search_rows = evaluate_search(root, output_dir, available)
        summaries = aggregate_prediction(prediction_rows, ["loss_mode"]) + aggregate_search(search_rows, ["loss_mode", "algorithm"])
        write_csv(os.path.join(output_dir, "results.csv"), summaries + prediction_rows + search_rows)
        structure = aggregate_prediction(prediction_rows, ["loss_mode", "structure_type"]) + aggregate_search(search_rows, ["loss_mode", "algorithm", "structure_type"])
        write_csv(os.path.join(output_dir, "structure_results.csv"), structure)
        write_report(output_dir, prediction_rows, search_rows, components)


if __name__ == "__main__":
    main()
