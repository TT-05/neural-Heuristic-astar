"""Ranking-loss lambda sweep on the fixed structured U-Net dataset."""

import argparse
import csv
import os
from collections import defaultdict

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
    SEED,
    StructuredDistanceDataset,
    pairwise_accuracy,
    ranking_loss,
    sample_pairs,
    tie_accuracy,
    write_csv,
)
from train_unet import set_global_seed


LAMBDA_TEXT = ("0.1", "0.25", "0.5", "0.75", "1.0", "2.0")


def checkpoint_path(output_dir, lambda_text):
    return os.path.join(output_dir, f"lambda_{lambda_text}_best.pt")


def masked_mse(prediction, target, mask):
    return (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


def run_epoch(model, loader, ranking_lambda, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    examples = valid_cells = 0.0
    for model_input, target, mask, _goal, _start, pairs in loader:
        if training:
            optimizer.zero_grad()
        prediction = model(model_input)
        mse = masked_mse(prediction, target, mask)
        rank = ranking_loss(prediction, target, pairs)
        loss = mse + ranking_lambda * rank
        if training:
            loss.backward()
            optimizer.step()
        batch = model_input.size(0)
        examples += batch
        valid_cells += mask.sum().item()
        totals["loss"] += loss.item() * batch
        totals["mse_loss"] += mse.item() * batch
        totals["ranking_loss"] += rank.item() * batch
        error = (prediction.detach() - target) * mask
        totals["mae"] += torch.abs(error).sum().item()
        totals["mse"] += (error ** 2).sum().item()
    return {
        "loss": totals["loss"] / examples,
        "mse_loss": totals["mse_loss"] / examples,
        "ranking_loss": totals["ranking_loss"] / examples,
        "mae": totals["mae"] / valid_cells,
        "mse": totals["mse"] / valid_cells,
    }


def train_variant(root, output_dir, lambda_text):
    ranking_lambda = float(lambda_text)
    archive = os.path.join(root, "outputs/expanded_dataset_training/dataset/expanded_unet_dataset.npz")
    manifest = os.path.join(root, "outputs/expanded_dataset_training/dataset/dataset_manifest.csv")
    set_global_seed(SEED)
    train_set = StructuredDistanceDataset(archive, manifest, "train")
    validation_set = StructuredDistanceDataset(archive, manifest, "val")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)
    model = UNetHeuristic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = float("inf")
    best_components = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        train = run_epoch(model, train_loader, ranking_lambda, optimizer)
        with torch.no_grad():
            validation = run_epoch(model, validation_loader, ranking_lambda)
        history.append({"ranking_lambda": lambda_text, "epoch": epoch, **{f"train_{key}": value for key, value in train.items()}, **{f"val_{key}": value for key, value in validation.items()}})
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_components = {"ranking_lambda": lambda_text, "best_epoch": epoch, **{f"validation_{key}": value for key, value in validation.items()}}
            torch.save({
                "model_state_dict": model.state_dict(), "loss": "mse_plus_lambda_ranking", "ranking_lambda": ranking_lambda,
                "epoch": epoch, "validation_loss": validation["loss"], "dataset_archive": archive, "dataset_manifest": manifest,
                "train_examples": len(train_set), "validation_examples": len(validation_set), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                "seed": SEED, "optimizer": "Adam(lr=0.001)", "ranking_pairs_per_map": PAIR_COUNT,
            }, checkpoint_path(output_dir, lambda_text))
        print(f"lambda={lambda_text} epoch {epoch:02d}/{EPOCHS}: train={train['loss']:.6f} val={validation['loss']:.6f}")
    write_csv(os.path.join(output_dir, f"training_history_lambda_{lambda_text}.csv"), history)
    return history, best_components


def prediction_table(model, grid, goal):
    heuristic = make_unet_heuristic(model, grid, goal)
    return {cell: float(heuristic(cell, goal)) for cell in free_cells(grid)}


def evaluate_prediction(root, output_dir, lambda_values):
    archive = os.path.join(root, "outputs/expanded_dataset_training/dataset/expanded_unet_dataset.npz")
    manifest = os.path.join(root, "outputs/expanded_dataset_training/dataset/dataset_manifest.csv")
    dataset = StructuredDistanceDataset(archive, manifest, "test")
    rows = []
    for lambda_text in lambda_values:
        model = load_unet_heuristic(checkpoint_path(output_dir, lambda_text))
        for record in dataset.records:
            item = int(record["index"])
            grid = dataset.grids[item].tolist()
            start = tuple(int(value) for value in dataset.starts[item])
            goal = tuple(int(value) for value in dataset.goals[item])
            labels = dataset.labels[item].tolist()
            table = prediction_table(model, grid, goal)
            if item not in dataset.pairs:
                dataset.pairs[item] = sample_pairs(grid, goal, start, dataset.labels[item], SEED + item * 37)
            correct, total = pairwise_accuracy(table, grid, dataset.pairs[item], labels)
            trace = trace_search(grid, start, goal, table, labels, set(), "unet_tiebreak")
            if trace["cost"] != labels[start[0]][start[1]]:
                raise AssertionError("U-Net tie-break lost held-out optimality")
            tie_correct, tie_total = tie_accuracy(trace)
            cells = list(table)
            rows.append({
                "ranking_lambda": lambda_text, "scope": "prediction", "case_id": f"test_{item}", "structure_type": record["map_type"],
                "mae": mean(abs(table[cell] - labels[cell[0]][cell[1]]) for cell in cells),
                "mse": mean((table[cell] - labels[cell[0]][cell[1]]) ** 2 for cell in cells),
                "pairwise_ordering_accuracy": correct / max(1, total), "pairwise_pairs": total,
                "tie_set_ordering_accuracy": tie_correct / max(1, tie_total), "tie_set_decisions": tie_total,
            })
    return rows


def evaluate_search(root, output_dir, lambda_values):
    cases = benchmark_cases(root, "outputs/experiments/results_structured_tiebreak_controls_100.csv", "outputs/experiments/results_random_tiebreak_controls_100.csv")
    rows = []
    for lambda_text in lambda_values:
        model = load_unet_heuristic(checkpoint_path(output_dir, lambda_text))
        for number, source in enumerate(cases, 1):
            grid, start, goal = rebuild_case(source)
            optimal = compute_distance_to_goal(grid, goal)[start[0]][start[1]]
            table = prediction_table(model, grid, goal)
            heuristic = lambda node, unused: table[node]
            for algorithm in ("direct_unet", "unet_tiebreak"):
                result = astar_search(grid, start, goal, heuristic) if algorithm == "direct_unet" else astar_search(grid, start, goal, manhattan_heuristic, secondary_heuristic=heuristic)
                rows.append({"ranking_lambda": lambda_text, "scope": "search", "case_id": f"{source['analysis_structure']}_seed{source['seed']}", "structure_type": source["analysis_structure"], "algorithm": algorithm, "expanded_nodes": result["expanded"], "path_cost": result["cost"], "optimal_cost": optimal, "optimal": result["cost"] == optimal})
            if number % 100 == 0:
                print(f"lambda={lambda_text} evaluation {number}/{len(cases)}")
    return rows


def prediction_summary(rows):
    result = {}
    for lambda_text in LAMBDA_TEXT:
        values = [row for row in rows if row["ranking_lambda"] == lambda_text]
        result[lambda_text] = {
            "ranking_lambda": lambda_text, "scope": "prediction_summary", "cases": len(values),
            "mae": mean(row["mae"] for row in values), "mse": mean(row["mse"] for row in values),
            "tie_set_ordering_accuracy": sum(row["tie_set_ordering_accuracy"] * row["tie_set_decisions"] for row in values) / max(1, sum(row["tie_set_decisions"] for row in values)),
        }
    return result


def search_summary(rows):
    result = {}
    for lambda_text in LAMBDA_TEXT:
        for algorithm in ("direct_unet", "unet_tiebreak"):
            values = [row for row in rows if row["ranking_lambda"] == lambda_text and row["algorithm"] == algorithm]
            result[(lambda_text, algorithm)] = {"ranking_lambda": lambda_text, "algorithm": algorithm, "scope": "search_summary", "cases": len(values), "mean_expanded_nodes": mean(row["expanded_nodes"] for row in values), "optimality_rate": mean(float(row["optimal"]) for row in values)}
    return result


def write_report(output_dir, prediction_rows, search_rows, components):
    prediction = prediction_summary(prediction_rows)
    search = search_summary(search_rows)
    summaries = []
    for lambda_text in LAMBDA_TEXT:
        summaries.append({
            **prediction[lambda_text], "scope": "summary",
            "direct_unet_mean_expanded_nodes": search[(lambda_text, "direct_unet")]["mean_expanded_nodes"],
            "direct_unet_optimality_rate": search[(lambda_text, "direct_unet")]["optimality_rate"],
            "unet_tiebreak_mean_expanded_nodes": search[(lambda_text, "unet_tiebreak")]["mean_expanded_nodes"],
            "unet_tiebreak_optimality_rate": search[(lambda_text, "unet_tiebreak")]["optimality_rate"],
        })
    write_csv(os.path.join(output_dir, "results.csv"), summaries + prediction_rows + search_rows)
    by_lambda = {row["ranking_lambda"]: row for row in summaries}
    best_direct = min(summaries, key=lambda row: row["direct_unet_mean_expanded_nodes"])
    best_tie = min(summaries, key=lambda row: row["unet_tiebreak_mean_expanded_nodes"])
    best_prediction = min(summaries, key=lambda row: row["mse"])
    lines = ["# Ranking-Loss Lambda Sweep", "", "All runs use the unchanged 5,000-map split, U-Net architecture, optimizer, batch size, 50-epoch schedule, ranking-pair formulation, and 2,000 benchmark cases. Only lambda in `L = MSE + lambda * ranking_loss` changes.", "", "| Lambda | MAE | MSE | Tie-set ordering | Direct expanded | Direct optimality | Tie-break expanded | Tie-break optimality |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lambda_text in LAMBDA_TEXT:
        row = by_lambda[lambda_text]
        lines.append(f"| {lambda_text} | {row['mae']:.3f} | {row['mse']:.3f} | {row['tie_set_ordering_accuracy']:.3f} | {row['direct_unet_mean_expanded_nodes']:.2f} | {row['direct_unet_optimality_rate']:.4f} | {row['unet_tiebreak_mean_expanded_nodes']:.2f} | {row['unet_tiebreak_optimality_rate']:.4f} |")
    lines += ["", "## Main Question", "", f"The lowest Direct U-Net mean expansion count occurs at lambda={best_direct['ranking_lambda']} ({best_direct['direct_unet_mean_expanded_nodes']:.2f}, optimality {best_direct['direct_unet_optimality_rate']:.4f}).", f"The lowest U-Net tie-break mean expansion count occurs at lambda={best_tie['ranking_lambda']} ({best_tie['unet_tiebreak_mean_expanded_nodes']:.2f}).", f"The lowest held-out prediction MSE occurs at lambda={best_prediction['ranking_lambda']} ({best_prediction['mse']:.3f}).", "These may be different lambdas; the sweep reports the observed trade-off rather than defining one universal optimum.", "", "See `results.csv` for per-case data and summary rows.", ""]
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep the U-Net ranking-loss weight.")
    parser.add_argument("--ranking_lambda", choices=[*LAMBDA_TEXT, "all"], default="all")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--output-dir", default="outputs/ranking_lambda_sweep")
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    values = LAMBDA_TEXT if args.ranking_lambda == "all" else (args.ranking_lambda,)
    components = []
    if not args.evaluate:
        history = []
        for lambda_text in values:
            rows, best = train_variant(root, output_dir, lambda_text)
            history.extend(rows)
            components.append(best)
        write_csv(os.path.join(output_dir, "training_history.csv"), history)
        write_csv(os.path.join(output_dir, "best_validation_components.csv"), components)
    else:
        components = list(csv.DictReader(open(os.path.join(output_dir, "best_validation_components.csv"), newline="", encoding="utf-8")))
    available = [lambda_text for lambda_text in LAMBDA_TEXT if os.path.exists(checkpoint_path(output_dir, lambda_text))]
    if args.evaluate or args.ranking_lambda == "all":
        if set(available) != set(LAMBDA_TEXT):
            raise FileNotFoundError("Evaluation requires all six lambda checkpoints.")
        prediction_rows = evaluate_prediction(root, output_dir, available)
        search_rows = evaluate_search(root, output_dir, available)
        write_report(output_dir, prediction_rows, search_rows, components)


if __name__ == "__main__":
    main()
