"""Reusable loss-ablation training and evaluation for the fixed structured U-Net dataset."""

import argparse
import csv
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from analyze_direct_vs_tiebreak import trace_search
from analyze_unet_structure_behavior import benchmark_cases, free_cells, mean, rebuild_case
from astar import astar_search
from bfs_label import compute_distance_to_goal
from expanded_unet_dataset import MAP_TYPES
from model import UNetHeuristic, distance_normalizer_for_grid, grid_goal_tensor, load_unet_heuristic, make_unet_heuristic, manhattan_heuristic
from train_unet import BATCH_SIZE, EPOCHS, SEED, set_global_seed


LOSS_MODES = ("mse", "ranking", "consistency", "combined")
PAIR_COUNT = 512
RANK_WEIGHT = 0.5
CONSISTENCY_WEIGHT = 0.2


def canonical_loss(name):
    return "combined" if name == "ranking_consistency" else name


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields: fields.append(field)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_path(output_dir, mode):
    return os.path.join(output_dir, "checkpoints", f"unet_{mode}_best.pt")


def sample_pairs(grid, goal, start, distances, seed):
    """Sample the requested 40/30/30 pair mixture once per fixed training map."""
    cells = [(row, col) for row, values in enumerate(distances) for col, value in enumerate(values) if value >= 0]
    coordinates = np.asarray(cells, dtype=np.int16)
    true = np.asarray([distances[row][col] for row, col in cells], dtype=np.float32)
    flat = np.asarray([row * len(grid[0]) + col for row, col in cells], dtype=np.int64)
    n = len(cells)
    row, col = coordinates[:, 0], coordinates[:, 1]
    manhattan_f = np.abs(row - start[0]) + np.abs(col - start[1]) + np.abs(row - goal[0]) + np.abs(col - goal[1])
    first, second = np.indices((n, n))
    different = (first != second) & (true[second] != true[first])
    coordinate_distance = np.abs(row[first] - row[second]) + np.abs(col[first] - col[second])
    groups = [
        different & (np.abs(manhattan_f[first] - manhattan_f[second]) <= 1),
        different & (np.abs(true[second] - true[first]) >= 1) & (np.abs(true[second] - true[first]) <= 3),
        different & (coordinate_distance > 1),
    ]
    all_distinct = np.flatnonzero(different.ravel())
    if not len(all_distinct):
        # A singleton reachable component has no ranking relation; use a harmless
        # self-pair so its rank contribution is exactly zero while MSE remains.
        only = flat[0]
        return np.tile(np.asarray([[only, only]], dtype=np.int64), (PAIR_COUNT, 1))
    counts = [205, 154, 153]
    rng = np.random.default_rng(seed)
    result = []
    for candidates, count in zip(groups, counts):
        available = np.flatnonzero(candidates.ravel())
        # A few disconnected maps have a tiny goal component. For these only,
        # retain the requested count with distinct-distance reachable pairs.
        if not len(available): available = all_distinct
        chosen = rng.choice(available, size=count, replace=len(available) < count)
        a, b = np.unravel_index(chosen, candidates.shape)
        result.extend(zip(flat[a], flat[b]))
    return np.asarray(result, dtype=np.int64)


class StructuredDistanceDataset(Dataset):
    """The existing NPZ/manifest split with fixed ranking-pair metadata added."""
    def __init__(self, archive_path, manifest_path, split):
        archive = np.load(archive_path)
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            manifest = list(csv.DictReader(handle))
        self.records = [row for row in manifest if row["split"] == split]
        self.grids, self.goals = archive["grids"], archive["goals"]
        self.labels, self.starts = archive["labels"], archive["reference_starts"]
        self.pairs = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]; item = int(record["index"])
        grid = self.grids[item].tolist()
        goal = tuple(int(value) for value in self.goals[item])
        start = tuple(int(value) for value in self.starts[item])
        label = self.labels[item].astype(np.float32)
        mask = (label >= 0).astype(np.float32)
        target = np.clip(label, 0.0, None) / distance_normalizer_for_grid(grid)
        if item not in self.pairs:
            self.pairs[item] = sample_pairs(grid, goal, start, self.labels[item], SEED + item * 37)
        return (
            grid_goal_tensor(grid, goal), torch.from_numpy(target), torch.from_numpy(mask),
            torch.tensor(goal, dtype=torch.long), torch.tensor(start, dtype=torch.long),
            torch.from_numpy(self.pairs[item]),
        )


def masked_mse(prediction, target, mask):
    return (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


def ranking_loss(prediction, target, pairs):
    """Hinge ranking loss in original grid-distance units; no squared hinge."""
    normalizer = float(prediction.shape[-2] + prediction.shape[-1])
    predicted = (prediction * normalizer).flatten(1)
    truth = (target * normalizer).flatten(1)
    first, second = pairs[:, :, 0], pairs[:, :, 1]
    prediction_a = torch.gather(predicted, 1, first)
    prediction_b = torch.gather(predicted, 1, second)
    truth_a = torch.gather(truth, 1, first)
    truth_b = torch.gather(truth, 1, second)
    delta = truth_b - truth_a
    sign = torch.sign(delta)
    rows, cols = prediction.shape[-2:]
    first_row, first_col = torch.div(first, cols, rounding_mode="floor"), first % cols
    second_row, second_col = torch.div(second, cols, rounding_mode="floor"), second % cols
    adjacent = (torch.abs(first_row - second_row) + torch.abs(first_col - second_col)) == 1
    margin = torch.minimum(torch.ones_like(delta), .25 * torch.abs(delta))
    margin = torch.where(adjacent, torch.zeros_like(margin), margin)
    hinge = torch.relu(margin - sign * (prediction_b - prediction_a))
    valid = delta != 0
    return hinge[valid].mean() if valid.any() else prediction.sum() * 0.0


def consistency_loss(prediction, mask):
    normalizer = float(prediction.shape[-2] + prediction.shape[-1])
    prediction = prediction * normalizer
    horizontal_mask, vertical_mask = mask[:, :, 1:] * mask[:, :, :-1], mask[:, 1:, :] * mask[:, :-1, :]
    horizontal = torch.relu(torch.abs(prediction[:, :, 1:] - prediction[:, :, :-1]) - 1.0) ** 2 * horizontal_mask
    vertical = torch.relu(torch.abs(prediction[:, 1:, :] - prediction[:, :-1, :]) - 1.0) ** 2 * vertical_mask
    return (horizontal.sum() + vertical.sum()) / (horizontal_mask.sum() + vertical_mask.sum()).clamp(min=1.0)


def loss_components(prediction, target, mask, pairs, mode):
    mse = masked_mse(prediction, target, mask)
    rank = ranking_loss(prediction, target, pairs) if mode in ("ranking", "combined") else prediction.sum() * 0.0
    consistency = consistency_loss(prediction, mask) if mode in ("consistency", "combined") else prediction.sum() * 0.0
    total = mse + (RANK_WEIGHT * rank if mode in ("ranking", "combined") else 0.0) + (CONSISTENCY_WEIGHT * consistency if mode in ("consistency", "combined") else 0.0)
    return total, mse, rank, consistency


def run_epoch(model, loader, mode, optimizer=None):
    training = optimizer is not None; model.train(training)
    totals = defaultdict(float); valid = 0.0; examples = 0
    for model_input, target, mask, _goal, _start, pairs in loader:
        if training: optimizer.zero_grad()
        prediction = model(model_input)
        total, mse, rank, consistency = loss_components(prediction, target, mask, pairs, mode)
        if training:
            total.backward(); optimizer.step()
        count = model_input.size(0); examples += count
        totals["loss"] += total.item() * count; totals["mse_loss"] += mse.item() * count
        totals["ranking_loss"] += rank.item() * count; totals["consistency_loss"] += consistency.item() * count
        error = (prediction.detach() - target) * mask
        totals["mae"] += torch.abs(error).sum().item(); totals["mse"] += (error ** 2).sum().item(); valid += mask.sum().item()
    return {"loss": totals["loss"] / examples, "mse_loss": totals["mse_loss"] / examples,
            "ranking_loss": totals["ranking_loss"] / examples, "consistency_loss": totals["consistency_loss"] / examples,
            "mae": totals["mae"] / valid, "mse": totals["mse"] / valid}


def payload(model, mode, epoch, train_stats, val_stats, archive_path, manifest_path):
    return {"model_state_dict": model.state_dict(), "loss": mode, "epoch": epoch, "validation_loss": val_stats["loss"],
            "training_loss": train_stats["loss"], "dataset_archive": archive_path, "dataset_manifest": manifest_path,
            "train_examples": 4000, "val_examples": 500, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "seed": SEED, "optimizer": "Adam(lr=0.001)", "loss_weights": {"ranking": RANK_WEIGHT, "consistency": CONSISTENCY_WEIGHT},
            "ranking_pairs_per_map": PAIR_COUNT, "ranking_pair_mix": "40% Manhattan-f nearby, 30% true-distance 1-3, 30% random non-adjacent"}


def train_variant(project_root, output_dir, mode):
    archive_path = os.path.join(project_root, "outputs/expanded_dataset_training/dataset/expanded_unet_dataset.npz")
    manifest_path = os.path.join(project_root, "outputs/expanded_dataset_training/dataset/dataset_manifest.csv")
    if not os.path.exists(archive_path): raise FileNotFoundError("The fixed 5,000-map structured dataset is missing")
    set_global_seed(SEED)
    train_set, val_set = StructuredDistanceDataset(archive_path, manifest_path, "train"), StructuredDistanceDataset(archive_path, manifest_path, "val")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    model = UNetHeuristic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rows, best = [], float("inf")
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        train_stats = run_epoch(model, train_loader, mode, optimizer)
        with torch.no_grad(): val_stats = run_epoch(model, val_loader, mode)
        row = {"loss_variant": mode, "epoch": epoch, **{f"train_{key}": value for key, value in train_stats.items()}, **{f"val_{key}": value for key, value in val_stats.items()}}
        rows.append(row)
        if val_stats["loss"] < best:
            best = val_stats["loss"]
            torch.save(payload(model, mode, epoch, train_stats, val_stats, archive_path, manifest_path), checkpoint_path(output_dir, mode))
        print(f"{mode} epoch {epoch:02d}/{EPOCHS}: train={train_stats['loss']:.6f} val={val_stats['loss']:.6f}")
    write_csv(os.path.join(output_dir, f"training_history_{mode}.csv"), rows)
    return rows


def table_from_model(model, grid, goal):
    heuristic = make_unet_heuristic(model, grid, goal)
    return {cell: float(heuristic(cell, goal)) for cell in free_cells(grid)}


def pairwise_accuracy(table, grid, pairs, labels):
    cols = len(grid[0]); correct = total = 0
    for first, second in pairs:
        a, b = (int(first) // cols, int(first) % cols), (int(second) // cols, int(second) % cols)
        delta = labels[b[0]][b[1]] - labels[a[0]][a[1]]
        if delta == 0: continue
        correct += int((table[b] - table[a]) * delta > 0); total += 1
    return correct, total


def consistency_metrics(table, grid, labels):
    violations = 0; magnitude = 0.0
    for cell in table:
        for neighbor in ((cell[0] + 1, cell[1]), (cell[0], cell[1] + 1)):
            if neighbor not in table: continue
            excess = abs(table[cell] - table[neighbor]) - 1.0
            if excess > 1e-6: violations += 1; magnitude += excess
    return violations, magnitude


def tie_accuracy(trace):
    correct = total = 0
    for item in trace["trace"]:
        current = item["expanded"]; tied = [entry for entry in item["open_before"] if entry["f"] == current["f"]]
        if len(tied) < 2: continue
        total += 1; correct += int(current["true_distance"] == min(entry["true_distance"] for entry in tied))
    return correct, total


def evaluate_prediction_set(project_root, output_dir, variants):
    archive_path = os.path.join(project_root, "outputs/expanded_dataset_training/dataset/expanded_unet_dataset.npz")
    manifest_path = os.path.join(project_root, "outputs/expanded_dataset_training/dataset/dataset_manifest.csv")
    dataset = StructuredDistanceDataset(archive_path, manifest_path, "test")
    rows = []
    for variant in variants:
        model = load_unet_heuristic(checkpoint_path(output_dir, variant))
        for index, record in enumerate(dataset.records):
            item = int(record["index"]); grid = dataset.grids[item].tolist(); goal = tuple(int(v) for v in dataset.goals[item]); start = tuple(int(v) for v in dataset.starts[item])
            labels = dataset.labels[item].tolist(); table = table_from_model(model, grid, goal)
            reachable = list(table); predictions, targets = [table[cell] for cell in reachable], [labels[cell[0]][cell[1]] for cell in reachable]
            mae = mean(abs(a - b) for a, b in zip(predictions, targets)); mse = mean((a - b) ** 2 for a, b in zip(predictions, targets))
            if item not in dataset.pairs:
                dataset.pairs[item] = sample_pairs(grid, goal, start, dataset.labels[item], SEED + item * 37)
            pairs = dataset.pairs[item]
            pair_correct, pair_total = pairwise_accuracy(table, grid, pairs, labels)
            distance_from_start = compute_distance_to_goal(grid, start); optimal_cost = labels[start[0]][start[1]]
            trace = trace_search(grid, start, goal, table, labels, set(), "unet_tiebreak")
            if trace["cost"] != optimal_cost: raise AssertionError("Tie-break lost optimality on held-out prediction test set")
            tie_correct, tie_total = tie_accuracy(trace); violations, magnitude = consistency_metrics(table, grid, labels)
            rows.append({"variant": variant, "split": "prediction_test", "case_id": f"test_{item}", "structure_type": record["map_type"], "obstacle_rate": record["obstacle_rate"],
                         "mae": mae, "mse": mse, "pairwise_ordering_accuracy": pair_correct / max(1, pair_total), "pairwise_pairs": pair_total,
                         "tie_set_ordering_accuracy": tie_correct / max(1, tie_total), "tie_set_decisions": tie_total,
                         "consistency_violation_count": violations, "consistency_violation_magnitude": magnitude})
    return rows


def evaluate_search_cases(project_root, output_dir, variants):
    cases = benchmark_cases(project_root, "outputs/experiments/results_structured_tiebreak_controls_100.csv", "outputs/experiments/results_random_tiebreak_controls_100.csv")
    rows = []
    for variant in variants:
        model = load_unet_heuristic(checkpoint_path(output_dir, variant))
        for number, source in enumerate(cases, start=1):
            grid, start, goal = rebuild_case(source); distance = compute_distance_to_goal(grid, goal); optimal = distance[start[0]][start[1]]
            table = table_from_model(model, grid, goal); heuristic = lambda node, unused: table[node]
            for algorithm in ("direct_unet", "unet_tiebreak"):
                result = astar_search(grid, start, goal, heuristic) if algorithm == "direct_unet" else astar_search(grid, start, goal, manhattan_heuristic, secondary_heuristic=heuristic)
                rows.append({"variant": variant, "case_id": f"{source['analysis_structure']}_seed{source['seed']}_s{start[0]}-{start[1]}_g{goal[0]}-{goal[1]}", "structure_type": source["analysis_structure"], "algorithm": algorithm,
                             "expanded_nodes": result["expanded"], "path_cost": result["cost"], "optimal_cost": optimal, "optimal": result["cost"] == optimal})
            if number % 100 == 0: print(f"{variant} search evaluation {number}/{len(cases)}")
    return rows


def aggregate_prediction(rows, fields):
    grouped = defaultdict(list)
    for row in rows: grouped[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        result = {field: value for field, value in zip(fields, key)}; result.update({"scope": "prediction", "cases": len(values),
            "mae": mean(row["mae"] for row in values), "mse": mean(row["mse"] for row in values),
            "pairwise_ordering_accuracy": sum(row["pairwise_ordering_accuracy"] * row["pairwise_pairs"] for row in values) / max(1, sum(row["pairwise_pairs"] for row in values)),
            "tie_set_ordering_accuracy": sum(row["tie_set_ordering_accuracy"] * row["tie_set_decisions"] for row in values) / max(1, sum(row["tie_set_decisions"] for row in values)),
            "consistency_violation_count": mean(row["consistency_violation_count"] for row in values), "consistency_violation_magnitude": mean(row["consistency_violation_magnitude"] for row in values)})
        output.append(result)
    return output


def aggregate_search(rows, fields):
    grouped = defaultdict(list)
    for row in rows: grouped[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        result = {field: value for field, value in zip(fields, key)}; result.update({"scope": "search", "cases": len(values),
            "mean_expanded_nodes": mean(row["expanded_nodes"] for row in values), "optimality_rate": mean(float(row["optimal"]) for row in values)})
        output.append(result)
    return output


def write_report(output_dir, prediction, search):
    overall_prediction = {row["variant"]: row for row in aggregate_prediction(prediction, ["variant"])}
    overall_search = {(row["variant"], row["algorithm"]): row for row in aggregate_search(search, ["variant", "algorithm"])}
    baseline = overall_prediction["mse"]
    lines = ["# U-Net Loss Ablation v1", "", "All variants use the existing 5,000-map structured dataset, its fixed 4,000/500/500 split, unchanged U-Net architecture, Adam optimizer, batch size 16, and 50 epochs.", "", "## Overall", "", "| Variant | MAE | MSE | Pair ordering | Tie-set ordering | Consistency violations | Direct optimality | Direct expanded | Tie-break expanded |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in LOSS_MODES:
        prediction_row = overall_prediction[variant]; direct = overall_search[(variant, "direct_unet")]; tie = overall_search[(variant, "unet_tiebreak")]
        lines.append(f"| {variant} | {prediction_row['mae']:.3f} | {prediction_row['mse']:.3f} | {prediction_row['pairwise_ordering_accuracy']:.3f} | {prediction_row['tie_set_ordering_accuracy']:.3f} | {prediction_row['consistency_violation_count']:.2f} | {direct['optimality_rate']:.4f} | {direct['mean_expanded_nodes']:.2f} | {tie['mean_expanded_nodes']:.2f} |")
    ranking = overall_prediction["ranking"]; consistency = overall_prediction["consistency"]; combined = overall_prediction["combined"]
    combined_clear = combined['pairwise_ordering_accuracy'] > baseline['pairwise_ordering_accuracy'] and overall_search[('combined','direct_unet')]['mean_expanded_nodes'] <= overall_search[('mse','direct_unet')]['mean_expanded_nodes'] and overall_search[('combined','unet_tiebreak')]['mean_expanded_nodes'] <= overall_search[('mse','unet_tiebreak')]['mean_expanded_nodes']
    lines += ["", "## Answers", "", f"1. Ranking loss {'improved' if ranking['pairwise_ordering_accuracy'] > baseline['pairwise_ordering_accuracy'] else 'did not improve'} held-out pair ordering ({baseline['pairwise_ordering_accuracy']:.3f} -> {ranking['pairwise_ordering_accuracy']:.3f}) and changed Direct/tie-break mean expansions from {overall_search[('mse','direct_unet')]['mean_expanded_nodes']:.2f}/{overall_search[('mse','unet_tiebreak')]['mean_expanded_nodes']:.2f} to {overall_search[('ranking','direct_unet')]['mean_expanded_nodes']:.2f}/{overall_search[('ranking','unet_tiebreak')]['mean_expanded_nodes']:.2f}.", f"2. Consistency loss {'reduced' if consistency['consistency_violation_count'] < baseline['consistency_violation_count'] else 'did not reduce'} mean violations ({baseline['consistency_violation_count']:.2f} -> {consistency['consistency_violation_count']:.2f}) and improved Direct optimality from {overall_search[('mse','direct_unet')]['optimality_rate']:.4f} to {overall_search[('consistency','direct_unet')]['optimality_rate']:.4f}; it did not fully eliminate failures.", f"3. Combined loss {'outperformed' if combined_clear else 'did not clearly outperform'} MSE overall: it improves ordering and Direct optimality, but Direct mean expansions are {overall_search[('combined','direct_unet')]['mean_expanded_nodes']:.2f} versus {overall_search[('mse','direct_unet')]['mean_expanded_nodes']:.2f} for MSE.", "4. Lower prediction error does not imply better search performance here: combined has the lowest held-out MSE but substantially more Direct expansions than MSE, while ranking has higher MAE yet fewer Direct expansions.", "", "See `summary_by_structure.csv` for all map-type and benchmark-structure results.", ""]
    with open(os.path.join(output_dir, "loss_ablation_report.md"), "w", encoding="utf-8") as handle: handle.write("\n".join(lines))


def evaluate(project_root, output_dir, variants):
    prediction = evaluate_prediction_set(project_root, output_dir, variants)
    search = evaluate_search_cases(project_root, output_dir, variants)
    write_csv(os.path.join(output_dir, "prediction_metrics.csv"), prediction)
    write_csv(os.path.join(output_dir, "search_results.csv"), search)
    overall = []
    for row in aggregate_prediction(prediction, ["variant"]): overall.append(row)
    for row in aggregate_search(search, ["variant", "algorithm"]): overall.append(row)
    write_csv(os.path.join(output_dir, "summary_by_variant.csv"), overall)
    by_structure = aggregate_prediction(prediction, ["variant", "structure_type"]) + aggregate_search(search, ["variant", "algorithm", "structure_type"])
    write_csv(os.path.join(output_dir, "summary_by_structure.csv"), by_structure)
    write_report(output_dir, prediction, search)


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate fixed-dataset U-Net loss ablations.")
    parser.add_argument("--loss", choices=[*LOSS_MODES, "ranking_consistency", "all"], default="all")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate all available checkpoints after training.")
    parser.add_argument("--output-dir", default="outputs/loss_ablation_v1")
    return parser.parse_args()


def main():
    args = parse_args(); project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    requested = LOSS_MODES if args.loss == "all" else (canonical_loss(args.loss),)
    histories = []
    for mode in requested: histories.extend(train_variant(project_root, output_dir, mode))
    if histories: write_csv(os.path.join(output_dir, "training_history.csv"), histories)
    available = [mode for mode in LOSS_MODES if os.path.exists(checkpoint_path(output_dir, mode))]
    if args.evaluate or args.loss == "all":
        if set(available) != set(LOSS_MODES): raise FileNotFoundError("Evaluation requires mse, ranking, consistency, and combined checkpoints")
        evaluate(project_root, output_dir, available)


if __name__ == "__main__": main()
