"""Critical-decision weighted ranking-loss experiment for the fixed U-Net dataset."""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from analyze_critical_decisions import (
    WEAK_MARGIN_THRESHOLD,
    critical_events_for_case,
    optimal_path_nodes,
    trace_unet_tiebreak_astar,
)
from analyze_direct_vs_tiebreak import trace_search
from analyze_unet_structure_behavior import benchmark_cases, free_cells, mean, rebuild_case
from astar import astar_search
from bfs_label import compute_distance_to_goal
from model import UNetHeuristic, grid_goal_tensor, load_unet_heuristic, make_unet_heuristic, manhattan_heuristic
from run_loss_ablation import BATCH_SIZE, EPOCHS, PAIR_COUNT, RANK_WEIGHT, pairwise_accuracy, sample_pairs, tie_accuracy, write_csv
from train_unet import SEED, set_global_seed


MODES = ("random", "critical_weighted")
CRITICAL_WEIGHT = 5.0
CRITICAL_PAIR_BUDGET = 64
# Match the existing critical-decision analysis. Critical labels are fixed
# before training rather than refreshed as the experimental model changes.
BASELINE_CRITICAL_CHECKPOINT = "checkpoints/unet_heuristic.pt"


def checkpoint_path(output_dir, mode):
    name = "random" if mode == "random" else "critical_weighted"
    return os.path.join(output_dir, f"{name}_best.pt")


def masked_mse(prediction, target, mask):
    return (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


def weighted_ranking_loss(prediction, target, pairs, weights):
    """Original-unit hinge ranking loss with per-pair critical-decision weights."""
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
    cols = prediction.shape[-1]
    first_row, first_col = torch.div(first, cols, rounding_mode="floor"), first % cols
    second_row, second_col = torch.div(second, cols, rounding_mode="floor"), second % cols
    adjacent = (torch.abs(first_row - second_row) + torch.abs(first_col - second_col)) == 1
    margin = torch.minimum(torch.ones_like(delta), 0.25 * torch.abs(delta))
    margin = torch.where(adjacent, torch.zeros_like(margin), margin)
    hinge = torch.relu(margin - sign * (prediction_b - prediction_a))
    effective_weight = weights * (delta != 0).to(weights.dtype)
    return (hinge * effective_weight).sum() / effective_weight.sum().clamp(min=1.0)


def prediction_table(model, grid, goal):
    heuristic = make_unet_heuristic(model, grid, goal)
    return {cell: float(heuristic(cell, goal)) for cell in free_cells(grid)}


def critical_pairs_for_map(model, grid, start, goal, labels, case_id, structure_type):
    """Return source-model U-Net ordering mistakes with recovery cost >= 10."""
    optimal_cost = int(labels[start[0]][start[1]])
    if optimal_cost < 0:
        return []
    distance_to_goal = labels.tolist()
    distance_from_start = compute_distance_to_goal(grid, start)
    optimal_nodes = optimal_path_nodes(distance_from_start, distance_to_goal, optimal_cost)
    values = prediction_table(model, grid, goal)
    trace = trace_unet_tiebreak_astar(grid, start, goal, values, distance_to_goal, optimal_nodes)
    if trace["cost"] != optimal_cost:
        raise AssertionError(f"Source U-Net tie-break lost optimality for {case_id}")
    case = {"case_id": case_id, "structured_type": structure_type, "optimal_cost": optimal_cost}
    events = critical_events_for_case(case, trace, optimal_nodes, WEAK_MARGIN_THRESHOLD)
    rows, cols = len(grid), len(grid[0])
    pairs = []
    for event in events:
        # This is exactly the requested disagreement: true distance favors the
        # optimal candidate while the source U-Net ranked the wrong expansion first.
        if event["error_type"] != "A_unet_ordering_error" or int(event["recovery_cost"]) < 10:
            continue
        wrong_r, wrong_c = (int(value) for value in event["wrong_node"].split(","))
        correct_r, correct_c = (int(value) for value in event["correct_node"].split(","))
        pairs.append((wrong_r * cols + wrong_c, correct_r * cols + correct_c))
    return sorted(set(pairs))


class CriticalWeightedDataset(Dataset):
    """Existing split with all random pairs plus a bounded weighted critical subset."""

    def __init__(self, archive_path, manifest_path, split, mode, baseline_model, cache_dir):
        archive = np.load(archive_path)
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            manifest = list(csv.DictReader(handle))
        self.records = [row for row in manifest if row["split"] == split]
        self.grids, self.goals = archive["grids"], archive["goals"]
        self.labels, self.starts = archive["labels"], archive["reference_starts"]
        self.pairs, self.weights, self.critical_counts = {}, {}, {}
        for record in self.records:
            item = int(record["index"])
            grid = self.grids[item].tolist()
            start = tuple(int(value) for value in self.starts[item])
            goal = tuple(int(value) for value in self.goals[item])
            random_pairs = sample_pairs(grid, goal, start, self.labels[item], SEED + item * 37)
            critical_pairs = []
            if mode == "critical_weighted":
                critical_pairs = critical_pairs_for_map(
                    baseline_model, grid, start, goal, self.labels[item], f"{split}_{item}", record["map_type"]
                )
                if len(critical_pairs) > CRITICAL_PAIR_BUDGET:
                    rng = np.random.default_rng(SEED + item * 101)
                    selected = rng.choice(len(critical_pairs), size=CRITICAL_PAIR_BUDGET, replace=False)
                    critical_pairs = [critical_pairs[index] for index in selected]
            self.critical_counts[item] = len(critical_pairs)
            padding = CRITICAL_PAIR_BUDGET - len(critical_pairs)
            filler = random_pairs[:1] if len(random_pairs) else np.zeros((1, 2), dtype=np.int64)
            padded_critical = np.asarray(critical_pairs, dtype=np.int64) if critical_pairs else np.empty((0, 2), dtype=np.int64)
            if padding:
                padded_critical = np.concatenate([padded_critical, np.repeat(filler, padding, axis=0)], axis=0)
            self.pairs[item] = np.concatenate([random_pairs, padded_critical], axis=0)
            self.weights[item] = np.concatenate([
                np.ones(PAIR_COUNT, dtype=np.float32),
                np.concatenate([
                    np.full(len(critical_pairs), CRITICAL_WEIGHT, dtype=np.float32),
                    np.zeros(padding, dtype=np.float32),
                ]),
            ])
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(cache_dir, f"{split}_{mode}_pairs.npz"),
            indices=np.asarray([int(row["index"]) for row in self.records]),
            pairs=np.asarray([self.pairs[int(row["index"])] for row in self.records]),
            weights=np.asarray([self.weights[int(row["index"])] for row in self.records]),
            critical_counts=np.asarray([self.critical_counts[int(row["index"])] for row in self.records]),
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        item = int(record["index"])
        grid = self.grids[item].tolist()
        goal = tuple(int(value) for value in self.goals[item])
        label = self.labels[item].astype(np.float32)
        mask = (label >= 0).astype(np.float32)
        target = np.clip(label, 0.0, None) / float(len(grid) + len(grid[0]))
        return (
            grid_goal_tensor(grid, goal),
            torch.from_numpy(target),
            torch.from_numpy(mask),
            torch.from_numpy(self.pairs[item]),
            torch.from_numpy(self.weights[item]),
        )


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    examples = valid_cells = 0.0
    for model_input, target, mask, pairs, weights in loader:
        if training:
            optimizer.zero_grad()
        prediction = model(model_input)
        mse = masked_mse(prediction, target, mask)
        rank = weighted_ranking_loss(prediction, target, pairs, weights)
        loss = mse + RANK_WEIGHT * rank
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
    return {"loss": totals["loss"] / examples, "mse_loss": totals["mse_loss"] / examples, "ranking_loss": totals["ranking_loss"] / examples, "mae": totals["mae"] / valid_cells, "mse": totals["mse"] / valid_cells}


def train_variant(root, output_dir, mode, baseline_model):
    archive = os.path.join(root, "outputs/expanded_dataset_training/dataset/expanded_unet_dataset.npz")
    manifest = os.path.join(root, "outputs/expanded_dataset_training/dataset/dataset_manifest.csv")
    cache_dir = os.path.join(output_dir, "pair_caches")
    set_global_seed(SEED)
    train_set = CriticalWeightedDataset(archive, manifest, "train", mode, baseline_model, cache_dir)
    val_set = CriticalWeightedDataset(archive, manifest, "val", mode, baseline_model, cache_dir)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    model = UNetHeuristic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best = float("inf")
    history = []
    for epoch in range(1, EPOCHS + 1):
        train = run_epoch(model, train_loader, optimizer)
        with torch.no_grad():
            validation = run_epoch(model, val_loader)
        history.append({"variant": mode, "epoch": epoch, **{f"train_{key}": value for key, value in train.items()}, **{f"val_{key}": value for key, value in validation.items()}})
        if validation["loss"] < best:
            best = validation["loss"]
            torch.save({"model_state_dict": model.state_dict(), "loss": "mse_plus_weighted_ranking", "variant": mode, "epoch": epoch, "validation_loss": validation["loss"], "critical_weight": CRITICAL_WEIGHT}, checkpoint_path(output_dir, mode))
        print(f"{mode} epoch {epoch:02d}/{EPOCHS}: train={train['loss']:.6f} val={validation['loss']:.6f}")
    return history, {
        "variant": mode,
        "train_maps": len(train_set),
        "validation_maps": len(val_set),
        "random_pairs_per_map": PAIR_COUNT,
        "critical_pair_budget": CRITICAL_PAIR_BUDGET,
        "mean_critical_pairs_train": mean(train_set.critical_counts.values()),
        "maps_with_critical_pairs_train": sum(count > 0 for count in train_set.critical_counts.values()),
    }


def evaluate_prediction(root, output_dir, variants):
    archive = np.load(os.path.join(root, "outputs/expanded_dataset_training/dataset/expanded_unet_dataset.npz"))
    with open(os.path.join(root, "outputs/expanded_dataset_training/dataset/dataset_manifest.csv"), newline="", encoding="utf-8") as handle:
        records = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    rows = []
    for variant in variants:
        model = load_unet_heuristic(checkpoint_path(output_dir, variant))
        for record in records:
            item = int(record["index"])
            grid = archive["grids"][item].tolist()
            start = tuple(int(value) for value in archive["reference_starts"][item])
            goal = tuple(int(value) for value in archive["goals"][item])
            labels = archive["labels"][item].tolist()
            values = prediction_table(model, grid, goal)
            pairs = sample_pairs(grid, goal, start, labels, SEED + item * 37)
            correct, total = pairwise_accuracy(values, grid, pairs, labels)
            trace = trace_search(grid, start, goal, values, labels, set(), "unet_tiebreak")
            if trace["cost"] != labels[start[0]][start[1]]:
                raise AssertionError("Tie-break lost held-out optimality")
            tie_correct, tie_total = tie_accuracy(trace)
            cells = list(values)
            rows.append({"variant": variant, "scope": "prediction", "case_id": f"test_{item}", "structure_type": record["map_type"], "mae": mean(abs(values[cell] - labels[cell[0]][cell[1]]) for cell in cells), "mse": mean((values[cell] - labels[cell[0]][cell[1]]) ** 2 for cell in cells), "pairwise_ordering_accuracy": correct / max(1, total), "pairwise_pairs": total, "tie_set_ordering_accuracy": tie_correct / max(1, tie_total), "tie_set_decisions": tie_total})
    return rows


def critical_error_counts(grid, start, goal, values, optimal_cost, case_id, structure_type):
    distance_to_goal = compute_distance_to_goal(grid, goal)
    optimal_nodes = optimal_path_nodes(compute_distance_to_goal(grid, start), distance_to_goal, optimal_cost)
    trace = trace_unet_tiebreak_astar(grid, start, goal, values, distance_to_goal, optimal_nodes)
    events = critical_events_for_case({"case_id": case_id, "structured_type": structure_type, "optimal_cost": optimal_cost}, trace, optimal_nodes, WEAK_MARGIN_THRESHOLD)
    return {
        "high_impact_ordering_errors_ge10": sum(event["error_type"] == "A_unet_ordering_error" and int(event["recovery_cost"]) >= 10 for event in events),
        "high_impact_ordering_errors_ge20": sum(event["error_type"] == "A_unet_ordering_error" and int(event["recovery_cost"]) >= 20 for event in events),
    }


def evaluate_search(root, output_dir, variants):
    cases = benchmark_cases(root, "outputs/experiments/results_structured_tiebreak_controls_100.csv", "outputs/experiments/results_random_tiebreak_controls_100.csv")
    rows = []
    for variant in variants:
        model = load_unet_heuristic(checkpoint_path(output_dir, variant))
        for number, source in enumerate(cases, 1):
            grid, start, goal = rebuild_case(source)
            optimal = compute_distance_to_goal(grid, goal)[start[0]][start[1]]
            values = prediction_table(model, grid, goal)
            heuristic = lambda node, unused: values[node]
            case_id = f"{source['analysis_structure']}_seed{source['seed']}"
            for algorithm in ("direct_unet", "unet_tiebreak"):
                result = astar_search(grid, start, goal, heuristic) if algorithm == "direct_unet" else astar_search(grid, start, goal, manhattan_heuristic, secondary_heuristic=heuristic)
                rows.append({"variant": variant, "scope": "search", "case_id": case_id, "structure_type": source["analysis_structure"], "algorithm": algorithm, "expanded_nodes": result["expanded"], "path_cost": result["cost"], "optimal_cost": optimal, "optimal": result["cost"] == optimal})
            rows.append({"variant": variant, "scope": "high_impact_ordering", "case_id": case_id, "structure_type": source["analysis_structure"], **critical_error_counts(grid, start, goal, values, optimal, case_id, source["analysis_structure"])})
            if number % 100 == 0:
                print(f"{variant} evaluation {number}/{len(cases)}")
    return rows


def aggregate_prediction(rows, variant):
    scoped = [row for row in rows if row["variant"] == variant]
    return {"mae": mean(row["mae"] for row in scoped), "mse": mean(row["mse"] for row in scoped), "pairwise_ordering_accuracy": sum(row["pairwise_ordering_accuracy"] * row["pairwise_pairs"] for row in scoped) / max(1, sum(row["pairwise_pairs"] for row in scoped)), "tie_set_ordering_accuracy": sum(row["tie_set_ordering_accuracy"] * row["tie_set_decisions"] for row in scoped) / max(1, sum(row["tie_set_decisions"] for row in scoped))}


def aggregate_search(rows, variant, algorithm):
    scoped = [row for row in rows if row["variant"] == variant and row["algorithm"] == algorithm]
    return {"mean_expanded_nodes": mean(row["expanded_nodes"] for row in scoped), "optimality_rate": mean(float(row["optimal"]) for row in scoped)}


def report(output_dir, prediction_rows, search_rows, critical_rows, metadata):
    summary = []
    for variant in MODES:
        prediction = aggregate_prediction(prediction_rows, variant)
        direct = aggregate_search(search_rows, variant, "direct_unet")
        tie = aggregate_search(search_rows, variant, "unet_tiebreak")
        critical = [row for row in critical_rows if row["variant"] == variant]
        summary.append({"variant": variant, "scope": "summary", **prediction, "direct_unet_mean_expanded_nodes": direct["mean_expanded_nodes"], "direct_unet_optimality_rate": direct["optimality_rate"], "unet_tiebreak_mean_expanded_nodes": tie["mean_expanded_nodes"], "unet_tiebreak_optimality_rate": tie["optimality_rate"], "high_impact_ordering_errors_ge10": sum(row["high_impact_ordering_errors_ge10"] for row in critical), "high_impact_ordering_errors_ge20": sum(row["high_impact_ordering_errors_ge20"] for row in critical)})
    write_csv(os.path.join(output_dir, "results.csv"), summary + prediction_rows + search_rows + critical_rows)
    by_variant = {row["variant"]: row for row in summary}
    random, weighted = by_variant["random"], by_variant["critical_weighted"]
    lines = ["# Critical-Decision Weighted Ranking Loss", "", "Both variants use the same 5,000-map split, U-Net, Adam optimizer, batch size, epochs, and A* evaluation cases. The weighted variant keeps all 512 random pairs and appends up to 64 baseline-U-Net critical comparisons per map with weight 5.", "", "| Variant | MAE | MSE | Pair ordering | Tie-set ordering | Direct expanded | Direct optimality | Tie-break expanded | Tie-break optimality | Errors >=10 | Errors >=20 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['variant']} | {row['mae']:.3f} | {row['mse']:.3f} | {row['pairwise_ordering_accuracy']:.3f} | {row['tie_set_ordering_accuracy']:.3f} | {row['direct_unet_mean_expanded_nodes']:.2f} | {row['direct_unet_optimality_rate']:.4f} | {row['unet_tiebreak_mean_expanded_nodes']:.2f} | {row['unet_tiebreak_optimality_rate']:.4f} | {row['high_impact_ordering_errors_ge10']} | {row['high_impact_ordering_errors_ge20']} |")
    lines += ["", "## Interpretation", "", f"Critical weighting changed U-Net tie-break mean expansions from {random['unet_tiebreak_mean_expanded_nodes']:.2f} to {weighted['unet_tiebreak_mean_expanded_nodes']:.2f}, and high-impact ordering errors (recovery >= 10) from {random['high_impact_ordering_errors_ge10']} to {weighted['high_impact_ordering_errors_ge10']}.", "High-impact counts are measured on the Manhattan + U-Net tie-break trace because the weighted supervision targets its ordering decisions. Critical labels are mined once from the source checkpoint rather than refreshed as training changes the model. These results are observational and do not establish causal effects.", "", "## Critical-Pair Source", "", f"- Source checkpoint: `{BASELINE_CRITICAL_CHECKPOINT}`", f"- Training maps with at least one critical pair: {metadata['critical_weighted']['maps_with_critical_pairs_train']} / {metadata['critical_weighted']['train_maps']}", f"- Mean retained critical pairs per training map: {metadata['critical_weighted']['mean_critical_pairs_train']:.3f}", f"- Critical-pair weight: {CRITICAL_WEIGHT}", ""]
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Critical-decision weighted ranking-loss experiment.")
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--output-dir", default="outputs/critical_weighted_ranking")
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    variants = MODES if args.mode == "all" else (args.mode,)
    metadata = {}
    if not args.evaluate:
        source_path = os.path.join(root, BASELINE_CRITICAL_CHECKPOINT)
        baseline_model = load_unet_heuristic(source_path)
        all_history = []
        for variant in variants:
            history, metadata[variant] = train_variant(root, output_dir, variant, baseline_model)
            all_history.extend(history)
        write_csv(os.path.join(output_dir, "training_history.csv"), all_history)
        with open(os.path.join(output_dir, "training_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    else:
        with open(os.path.join(output_dir, "training_metadata.json"), encoding="utf-8") as handle:
            metadata = json.load(handle)
    available = [variant for variant in MODES if os.path.exists(checkpoint_path(output_dir, variant))]
    if args.evaluate or args.mode == "all":
        if set(available) != set(MODES):
            raise FileNotFoundError("Evaluation requires both trained checkpoints.")
        prediction_rows = evaluate_prediction(root, output_dir, available)
        evaluation_rows = evaluate_search(root, output_dir, available)
        search_rows = [row for row in evaluation_rows if row["scope"] == "search"]
        critical_rows = [row for row in evaluation_rows if row["scope"] == "high_impact_ordering"]
        report(output_dir, prediction_rows, search_rows, critical_rows, metadata)


if __name__ == "__main__":
    main()
