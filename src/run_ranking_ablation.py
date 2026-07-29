"""Pair-sampling ablation for fixed-data U-Net ranking loss."""

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from analyze_direct_vs_tiebreak import trace_search
from analyze_unet_structure_behavior import benchmark_cases, free_cells, mean, rebuild_case
from astar import astar_search
from bfs_label import compute_distance_to_goal
from model import UNetHeuristic, load_unet_heuristic, make_unet_heuristic, manhattan_heuristic
from run_loss_ablation import (
    BATCH_SIZE, EPOCHS, PAIR_COUNT, RANK_WEIGHT, SEED, StructuredDistanceDataset,
    aggregate_prediction, aggregate_search, pairwise_accuracy, ranking_loss, sample_pairs,
    tie_accuracy, consistency_metrics, write_csv,
)
from train_unet import set_global_seed


MODES = ("random", "search_aware", "tie_set")


def checkpoint_path(output_dir, mode):
    name = {"random": "random_ranking", "search_aware": "search_aware", "tie_set": "tie_set"}[mode]
    return os.path.join(output_dir, "checkpoints", f"unet_{name}_best.pt")


def masked_mse(prediction, target, mask):
    return (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


def ranking_pairs_from_trace(grid, start, goal, labels, mode):
    """Generate pairs from a Manhattan A* trace; supervision is true distance only."""
    table = {cell: 0.0 for cell in free_cells(grid)}
    trace = trace_search(grid, start, goal, table, labels, set(), "manhattan")
    cols = len(grid[0]); candidates = []
    for event in trace["trace"]:
        current = event["expanded"]
        open_nodes = event["open_before"]
        if mode == "search_aware":
            close = [item for item in open_nodes if abs(item["f"] - current["f"]) <= 1]
            for item in close:
                if item["node"] != current["node"]:
                    candidates.append((current["node"][0] * cols + current["node"][1], item["node"][0] * cols + item["node"][1]))
            for first, second in zip(close[::2], close[1::2]):
                candidates.append((first["node"][0] * cols + first["node"][1], second["node"][0] * cols + second["node"][1]))
        else:
            tied = [item for item in open_nodes if item["f"] == current["f"]]
            for position, first in enumerate(tied):
                for second in tied[position + 1:]:
                    candidates.append((first["node"][0] * cols + first["node"][1], second["node"][0] * cols + second["node"][1]))
    valid = []
    for first, second in candidates:
        a, b = (first // cols, first % cols), (second // cols, second % cols)
        if labels[a[0]][a[1]] >= 0 and labels[b[0]][b[1]] >= 0 and labels[a[0]][a[1]] != labels[b[0]][b[1]]:
            valid.append((first, second))
    if not valid:
        # Degenerate maps contribute no ranking gradient; their MSE remains intact.
        cell = start[0] * cols + start[1]
        return np.tile(np.asarray([[cell, cell]], dtype=np.int64), (PAIR_COUNT, 1)), 0
    rng = np.random.default_rng(SEED + start[0] * 997 + start[1] * 313 + goal[0] * 31 + goal[1])
    chosen = rng.choice(len(valid), size=PAIR_COUNT, replace=len(valid) < PAIR_COUNT)
    return np.asarray([valid[index] for index in chosen], dtype=np.int64), len(valid)


def build_pair_cache(grid, goal, start, labels, mode, item):
    if mode == "random": return sample_pairs(grid, goal, start, labels, SEED + item * 37), PAIR_COUNT
    return ranking_pairs_from_trace(grid, start, goal, labels, mode)


class RankingDataset(Dataset):
    def __init__(self, archive_path, manifest_path, split, mode, cache_dir):
        archive = np.load(archive_path)
        with open(manifest_path, newline="", encoding="utf-8") as handle: manifest = list(csv.DictReader(handle))
        self.records = [row for row in manifest if row["split"] == split]
        self.grids, self.goals, self.labels, self.starts = archive["grids"], archive["goals"], archive["labels"], archive["reference_starts"]
        self.mode, self.pairs, self.pair_counts = mode, {}, {}
        began = time.perf_counter()
        for record in self.records:
            item = int(record["index"]); grid = self.grids[item].tolist(); goal = tuple(int(v) for v in self.goals[item]); start = tuple(int(v) for v in self.starts[item])
            self.pairs[item], self.pair_counts[item] = build_pair_cache(grid, goal, start, self.labels[item], mode, item)
        self.generation_seconds = time.perf_counter() - began
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(os.path.join(cache_dir, f"{split}_{mode}_pairs.npz"), indices=np.asarray([int(r["index"]) for r in self.records]), pairs=np.asarray([self.pairs[int(r["index"])] for r in self.records]), candidate_counts=np.asarray([self.pair_counts[int(r["index"])] for r in self.records]))

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        item = int(self.records[index]["index"]); grid = self.grids[item].tolist(); goal = tuple(int(v) for v in self.goals[item]); start = tuple(int(v) for v in self.starts[item])
        label = self.labels[item].astype(np.float32); mask = (label >= 0).astype(np.float32); target = np.clip(label, 0.0, None) / float(len(grid) + len(grid[0]))
        from model import grid_goal_tensor
        return grid_goal_tensor(grid, goal), torch.from_numpy(target), torch.from_numpy(mask), torch.from_numpy(self.pairs[item])


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None; model.train(training); totals = defaultdict(float); examples = valid = 0.0
    for model_input, target, mask, pairs in loader:
        if training: optimizer.zero_grad()
        prediction = model(model_input); mse = masked_mse(prediction, target, mask); rank = ranking_loss(prediction, target, pairs); loss = mse + RANK_WEIGHT * rank
        if training: loss.backward(); optimizer.step()
        count = model_input.size(0); examples += count; valid += mask.sum().item()
        totals["loss"] += loss.item() * count; totals["mse_loss"] += mse.item() * count; totals["ranking_loss"] += rank.item() * count
        error = (prediction.detach() - target) * mask; totals["mae"] += torch.abs(error).sum().item(); totals["mse"] += (error ** 2).sum().item()
    return {"loss": totals["loss"] / examples, "mse_loss": totals["mse_loss"] / examples, "ranking_loss": totals["ranking_loss"] / examples, "mae": totals["mae"] / valid, "mse": totals["mse"] / valid}


def train_variant(root, output_dir, mode):
    archive = os.path.join(root, "outputs/expanded_dataset_training/dataset/expanded_unet_dataset.npz"); manifest = os.path.join(root, "outputs/expanded_dataset_training/dataset/dataset_manifest.csv")
    set_global_seed(SEED); cache_dir = os.path.join(output_dir, "pair_caches")
    train_set = RankingDataset(archive, manifest, "train", mode, cache_dir); val_set = RankingDataset(archive, manifest, "val", mode, cache_dir)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED)); val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    model = UNetHeuristic(); optimizer = torch.optim.Adam(model.parameters(), lr=0.001); rows = []; best = float("inf"); began = time.perf_counter(); os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        train = run_epoch(model, train_loader, optimizer)
        with torch.no_grad(): val = run_epoch(model, val_loader)
        rows.append({"ranking_mode": mode, "epoch": epoch, **{f"train_{key}": value for key, value in train.items()}, **{f"val_{key}": value for key, value in val.items()}})
        if val["loss"] < best:
            best = val["loss"]
            torch.save({"model_state_dict": model.state_dict(), "loss": "mse_plus_ranking", "ranking_mode": mode, "epoch": epoch, "validation_loss": val["loss"], "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "Adam(lr=0.001)", "ranking_pairs_per_map": PAIR_COUNT}, checkpoint_path(output_dir, mode))
        print(f"{mode} epoch {epoch:02d}/{EPOCHS}: train={train['loss']:.6f} val={val['loss']:.6f}")
    write_csv(os.path.join(output_dir, f"training_history_{mode}.csv"), rows)
    return rows, {"ranking_mode": mode, "train_maps": len(train_set), "validation_maps": len(val_set), "pairs_per_map": PAIR_COUNT, "mean_candidate_pairs_train": mean(train_set.pair_counts.values()), "pair_generation_seconds": train_set.generation_seconds + val_set.generation_seconds, "training_seconds": time.perf_counter() - began}


def table(model, grid, goal):
    heuristic = make_unet_heuristic(model, grid, goal)
    return {cell: float(heuristic(cell, goal)) for cell in free_cells(grid)}


def evaluate_prediction(root, output_dir, modes):
    archive = os.path.join(root, "outputs/expanded_dataset_training/dataset/expanded_unet_dataset.npz"); manifest = os.path.join(root, "outputs/expanded_dataset_training/dataset/dataset_manifest.csv")
    test = StructuredDistanceDataset(archive, manifest, "test"); rows = []
    for mode in modes:
        model = load_unet_heuristic(checkpoint_path(output_dir, mode))
        for record in test.records:
            item = int(record["index"]); grid = test.grids[item].tolist(); goal = tuple(int(v) for v in test.goals[item]); start = tuple(int(v) for v in test.starts[item]); labels = test.labels[item].tolist(); values = table(model, grid, goal); cells = list(values)
            if item not in test.pairs: test.pairs[item] = sample_pairs(grid, goal, start, test.labels[item], SEED + item * 37)
            correct, total = pairwise_accuracy(values, grid, test.pairs[item], labels); trace = trace_search(grid, start, goal, values, labels, set(), "unet_tiebreak")
            if trace["cost"] != labels[start[0]][start[1]]: raise AssertionError("Tie-break lost held-out optimality")
            tied, decisions = tie_accuracy(trace)
            violations, violation_magnitude = consistency_metrics(values, grid, labels)
            rows.append({"variant": mode, "scope": "prediction_test", "case_id": f"test_{item}", "structure_type": record["map_type"], "mae": mean(abs(values[c] - labels[c[0]][c[1]]) for c in cells), "mse": mean((values[c] - labels[c[0]][c[1]]) ** 2 for c in cells), "pairwise_ordering_accuracy": correct / max(1,total), "pairwise_pairs": total, "tie_set_ordering_accuracy": tied / max(1,decisions), "tie_set_decisions": decisions, "consistency_violation_count": violations, "consistency_violation_magnitude": violation_magnitude})
    return rows


def evaluate_search(root, output_dir, modes):
    cases = benchmark_cases(root, "outputs/experiments/results_structured_tiebreak_controls_100.csv", "outputs/experiments/results_random_tiebreak_controls_100.csv"); rows=[]
    for mode in modes:
        model = load_unet_heuristic(checkpoint_path(output_dir, mode))
        for number, source in enumerate(cases, 1):
            grid,start,goal=rebuild_case(source); optimal=compute_distance_to_goal(grid, goal)[start[0]][start[1]]; values=table(model,grid,goal); h=lambda node,unused: values[node]
            for algorithm in ("direct_unet","unet_tiebreak"):
                result=astar_search(grid,start,goal,h) if algorithm=="direct_unet" else astar_search(grid,start,goal,manhattan_heuristic,secondary_heuristic=h)
                rows.append({"variant":mode,"scope":"search","case_id":f"{source['analysis_structure']}_seed{source['seed']}","structure_type":source["analysis_structure"],"algorithm":algorithm,"expanded_nodes":result["expanded"],"path_cost":result["cost"],"optimal_cost":optimal,"optimal":result["cost"]==optimal})
            if number%100==0: print(f"{mode} search {number}/{len(cases)}")
    return rows


def write_summary(output_dir,prediction,search,times):
    overall_p=aggregate_prediction(prediction,["variant"]); overall_s=aggregate_search(search,["variant","algorithm"]); structure=aggregate_prediction(prediction,["variant","structure_type"])+aggregate_search(search,["variant","algorithm","structure_type"])
    write_csv(os.path.join(output_dir,"results.csv"),prediction+search); write_csv(os.path.join(output_dir,"structure_results.csv"),structure); write_csv(os.path.join(output_dir,"training_time.csv"),times)
    lookup_p={r["variant"]:r for r in overall_p}; lookup_s={(r["variant"],r["algorithm"]):r for r in overall_s}
    lines=["# Ranking Pair-Sampling Ablation","","All variants use `L = MSE + 0.5 * ranking hinge loss` with the same fixed 5,000-map dataset, U-Net, Adam optimizer, batch size, epochs, and A* benchmark.","","| Mode | MAE | MSE | Pair ordering | Tie-set ordering | Direct optimality | Direct expanded | Tie-break expanded |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        p=lookup_p[mode]; direct=lookup_s[(mode,"direct_unet")]; tie=lookup_s[(mode,"unet_tiebreak")]
        lines.append(f"| {mode} | {p['mae']:.3f} | {p['mse']:.3f} | {p['pairwise_ordering_accuracy']:.3f} | {p['tie_set_ordering_accuracy']:.3f} | {direct['optimality_rate']:.4f} | {direct['mean_expanded_nodes']:.2f} | {tie['mean_expanded_nodes']:.2f} |")
    random_tie=lookup_s[("random","unet_tiebreak")]; aware_tie=lookup_s[("search_aware","unet_tiebreak")]; tied_tie=lookup_s[("tie_set","unet_tiebreak")]
    random_direct=lookup_s[("random","direct_unet")]; aware_direct=lookup_s[("search_aware","direct_unet")]; tied_direct=lookup_s[("tie_set","direct_unet")]
    lines += [
        "", "## Answer", "",
        "On this fixed benchmark, neither search-aware nor tie-set-only sampling improved A* performance over generic random ranking pairs.",
        f"For U-Net tie-breaking, mean expansions increased from {random_tie['mean_expanded_nodes']:.2f} (random) to {aware_tie['mean_expanded_nodes']:.2f} (search-aware) and {tied_tie['mean_expanded_nodes']:.2f} (tie-set-only).",
        f"For Direct U-Net A*, mean expansions increased from {random_direct['mean_expanded_nodes']:.2f} to {aware_direct['mean_expanded_nodes']:.2f} and {tied_direct['mean_expanded_nodes']:.2f}; optimality fell from {random_direct['optimality_rate']:.4f} to {aware_direct['optimality_rate']:.4f} and {tied_direct['optimality_rate']:.4f}.",
        "Random sampling also had the highest held-out pairwise and tie-set ordering accuracy. These are measured associations, not a causal conclusion about all search-aware sampling designs.",
        "Training and pair-generation timing was not persisted before the prior interruption; `training_time.csv` marks this explicitly while retaining the recovered per-map pair statistics.", ""]
    with open(os.path.join(output_dir,"summary.md"),"w",encoding="utf-8") as h:h.write("\n".join(lines))


def parse_args():
    parser=argparse.ArgumentParser(description="Ranking pair-sampling ablation."); parser.add_argument("--ranking_mode",choices=[*MODES,"all"],default="all"); parser.add_argument("--evaluate",action="store_true"); parser.add_argument("--output-dir",default="outputs/ranking_ablation"); return parser.parse_args()


def main():
    args=parse_args(); root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); output=os.path.join(root,args.output_dir); os.makedirs(output,exist_ok=True); modes=MODES if args.ranking_mode=="all" else (args.ranking_mode,); histories=[]; times=[]
    if not args.evaluate:
        for mode in modes:
            history, timing=train_variant(root,output,mode); histories.extend(history); times.append(timing)
    if histories:
        write_csv(os.path.join(output,"training_history.csv"),histories)
        with open(os.path.join(output,"final_configuration.json"),"w",encoding="utf-8") as h:
            json.dump({"loss":"mse + 0.5 * ranking_hinge","ranking_modes":list(MODES),"pairs_per_map":PAIR_COUNT,"optimizer":"Adam(lr=0.001)","batch_size":BATCH_SIZE,"epochs":EPOCHS},h,indent=2)
    available=[mode for mode in MODES if os.path.exists(checkpoint_path(output,mode))]
    if args.evaluate or args.ranking_mode=="all":
        if set(available)!=set(MODES):raise FileNotFoundError("Evaluation requires all three ranking checkpoints")
        if not times:
            # Earlier interrupted runs may have completed training before writing
            # timing metadata. Preserve that distinction rather than estimating it.
            times=[]
            for mode in available:
                cache=np.load(os.path.join(output,"pair_caches",f"train_{mode}_pairs.npz"))
                times.append({"ranking_mode":mode,"pairs_per_map":PAIR_COUNT,"mean_candidate_pairs_train":float(cache["candidate_counts"].mean()),"training_seconds":None,"pair_generation_seconds":None,"measurement_status":"not recorded before interruption"})
        prediction=evaluate_prediction(root,output,available); search=evaluate_search(root,output,available); write_summary(output,prediction,search,times)


if __name__=="__main__":main()
