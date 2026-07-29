import argparse
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from bfs_label import compute_distance_to_goal
from gen_map import gen_map
from model import UNetHeuristic, distance_normalizer_for_grid, grid_goal_tensor
from train import choose_free_goal


NUM_MAPS = 500
EPOCHS = 50
BATCH_SIZE = 16
WIDTH = 20
HEIGHT = 20
OBSTACLE_RATE = 0.2
SEED = 1000
SMOOTH_L1_BETA = 0.1

USE_DISTANCE_LOSS = True
USE_GOAL_ANCHOR_LOSS = False
USE_OVERESTIMATION_PENALTY = False
USE_CONSISTENCY_REGULARIZATION = False

GOAL_ANCHOR_WEIGHT = 1.0
OVERESTIMATION_WEIGHT = 1.0
CONSISTENCY_WEIGHT = 1.0


class UNetHeuristicDataset(Dataset):
    def __init__(self, num_maps=NUM_MAPS, width=WIDTH, height=HEIGHT, obstacle_rate=OBSTACLE_RATE, seed=SEED):
        self.examples = []
        random.seed(seed)

        for map_index in range(num_maps):
            grid = gen_map(width, height, seed=seed + map_index, obstacle_rate=obstacle_rate)
            goal = choose_free_goal(grid)
            distance_grid = compute_distance_to_goal(grid, goal)
            self.examples.append((grid, goal, distance_grid))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        grid, goal, distance_grid = self.examples[index]
        model_input = grid_goal_tensor(grid, goal)

        target = torch.tensor(distance_grid, dtype=torch.float32)
        mask = (target >= 0).float()
        target = torch.clamp(target, min=0.0) / distance_normalizer_for_grid(grid)

        return model_input, target, mask, torch.tensor(goal, dtype=torch.long)


def masked_mse_loss(predictions, targets, mask):
    squared_error = (predictions - targets) ** 2
    masked_error = squared_error * mask
    return masked_error.sum() / mask.sum().clamp(min=1.0)


def masked_smooth_l1_loss(predictions, targets, mask, beta=SMOOTH_L1_BETA):
    loss = F.smooth_l1_loss(predictions, targets, beta=beta, reduction="none")
    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum().clamp(min=1.0)


def get_loss_fn(loss_name):
    if loss_name == "mse":
        return masked_mse_loss
    if loss_name == "smooth_l1":
        return masked_smooth_l1_loss
    raise ValueError(f"Unsupported loss: {loss_name}")


def goal_anchor_loss(predictions, goals):
    batch_indices = torch.arange(predictions.size(0), device=predictions.device)
    goal_rows = goals[:, 0].to(predictions.device)
    goal_cols = goals[:, 1].to(predictions.device)
    goal_predictions = predictions[batch_indices, goal_rows, goal_cols]
    return torch.mean(goal_predictions ** 2)


def overestimation_penalty(predictions, targets, mask):
    penalty = F.relu(predictions - targets) * mask
    return penalty.sum() / mask.sum().clamp(min=1.0)


def consistency_regularization(predictions, mask):
    normalizer = float(WIDTH + HEIGHT)
    horizontal_mask = mask[:, :, 1:] * mask[:, :, :-1]
    vertical_mask = mask[:, 1:, :] * mask[:, :-1, :]

    horizontal_diff = torch.abs(predictions[:, :, 1:] - predictions[:, :, :-1])
    vertical_diff = torch.abs(predictions[:, 1:, :] - predictions[:, :-1, :])

    horizontal_penalty = F.relu(horizontal_diff - 1.0 / normalizer) * horizontal_mask
    vertical_penalty = F.relu(vertical_diff - 1.0 / normalizer) * vertical_mask

    total_penalty = horizontal_penalty.sum() + vertical_penalty.sum()
    total_count = horizontal_mask.sum() + vertical_mask.sum()
    return total_penalty / total_count.clamp(min=1.0)


def compute_loss(predictions, targets, mask, goals, distance_loss_fn):
    loss = predictions.sum() * 0.0

    if USE_DISTANCE_LOSS:
        loss = loss + distance_loss_fn(predictions, targets, mask)

    if USE_GOAL_ANCHOR_LOSS:
        loss = loss + GOAL_ANCHOR_WEIGHT * goal_anchor_loss(predictions, goals)

    if USE_OVERESTIMATION_PENALTY:
        loss = loss + OVERESTIMATION_WEIGHT * overestimation_penalty(predictions, targets, mask)

    if USE_CONSISTENCY_REGULARIZATION:
        loss = loss + CONSISTENCY_WEIGHT * consistency_regularization(predictions, mask)

    return loss


def prediction_metrics(predictions, targets, mask):
    valid_count = mask.sum().clamp(min=1.0)
    error = (predictions - targets) * mask
    abs_error = torch.abs(error)
    squared_error = error ** 2
    overestimate_count = ((predictions > targets) * mask.bool()).sum()

    return {
        "mae": abs_error.sum().item() / valid_count.item(),
        "mse": squared_error.sum().item() / valid_count.item(),
        "overestimate_rate": overestimate_count.item() / valid_count.item(),
        "valid_count": valid_count.item(),
    }


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    total_abs_error = 0.0
    total_squared_error = 0.0
    total_overestimates = 0.0
    total_valid = 0.0

    for model_input, target, mask, goals in loader:
        if training:
            optimizer.zero_grad()

        predictions = model(model_input)
        loss = compute_loss(predictions, target, mask, goals, loss_fn)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * model_input.size(0)
        total_examples += model_input.size(0)
        metrics = prediction_metrics(predictions.detach(), target, mask)
        total_abs_error += metrics["mae"] * metrics["valid_count"]
        total_squared_error += metrics["mse"] * metrics["valid_count"]
        total_overestimates += metrics["overestimate_rate"] * metrics["valid_count"]
        total_valid += metrics["valid_count"]

    return {
        "loss": total_loss / total_examples,
        "mae": total_abs_error / max(total_valid, 1.0),
        "mse": total_squared_error / max(total_valid, 1.0),
        "overestimate_rate": total_overestimates / max(total_valid, 1.0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the U-Net heuristic model.")
    parser.add_argument(
        "--loss",
        choices=["mse", "smooth_l1"],
        default="mse",
        help="Loss function for distance-map regression.",
    )
    return parser.parse_args()


def set_global_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def checkpoint_payload(model, epoch, val_stats, args, train_size, val_size):
    return {
        "model_state_dict": model.state_dict(),
        "input_channels": ["obstacle_map", "goal_map"],
        "output": "normalized predicted distance-to-goal map",
        "normalizer": "rows + cols",
        "epoch": epoch,
        "validation_loss": val_stats["loss"],
        "validation_mae": val_stats["mae"],
        "validation_mse": val_stats["mse"],
        "validation_overestimate_rate": val_stats["overestimate_rate"],
        "train_examples": train_size,
        "val_examples": val_size,
        "num_maps": NUM_MAPS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "width": WIDTH,
        "height": HEIGHT,
        "obstacle_rate": OBSTACLE_RATE,
        "seed": SEED,
        "loss": args.loss,
        "smooth_l1_beta": SMOOTH_L1_BETA if args.loss == "smooth_l1" else None,
        "loss_components": {
            "distance": USE_DISTANCE_LOSS,
            "goal_anchor": USE_GOAL_ANCHOR_LOSS,
            "overestimation_penalty": USE_OVERESTIMATION_PENALTY,
            "consistency_regularization": USE_CONSISTENCY_REGULARIZATION,
        },
        "loss_weights": {
            "goal_anchor": GOAL_ANCHOR_WEIGHT,
            "overestimation": OVERESTIMATION_WEIGHT,
            "consistency": CONSISTENCY_WEIGHT,
        },
    }


def save_checkpoint(path, model, epoch, val_stats, args, train_size, val_size):
    torch.save(
        checkpoint_payload(model, epoch, val_stats, args, train_size, val_size),
        path,
    )


def train_unet():
    args = parse_args()
    set_global_seed(SEED)
    loss_fn = get_loss_fn(args.loss)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    latest_checkpoint_path = os.path.join(checkpoints_dir, "unet_heuristic_latest.pt")
    best_checkpoint_path = os.path.join(checkpoints_dir, "unet_heuristic_best.pt")
    compatible_checkpoint_path = os.path.join(checkpoints_dir, "unet_heuristic.pt")
    os.makedirs(checkpoints_dir, exist_ok=True)

    dataset = UNetHeuristicDataset()
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=train_generator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNetHeuristic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        train_stats = run_epoch(model, train_loader, loss_fn, optimizer=optimizer)
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, loss_fn)

        save_checkpoint(latest_checkpoint_path, model, epoch, val_stats, args, train_size, val_size)
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            save_checkpoint(best_checkpoint_path, model, epoch, val_stats, args, train_size, val_size)
            save_checkpoint(compatible_checkpoint_path, model, epoch, val_stats, args, train_size, val_size)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} - "
            f"loss: {args.loss} - "
            f"train_loss: {train_stats['loss']:.6f} - "
            f"val_loss: {val_stats['loss']:.6f} - "
            f"val_mae: {val_stats['mae']:.6f} - "
            f"val_mse: {val_stats['mse']:.6f} - "
            f"val_overestimate_rate: {val_stats['overestimate_rate']:.4f}"
        )

    print(f"Saved latest checkpoint to {latest_checkpoint_path}")
    print(f"Saved best checkpoint to {best_checkpoint_path}")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Compatible checkpoint for evaluation: {compatible_checkpoint_path}")


if __name__ == "__main__":
    # The original 500-map trainer remains the no-argument/default path.  The
    # named loss-ablation modes explicitly select the fixed 5,000-map framework.
    import sys

    if "--ranking_mode" in sys.argv:
        from run_ranking_ablation import main as run_ranking_ablation

        run_ranking_ablation()
        raise SystemExit

    if "--loss_mode" in sys.argv:
        from run_combined_loss_ablation import main as run_combined_loss_ablation

        run_combined_loss_ablation()
        raise SystemExit

    ablation_modes = {"mse", "ranking", "consistency", "combined", "ranking_consistency", "all"}
    requested = sys.argv[sys.argv.index("--loss") + 1] if "--loss" in sys.argv and sys.argv.index("--loss") + 1 < len(sys.argv) else None
    if requested in ablation_modes:
        from run_loss_ablation import main as run_loss_ablation

        run_loss_ablation()
    else:
        train_unet()
