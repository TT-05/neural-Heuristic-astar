import os
import random

import torch
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

        return model_input, target, mask


def masked_mse_loss(predictions, targets, mask):
    squared_error = (predictions - targets) ** 2
    masked_error = squared_error * mask
    return masked_error.sum() / mask.sum().clamp(min=1.0)


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0

    for model_input, target, mask in loader:
        if training:
            optimizer.zero_grad()

        predictions = model(model_input)
        loss = masked_mse_loss(predictions, target, mask)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * model_input.size(0)
        total_examples += model_input.size(0)

    return total_loss / total_examples


def train_unet():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(project_root, "checkpoints", "unet_heuristic.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    dataset = UNetHeuristicDataset()
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(0)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNetHeuristic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer=optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} - "
            f"train_masked_mse: {train_loss:.6f} - val_masked_mse: {val_loss:.6f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_channels": ["obstacle_map", "goal_map"],
            "output": "normalized predicted distance-to-goal map",
            "normalizer": "rows + cols",
            "train_examples": train_size,
            "val_examples": val_size,
            "num_maps": NUM_MAPS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "width": WIDTH,
            "height": HEIGHT,
            "obstacle_rate": OBSTACLE_RATE,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    train_unet()
