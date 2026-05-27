import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bfs_label import compute_distance_to_goal
from gen_map import gen_map
from model import UNetHeuristic, grid_goal_tensor
from train import choose_free_goal


class UNetHeuristicDataset(Dataset):
    def __init__(self, num_maps=100, width=20, height=20, obstacle_rate=0.2, seed=1000):
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
        target = torch.clamp(target, min=0.0)

        return model_input, target, mask


def masked_mse_loss(predictions, targets, mask):
    squared_error = (predictions - targets) ** 2
    masked_error = squared_error * mask
    return masked_error.sum() / mask.sum().clamp(min=1.0)


def train_unet():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(project_root, "checkpoints", "unet_heuristic.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    dataset = UNetHeuristicDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNetHeuristic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for model_input, target, mask in loader:
            optimizer.zero_grad()
            predictions = model(model_input)
            loss = masked_mse_loss(predictions, target, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * model_input.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d}/{epochs} - train_masked_mse: {avg_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_channels": ["obstacle_map", "goal_map"],
            "output": "predicted distance-to-goal map",
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    train_unet()
