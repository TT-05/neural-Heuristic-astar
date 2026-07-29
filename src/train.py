import os
import random

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("PyTorch is required for training. Install torch before running train.py.") from exc

from dataset import build_samples_from_grid
from gen_map import gen_map
from model import MLPHeuristic, heuristic_features


class HeuristicDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        features = heuristic_features(sample["current"], sample["goal"])
        label = sample["distance_to_goal"]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def choose_free_goal(grid):
    free_cells = []
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == 0:
                free_cells.append((r, c))
    if not free_cells:
        raise ValueError("Generated grid has no free cells.")
    return random.choice(free_cells)


def build_training_samples(num_maps=50, width=20, height=20, obstacle_rate=0.2, seed=0):
    random.seed(seed)
    all_samples = []

    for map_index in range(num_maps):
        grid = gen_map(width, height, seed=seed + map_index, obstacle_rate=obstacle_rate)
        goal = choose_free_goal(grid)
        samples = build_samples_from_grid(grid, goal)
        all_samples.extend(samples)

    if not all_samples:
        raise ValueError("No training samples were generated.")

    return all_samples


def train():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(project_root, "checkpoints", "mlp_heuristic.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    samples = build_training_samples()
    dataset = HeuristicDataset(samples)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = MLPHeuristic()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for features, labels in loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d}/{epochs} - train_mse: {avg_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_features": "[abs(current_row - goal_row), abs(current_col - goal_col)]",
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    import sys

    if "--ranking_lambda" in sys.argv:
        from run_ranking_lambda_sweep import main as run_ranking_lambda_sweep

        run_ranking_lambda_sweep()
    else:
        train()
