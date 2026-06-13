import torch
from torch import nn


class MLPHeuristic(nn.Module):
    """Small MLP that predicts distance-to-goal from row/column deltas."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


class UNetHeuristic(nn.Module):
    """Small U-Net that predicts a distance-to-goal map from grid and goal maps."""

    def __init__(self, in_channels=2):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.up2(bottleneck)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return self.output(dec1).squeeze(1)


def heuristic_features(current, goal):
    current_r, current_c = current
    goal_r, goal_c = goal
    return [abs(current_r - goal_r), abs(current_c - goal_c)]


def manhattan_heuristic(current, goal):
    dr, dc = heuristic_features(current, goal)
    return dr + dc


def grid_goal_tensor(grid, goal, device="cpu"):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    obstacle = torch.zeros((rows, cols), dtype=torch.float32, device=device)
    goal_map = torch.zeros((rows, cols), dtype=torch.float32, device=device)

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            obstacle[r, c] = float(value)

    goal_map[goal[0], goal[1]] = 1.0
    return torch.stack([obstacle, goal_map], dim=0)


def distance_normalizer_for_grid(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    return float(rows + cols)


def load_mlp_heuristic(checkpoint_path, device="cpu"):
    model = MLPHeuristic().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_unet_heuristic(checkpoint_path, device="cpu"):
    model = UNetHeuristic().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def make_mlp_heuristic(model, device="cpu"):
    def heuristic(current, goal):
        features = torch.tensor([heuristic_features(current, goal)], dtype=torch.float32, device=device)
        with torch.no_grad():
            prediction = model(features).item()
        return max(0.0, prediction)

    return heuristic


def make_mlp_table_heuristic(model, grid, goal, device="cpu"):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    features = []

    for r in range(rows):
        for c in range(cols):
            features.append(heuristic_features((r, c), goal))

    feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    with torch.no_grad():
        predictions = model(feature_tensor).cpu()

    prediction_grid = predictions.reshape(rows, cols)

    def heuristic(current, unused_goal):
        return max(0.0, float(prediction_grid[current[0], current[1]]))

    return heuristic


def make_unet_heuristic(model, grid, goal, device="cpu"):
    model_input = grid_goal_tensor(grid, goal, device=device).unsqueeze(0)
    normalizer = distance_normalizer_for_grid(grid)
    with torch.no_grad():
        prediction_grid = model(model_input).squeeze(0).cpu() * normalizer

    def heuristic(current, unused_goal):
        return max(0.0, float(prediction_grid[current[0], current[1]]))

    return heuristic
