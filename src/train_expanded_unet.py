"""Train the unchanged U-Net on the expanded, stratified map dataset."""

import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader

from expanded_unet_dataset import ExpandedUNetDataset, generate_expanded_dataset
from model import UNetHeuristic
from train_unet import BATCH_SIZE, EPOCHS, SEED, get_loss_fn, run_epoch, set_global_seed


def write_history(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_payload(model, epoch, train_stats, val_stats, archive_path, manifest_path):
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
        "training_loss": train_stats["loss"],
        "training_mae": train_stats["mae"],
        "dataset_archive": archive_path,
        "dataset_manifest": manifest_path,
        "train_examples": 4000,
        "val_examples": 500,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "loss": "mse",
        "optimizer": "Adam(lr=0.001)",
    }


def save_checkpoint(path, model, epoch, train_stats, val_stats, archive_path, manifest_path):
    torch.save(checkpoint_payload(model, epoch, train_stats, val_stats, archive_path, manifest_path), path)


def train(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output_dir)
    dataset_dir = os.path.join(output_dir, "dataset")
    archive_path = os.path.join(dataset_dir, "expanded_unet_dataset.npz")
    manifest_path = os.path.join(dataset_dir, "dataset_manifest.csv")
    os.makedirs(output_dir, exist_ok=True)
    if not (os.path.exists(archive_path) and os.path.exists(manifest_path)):
        generate_expanded_dataset(dataset_dir, SEED)

    set_global_seed(SEED)
    train_dataset = ExpandedUNetDataset(archive_path, manifest_path, "train")
    val_dataset = ExpandedUNetDataset(archive_path, manifest_path, "val")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = UNetHeuristic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = get_loss_fn("mse")
    history, best_loss = [], float("inf")
    best_path = os.path.join(output_dir, "unet_heuristic_expanded_best.pt")
    latest_path = os.path.join(output_dir, "unet_heuristic_expanded_latest.pt")

    for epoch in range(1, EPOCHS + 1):
        train_stats = run_epoch(model, train_loader, loss_fn, optimizer=optimizer)
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, loss_fn)
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train_stats.items()}, **{f"val_{key}": value for key, value in val_stats.items()}}
        history.append(row)
        save_checkpoint(latest_path, model, epoch, train_stats, val_stats, archive_path, manifest_path)
        if val_stats["loss"] < best_loss:
            best_loss = val_stats["loss"]
            save_checkpoint(best_path, model, epoch, train_stats, val_stats, archive_path, manifest_path)
        print(f"Epoch {epoch:02d}/{EPOCHS}: train_loss={train_stats['loss']:.6f} val_loss={val_stats['loss']:.6f} val_mae={val_stats['mae']:.6f}")

    write_history(os.path.join(output_dir, "training_history.csv"), history)
    print(f"Saved best expanded checkpoint to {best_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train unchanged U-Net on expanded data.")
    parser.add_argument("--output-dir", default="outputs/expanded_dataset_training")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
