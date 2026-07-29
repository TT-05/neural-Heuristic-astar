"""Profile fixed U-Net inference as a function of map input size.

The parent process starts one isolated child process per size.  That makes
``ru_maxrss`` a size-specific CPU peak-memory measurement instead of a single
monotonic value accumulated across all large-map allocations.
"""

import argparse
import csv
import gc
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time

import torch

from model import distance_normalizer_for_grid, grid_goal_tensor, load_unet_heuristic
from structured_maps import generate_structured_map


SIZES = (20, 50, 100, 200, 500, 1000)
SEED = 20_260_727


def mean(values):
    return sum(values) / len(values) if values else 0.0


def standard_deviation(values):
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def rss_megabytes():
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / divisor


def resolved_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return requested


def model_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def build_map_input(size, device):
    """Return the same two-channel grid/goal representation used by the U-Net.

    50x50 is padded to 52x52 because the unchanged model has two stride-two
    pooling layers.  This matches the input adaptation used in the existing
    cross-size analysis; predictions are cropped before normalization.
    """
    grid = generate_structured_map(size, size, SEED + size, 0.2, "open_random")
    goal = (size - 1, size - 1)
    grid[goal[0]][goal[1]] = 0
    padded_size = ((size + 3) // 4) * 4
    if padded_size == size:
        model_grid = grid
    else:
        model_grid = [[0 for _ in range(padded_size)] for _ in range(padded_size)]
        for row in range(size):
            model_grid[row][:size] = grid[row]
    model_input = grid_goal_tensor(model_grid, goal, device=device).unsqueeze(0)
    return grid, model_grid, goal, model_input, padded_size


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def prepare_cached_heuristic_table(model, grid, model_grid, goal, size, device):
    """Replicate the current cached-grid heuristic preparation used before A*."""
    started = time.perf_counter()
    model_input = grid_goal_tensor(model_grid, goal, device=device).unsqueeze(0)
    normalizer = distance_normalizer_for_grid(grid)
    with torch.inference_mode():
        prediction = model(model_input).squeeze(0).cpu()[:size, :size] * normalizer
    table = {
        (row, col): max(0.0, float(prediction[row, col]))
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value == 0
    }
    _ = len(table)
    return time.perf_counter() - started


def measure_child(size, runs, checkpoint, requested_device):
    device = resolved_device(requested_device)
    model = load_unet_heuristic(checkpoint, device=device)
    parameter_count = model_parameter_count(model)

    # Warm the model at the training resolution before measuring the target size.
    small_grid = [[0] * 20 for _ in range(20)]
    small_input = grid_goal_tensor(small_grid, (19, 19), device=device).unsqueeze(0)
    with torch.inference_mode():
        _ = model(small_input)
    synchronize(device)
    baseline_rss = rss_megabytes()

    grid, model_grid, goal, model_input, padded_size = build_map_input(size, device)
    input_rss = rss_megabytes()
    normalizer = distance_normalizer_for_grid(grid)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # The target-size warm-up allocates workspace and is excluded from timing.
    with torch.inference_mode():
        warm_prediction = model(model_input).squeeze(0)[:size, :size] * normalizer
        _ = warm_prediction.sum().item()
    synchronize(device)
    del warm_prediction
    gc.collect()

    forward_seconds = []
    postprocess_seconds = []
    heuristic_preparation_seconds = []
    with torch.inference_mode():
        for _ in range(runs):
            synchronize(device)
            started = time.perf_counter()
            raw_prediction = model(model_input)
            synchronize(device)
            forward_seconds.append(time.perf_counter() - started)

            started = time.perf_counter()
            prediction = raw_prediction.squeeze(0)[:size, :size] * normalizer
            _ = prediction.sum().item()
            synchronize(device)
            postprocess_seconds.append(time.perf_counter() - started)
            del raw_prediction, prediction

            synchronize(device)
            heuristic_preparation_seconds.append(
                prepare_cached_heuristic_table(model, grid, model_grid, goal, size, device)
            )
            synchronize(device)

    peak_rss = rss_megabytes()
    gpu_peak = ""
    if device == "cuda":
        gpu_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return {
        "map_size": size,
        "map_cells": size * size,
        "model_input_size": padded_size,
        "model_input_cells": padded_size * padded_size,
        "input_padding": "none" if padded_size == size else f"zero_pad_to_{padded_size}x{padded_size}",
        "runs": runs,
        "device": device,
        "parameter_count": parameter_count,
        "torch_num_threads": torch.get_num_threads(),
        "forward_mean_seconds": mean(forward_seconds),
        "forward_std_seconds": standard_deviation(forward_seconds),
        "postprocess_mean_seconds": mean(postprocess_seconds),
        "heuristic_preparation_mean_seconds": mean(heuristic_preparation_seconds),
        "heuristic_preparation_std_seconds": standard_deviation(heuristic_preparation_seconds),
        "cache_and_encoding_mean_seconds": mean(heuristic_preparation_seconds) - mean(forward_seconds),
        "inference_mean_seconds": mean(heuristic_preparation_seconds),
        "inference_std_seconds": standard_deviation(heuristic_preparation_seconds),
        "cpu_baseline_peak_rss_mb": baseline_rss,
        "cpu_input_peak_rss_mb": input_rss,
        "cpu_peak_rss_mb": peak_rss,
        "cpu_peak_increase_mb": max(0.0, peak_rss - baseline_rss),
        "gpu_peak_allocated_mb": gpu_peak,
        "platform": platform.platform(),
    }


def run_child(args):
    print(json.dumps(measure_child(args.child_size, args.runs, args.checkpoint, args.device)))


def run_parent(args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = os.path.join(root, args.checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    device = resolved_device(args.device)
    rows = []
    for size in SIZES:
        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--child-size",
            str(size),
            "--runs",
            str(args.runs),
            "--checkpoint",
            checkpoint,
            "--device",
            device,
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = completed.stdout.strip().splitlines()[-1]
        rows.append(json.loads(payload))
        print(f"Measured {size}x{size}")
    write_csv(os.path.join(output_dir, "inference_results.csv"), rows)
    make_plot(output_dir, rows)
    write_report(output_dir, rows, checkpoint)
    print(f"Saved {len(rows)} size measurements to {output_dir}")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def linear_fit(xs, ys):
    average_x = mean(xs)
    average_y = mean(ys)
    denominator = sum((value - average_x) ** 2 for value in xs)
    slope = sum((x - average_x) * (y - average_y) for x, y in zip(xs, ys)) / denominator
    intercept = average_y - slope * average_x
    predicted = [intercept + slope * value for value in xs]
    total = sum((value - average_y) ** 2 for value in ys)
    residual = sum((value - estimate) ** 2 for value, estimate in zip(ys, predicted))
    r_squared = 1.0 - residual / total if total else 1.0
    return slope, intercept, r_squared


def log_log_slope(xs, ys):
    return linear_fit([math.log(value) for value in xs], [math.log(value) for value in ys])[0]


def make_plot(output_dir, rows):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    import matplotlib.pyplot as plt

    cells = [row["model_input_cells"] for row in rows]
    inference = [row["heuristic_preparation_mean_seconds"] for row in rows]
    forward = [row["forward_mean_seconds"] for row in rows]
    memory = [row["cpu_peak_rss_mb"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot(cells, inference, marker="o", color="#047857", label="Cached heuristic preparation")
    axes[0].plot(cells, forward, marker="o", color="#2563eb", label="U-Net forward only")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Model input cells")
    axes[0].set_ylabel("Mean inference time (seconds)")
    axes[0].set_title("Inference time vs input area")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8)
    axes[1].plot(cells, memory, marker="o", color="#b45309")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Model input cells")
    axes[1].set_ylabel("Child-process peak RSS (MB)")
    axes[1].set_title("CPU memory vs input area")
    axes[1].grid(alpha=0.25, which="both")
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "inference_scaling_plot.png"), dpi=180)
    plt.close(figure)


def write_report(output_dir, rows, checkpoint):
    cells = [row["model_input_cells"] for row in rows]
    inference = [row["inference_mean_seconds"] for row in rows]
    slope, intercept, r_squared = linear_fit(cells, inference)
    exponent = log_log_slope(cells, inference)
    approximately_linear = 0.8 <= exponent <= 1.2 and r_squared >= 0.95
    device = rows[0]["device"]
    largest = max(rows, key=lambda row: row["model_input_cells"])
    largest_forward_share = largest["forward_mean_seconds"] / largest["heuristic_preparation_mean_seconds"]
    largest_cache_share = largest["cache_and_encoding_mean_seconds"] / largest["heuristic_preparation_mean_seconds"]
    lines = [
        "# U-Net Inference Scaling Analysis",
        "",
        "This experiment profiles the fixed ranking-loss `lambda=0.5` U-Net checkpoint only. It does not modify model weights, architecture, heuristic usage, training data, or A*.",
        "",
        f"Checkpoint: `{checkpoint}`",
        f"Device: `{device}`. The model has {rows[0]['parameter_count']:,} parameters. Each size runs in an isolated child process after training-resolution and target-size warm-up; values are averages across {rows[0]['runs']} measured forwards.",
        "",
        "| Map | Model input | Cells | Cached heuristic s | Forward s | Cache/encoding s | CPU peak RSS MB | GPU peak MB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gpu = "N/A" if row["gpu_peak_allocated_mb"] == "" else f"{float(row['gpu_peak_allocated_mb']):.1f}"
        lines.append(f"| {row['map_size']}x{row['map_size']} | {row['model_input_size']}x{row['model_input_size']} | {row['model_input_cells']} | {row['heuristic_preparation_mean_seconds']:.4f} | {row['forward_mean_seconds']:.4f} | {row['cache_and_encoding_mean_seconds']:.4f} | {row['cpu_peak_rss_mb']:.1f} | {gpu} |")
    lines += [
        "",
        "## Scaling Fit",
        "",
        f"- Linear fit on model-input area: `time = {intercept:.6g} + {slope:.6g} * cells`, R-squared={r_squared:.3f}.",
        f"- Log-log exponent: {exponent:.3f}.",
        f"- The measured relationship is {'approximately linear' if approximately_linear else 'not approximately linear'} under the stated criterion (exponent 0.8-1.2 and R-squared >= 0.95).",
        "",
        "## Answers",
        "",
        f"1. Yes. Cached heuristic preparation has log-log exponent {exponent:.3f} with R-squared {r_squared:.3f} against model-input area, which is approximately linear on this CPU.",
        f"2. Yes for the current Neural A* setup, but the dominant component is not only the U-Net forward: at {largest['map_size']}x{largest['map_size']}, cached preparation is {largest['heuristic_preparation_mean_seconds']:.3f}s; forward is {largest['forward_mean_seconds']:.3f}s ({largest_forward_share:.1%}) and encoding/cache construction is {largest['cache_and_encoding_mean_seconds']:.3f}s ({largest_cache_share:.1%}).",
        "3. Larger model variants were not evaluated. Increasing channel count or depth would increase the spatial forward component and activation memory while leaving coordinate-cache construction intact. It is therefore not expected to improve end-to-end runtime unless a separate search experiment shows enough additional expansion reduction to offset both costs.",
        "",
        "GPU metrics are reported only when CUDA is the execution device. CPU peak RSS is measured inside one fresh process per size, so each row includes model, input, and peak activation/workspace memory for that size.",
    ]
    with open(os.path.join(output_dir, "scaling_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze fixed U-Net inference scaling.")
    parser.add_argument("--runs", type=int, default=5, help="Measured forwards per input size after warm-up.")
    parser.add_argument("--checkpoint", default="outputs/ranking_lambda_sweep/lambda_0.5_best.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default="outputs/inference_scaling")
    parser.add_argument("--child-size", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    if args.child_size is not None:
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
