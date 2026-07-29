import argparse
import csv
import math
import os
import re
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from analyze_tie_set_counterfactual_penalty import result_expanded
from analyze_tie_set_ordering import build_grid, checkpoint_path, group_maps, pearson, prediction_table, read_csv, spearman
from analyze_tie_set_weighted_ordering import reconstruct_optimal_path
from bfs_label import compute_distance_to_goal
from generate_maze_case_studies import case_identity, simulate_manhattan_expansion, simulate_secondary_expansion
from model import load_mlp_heuristic, load_unet_heuristic, make_mlp_table_heuristic, make_unet_heuristic


METHODS = ["manhattan", "mlp", "unet"]
FRACTIONS = [("all", 1.0), ("first25", 0.25), ("first50", 0.50)]
CORRELATION_METRICS = [
    "mean_distance_to_path_gap",
    "off_path_ge2_count_gap",
    "off_path_ge3_count_gap",
    "off_path_ge2_fraction_gap",
    "first25_off_path_ge2_fraction_gap",
    "first50_off_path_ge2_fraction_gap",
    "frontier_compactness_gap",
    "bounding_box_area_gap",
    "first_disagreement_step",
    "disagreement_count",
]


def write_csv(path, rows):
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def read_docx_text(path):
    if not os.path.exists(path):
        return ""
    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", ns):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns)).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def clean_text(text):
    text = re.sub(r"\[oai_citation:[^\]]+\]\([^)]+\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_manual_records(docx_path):
    text = read_docx_text(docx_path)
    if not text:
        return []
    chunks = re.split(r"(?=Case\s*\d+\s*:)", text)
    records = []
    for chunk in chunks:
        if not chunk.strip().startswith("Case"):
            continue
        case_match = re.search(r"Case\s*(\d+)\s*:", chunk)
        case_label = f"Case{case_match.group(1)}" if case_match else ""
        map_match = re.search(r"case_id:\s*(maze_like_[^\s]+)", chunk)
        winner_match = re.search(r"winner:\s*([^\n]+)", chunk)
        first_match = re.search(r"first_disagreement_step\s*=\s*([0-9]+)", chunk)
        trajectory_match = re.search(r"trajectory-level guidance:\s*([^\n]+)", chunk)
        expanded = {}
        for method in ["Manhattan", "MLP", "U-Net"]:
            expanded_match = re.search(rf"{method}:\s*([0-9]+)", chunk)
            expanded[method] = expanded_match.group(1) if expanded_match else ""
        records.append(
            {
                "manual_case_label": case_label,
                "manual_map_id": map_match.group(1).strip() if map_match else "",
                "manual_manhattan_expanded": expanded["Manhattan"],
                "manual_mlp_expanded": expanded["MLP"],
                "manual_unet_expanded": expanded["U-Net"],
                "manual_winner": clean_text(winner_match.group(1)) if winner_match else "",
                "manual_first_disagreement_step": first_match.group(1) if first_match else "",
                "manual_trajectory_guidance": clean_text(trajectory_match.group(1)) if trajectory_match else "",
                "label_less_wrong_branch": int("less wrong-branch exploration" in chunk or "更少错误分支探索" in chunk),
                "label_global_directionality": int("stronger global directionality" in chunk or "更强全局方向感" in chunk),
                "label_early_commitment": int("earlier commitment" in chunk or "更早集中" in chunk),
                "label_dead_end_pruning": int("dead-end" in chunk or "死路" in chunk),
                "cleaned_notes": clean_text(chunk[:1200]),
            }
        )
    return records


def cleaned_labels(case_index, manual_records):
    by_expanded = {
        (
            str(int(float(case["manhattan_expanded"]))),
            str(int(float(case["mlp_expanded"]))),
            str(int(float(case["unet_expanded"]))),
        ): case["map_id"]
        for case in case_index
    }
    by_map = {}
    duplicate_maps = set()
    for record in manual_records:
        map_id = record["manual_map_id"]
        if not map_id:
            key = (
                record.get("manual_manhattan_expanded", ""),
                record.get("manual_mlp_expanded", ""),
                record.get("manual_unet_expanded", ""),
            )
            map_id = by_expanded.get(key, "")
            record["manual_map_id"] = map_id
            if map_id:
                record["cleaning_record_note"] = "missing_map_id_inferred_from_expanded_counts"
        if not map_id:
            continue
        if map_id in by_map:
            duplicate_maps.add(map_id)
            continue
        by_map[map_id] = record

    rows = []
    for case in case_index:
        map_id = case["map_id"]
        record = by_map.get(map_id, {})
        winner_index = "U-Net" if float(case["expanded_gap"]) < 0 else "MLP"
        notes = []
        if not record:
            notes.append("manual_record_missing_or_unmatched")
        if map_id in duplicate_maps:
            notes.append("duplicate_manual_record_removed")
        if record.get("cleaning_record_note"):
            notes.append(record["cleaning_record_note"])
        rows.append(
            {
                "case_id": case["case_id"],
                "map_id": map_id,
                "winner_index": winner_index,
                "manual_case_label": record.get("manual_case_label", ""),
                "manual_winner": record.get("manual_winner", ""),
                "manual_first_disagreement_step": record.get("manual_first_disagreement_step", ""),
                "manual_trajectory_guidance": record.get("manual_trajectory_guidance", ""),
                "label_less_wrong_branch": record.get("label_less_wrong_branch", ""),
                "label_global_directionality": record.get("label_global_directionality", ""),
                "label_early_commitment": record.get("label_early_commitment", ""),
                "label_dead_end_pruning": record.get("label_dead_end_pruning", ""),
                "cleaning_notes": ";".join(notes),
            }
        )
    return rows


def distance_to_path_table(grid, path):
    free_cells = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 0]
    if not path:
        return {cell: 0 for cell in free_cells}
    return {
        cell: min(abs(cell[0] - path_cell[0]) + abs(cell[1] - path_cell[1]) for path_cell in path)
        for cell in free_cells
    }


def expansion_metrics(expanded, distance_table, fraction):
    if not expanded:
        return {
            "expanded_count": 0,
            "mean_distance_to_path": 0.0,
            "off_path_ge2_count": 0,
            "off_path_ge3_count": 0,
            "off_path_ge2_fraction": 0.0,
            "off_path_ge3_fraction": 0.0,
            "frontier_compactness": 0.0,
            "bounding_box_area": 0,
        }
    count = len(expanded) if fraction >= 1.0 else max(1, math.ceil(len(expanded) * fraction))
    cells = expanded[:count]
    distances = [distance_table.get(cell, 0) for cell in cells]
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    off2 = sum(1 for value in distances if value >= 2)
    off3 = sum(1 for value in distances if value >= 3)
    return {
        "expanded_count": len(cells),
        "mean_distance_to_path": mean(distances),
        "off_path_ge2_count": off2,
        "off_path_ge3_count": off3,
        "off_path_ge2_fraction": off2 / len(cells),
        "off_path_ge3_fraction": off3 / len(cells),
        "frontier_compactness": len(cells) / bbox_area if bbox_area else 0.0,
        "bounding_box_area": bbox_area,
    }


def flatten_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def build_method_groups(rows):
    groups = {}
    for methods in group_maps(rows, "structured"):
        sample = methods["manhattan"]
        if sample.get("structured_type") == "maze_like":
            groups[case_identity(sample)] = methods
    return groups


def case_metrics(case, methods, mlp_model, unet_model):
    sample = methods["manhattan"]
    grid, start, goal = build_grid(sample)
    distance_grid = compute_distance_to_goal(grid, goal)
    path = reconstruct_optimal_path(grid, distance_grid, start, goal)
    path_distance = distance_to_path_table(grid, path)
    mlp_table = prediction_table(grid, goal, make_mlp_table_heuristic(mlp_model, grid, goal))
    unet_table = prediction_table(grid, goal, make_unet_heuristic(unet_model, grid, goal))
    expanded = {
        "manhattan": simulate_manhattan_expansion(grid, start, goal)[0],
        "mlp": simulate_secondary_expansion(grid, start, goal, mlp_table)[0],
        "unet": simulate_secondary_expansion(grid, start, goal, unet_table)[0],
    }

    row = {
        "row_type": "case_metrics",
        "case_id": case["case_id"],
        "map_id": case["map_id"],
        "winner_group": "unet_win" if float(case["expanded_gap"]) < 0 else "mlp_win",
        "expanded_gap": float(case["expanded_gap"]),
        "first_disagreement_step": float(case["first_disagreement_step"] or 0),
        "disagreement_count": float(case["disagreement_count"] or 0),
        "path_length": len(path) - 1 if path else -1,
        "manhattan_expanded": result_expanded(methods, "manhattan"),
        "mlp_expanded": result_expanded(methods, "manhattan_mlp_tiebreak"),
        "unet_expanded": result_expanded(methods, "manhattan_unet_tiebreak"),
    }

    for method_name in METHODS:
        for fraction_name, fraction in FRACTIONS:
            prefix = f"{method_name}_{fraction_name}"
            row.update(flatten_metrics(prefix, expansion_metrics(expanded[method_name], path_distance, fraction)))

    for metric in [
        "mean_distance_to_path",
        "off_path_ge2_count",
        "off_path_ge3_count",
        "off_path_ge2_fraction",
        "off_path_ge3_fraction",
        "frontier_compactness",
        "bounding_box_area",
    ]:
        row[f"{metric}_gap"] = row[f"unet_all_{metric}"] - row[f"mlp_all_{metric}"]
        row[f"first25_{metric}_gap"] = row[f"unet_first25_{metric}"] - row[f"mlp_first25_{metric}"]
        row[f"first50_{metric}_gap"] = row[f"unet_first50_{metric}"] - row[f"mlp_first50_{metric}"]
    return row


def correlation_rows(case_rows):
    rows = []
    ys = [row["expanded_gap"] for row in case_rows]
    for metric in CORRELATION_METRICS:
        xs = [row[metric] for row in case_rows]
        rows.append(
            {
                "row_type": "correlation",
                "metric": metric,
                "y": "expanded_gap",
                "n": len(case_rows),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
        )
    return rows


def group_mean_rows(case_rows):
    rows = []
    for group in ["unet_win", "mlp_win"]:
        scoped = [row for row in case_rows if row["winner_group"] == group]
        output = {"row_type": "group_mean", "winner_group": group, "n": len(scoped)}
        for key in [
            "expanded_gap",
            "mean_distance_to_path_gap",
            "off_path_ge2_count_gap",
            "off_path_ge2_fraction_gap",
            "first25_off_path_ge2_fraction_gap",
            "first50_off_path_ge2_fraction_gap",
            "frontier_compactness_gap",
            "bounding_box_area_gap",
            "first_disagreement_step",
            "disagreement_count",
        ]:
            output[key] = mean(row[key] for row in scoped)
        rows.append(output)
    return rows


def save_plots(path, case_rows, correlations):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(path), "matplotlib_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter([row["off_path_ge2_fraction_gap"] for row in case_rows], [row["expanded_gap"] for row in case_rows], s=35)
    axes[0].set_title("Off-path gap vs expanded gap")
    axes[0].set_xlabel("U-Net - MLP off-path fraction")
    axes[0].set_ylabel("U-Net - MLP expanded")

    axes[1].scatter([row["first25_off_path_ge2_fraction_gap"] for row in case_rows], [row["expanded_gap"] for row in case_rows], s=35)
    axes[1].set_title("Early off-path gap")
    axes[1].set_xlabel("first 25% off-path gap")
    axes[1].set_ylabel("expanded gap")

    plot_metrics = ["off_path_ge2_fraction_gap", "first25_off_path_ge2_fraction_gap", "frontier_compactness_gap", "disagreement_count"]
    selected = [row for row in correlations if row["metric"] in plot_metrics]
    axes[2].bar([row["metric"].replace("_gap", "") for row in selected], [float(row["spearman"]) for row in selected])
    axes[2].set_title("Spearman with expanded gap")
    axes[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def find_corr(correlations, metric):
    return next(row for row in correlations if row["metric"] == metric)


def write_summary(path, correlations, group_rows):
    group = {row["winner_group"]: row for row in group_rows}
    off_corr = find_corr(correlations, "off_path_ge2_fraction_gap")
    early_corr = find_corr(correlations, "first25_off_path_ge2_fraction_gap")
    compact_corr = find_corr(correlations, "frontier_compactness_gap")
    first_dis_corr = find_corr(correlations, "first_disagreement_step")
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Maze Visual Mechanism Analysis\n\n")
        file.write(
            "This analysis quantifies the mechanisms observed in the maze_like visual case studies. "
            "It uses the 20 selected case-study maps and objective expansion-trajectory metrics.\n\n"
        )
        file.write("## Group Comparison\n\n")
        file.write("| Group | n | expanded gap | off-path fraction gap | early off-path gap | compactness gap | first disagreement | disagreement count |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for name in ["unet_win", "mlp_win"]:
            row = group[name]
            file.write(
                f"| {name} | {row['n']} | {row['expanded_gap']:.3f} | {row['off_path_ge2_fraction_gap']:.3f} | "
                f"{row['first25_off_path_ge2_fraction_gap']:.3f} | {row['frontier_compactness_gap']:.3f} | "
                f"{row['first_disagreement_step']:.3f} | {row['disagreement_count']:.3f} |\n"
            )
        file.write("\n## Correlations With Expanded Gap\n\n")
        file.write("| Metric | Pearson | Spearman |\n")
        file.write("|---|---:|---:|\n")
        for row in correlations:
            file.write(f"| {row['metric']} | {row['pearson']:.3f} | {row['spearman']:.3f} |\n")
        file.write("\n## Answers\n\n")
        file.write(
            f"- Reduced off-path exploration: supported if positive off-path gaps track positive expanded gaps. "
            f"Spearman={off_corr['spearman']:.3f} for off-path fraction gap.\n"
        )
        file.write(
            f"- Early off-path exploration: first-25% off-path Spearman={early_corr['spearman']:.3f}; "
            "compare with full-trajectory off-path to assess whether early errors dominate.\n"
        )
        file.write(
            f"- Concentrated search frontier: compactness-gap Spearman={compact_corr['spearman']:.3f}. "
            "A negative compactness gap in U-Net wins means U-Net may expand fewer cells without necessarily being denser by this metric.\n"
        )
        file.write(
            f"- First disagreement timing: Spearman={first_dis_corr['spearman']:.3f}; early disagreement alone is not enough unless it aligns with reduced off-path expansion.\n"
        )
        file.write(
            "- MLP-win interpretation: if MLP-win cases show lower off-path or bounding-box gaps for MLP, they are more consistent "
            "with local ordering advantages than with U-Net-style global spatial pruning.\n\n"
        )
        file.write("## Design Implication\n\n")
        file.write(
            "Future learned-heuristic search designs should preserve U-Net's ability to suppress wrong-branch exploration in maze-like maps, "
            "but should evaluate this with trajectory-level diagnostics rather than only global heuristic error. The useful signal appears "
            "to be spatial route bias over connected maze structure, not just cellwise distance regression.\n"
        )


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "maze_visual_mechanisms")
    os.makedirs(output_dir, exist_ok=True)
    maze_dir = os.path.join(project_root, "outputs", "maze_case_studies")
    index_path = os.path.join(maze_dir, "index.csv")
    docx_path = os.path.join(maze_dir, "maze-like case analysis summary.docx")

    case_index = read_csv(index_path)
    manual_records = parse_manual_records(docx_path)
    labels = cleaned_labels(case_index, manual_records)

    method_groups = build_method_groups(read_csv(args.structured_results))
    mlp_model = load_mlp_heuristic(checkpoint_path(project_root, args.checkpoint, "mlp"))
    unet_model = load_unet_heuristic(checkpoint_path(project_root, args.checkpoint, "unet"))
    case_rows = [case_metrics(case, method_groups[case["map_id"]], mlp_model, unet_model) for case in case_index]
    correlations = correlation_rows(case_rows)
    group_rows = group_mean_rows(case_rows)

    write_csv(os.path.join(output_dir, "cleaned_case_labels.csv"), labels)
    write_csv(os.path.join(output_dir, "maze_visual_mechanism_statistics.csv"), case_rows + correlations + group_rows)
    save_plots(os.path.join(output_dir, "maze_visual_mechanism_plots.png"), case_rows, correlations)
    write_summary(os.path.join(output_dir, "maze_visual_mechanism_summary.md"), correlations, group_rows)
    print(f"Saved maze visual mechanism outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze mechanisms from maze-like visual case studies.")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    parser.add_argument("--checkpoint", default="best")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
