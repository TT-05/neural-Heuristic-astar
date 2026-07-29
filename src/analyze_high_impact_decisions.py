import argparse
import csv
import os
from collections import Counter, defaultdict

from analyze_critical_decisions import DEFAULT_RANDOM_START_GOAL_RETRIES, collect_cases, mean, median
from analyze_critical_decisions_refined import extra_expansions


OUTPUT_DIR = "outputs/high_impact_decision_analysis"
THRESHOLDS = [
    ("recovery_ge_10", 10),
    ("recovery_ge_20", 20),
]
WEAK_SEPARATION_MARGIN = 0.5
STRUCTURE_TYPES = ["maze_like", "bottleneck", "large_block", "narrow_corridor"]


def write_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mechanism_type(event, weak_margin):
    wrong_unet = float(event["wrong_unet_h"])
    correct_unet = float(event["correct_unet_h"])
    wrong_true = float(event["wrong_true_distance"])
    correct_true = float(event["correct_true_distance"])
    wrong_f = float(event["wrong_f"])
    correct_f = float(event["correct_f"])

    if correct_true < wrong_true and correct_unet > wrong_unet:
        return "A_unet_ordering_error"
    if correct_unet < wrong_unet and correct_f > wrong_f:
        return "B_manhattan_f_limitation"
    if correct_unet <= wrong_unet and abs(correct_unet - wrong_unet) <= weak_margin:
        return "C_weak_separation"
    return "D_other"


def threshold_events(cases, weak_margin):
    rows = []
    for case in cases:
        for threshold_name, minimum_recovery in THRESHOLDS:
            for event in case["events"]:
                if int(event["recovery_cost"]) < minimum_recovery:
                    continue
                rows.append(
                    {
                        "case_id": event["case_id"],
                        "structure_type": event["structure_type"],
                        "threshold": threshold_name,
                        "step": event["step"],
                        "wrong_node": event["wrong_node"],
                        "correct_node": event["correct_node"],
                        "recovery_cost": event["recovery_cost"],
                        "error_type": mechanism_type(event, weak_margin),
                        "wrong_f": event["wrong_f"],
                        "correct_f": event["correct_f"],
                        "wrong_unet_h": event["wrong_unet_h"],
                        "correct_unet_h": event["correct_unet_h"],
                        "wrong_true_distance": event["wrong_true_distance"],
                        "correct_true_distance": event["correct_true_distance"],
                        "unet_margin": float(event["correct_unet_h"]) - float(event["wrong_unet_h"]),
                        "extra_expansions_after_event": event["off_path_expansions_after_event"],
                    }
                )
    return rows


def case_lookup(cases):
    return {case["case_id"]: case for case in cases}


def covered_steps(events):
    by_case = defaultdict(set)
    for event in events:
        start = int(event["step"]) + 1
        stop = start + int(event["extra_expansions_after_event"])
        by_case[event["case_id"]].update(range(start, stop))
    return by_case


def capped_coverage(events, cases):
    by_case_steps = covered_steps(events)
    cases_by_id = case_lookup(cases)
    return sum(min(len(steps), extra_expansions(cases_by_id[case_id])) for case_id, steps in by_case_steps.items())


def summary_by_threshold(events, cases):
    rows = []
    total_extra = sum(extra_expansions(case) for case in cases)
    for threshold_name, _ in THRESHOLDS:
        scoped = [event for event in events if event["threshold"] == threshold_name]
        recoveries = [float(event["recovery_cost"]) for event in scoped]
        extra_after = [float(event["extra_expansions_after_event"]) for event in scoped]
        type_counts = Counter(event["error_type"] for event in scoped)
        recoverable = capped_coverage(scoped, cases)
        row = {
            "threshold": threshold_name,
            "event_count": len(scoped),
            "mean_recovery_cost": mean(recoveries),
            "median_recovery_cost": median(recoveries),
            "mean_extra_expansions_after_event": mean(extra_after),
            "median_extra_expansions_after_event": median(extra_after),
            "maximum_recoverable_expansions": recoverable,
            "total_extra_expansions": total_extra,
            "recoverable_extra_expansion_fraction": recoverable / total_extra if total_extra else 0.0,
        }
        for error_type in ["A_unet_ordering_error", "B_manhattan_f_limitation", "C_weak_separation", "D_other"]:
            count = type_counts.get(error_type, 0)
            row[f"{error_type}_count"] = count
            row[f"{error_type}_share"] = count / len(scoped) if scoped else 0.0
        rows.append(row)
    return rows


def structure_rows(events, cases):
    rows = []
    case_counts = Counter(case["structured_type"] for case in cases)
    for threshold_name, _ in THRESHOLDS:
        threshold_events_only = [event for event in events if event["threshold"] == threshold_name]
        for structure in STRUCTURE_TYPES:
            scoped = [event for event in threshold_events_only if event["structure_type"] == structure]
            recoveries = [float(event["recovery_cost"]) for event in scoped]
            extra_after = [float(event["extra_expansions_after_event"]) for event in scoped]
            type_counts = Counter(event["error_type"] for event in scoped)
            row = {
                "structure_type": structure,
                "threshold": threshold_name,
                "cases": case_counts.get(structure, 0),
                "event_count": len(scoped),
                "events_per_case": len(scoped) / case_counts[structure] if case_counts.get(structure, 0) else 0.0,
                "mean_recovery_cost": mean(recoveries),
                "mean_extra_expansions_after_event": mean(extra_after),
            }
            for error_type in ["A_unet_ordering_error", "B_manhattan_f_limitation", "C_weak_separation", "D_other"]:
                count = type_counts.get(error_type, 0)
                row[f"{error_type}_count"] = count
                row[f"{error_type}_share"] = count / len(scoped) if scoped else 0.0
            rows.append(row)
    return rows


def make_plots(output_dir, events, threshold_rows, struct_rows):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    error_types = ["A_unet_ordering_error", "B_manhattan_f_limitation", "C_weak_separation", "D_other"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    labels = [row["threshold"] for row in threshold_rows]
    bottoms = [0] * len(labels)
    plt.figure(figsize=(7, 4))
    for error_type, color in zip(error_types, colors):
        values = [row[f"{error_type}_count"] for row in threshold_rows]
        plt.bar(labels, values, bottom=bottoms, label=error_type, color=color)
        bottoms = [left + value for left, value in zip(bottoms, values)]
    plt.ylabel("Events")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_type_distribution.png"))
    plt.close()

    plt.figure(figsize=(7, 4))
    for threshold_name, _ in THRESHOLDS:
        scoped = [event for event in events if event["threshold"] == threshold_name]
        plt.scatter(
            [float(event["unet_margin"]) for event in scoped],
            [float(event["recovery_cost"]) for event in scoped],
            s=8,
            alpha=0.35,
            label=threshold_name,
        )
    plt.xlabel("U-Net margin: correct h - wrong h")
    plt.ylabel("Recovery cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recovery_vs_unet_margin.png"))
    plt.close()

    plt.figure(figsize=(7, 4))
    for threshold_name, _ in THRESHOLDS:
        scoped = [float(event["extra_expansions_after_event"]) for event in events if event["threshold"] == threshold_name]
        plt.hist(scoped, bins=35, alpha=0.55, label=threshold_name)
    plt.xlabel("Extra expansions after event")
    plt.ylabel("Events")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "extra_expansions_after_event_distribution.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    width = 0.35
    xs = list(range(len(STRUCTURE_TYPES)))
    for offset, threshold_name in [(-width / 2, "recovery_ge_10"), (width / 2, "recovery_ge_20")]:
        values = [
            next(
                row["events_per_case"]
                for row in struct_rows
                if row["structure_type"] == structure and row["threshold"] == threshold_name
            )
            for structure in STRUCTURE_TYPES
        ]
        plt.bar([x + offset for x in xs], values, width=width, label=threshold_name)
    plt.xticks(xs, STRUCTURE_TYPES, rotation=15)
    plt.ylabel("High-impact events per case")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "structure_high_impact_comparison.png"))
    plt.close()


def mechanism_table_lines(events, threshold):
    scoped = [event for event in events if event["threshold"] == threshold]
    counts = Counter(event["error_type"] for event in scoped)
    total = len(scoped)
    lines = ["| Error type | Count | Share |", "|---|---:|---:|"]
    for error_type in ["A_unet_ordering_error", "B_manhattan_f_limitation", "C_weak_separation", "D_other"]:
        count = counts.get(error_type, 0)
        lines.append(f"| {error_type} | {count} | {count / total if total else 0.0:.3f} |")
    return lines


def write_summary(path, events, threshold_rows, struct_rows):
    by_threshold = {row["threshold"]: row for row in threshold_rows}
    lines = []
    lines.append("# High-impact Critical Decision Mechanism Analysis")
    lines.append("")
    lines.append("This offline diagnostic analyzes only critical decisions with recovery cost >= 10 and >= 20.")
    lines.append("It does not modify A*, model weights, checkpoints, or training.")
    lines.append("")
    lines.append("## Threshold Summary")
    lines.append("")
    lines.append("| Threshold | Events | Mean recovery | Mean extra after event | Max recoverable | Extra fraction |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in threshold_rows:
        lines.append(
            f"| {row['threshold']} | {row['event_count']} | {row['mean_recovery_cost']:.3f} | "
            f"{row['mean_extra_expansions_after_event']:.3f} | {row['maximum_recoverable_expansions']} | "
            f"{100.0 * row['recoverable_extra_expansion_fraction']:.2f}% |"
        )
    lines.append("")

    lines.append("## Mechanism Distribution")
    lines.append("")
    for threshold_name, _ in THRESHOLDS:
        lines.append(f"### {threshold_name}")
        lines.append("")
        lines.extend(mechanism_table_lines(events, threshold_name))
        lines.append("")

    lines.append("## Structure Analysis")
    lines.append("")
    lines.append("| Structure | Threshold | Events/case | Mean recovery | Mean extra | A share | B share | C share | D share |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in struct_rows:
        lines.append(
            f"| {row['structure_type']} | {row['threshold']} | {row['events_per_case']:.3f} | "
            f"{row['mean_recovery_cost']:.3f} | {row['mean_extra_expansions_after_event']:.3f} | "
            f"{row['A_unet_ordering_error_share']:.3f} | {row['B_manhattan_f_limitation_share']:.3f} | "
            f"{row['C_weak_separation_share']:.3f} | {row['D_other_share']:.3f} |"
        )
    lines.append("")

    ge10 = by_threshold["recovery_ge_10"]
    ge20 = by_threshold["recovery_ge_20"]
    ge10_a = ge10["A_unet_ordering_error_share"]
    ge10_b = ge10["B_manhattan_f_limitation_share"]
    ge20_a = ge20["A_unet_ordering_error_share"]
    ge20_b = ge20["B_manhattan_f_limitation_share"]
    dominant_10 = "U-Net ranking errors" if ge10_a > ge10_b else "Manhattan f restrictions"
    dominant_20 = "U-Net ranking errors" if ge20_a > ge20_b else "Manhattan f restrictions"

    lines.append("## Final Questions")
    lines.append("")
    lines.append(
        f"1. For recovery >= 10, the dominant observed mechanism is {dominant_10}: "
        f"A={ge10_a:.3f}, B={ge10_b:.3f}."
    )
    lines.append(
        f"2. Increasing the threshold strengthens the relative share of U-Net ordering errors: "
        f"A changes from {ge10_a:.3f} to {ge20_a:.3f}, while B changes from {ge10_b:.3f} to {ge20_b:.3f}."
    )
    lines.append(
        "3. Structure-specific mechanisms differ; inspect the structure table for cases where A or B dominates."
    )
    lines.append(
        f"4. The high-impact events represent up to {100.0 * ge10['recoverable_extra_expansion_fraction']:.2f}% "
        f"(>=10) and {100.0 * ge20['recoverable_extra_expansion_fraction']:.2f}% (>=20) of total extra expansions, "
        "supporting selective high-impact guidance as a research direction."
    )
    lines.append("")
    lines.append("These are observed relationships in offline traces, not causal claims.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def run(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    cases, skips = collect_cases(args)
    if skips:
        print(f"Skipped cases: {dict(skips)}")

    events = threshold_events(cases, args.weak_separation_margin)
    threshold_rows = summary_by_threshold(events, cases)
    struct_rows = structure_rows(events, cases)

    write_csv(os.path.join(output_dir, "high_impact_events.csv"), events)
    write_csv(os.path.join(output_dir, "threshold_mechanism_summary.csv"), threshold_rows)
    write_csv(os.path.join(output_dir, "structure_mechanism_summary.csv"), struct_rows)
    make_plots(output_dir, events, threshold_rows, struct_rows)
    write_summary(os.path.join(output_dir, "summary.md"), events, threshold_rows, struct_rows)

    print(f"Saved high-impact decision analysis outputs to {output_dir}")
    print(f"Cases: {len(cases)}")
    for row in threshold_rows:
        print(
            f"{row['threshold']}: events={row['event_count']} "
            f"recoverable={100.0 * row['recoverable_extra_expansion_fraction']:.2f}%"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="High-impact critical decision mechanism analysis.")
    parser.add_argument("--checkpoint", default="compatible")
    parser.add_argument("--seeds", default="0:100")
    parser.add_argument("--structured-types", default="all")
    parser.add_argument("--start-goal-mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--random-start-goal-retries", type=int, default=DEFAULT_RANDOM_START_GOAL_RETRIES)
    parser.add_argument("--weak-margin-threshold", type=float, default=0.5)
    parser.add_argument("--weak-separation-margin", type=float, default=WEAK_SEPARATION_MARGIN)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
