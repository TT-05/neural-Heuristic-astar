import argparse
import csv
import os
from collections import Counter, defaultdict

from analyze_critical_decisions import (
    DEFAULT_RANDOM_START_GOAL_RETRIES,
    STRUCTURED_TYPES,
    collect_cases,
    mean,
    median,
)


OUTPUT_DIR = "outputs/critical_decision_refined"
THRESHOLDS = [
    ("level0", 0),
    ("level1", 5),
    ("level2", 10),
    ("level3", 20),
]


def write_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def event_recovery(event):
    return int(event["recovery_cost"])


def threshold_label(name, minimum_recovery):
    if minimum_recovery == 0:
        return f"{name}_baseline"
    return f"{name}_recovery_ge_{minimum_recovery}"


def extra_expansions(case):
    return max(0, int(case["trace_result"]["expanded"]) - (int(case["optimal_cost"]) + 1))


def covered_steps_for_events(events):
    steps = set()
    for event in events:
        start = int(event["step"]) + 1
        stop = start + int(event["off_path_expansions_after_event"])
        steps.update(range(start, stop))
    return len(steps)


def filtered_events(case, minimum_recovery):
    return [event for event in case["events"] if event_recovery(event) >= minimum_recovery]


def threshold_case_rows(cases):
    rows = []
    for case in cases:
        total_expanded = int(case["trace_result"]["expanded"])
        extra = extra_expansions(case)
        for name, minimum_recovery in THRESHOLDS:
            events = filtered_events(case, minimum_recovery)
            recovery_costs = [event_recovery(event) for event in events]
            covered = covered_steps_for_events(events)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "structure_type": case["structured_type"],
                    "threshold": threshold_label(name, minimum_recovery),
                    "critical_event_count": len(events),
                    "total_expansions": total_expanded,
                    "extra_expansions": extra,
                    "covered_extra_expansions": min(covered, extra),
                    "coverage_ratio": min(covered, extra) / extra if extra else 0.0,
                    "mean_recovery_cost": mean(recovery_costs),
                    "median_recovery_cost": median(recovery_costs),
                }
            )
    return rows


def event_detail_rows(cases):
    rows = []
    for case in cases:
        for name, minimum_recovery in THRESHOLDS:
            label = threshold_label(name, minimum_recovery)
            for event in filtered_events(case, minimum_recovery):
                rows.append(
                    {
                        "case_id": event["case_id"],
                        "structure_type": event["structure_type"],
                        "threshold": label,
                        "step": event["step"],
                        "wrong_node": event["wrong_node"],
                        "correct_candidate": event["correct_node"],
                        "recovery_cost": event["recovery_cost"],
                        "off_path_expansions_after_event": event["off_path_expansions_after_event"],
                        "error_type": event["error_type"],
                        "wrong_unet_h": event["wrong_unet_h"],
                        "correct_unet_h": event["correct_unet_h"],
                        "wrong_true_distance": event["wrong_true_distance"],
                        "correct_true_distance": event["correct_true_distance"],
                    }
                )
    return rows


def aggregate_thresholds(case_rows, event_rows):
    grouped = defaultdict(list)
    for row in case_rows:
        grouped[row["threshold"]].append(row)
    events_by_threshold = defaultdict(list)
    for row in event_rows:
        events_by_threshold[row["threshold"]].append(row)

    rows = []
    for threshold, items in grouped.items():
        events = sum(int(row["critical_event_count"]) for row in items)
        total = sum(int(row["total_expansions"]) for row in items)
        extra = sum(int(row["extra_expansions"]) for row in items)
        covered = sum(int(row["covered_extra_expansions"]) for row in items)
        recovery_values = [float(row["recovery_cost"]) for row in events_by_threshold[threshold]]
        rows.append(
            {
                "scope": "all",
                "threshold": threshold,
                "cases": len(items),
                "critical_event_count": events,
                "events_per_case": events / len(items) if items else 0.0,
                "total_expansions": total,
                "extra_expansions": extra,
                "covered_extra_expansions": covered,
                "coverage_ratio": covered / extra if extra else 0.0,
                "mean_recovery_cost": mean(recovery_values),
            }
        )
    return sorted(rows, key=lambda row: threshold_order(row["threshold"]))


def aggregate_by_structure(case_rows, event_rows):
    grouped = defaultdict(list)
    for row in case_rows:
        grouped[(row["structure_type"], row["threshold"])].append(row)
    events_by_group = defaultdict(list)
    for row in event_rows:
        events_by_group[(row["structure_type"], row["threshold"])].append(row)

    rows = []
    for (structure, threshold), items in grouped.items():
        events = sum(int(row["critical_event_count"]) for row in items)
        extra = sum(int(row["extra_expansions"]) for row in items)
        covered = sum(int(row["covered_extra_expansions"]) for row in items)
        recovery_values = [float(row["recovery_cost"]) for row in events_by_group[(structure, threshold)]]
        rows.append(
            {
                "structure_type": structure,
                "threshold": threshold,
                "cases": len(items),
                "critical_event_count": events,
                "events_per_case": events / len(items) if items else 0.0,
                "extra_expansions": extra,
                "covered_extra_expansions": covered,
                "coverage_ratio": covered / extra if extra else 0.0,
                "mean_recovery_cost": mean(recovery_values),
            }
        )
    return sorted(rows, key=lambda row: (row["structure_type"], threshold_order(row["threshold"])))


def threshold_order(label):
    for index, (name, minimum) in enumerate(THRESHOLDS):
        if label == threshold_label(name, minimum):
            return index
    return len(THRESHOLDS)


def pareto_rows(cases):
    all_events = []
    case_extra = {case["case_id"]: extra_expansions(case) for case in cases}
    total_extra = sum(case_extra.values())
    for case in cases:
        for event in case["events"]:
            all_events.append((case["case_id"], event))
    all_events.sort(key=lambda item: int(item[1]["off_path_expansions_after_event"]), reverse=True)

    rows = []
    impacted = defaultdict(set)
    total_events = len(all_events)
    for index, (case_id, event) in enumerate(all_events, start=1):
        start = int(event["step"]) + 1
        stop = start + int(event["off_path_expansions_after_event"])
        impacted[case_id].update(range(start, stop))
        if index == 1 or index == total_events or index % max(1, total_events // 200) == 0:
            covered = sum(min(len(steps), case_extra.get(case_id, 0)) for case_id, steps in impacted.items())
            rows.append(
                {
                    "selected_event_count": index,
                    "selected_event_fraction": index / total_events if total_events else 0.0,
                    "covered_extra_expansions": min(covered, total_extra),
                    "extra_expansion_coverage": min(covered, total_extra) / total_extra if total_extra else 0.0,
                }
            )
    return rows


def make_plots(output_dir, aggregate_rows, event_rows, pareto):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, "matplotlib_cache"))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    labels = [row["threshold"] for row in aggregate_rows]
    xs = list(range(len(labels)))

    plt.figure(figsize=(7, 4))
    plt.plot(xs, [row["critical_event_count"] for row in aggregate_rows], marker="o")
    plt.xticks(xs, labels, rotation=20, ha="right")
    plt.ylabel("Critical event count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "critical_event_count_vs_threshold.png"))
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(
        [row["critical_event_count"] for row in aggregate_rows],
        [100.0 * row["coverage_ratio"] for row in aggregate_rows],
        s=45,
    )
    for row in aggregate_rows:
        plt.annotate(row["threshold"], (row["critical_event_count"], 100.0 * row["coverage_ratio"]), fontsize=8)
    plt.xlabel("Detected critical events")
    plt.ylabel("Extra expansion coverage (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "coverage_vs_detected_events.png"))
    plt.close()

    recoveries = [int(row["recovery_cost"]) for row in event_rows if row["threshold"] == "level0_baseline"]
    plt.figure(figsize=(7, 4))
    plt.hist(recoveries, bins=40)
    plt.xlabel("Recovery cost")
    plt.ylabel("Events")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recovery_cost_distribution.png"))
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(
        [100.0 * row["selected_event_fraction"] for row in pareto],
        [100.0 * row["extra_expansion_coverage"] for row in pareto],
    )
    plt.xlabel("Selected critical decisions (%)")
    plt.ylabel("Extra expansions explained (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_critical_decisions.png"))
    plt.close()


def error_distribution(event_rows, threshold):
    counts = Counter(row["error_type"] for row in event_rows if row["threshold"] == threshold)
    total = sum(counts.values())
    return {
        error_type: {
            "count": count,
            "share": count / total if total else 0.0,
        }
        for error_type, count in counts.most_common()
    }


def write_summary(path, aggregate_rows, structure_rows, event_rows, pareto):
    by_threshold = {row["threshold"]: row for row in aggregate_rows}
    lines = []
    lines.append("# Refined Critical Decision Analysis")
    lines.append("")
    lines.append("This offline diagnostic reuses the existing U-Net tie-break trace analysis and applies recovery-cost thresholds.")
    lines.append("It does not modify A*, model weights, checkpoints, or training.")
    lines.append("")
    lines.append("## Threshold Summary")
    lines.append("")
    lines.append("| Threshold | Events | Events/case | Extra coverage | Mean recovery |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in aggregate_rows:
        lines.append(
            f"| {row['threshold']} | {row['critical_event_count']} | {row['events_per_case']:.3f} | "
            f"{100.0 * row['coverage_ratio']:.2f}% | {row['mean_recovery_cost']:.3f} |"
        )
    lines.append("")

    lines.append("## Error Type Shift")
    lines.append("")
    for threshold in ["level0_baseline", "level2_recovery_ge_10", "level3_recovery_ge_20"]:
        dist = error_distribution(event_rows, threshold)
        lines.append(f"### {threshold}")
        lines.append("")
        lines.append("| Error type | Count | Share |")
        lines.append("|---|---:|---:|")
        for error_type, values in dist.items():
            lines.append(f"| {error_type} | {values['count']} | {values['share']:.3f} |")
        lines.append("")

    lines.append("## Structure Summary")
    lines.append("")
    lines.append("| Structure | Threshold | Events/case | Extra coverage | Mean recovery |")
    lines.append("|---|---|---:|---:|---:|")
    for row in structure_rows:
        lines.append(
            f"| {row['structure_type']} | {row['threshold']} | {row['events_per_case']:.3f} | "
            f"{100.0 * row['coverage_ratio']:.2f}% | {row['mean_recovery_cost']:.3f} |"
        )
    lines.append("")

    level0 = by_threshold["level0_baseline"]
    level2 = by_threshold["level2_recovery_ge_10"]
    level3 = by_threshold["level3_recovery_ge_20"]
    pareto_10 = next((row for row in pareto if row["selected_event_fraction"] >= 0.10), pareto[-1] if pareto else None)
    pareto_20 = next((row for row in pareto if row["selected_event_fraction"] >= 0.20), pareto[-1] if pareto else None)

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"- Events become substantially sparser with stricter thresholds: {level0['critical_event_count']} at Level 0, "
        f"{level2['critical_event_count']} at Level 2, and {level3['critical_event_count']} at Level 3."
    )
    lines.append(
        f"- High-impact events still cover a meaningful fraction of extra expansions: Level 2 covers "
        f"{100.0 * level2['coverage_ratio']:.2f}% and Level 3 covers {100.0 * level3['coverage_ratio']:.2f}%."
    )
    if pareto_10:
        lines.append(
            f"- Pareto diagnostic: the top {100.0 * pareto_10['selected_event_fraction']:.1f}% of baseline events cover "
            f"{100.0 * pareto_10['extra_expansion_coverage']:.2f}% of extra expansions."
        )
    if pareto_20:
        lines.append(
            f"- The top {100.0 * pareto_20['selected_event_fraction']:.1f}% of baseline events cover "
            f"{100.0 * pareto_20['extra_expansion_coverage']:.2f}% of extra expansions."
        )
    lines.append(
        "- These are observed associations from offline traces. They support prioritizing selected high-impact decision points "
        "as a research direction, but they do not prove that changing those decisions would causally recover the measured waste."
    )
    lines.append("")
    return "\n".join(lines)


def run(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    cases, skips = collect_cases(args)
    if skips:
        print(f"Skipped cases: {dict(skips)}")

    case_rows = threshold_case_rows(cases)
    event_rows = event_detail_rows(cases)
    aggregate_rows = aggregate_thresholds(case_rows, event_rows)
    structure_rows = aggregate_by_structure(case_rows, event_rows)
    pareto = pareto_rows(cases)

    write_csv(os.path.join(output_dir, "critical_decision_thresholds.csv"), case_rows)
    write_csv(os.path.join(output_dir, "event_details.csv"), event_rows)
    write_csv(os.path.join(output_dir, "threshold_summary.csv"), aggregate_rows)
    write_csv(os.path.join(output_dir, "structure_threshold_summary.csv"), structure_rows)
    write_csv(os.path.join(output_dir, "pareto_curve.csv"), pareto)
    make_plots(output_dir, aggregate_rows, event_rows, pareto)
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as file:
        file.write(write_summary(os.path.join(output_dir, "summary.md"), aggregate_rows, structure_rows, event_rows, pareto))

    print(f"Saved refined critical decision outputs to {output_dir}")
    print(f"Cases: {len(cases)}")
    for row in aggregate_rows:
        print(
            f"{row['threshold']}: events={row['critical_event_count']} "
            f"coverage={100.0 * row['coverage_ratio']:.2f}%"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Refined critical decision threshold analysis for U-Net A* traces.")
    parser.add_argument("--checkpoint", default="compatible")
    parser.add_argument("--seeds", default="0:100")
    parser.add_argument("--structured-types", default="all")
    parser.add_argument("--start-goal-mode", choices=["fixed", "random"], default="random")
    parser.add_argument("--random-start-goal-retries", type=int, default=DEFAULT_RANDOM_START_GOAL_RETRIES)
    parser.add_argument("--weak-margin-threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
