import argparse
import csv
import os


STRUCTURED_TYPES = ["maze_like", "bottleneck", "large_block", "narrow_corridor"]
METHODS = [
    "manhattan",
    "manhattan_large_g_tiebreak",
    "manhattan_mlp_tiebreak",
    "manhattan_unet_tiebreak",
    "manhattan_true_distance_tiebreak",
]
MLP_METHOD = "manhattan_mlp_tiebreak"
UNET_METHOD = "manhattan_unet_tiebreak"
ORACLE_METHOD = "oracle_structure_selector"
COARSE_ORACLE_METHOD = "oracle_coarse_selector"


def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def to_float(row, key, default=0.0):
    value = row.get(key, "")
    if value == "":
        return default
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    return float(value)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def map_key(row):
    return (
        row["seed"],
        row["map_size"],
        row["obstacle_rate"],
        row.get("map_mode", "random"),
        row.get("structured_type", "random"),
        row.get("start_goal_mode", ""),
        row.get("start_row", ""),
        row.get("start_col", ""),
        row.get("goal_row", ""),
        row.get("goal_col", ""),
    )


def group_maps(rows):
    grouped = {}
    needed = set(METHODS)
    for row in rows:
        if row.get("heuristic") not in needed:
            continue
        grouped.setdefault(map_key(row), {})[row["heuristic"]] = row
    return [methods for methods in grouped.values() if needed.issubset(methods.keys()) and not methods["manhattan"].get("skip_reason")]


def cost_gap(row):
    return max(0.0, to_float(row, "path_length") - to_float(row, "optimal_cost"))


def stats_for_rows(rows):
    return {
        "maps": len(rows),
        "mean_expanded": mean(to_float(row, "expanded_nodes") for row in rows),
        "optimality_rate": mean(1.0 if row["optimal"] == "True" else 0.0 for row in rows),
        "mean_cost_gap": mean(cost_gap(row) for row in rows),
        "mean_runtime": mean(to_float(row, "runtime_seconds") for row in rows),
    }


def stats_for_method(maps, method):
    return stats_for_rows([methods[method] for methods in maps])


def choose_by_type(maps):
    choices = {}
    type_rows = []
    for structured_type in STRUCTURED_TYPES:
        scoped = [methods for methods in maps if methods["manhattan"].get("structured_type") == structured_type]
        mlp_stats = stats_for_method(scoped, MLP_METHOD)
        unet_stats = stats_for_method(scoped, UNET_METHOD)
        choice = UNET_METHOD if unet_stats["mean_expanded"] < mlp_stats["mean_expanded"] else MLP_METHOD
        choices[structured_type] = choice
        type_rows.append(
            {
                "row_type": "structured_type_choice",
                "structured_type": structured_type,
                "maps": len(scoped),
                "mean_expanded_mlp_tiebreak": mlp_stats["mean_expanded"],
                "mean_expanded_unet_tiebreak": unet_stats["mean_expanded"],
                "mlp_optimality_rate": mlp_stats["optimality_rate"],
                "unet_optimality_rate": unet_stats["optimality_rate"],
                "mlp_cost_gap": mlp_stats["mean_cost_gap"],
                "unet_cost_gap": unet_stats["mean_cost_gap"],
                "oracle_choice": choice,
            }
        )
    return choices, type_rows


def coarse_type(structured_type):
    if structured_type in {"maze_like", "bottleneck"}:
        return "maze_like+bottleneck"
    return "large_block+narrow_corridor"


def choose_by_coarse_type(maps):
    choices = {}
    rows = []
    for label in ["maze_like+bottleneck", "large_block+narrow_corridor"]:
        scoped = [methods for methods in maps if coarse_type(methods["manhattan"].get("structured_type")) == label]
        mlp_stats = stats_for_method(scoped, MLP_METHOD)
        unet_stats = stats_for_method(scoped, UNET_METHOD)
        choice = UNET_METHOD if unet_stats["mean_expanded"] < mlp_stats["mean_expanded"] else MLP_METHOD
        choices[label] = choice
        rows.append(
            {
                "row_type": "coarse_choice",
                "structured_type": label,
                "maps": len(scoped),
                "mean_expanded_mlp_tiebreak": mlp_stats["mean_expanded"],
                "mean_expanded_unet_tiebreak": unet_stats["mean_expanded"],
                "mlp_optimality_rate": mlp_stats["optimality_rate"],
                "unet_optimality_rate": unet_stats["optimality_rate"],
                "mlp_cost_gap": mlp_stats["mean_cost_gap"],
                "unet_cost_gap": unet_stats["mean_cost_gap"],
                "oracle_choice": choice,
            }
        )
    return choices, rows


def selected_rows(maps, choices):
    selected = []
    for methods in maps:
        structured_type = methods["manhattan"].get("structured_type")
        selected.append(methods[choices[structured_type]])
    return selected


def coarse_selected_rows(maps, choices):
    selected = []
    for methods in maps:
        label = coarse_type(methods["manhattan"].get("structured_type"))
        selected.append(methods[choices[label]])
    return selected


def comparison_rows(maps, exact_choices, coarse_choices):
    rows = []
    for method in METHODS:
        stats = stats_for_method(maps, method)
        rows.append({"row_type": "comparison", "method": method, **stats})
    rows.append({"row_type": "comparison", "method": ORACLE_METHOD, **stats_for_rows(selected_rows(maps, exact_choices))})
    rows.append({"row_type": "comparison", "method": COARSE_ORACLE_METHOD, **stats_for_rows(coarse_selected_rows(maps, coarse_choices))})
    return rows


def add_gain_rows(comparisons):
    by_method = {row["method"]: row for row in comparisons if row["row_type"] == "comparison"}
    oracle = by_method[ORACLE_METHOD]
    coarse = by_method[COARSE_ORACLE_METHOD]
    mlp = by_method[MLP_METHOD]
    unet = by_method[UNET_METHOD]
    best_single = mlp if mlp["mean_expanded"] <= unet["mean_expanded"] else unet
    true_oracle = by_method["manhattan_true_distance_tiebreak"]
    return [
        {
            "row_type": "gain",
            "method": ORACLE_METHOD,
            "oracle_selector_gain_vs_mlp": mlp["mean_expanded"] - oracle["mean_expanded"],
            "oracle_selector_gain_vs_unet": unet["mean_expanded"] - oracle["mean_expanded"],
            "oracle_selector_gain_vs_best_single_method": best_single["mean_expanded"] - oracle["mean_expanded"],
            "oracle_gap_to_true_distance_oracle": oracle["mean_expanded"] - true_oracle["mean_expanded"],
            "coarse_oracle_gap_to_exact_oracle": coarse["mean_expanded"] - oracle["mean_expanded"],
        }
    ]


def write_csv(path, rows):
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, comparisons):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(path)), "matplotlib_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    ordered = [
        "manhattan",
        "manhattan_large_g_tiebreak",
        MLP_METHOD,
        UNET_METHOD,
        ORACLE_METHOD,
        COARSE_ORACLE_METHOD,
        "manhattan_true_distance_tiebreak",
    ]
    by_method = {row["method"]: row for row in comparisons if row["row_type"] == "comparison"}
    labels = [method.replace("manhattan_", "").replace("_tiebreak", "") for method in ordered]
    values = [by_method[method]["mean_expanded"] for method in ordered]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Mean expanded nodes")
    ax.set_title("Oracle structure selector upper bound")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_summary(path, type_rows, coarse_rows, comparisons, gain_rows):
    by_method = {row["method"]: row for row in comparisons if row["row_type"] == "comparison"}
    gain = gain_rows[0]
    exact = by_method[ORACLE_METHOD]
    coarse = by_method[COARSE_ORACLE_METHOD]
    mlp = by_method[MLP_METHOD]
    unet = by_method[UNET_METHOD]
    true_oracle = by_method["manhattan_true_distance_tiebreak"]
    best_single_name = MLP_METHOD if mlp["mean_expanded"] <= unet["mean_expanded"] else UNET_METHOD

    with open(path, "w", encoding="utf-8") as file:
        file.write("# Oracle Structure-Selector Analysis\n\n")
        file.write(
            "This offline analysis estimates the upper bound of choosing between MLP and U-Net tie-breaking using known "
            "`structured_type` labels. It does not implement adaptation and does not assume these labels are available at inference time.\n\n"
        )
        file.write("## Per-Structure Choices\n\n")
        file.write("| Structured type | MLP expanded | U-Net expanded | Oracle choice |\n")
        file.write("|---|---:|---:|---|\n")
        for row in type_rows:
            file.write(
                f"| {row['structured_type']} | {row['mean_expanded_mlp_tiebreak']:.3f} | "
                f"{row['mean_expanded_unet_tiebreak']:.3f} | {row['oracle_choice']} |\n"
            )
        file.write("\n")

        file.write("## Aggregate Comparison\n\n")
        file.write("| Method | Mean expanded | Optimality | Cost gap |\n")
        file.write("|---|---:|---:|---:|\n")
        for method in [
            "manhattan",
            "manhattan_large_g_tiebreak",
            MLP_METHOD,
            UNET_METHOD,
            ORACLE_METHOD,
            COARSE_ORACLE_METHOD,
            "manhattan_true_distance_tiebreak",
        ]:
            row = by_method[method]
            file.write(
                f"| {method} | {row['mean_expanded']:.3f} | {row['optimality_rate']:.3f} | {row['mean_cost_gap']:.3f} |\n"
            )
        file.write("\n")

        file.write("## Adaptive Headroom\n\n")
        file.write(f"- Oracle selector gain vs MLP tie-break: {gain['oracle_selector_gain_vs_mlp']:.3f} expanded nodes.\n")
        file.write(f"- Oracle selector gain vs U-Net tie-break: {gain['oracle_selector_gain_vs_unet']:.3f} expanded nodes.\n")
        file.write(
            f"- Oracle selector gain vs best single method ({best_single_name}): "
            f"{gain['oracle_selector_gain_vs_best_single_method']:.3f} expanded nodes.\n"
        )
        file.write(
            f"- Exact oracle gap to true-distance oracle: {gain['oracle_gap_to_true_distance_oracle']:.3f} expanded nodes.\n"
        )
        file.write(f"- Coarse two-category oracle expanded nodes: {coarse['mean_expanded']:.3f}.\n")
        file.write(
            f"- Coarse oracle gap to exact structured-type oracle: {gain['coarse_oracle_gap_to_exact_oracle']:.3f} expanded nodes.\n\n"
        )

        file.write("## Key Questions\n\n")
        file.write(
            f"Q1: Perfect structure-aware MLP/U-Net selection improves over the best single learned tie-break by "
            f"{gain['oracle_selector_gain_vs_best_single_method']:.3f} expanded nodes on average.\n\n"
        )
        if gain["oracle_selector_gain_vs_best_single_method"] < 1.0:
            file.write(
                "Q2: The gain is small, so adaptive selection by these four structure labels alone is unlikely to justify much "
                "additional algorithmic complexity.\n\n"
            )
        else:
            file.write(
                "Q2: The gain is non-trivial, so structure-aware heuristic selection may be worth further investigation.\n\n"
            )
        unet_types = [row["structured_type"] for row in type_rows if row["oracle_choice"] == UNET_METHOD]
        mlp_types = [row["structured_type"] for row in type_rows if row["oracle_choice"] == MLP_METHOD]
        file.write(f"Q3: Structured types preferring U-Net: {', '.join(unet_types) if unet_types else 'none'}.\n\n")
        file.write(f"Q4: Structured types preferring MLP: {', '.join(mlp_types) if mlp_types else 'none'}.\n\n")
        file.write(
            f"Q5: The exact selector remains {exact['mean_expanded'] - true_oracle['mean_expanded']:.3f} expanded nodes above the "
            "true-distance tie-break oracle, so useful tie-breaking headroom remains.\n\n"
        )

        file.write("## Coarse Category Upper Bound\n\n")
        file.write("| Coarse category | MLP expanded | U-Net expanded | Choice |\n")
        file.write("|---|---:|---:|---|\n")
        for row in coarse_rows:
            file.write(
                f"| {row['structured_type']} | {row['mean_expanded_mlp_tiebreak']:.3f} | "
                f"{row['mean_expanded_unet_tiebreak']:.3f} | {row['oracle_choice']} |\n"
            )
        file.write("\n")

        file.write("## Interpretation\n\n")
        file.write(
            "This analysis estimates the value of adaptation only. It does not implement an adaptive selector and does not claim "
            "that structured_type labels are available during inference. The observed exact-structure oracle gain should be treated "
            "as an upper bound for choosing between MLP and U-Net tie-breaking using this label set.\n"
        )


def analyze(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "oracle_selector_analysis")
    os.makedirs(output_dir, exist_ok=True)

    structured_rows = read_csv(args.structured_results)
    maps = group_maps(structured_rows)
    exact_choices, type_rows = choose_by_type(maps)
    coarse_choices, coarse_rows = choose_by_coarse_type(maps)
    comparisons = comparison_rows(maps, exact_choices, coarse_choices)
    gains = add_gain_rows(comparisons)
    all_rows = type_rows + coarse_rows + comparisons + gains

    write_csv(os.path.join(output_dir, "oracle_selector_statistics.csv"), all_rows)
    write_summary(os.path.join(output_dir, "oracle_selector_summary.md"), type_rows, coarse_rows, comparisons, gains)
    save_plot(os.path.join(output_dir, "oracle_selector_plots.png"), comparisons)
    print(f"Saved oracle selector analysis outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze oracle structure-aware selection between MLP and U-Net tie-breaking.")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_tiebreak_controls_100.csv")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_tiebreak_controls_100.csv")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
