import argparse
import csv
import math
import os

import numpy as np

from analyze_search_behavior_state import load_behavior_rows, read_csv, to_float, write_csv


CANDIDATES = [
    "goal_progress_rate",
    "frontier_spread",
    "frontier_growth_rate",
    "local_branching",
    "tie_set_density",
    "expansion_persistence",
    "heuristic_disagreement",
    "heuristic_variance",
]
DOWNSTREAM = ["expanded_nodes", "route_bias", "off_path_exploration", "search_depth"]
ALL_VARIABLES = CANDIDATES + DOWNSTREAM


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def std(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / len(values))


def pearson(xs, ys):
    xs = list(xs)
    ys = list(ys)
    if len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom_x = math.sqrt(sum(x * x for x in dx))
    denom_y = math.sqrt(sum(y * y for y in dy))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def standardize_matrix(matrix):
    matrix = np.array(matrix, dtype=float)
    return (matrix - matrix.mean(axis=0)) / np.where(matrix.std(axis=0) == 0, 1.0, matrix.std(axis=0))


def residualize(y, controls):
    y = np.array(y, dtype=float)
    if controls.size == 0:
        return y - y.mean()
    x = np.column_stack([np.ones(len(y)), controls])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ coef


def partial_corr(x, y, controls):
    rx = residualize(x, controls)
    ry = residualize(y, controls)
    return pearson(rx, ry)


def regression_r2(y, x):
    y = np.array(y, dtype=float)
    if x.size == 0:
        prediction = np.repeat(y.mean(), len(y))
    else:
        design = np.column_stack([np.ones(len(y)), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        prediction = design @ coef
    denom = np.sum((y - y.mean()) ** 2)
    if denom == 0:
        return 0.0
    return 1.0 - float(np.sum((y - prediction) ** 2) / denom)


def prepare_rows(online_rows, oracle_rows):
    rows = load_behavior_rows(online_rows, oracle_rows)
    for row in rows:
        row["expanded_nodes"] = row["best_expanded"]
    return rows


def values(rows, variable):
    return [to_float(row, variable) for row in rows]


def dependency_matrix(rows):
    output = []
    data = {variable: values(rows, variable) for variable in ALL_VARIABLES if variable in rows[0]}
    candidates_present = [variable for variable in CANDIDATES if variable in data]
    controls_all = standardize_matrix([ [row[var] for var in candidates_present] for row in rows ])
    for source in data:
        for target in data:
            if source == target:
                continue
            raw = pearson(data[source], data[target])
            controls = [var for var in candidates_present if var not in {source, target}]
            control_matrix = standardize_matrix([[row[var] for var in controls] for row in rows]) if controls else np.empty((len(rows), 0))
            partial = partial_corr(data[source], data[target], control_matrix)
            output.append(
                {
                    "source": source,
                    "target": target,
                    "raw_correlation": raw,
                    "partial_correlation": partial,
                    "abs_partial_correlation": abs(partial),
                    "edge_supported": int(abs(partial) >= 0.15 and abs(raw) >= 0.20),
                    "controls": ";".join(controls),
                }
            )
    return output


def mediation(rows):
    output = []
    y = np.array(values(rows, "expanded_nodes"), dtype=float)
    for source in CANDIDATES:
        if source not in rows[0]:
            continue
        x = np.array(values(rows, source), dtype=float)
        total = pearson(x, y)
        for mediator in CANDIDATES:
            if mediator == source or mediator not in rows[0]:
                continue
            m = np.array(values(rows, mediator), dtype=float)
            direct = partial_corr(x, y, standardize_matrix(m.reshape(-1, 1)))
            mediated_fraction = 0.0
            if abs(total) > 1e-9:
                mediated_fraction = max(0.0, min(1.0, 1.0 - abs(direct) / abs(total)))
            output.append(
                {
                    "source": source,
                    "mediator": mediator,
                    "target": "expanded_nodes",
                    "source_target_corr": total,
                    "source_target_partial_given_mediator": direct,
                    "mediated_fraction_abs": mediated_fraction,
                    "mediation_supported": int(abs(total) >= 0.20 and mediated_fraction >= 0.30),
                }
            )
    return sorted(output, key=lambda row: row["mediated_fraction_abs"], reverse=True)


def role_classification(matrix_rows, mediation_rows):
    supported = [row for row in matrix_rows if row["edge_supported"]]
    incoming = {var: 0 for var in ALL_VARIABLES}
    outgoing = {var: 0 for var in ALL_VARIABLES}
    for row in supported:
        source = row["source"]
        target = row["target"]
        if source in CANDIDATES and target in ALL_VARIABLES:
            outgoing[source] += 1
            incoming[target] += 1
    mediated = {
        row["mediator"]
        for row in mediation_rows
        if row["mediation_supported"] and row["mediator"] in CANDIDATES
    }
    roles = []
    for variable in ALL_VARIABLES:
        if variable not in incoming:
            continue
        if variable in DOWNSTREAM:
            role = "downstream_consequence"
        elif variable in mediated:
            role = "mediator"
        elif outgoing[variable] >= incoming[variable]:
            role = "likely_upstream"
        else:
            role = "mixed_or_downstream_behavior"
        roles.append(
            {
                "variable": variable,
                "role": role,
                "supported_incoming_edges": incoming[variable],
                "supported_outgoing_edges": outgoing[variable],
            }
        )
    return roles


def redundancy_groups(rows):
    groups = []
    seen = set()
    for i, left in enumerate(CANDIDATES):
        if left in seen or left not in rows[0]:
            continue
        group = [left]
        for right in CANDIDATES[i + 1 :]:
            if right in rows[0] and abs(pearson(values(rows, left), values(rows, right))) >= 0.85:
                group.append(right)
                seen.add(right)
        seen.add(left)
        groups.append(group)
    return groups


def minimal_control_state(rows):
    y = np.array(values(rows, "expanded_nodes"), dtype=float)
    candidates = [var for var in CANDIDATES if var in rows[0]]
    selected = []
    remaining = list(candidates)
    history = []
    current_r2 = 0.0
    while remaining and len(selected) < 5:
        best = None
        for variable in remaining:
            trial = selected + [variable]
            x = standardize_matrix([[row[var] for var in trial] for row in rows])
            r2 = regression_r2(y, x)
            gain = r2 - current_r2
            candidate = (gain, r2, variable)
            if best is None or candidate > best:
                best = candidate
        gain, r2, variable = best
        if gain < 0.02 and selected:
            break
        selected.append(variable)
        remaining.remove(variable)
        current_r2 = r2
        history.append({"step": len(selected), "added_variable": variable, "r2": r2, "incremental_r2": gain})
    return selected, history


def write_minimal_state(path, selected, history, roles, groups):
    role_lookup = {row["variable"]: row["role"] for row in roles}
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Minimal Candidate Control State\n\n")
        file.write(
            "This is a compact online-measurable behavior state selected by incremental explained variance over expanded nodes. "
            "It is not an algorithm proposal.\n\n"
        )
        file.write("## Selected Variables\n\n")
        for variable in selected:
            file.write(f"- {variable}: {role_lookup.get(variable, 'unknown')}\n")
        file.write("\n## Incremental Fit\n\n")
        file.write("| step | added variable | R2 | incremental R2 |\n|---:|---|---:|---:|\n")
        for row in history:
            file.write(f"| {row['step']} | {row['added_variable']} | {row['r2']:.3f} | {row['incremental_r2']:.3f} |\n")
        file.write("\n## Redundancy Groups\n\n")
        for group in groups:
            if len(group) > 1:
                file.write(f"- {' / '.join(group)}\n")


def save_graph(path, roles, matrix_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    role_lookup = {row["variable"]: row["role"] for row in roles}
    upstream = [row["variable"] for row in roles if row["role"] == "likely_upstream"]
    mediators = [row["variable"] for row in roles if row["role"] == "mediator"]
    downstream = [var for var in ["route_bias", "off_path_exploration", "search_depth", "expanded_nodes"] if var in role_lookup]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")
    columns = [(0.15, upstream, "#e8eef8", "upstream"), (0.5, mediators, "#fff4d6", "mediators"), (0.85, downstream, "#e8f6ea", "downstream")]
    positions = {}
    for x, variables, color, title in columns:
        ys = np.linspace(0.85, 0.15, max(1, len(variables)))
        ax.text(x, 0.96, title, ha="center", va="center", fontsize=12, fontweight="bold")
        for y, variable in zip(ys, variables):
            positions[variable] = (x, y)
            ax.text(x, y, variable, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round", fc=color))
    edges = sorted(
        [row for row in matrix_rows if row["edge_supported"]],
        key=lambda row: abs(row["partial_correlation"]),
        reverse=True,
    )[:24]
    for row in edges:
        source = row["source"]
        target = row["target"]
        if source not in positions or target not in positions or positions[source][0] >= positions[target][0]:
            continue
        color = "#2f6f44" if row["partial_correlation"] > 0 else "#9d3a35"
        ax.add_patch(
            FancyArrowPatch(
                positions[source],
                positions[target],
                arrowstyle="->",
                mutation_scale=10,
                alpha=0.45,
                color=color,
            )
        )
    ax.set_title("Behavior Dependency Graph (partial-correlation supported)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_summary(path, roles, matrix_rows, mediation_rows, selected, history, groups):
    upstream = [row["variable"] for row in roles if row["role"] == "likely_upstream"]
    mediators = [row["variable"] for row in roles if row["role"] == "mediator"]
    downstream = [row["variable"] for row in roles if row["role"] == "downstream_consequence"]
    strongest_to_expanded = sorted(
        [row for row in matrix_rows if row["target"] == "expanded_nodes" and row["source"] in CANDIDATES],
        key=lambda row: abs(row["partial_correlation"]),
        reverse=True,
    )[:8]
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Behavior Dependency Analysis\n\n")
        file.write(
            "This analysis estimates dependency structure among behavior-state variables using correlation, partial correlation, "
            "and simple mediation diagnostics. It does not prove causality and does not propose a new algorithm.\n\n"
        )
        file.write("## Roles\n\n")
        file.write(f"- Likely upstream variables: {', '.join(upstream) if upstream else 'none'}.\n")
        file.write(f"- Mediator variables: {', '.join(mediators) if mediators else 'none'}.\n")
        file.write(f"- Downstream consequences: {', '.join(downstream) if downstream else 'none'}.\n\n")
        file.write("## Direct Relationship With Expanded Nodes\n\n")
        file.write("| variable | raw corr | partial corr |\n|---|---:|---:|\n")
        for row in strongest_to_expanded:
            file.write(f"| {row['source']} | {row['raw_correlation']:.3f} | {row['partial_correlation']:.3f} |\n")
        file.write("\n")
        file.write("## Strongest Mediation Patterns\n\n")
        file.write("| source | mediator | total corr | direct corr | mediated fraction |\n|---|---|---:|---:|---:|\n")
        for row in mediation_rows[:10]:
            file.write(
                f"| {row['source']} | {row['mediator']} | {row['source_target_corr']:.3f} "
                f"| {row['source_target_partial_given_mediator']:.3f} | {row['mediated_fraction_abs']:.3f} |\n"
            )
        file.write("\n")
        file.write("## Redundant Variables\n\n")
        for group in groups:
            if len(group) > 1:
                file.write(f"- {' / '.join(group)}\n")
        file.write("\n## Minimal Candidate Control State\n\n")
        file.write(", ".join(selected) + "\n\n")
        file.write("Incremental R2 path:\n\n")
        for row in history:
            file.write(f"- step {row['step']}: add {row['added_variable']}, R2={row['r2']:.3f}, gain={row['incremental_r2']:.3f}\n")
        file.write("\n## Critical Answers\n\n")
        file.write(
            "- Upstream candidates are variables whose partial relationships remain after conditioning on other behavior variables; "
            "they should be treated as candidate controls, not causal facts.\n"
        )
        file.write(
            "- Consequence variables include expanded_nodes and oracle-only diagnostics such as route_bias/off_path_exploration; "
            "these are useful readouts but poor direct controls.\n"
        )
        file.write(
            "- Mediators are variables that absorb part of another variable's relationship with expanded_nodes; they are useful for "
            "understanding dependency chains.\n"
        )
        file.write(
            "- The minimal state is the smallest online-measurable set that explains most observable variation under this linear diagnostic.\n"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze dependencies among search behavior variables.")
    parser.add_argument("--behavior-correlations", default="outputs/search_behavior_state/behavior_correlations.csv")
    parser.add_argument("--online-features", default="outputs/trust_signal_validation/online_features.csv")
    parser.add_argument("--oracle-features", default="outputs/trust_signal_validation/oracle_diagnostic_features.csv")
    parser.add_argument("--output-dir", default="outputs/behavior_dependency")
    return parser.parse_args()


def main(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Read behavior_correlations.csv to validate dependency on prior analysis outputs.
    behavior_corr = read_csv(os.path.join(project_root, args.behavior_correlations))
    if not behavior_corr:
        raise RuntimeError("behavior_correlations.csv is required.")

    online_rows = read_csv(os.path.join(project_root, args.online_features))
    oracle_rows = read_csv(os.path.join(project_root, args.oracle_features))
    rows = prepare_rows(online_rows, oracle_rows)

    dep_rows = dependency_matrix(rows)
    med_rows = mediation(rows)
    roles = role_classification(dep_rows, med_rows)
    groups = redundancy_groups(rows)
    selected, history = minimal_control_state(rows)

    write_csv(os.path.join(output_dir, "dependency_matrix.csv"), dep_rows)
    write_csv(os.path.join(output_dir, "mediation_results.csv"), med_rows)
    write_minimal_state(os.path.join(output_dir, "minimal_control_state.md"), selected, history, roles, groups)
    write_summary(os.path.join(output_dir, "behavior_dependency_summary.md"), roles, dep_rows, med_rows, selected, history, groups)
    save_graph(os.path.join(output_dir, "behavior_dependency_graph.png"), roles, dep_rows)
    print(f"Saved behavior dependency analysis to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
