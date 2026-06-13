import argparse
import csv
import os
import re


METHODS = ["dijkstra", "manhattan", "mlp_table", "unet"]
STRUCTURED_TYPES = ["maze_like", "bottleneck", "large_block", "narrow_corridor"]


def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_text(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


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


def group_by(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in keys), []).append(row)
    return groups


def summarize_methods(rows):
    output = {}
    for method in METHODS:
        scoped = [row for row in rows if row.get("heuristic") == method and not row.get("skip_reason")]
        output[method] = {
            "runs": len(scoped),
            "optimality": mean(to_float(row, "optimal") for row in scoped),
            "expanded": mean(to_float(row, "expanded_nodes") for row in scoped),
            "path_length": mean(to_float(row, "path_length") for row in scoped),
            "overestimate": mean(to_float(row, "overestimate_rate") for row in scoped),
            "mae": mean(to_float(row, "mae") for row in scoped),
            "rmse": mean(to_float(row, "rmse") for row in scoped),
        }
    return output


def summarize_structured(rows):
    output = {}
    for structured_type in STRUCTURED_TYPES:
        scoped = [row for row in rows if row.get("structured_type") == structured_type]
        output[structured_type] = summarize_methods(scoped)
    return output


def extract_section(text, heading):
    pattern = rf"(^## {re.escape(heading)}\n.*?)(?=^## |\Z)"
    match = re.search(pattern, text, flags=re.M | re.S)
    return match.group(1).strip() if match else ""


def extract_bullets(text, limit=6):
    bullets = []
    for line in text.splitlines():
        if line.startswith("- "):
            bullets.append(line)
        if len(bullets) >= limit:
            break
    return bullets


def extract_group_table_rows(text):
    section = extract_section(text, "Concise Group Table")
    rows = []
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) >= 5:
            rows.append(
                {
                    "group": parts[0].strip("`"),
                    "behavior": parts[1],
                    "mechanism": parts[3],
                    "evidence": parts[4],
                }
            )
    return rows


def pct_reduction(baseline, value):
    if baseline == 0:
        return 0.0
    return 100.0 * (baseline - value) / baseline


def method_sentence(name, stats):
    return (
        f"{name}: expanded={stats['expanded']:.2f}, optimality={stats['optimality']:.3f}, "
        f"path_length={stats['path_length']:.2f}, overestimate={stats['overestimate']:.3f}"
    )


def write_method_table(file, summary):
    file.write("| Method | Mean expanded | Optimality | Mean path length | Overestimate rate | MAE | RMSE |\n")
    file.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for method in METHODS:
        stats = summary[method]
        file.write(
            f"| {method} | {stats['expanded']:.2f} | {stats['optimality']:.3f} | "
            f"{stats['path_length']:.2f} | {stats['overestimate']:.3f} | {stats['mae']:.3f} | {stats['rmse']:.3f} |\n"
        )
    file.write("\n")


def write_structured_table(file, structured_summary):
    file.write("| Structure | U-Net expanded | MLP expanded | Manhattan expanded | U-Net optimality | U-Net minus MLP expanded |\n")
    file.write("|---|---:|---:|---:|---:|---:|\n")
    for structured_type in STRUCTURED_TYPES:
        stats = structured_summary[structured_type]
        unet = stats["unet"]
        mlp = stats["mlp_table"]
        manhattan = stats["manhattan"]
        file.write(
            f"| {structured_type} | {unet['expanded']:.2f} | {mlp['expanded']:.2f} | "
            f"{manhattan['expanded']:.2f} | {unet['optimality']:.3f} | {unet['expanded'] - mlp['expanded']:.2f} |\n"
        )
    file.write("\n")


def load_optional_summary(path):
    return read_csv(path) if path and os.path.exists(path) else []


def write_benchmark_evolution(file, fixed_summary_rows, structure_summary_rows):
    file.write("## 2. Benchmark Evolution\n\n")
    file.write(
        "The evaluation evolved from a fixed start-goal protocol to random start-goal sampling, then to structure-aware "
        "analysis and controlled structured maps. This progression matters because the fixed protocol overemphasized "
        "global coordinate regularities and made the MLP table heuristic appear stronger than it remained under more "
        "varied start-goal conditions.\n\n"
    )
    if fixed_summary_rows:
        fixed_unet = [row for row in fixed_summary_rows if row.get("heuristic") == "unet"]
        fixed_mlp = [row for row in fixed_summary_rows if row.get("heuristic") == "mlp_table"]
        file.write(
            f"In the fixed start-goal benchmark, U-Net averaged {mean(to_float(r, 'mean_expanded_nodes') for r in fixed_unet):.2f} "
            f"expanded nodes, while MLP table averaged {mean(to_float(r, 'mean_expanded_nodes') for r in fixed_mlp):.2f}. "
            "This was the setting where the MLP advantage was most pronounced.\n\n"
        )
    file.write(
        "Random start-goal evaluation reduced that gap by removing the single-route bias. The controlled structured benchmark "
        "then separated geometric regimes that were mixed together in random maps.\n\n"
    )
    if structure_summary_rows:
        file.write(
            "The structure-aware benchmark grouped random maps into labels such as open space, sparse obstacles, dense obstacles, "
            "maze-like, bottleneck, large obstacle block, narrow corridor, and multiple alternative routes. These categories "
            "showed that learned heuristics should not be judged only by aggregate averages: the same heuristic can be useful "
            "in one geometry class and fragile in another.\n\n"
        )


def create_report(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "research_synthesis")
    os.makedirs(output_dir, exist_ok=True)

    random_rows = read_csv(args.random_results)
    structured_rows = read_csv(args.structured_results)
    qualitative = read_text(args.qualitative_analysis)
    mechanism = read_text(args.mechanism_summary)
    route_critical = read_text(args.route_critical_summary)
    fixed_summary_rows = load_optional_summary(args.fixed_summary)
    structure_summary_rows = load_optional_summary(args.structure_summary)

    random_summary = summarize_methods(random_rows)
    structured_summary = summarize_methods(structured_rows)
    structured_by_type = summarize_structured(structured_rows)
    all_rows = random_rows + structured_rows
    all_summary = summarize_methods(all_rows)

    output_path = os.path.join(output_dir, "research_synthesis.md")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("# Research Synthesis: Learned Heuristics for A* Search\n\n")
        file.write(
            "This report synthesizes the benchmark, qualitative, mechanism-validation, and route-critical-cell analyses. "
            "It is written as a Results and Discussion narrative and does not propose or evaluate any new search algorithm.\n\n"
        )

        file.write("## 1. Experimental Setup\n\n")
        file.write(
            "The experiments compare four A* heuristic settings: Dijkstra/zero heuristic, Manhattan distance, an MLP table "
            "heuristic, and a U-Net heuristic. The synthesis uses 400 random start-goal maps and 1600 controlled structured "
            "maps, for 2000 evaluated maps in the mechanism-level analyses. The structured benchmark contains four controlled "
            "map families: maze_like, bottleneck, large_block, and narrow_corridor.\n\n"
        )
        file.write("Random start-goal benchmark summary:\n\n")
        write_method_table(file, random_summary)
        file.write("Controlled structured benchmark summary:\n\n")
        write_method_table(file, structured_summary)

        write_benchmark_evolution(file, fixed_summary_rows, structure_summary_rows)

        file.write("## 3. Main Empirical Findings\n\n")
        dijkstra = all_summary["dijkstra"]
        manhattan = all_summary["manhattan"]
        mlp = all_summary["mlp_table"]
        unet = all_summary["unet"]
        file.write(
            f"Across the random and controlled structured benchmarks, U-Net substantially reduces search relative to Dijkstra: "
            f"{method_sentence('Dijkstra', dijkstra)}; {method_sentence('U-Net', unet)}. "
            f"This corresponds to a {pct_reduction(dijkstra['expanded'], unet['expanded']):.1f}% reduction in mean expanded nodes.\n\n"
        )
        file.write(
            f"Relative to Manhattan, U-Net also reduces expansions on average: Manhattan expanded {manhattan['expanded']:.2f} nodes "
            f"versus U-Net {unet['expanded']:.2f}, a {pct_reduction(manhattan['expanded'], unet['expanded']):.1f}% reduction. "
            "However, unlike Manhattan, U-Net is not guaranteed admissible in these experiments and has a lower optimality rate.\n\n"
        )
        file.write(
            f"Relative to MLP table, U-Net remains slightly weaker on the aggregate expansion metric: MLP expanded {mlp['expanded']:.2f} "
            f"nodes versus U-Net {unet['expanded']:.2f}. This confirms that U-Net is competitive but not uniformly dominant. "
            "The important result is therefore not that U-Net is always the best heuristic, but that its behavior is strongly "
            "conditioned by map structure and by how its predicted field orders route-relevant regions.\n\n"
        )

        file.write("## 4. Structure-Dependent Behavior\n\n")
        file.write(
            "The structured benchmark shows that learned-heuristic quality is geometry-dependent. The aggregate table below "
            "summarizes the controlled structured families.\n\n"
        )
        write_structured_table(file, structured_by_type)
        file.write(
            "In geometry_easy or open-space-like settings, MLP table often has a strong advantage because coordinate-distance "
            "regularities are sufficient and obstacles do not require much global scene interpretation. In obstacle_structured "
            "settings, including bottlenecks, large blocks, and corridors, U-Net sometimes benefits from spatial context but also "
            "becomes vulnerable to localized overestimation. Maze-like maps are the most favorable controlled structure for U-Net: "
            "its expansion gap relative to MLP is near zero and sometimes negative, suggesting useful global route bias. Large-block "
            "and narrow-corridor maps expose the clearest failure modes, with higher U-Net expansion and lower optimality.\n\n"
        )

        file.write("## 5. Qualitative Case-Study Findings\n\n")
        qualitative_groups = extract_group_table_rows(qualitative)
        if qualitative_groups:
            file.write("| Case group | Typical behavior | Mechanism interpretation |\n")
            file.write("|---|---|---|\n")
            for row in qualitative_groups:
                file.write(f"| {row['group']} | {row['behavior']} | {row['mechanism']} |\n")
            file.write("\n")
        file.write(
            "The qualitative cases support a consistent interpretation: U-Net succeeds when its field provides a useful global "
            "bias toward the route family that contains the solution, and fails when positive error regions make necessary passages "
            "or detours look artificially expensive. Maze-like U-Net wins are mostly route-bias cases; large-block and narrow-corridor "
            "failures are mostly barrier-like cases.\n\n"
        )

        file.write("## 6. Mechanism Validation Findings\n\n")
        file.write(
            "The route-ordering hypothesis is supported: U-Net-win maps have better relative ordering than MLP-win maps by 0.029. "
            "This is the clearest mechanism-level result because it directly connects search efficiency with the relative ordering "
            "of useful regions rather than only absolute regression error.\n\n"
        )
        file.write(
            "The barrier hypothesis is also supported: non-optimal U-Net maps have higher overestimation, with an overestimate gap "
            "of 0.122 and a large-overestimate gap of 0.114. This supports the interpretation that localized positive error can "
            "make useful routes appear too expensive.\n\n"
        )
        file.write(
            "The corridor smoothness hypothesis remains suggestive rather than strongly supported. The roughness relationship in "
            "high-corridor maps is weak, with Spearman 0.028, so global smoothness is probably too coarse for corridor-specific "
            "failure analysis.\n\n"
        )

        file.write("## 7. Route-Critical-Cell Findings\n\n")
        route_q1 = extract_section(route_critical, "Q1: U-Net Wins")
        route_q2 = extract_section(route_critical, "Q2: U-Net Non-Optimality")
        route_q3 = extract_section(route_critical, "Q3: Predictive Power")
        for section in [route_q1, route_q2, route_q3]:
            if section:
                file.write(section.replace("## ", "### "))
                file.write("\n\n")
        file.write(
            "These results refine the mechanism story. Route-critical cells are more informative for U-Net optimality failures, "
            "especially through large overestimation near critical regions. They are not yet sufficient to explain expansion "
            "reduction: in the current analysis, global MAE remains more predictive of U-Net-minus-MLP expansion gap than the tested "
            "route-critical metrics.\n\n"
        )

        file.write("## 8. Supported Claims\n\n")
        file.write("### Strongly Supported Conclusions\n\n")
        file.write(
            "- U-Net substantially improves over Dijkstra and usually improves over Manhattan in expansion efficiency, but sacrifices some optimality.\n"
            "- The MLP table heuristic remains slightly better than U-Net on aggregate expansion, especially in easy geometric settings.\n"
            "- U-Net behavior is structure-dependent; maze-like maps are comparatively favorable, while large-block and narrow-corridor maps are more fragile.\n"
            "- Route ordering is a supported mechanism for U-Net wins, and overestimation barriers are supported for U-Net non-optimality.\n\n"
        )
        file.write("### Moderately Supported Conclusions\n\n")
        file.write(
            "- U-Net success is better described as useful global route bias than as uniformly better distance regression.\n"
            "- Route-critical overestimation is more predictive of optimality gap than global overestimation alone.\n"
            "- Qualitative corridor failures likely involve localized passage barriers, although global roughness does not strongly validate this.\n\n"
        )
        file.write("### Speculative Hypotheses\n\n")
        file.write(
            "- A learned heuristic may need route-critical calibration more than global calibration to preserve optimality.\n"
            "- Some expansion advantages may depend on broad global field shape, while optimality failures depend on local high-error cells.\n"
            "- Corridor failures may require passage-aware local metrics rather than map-wide smoothness or consistency summaries.\n\n"
        )

        file.write("## 9. Limitations\n\n")
        file.write(
            "- The mechanism metrics are observational and do not prove causality.\n"
            "- Ordering metrics are computed over map cells, not over the dynamic A* open list.\n"
            "- Route-critical cells are approximated from one reconstructed optimal path and local structural proxies.\n"
            "- Controlled structured maps are simplified generators, not a complete path-planning distribution.\n"
            "- Runtime values are small and implementation-dependent, so expanded nodes are the more stable efficiency measure.\n\n"
        )

        file.write("## 10. Future Research Directions\n\n")
        file.write(
            "Future analysis should focus on sharper local mechanism measurements rather than algorithm changes. The most useful next "
            "questions are whether overestimation concentrates on all optimal paths or only one reconstructed path, whether bottleneck "
            "and gap cells can be detected more robustly, and whether A* open-list ordering agrees with the static map-cell ordering "
            "metrics. Additional benchmarks should vary map scale, obstacle topology, and start-goal distance while preserving the "
            "current separation between global field quality, route-critical calibration, and search behavior.\n\n"
        )

        file.write("## Overall Interpretation\n\n")
        file.write(
            "The evidence supports a nuanced view of learned heuristics for A*. U-Net is not simply a better distance estimator than "
            "MLP table, nor is average prediction error sufficient to explain search behavior. Its value appears when spatial context "
            "creates useful route ordering, especially in structured maps where Manhattan is under-informed. Its main risk appears when "
            "localized overestimation creates artificial barriers near route-critical cells. The central lesson is that learned heuristic "
            "evaluation should measure both global search guidance and local route-critical reliability.\n"
        )

    print(f"Saved research synthesis to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a research synthesis report for learned A* heuristics.")
    parser.add_argument("--random-results", default="outputs/experiments/results_random_sg_100.csv")
    parser.add_argument("--structured-results", default="outputs/experiments/results_structured_controlled_100.csv")
    parser.add_argument("--qualitative-analysis", default="outputs/case_studies/structured_controlled_100/qualitative_analysis.md")
    parser.add_argument("--mechanism-summary", default="outputs/mechanism_validation/mechanism_validation_summary.md")
    parser.add_argument("--route-critical-summary", default="outputs/route_critical_analysis/route_critical_summary.md")
    parser.add_argument("--fixed-summary", default="outputs/experiments/summary_fixed_100.csv")
    parser.add_argument("--structure-summary", default="outputs/structure_benchmark/results_random_sg_100/structure_summary.csv")
    return parser.parse_args()


if __name__ == "__main__":
    create_report(parse_args())
