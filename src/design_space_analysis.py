import argparse
import os
import re


def read_text(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def first_match(text, pattern, default=""):
    match = re.search(pattern, text, flags=re.S)
    return match.group(1).strip().rstrip(".") if match else default


def evidence_values(research, mechanism, route_critical):
    return {
        "unet_vs_dijkstra_reduction": first_match(
            research, r"corresponds to a ([0-9.]+)% reduction in mean expanded nodes"
        ),
        "unet_vs_manhattan_reduction": first_match(
            research, r"Manhattan expanded [0-9.]+ nodes versus U-Net [0-9.]+, a ([0-9.]+)% reduction"
        ),
        "ordering_gap": first_match(mechanism, r"relative ordering than MLP-win maps by ([0-9.]+)"),
        "over_gap": first_match(mechanism, r"over gap ([0-9.]+)"),
        "large_over_gap": first_match(mechanism, r"large-over gap ([0-9.]+)"),
        "roughness_spearman": first_match(mechanism, r"Roughness relationship is weak.*?Spearman ([0-9.]+)"),
        "critical_ordering_gap": first_match(
            route_critical, r"critical ordering advantage gap ([0-9.]+)"
        ),
        "global_ordering_gap": first_match(
            route_critical, r"global ordering advantage gap ([0-9.]+)"
        ),
        "nonoptimal_critical_over": first_match(
            route_critical, r"non-optimal U-Net maps, critical minus global overestimate rate is ([0-9.]+)"
        ),
        "optimal_critical_over": first_match(
            route_critical, r"optimal U-Net maps it is ([0-9.]+)"
        ),
        "global_expansion_spearman": first_match(
            route_critical, r"expansion gap, best global Spearman is (-?[0-9.]+)"
        ),
        "critical_expansion_spearman": first_match(
            route_critical, r"expansion gap, best global Spearman is -?[0-9.]+ .*?best critical Spearman is (-?[0-9.]+)"
        ),
        "global_gap_spearman": first_match(
            route_critical, r"optimality gap, best global Spearman is ([0-9.]+)"
        ),
        "critical_gap_spearman": first_match(
            route_critical, r"optimality gap, best global Spearman is [0-9.]+ .*?best critical Spearman is ([0-9.]+)"
        ),
    }


def write_report(path, values):
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Design-Space Analysis For Future Learned-Heuristic Search\n\n")
        file.write(
            "This report uses only existing benchmark and mechanism-analysis evidence. It does not modify A*, retrain models, "
            "or propose a specific new algorithm. Its purpose is to identify what a future A*-based learned-heuristic method "
            "should preserve and what it should avoid.\n\n"
        )

        file.write("## Evidence Base\n\n")
        file.write(
            f"The current evidence shows that U-Net reduces expansions relative to Dijkstra by {values['unet_vs_dijkstra_reduction']}% "
            f"and relative to Manhattan by {values['unet_vs_manhattan_reduction']}%, while remaining weaker than MLP table on aggregate. "
            "The key design-space question is therefore not whether learned heuristics can help, but which learned-heuristic properties "
            "produce useful search guidance without damaging optimality.\n\n"
        )

        file.write("## 1. Properties Associated With Expansion Reduction\n\n")
        file.write(
            "The clearest expansion-related property is ordering quality. Benchmark-wide mechanism validation found that U-Net-win maps "
            f"have better relative ordering than MLP-win maps by {values['ordering_gap']}. Route-critical analysis sharpened this result: "
            f"the route-critical ordering advantage gap is {values['critical_ordering_gap']}, compared with a global ordering advantage "
            f"gap of {values['global_ordering_gap']}.\n\n"
        )
        file.write(
            "Route bias also appears beneficial. The qualitative maze-like successes show that U-Net can reduce exploration when its "
            "field gives A* a useful global preference toward the route family containing the solution. This supports preserving global "
            "spatial context, especially in structured maps where Manhattan distance is under-informed.\n\n"
        )
        file.write(
            f"Global MAE remains relevant. In route-critical analysis, the best global predictor of U-Net-minus-MLP expansion gap had "
            f"Spearman {values['global_expansion_spearman']}, while the best tested critical predictor had Spearman "
            f"{values['critical_expansion_spearman']}. This means expansion efficiency is not explained only by route-critical cells; "
            "broad field quality still matters.\n\n"
        )

        file.write("## 2. Properties Associated With Optimality Failures\n\n")
        file.write(
            f"Overestimation is the strongest failure signal. Non-optimal U-Net maps have higher overestimation, with an overestimate "
            f"gap of {values['over_gap']} and a large-overestimate gap of {values['large_over_gap']}. This supports the artificial-barrier "
            "interpretation: positive error can make useful paths appear too expensive.\n\n"
        )
        file.write(
            f"Route-critical overestimation is more diagnostic than global overestimation for optimality failures. Critical minus global "
            f"overestimate rate is {values['nonoptimal_critical_over']} on non-optimal U-Net maps and {values['optimal_critical_over']} "
            f"on optimal U-Net maps. For U-Net optimality gap, the best critical Spearman is {values['critical_gap_spearman']}, stronger "
            f"than the best global Spearman of {values['global_gap_spearman']}.\n\n"
        )
        file.write(
            "Artificial barriers near corridors, gaps, and detours appear harmful in qualitative cases. However, global corridor roughness "
            f"is weak as an explanatory metric, with Spearman {values['roughness_spearman']}. The harmful property is therefore not simply "
            "roughness everywhere; it is localized positive error at route-critical passages.\n\n"
        )

        file.write("## 3. Properties To Preserve\n\n")
        file.write("- Useful route ordering: predicted fields should rank route-relevant regions in a way that helps A* avoid unnecessary exploration.\n")
        file.write("- Global route bias: spatial context should remain available, especially in maze-like or obstacle-structured maps.\n")
        file.write("- Broad field quality: global MAE still correlates with expansion behavior and should not be ignored.\n")
        file.write("- Route-critical ordering: ordering near optimal paths and local passage neighborhoods appears especially relevant for U-Net wins.\n")
        file.write("- Compatibility with standard A*: beneficial heuristic behavior should improve guidance without requiring a different search objective in this evidence base.\n\n")

        file.write("## 4. Properties To Constrain\n\n")
        file.write("- Large overestimation: this is directly associated with U-Net optimality failures.\n")
        file.write("- Route-critical overestimation: large positive error near optimal paths, bottlenecks, gaps, or corridors is especially harmful.\n")
        file.write("- Artificial barriers: predicted fields should not make necessary passages or detours look prohibitively expensive.\n")
        file.write("- Structure-specific fragility: large-block and narrow-corridor failures indicate that average benchmark gains can hide local risks.\n")
        file.write("- Overreliance on global smoothness: map-wide roughness is too coarse to reliably detect corridor failures.\n\n")

        file.write("## 5. Taxonomy\n\n")
        file.write("### Beneficial\n\n")
        file.write("- High route-ordering quality.\n")
        file.write("- Useful global route bias.\n")
        file.write("- Low global MAE when it improves broad field reliability.\n")
        file.write("- Good route-critical ordering.\n")
        file.write("- Mild underestimation or mild error that does not create barriers.\n\n")
        file.write("### Harmful\n\n")
        file.write("- High large-overestimate rate.\n")
        file.write("- Route-critical overestimation.\n")
        file.write("- Local artificial heuristic barriers.\n")
        file.write("- Positive error concentrated near bottlenecks, corridors, gaps, and necessary detours.\n")
        file.write("- Geometry-specific failure modes hidden by aggregate averages.\n\n")
        file.write("### Mixed\n\n")
        file.write("- Global overestimation rate: MLP can overestimate frequently while remaining optimal, so frequency alone is less informative than location and magnitude.\n")
        file.write("- Global MAE: useful for expansion prediction, but insufficient to explain optimality failures.\n")
        file.write("- Field roughness: qualitatively plausible in corridors, but weak as a global metric.\n")
        file.write("- Spatial context: beneficial in maze-like maps, but can produce harmful barriers in large-block and narrow-corridor maps.\n\n")

        file.write("## 6. Future Algorithm-Design Principles\n\n")
        file.write(
            "Based only on current evidence, a future A*-based learned-heuristic algorithm should preserve learned global route guidance "
            "and route ordering, while explicitly avoiding route-critical overestimation and artificial barriers. It should treat "
            "expansion efficiency and optimality reliability as separate design targets: broad field quality helps explain expansion, "
            "whereas local route-critical errors better explain optimality failures.\n\n"
        )
        file.write(
            "The most promising design principles are therefore: preserve useful ordering, preserve spatial route bias, constrain large "
            "positive error near route-critical cells, evaluate behavior by structure class, and use local passage-aware diagnostics "
            "rather than relying only on global averages. These are principles, not an algorithm proposal.\n\n"
        )

        file.write("## Direct Answer\n\n")
        file.write(
            "If a new A*-based learned-heuristic algorithm is eventually designed, it should try to preserve global route bias, high "
            "ordering quality, route-critical ordering, and broad field reliability. It should try to avoid large overestimation, "
            "route-critical overestimation, artificial barriers near necessary passages, and failure modes that only appear under "
            "specific map structures.\n"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Create a learned-heuristic design-space analysis report.")
    parser.add_argument("--research-synthesis", default="outputs/research_synthesis/research_synthesis.md")
    parser.add_argument("--mechanism-summary", default="outputs/mechanism_validation/mechanism_validation_summary.md")
    parser.add_argument("--route-critical-summary", default="outputs/route_critical_analysis/route_critical_summary.md")
    return parser.parse_args()


def main(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "design_space_analysis")
    os.makedirs(output_dir, exist_ok=True)

    research = read_text(os.path.join(project_root, args.research_synthesis))
    mechanism = read_text(os.path.join(project_root, args.mechanism_summary))
    route_critical = read_text(os.path.join(project_root, args.route_critical_summary))
    values = evidence_values(research, mechanism, route_critical)

    output_path = os.path.join(output_dir, "design_space_analysis.md")
    write_report(output_path, values)
    print(f"Saved design-space analysis to {output_path}")


if __name__ == "__main__":
    main(parse_args())
