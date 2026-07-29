import argparse
import os
import re


def read_text(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def extract(text, pattern, default=""):
    match = re.search(pattern, text, flags=re.S)
    return match.group(1).strip().rstrip(".") if match else default


def collect_evidence(design_space, synthesis, mechanism, route_critical):
    return {
        "dijkstra_reduction": extract(
            synthesis, r"corresponds to a ([0-9.]+)% reduction in mean expanded nodes"
        ),
        "manhattan_reduction": extract(
            synthesis, r"Manhattan expanded [0-9.]+ nodes versus U-Net [0-9.]+, a ([0-9.]+)% reduction"
        ),
        "ordering_gap": extract(mechanism, r"relative ordering than MLP-win maps by ([0-9.]+)"),
        "critical_ordering_gap": extract(
            route_critical, r"critical ordering advantage gap ([0-9.]+)"
        ),
        "global_ordering_gap": extract(route_critical, r"global ordering advantage gap (?:of )?([0-9.]+)"),
        "global_expansion_spearman": extract(
            route_critical, r"expansion gap, best global Spearman is (-?[0-9.]+)"
        ),
        "critical_expansion_spearman": extract(
            route_critical,
            r"expansion gap, best global Spearman is -?[0-9.]+ .*?best critical Spearman is (-?[0-9.]+)",
        ),
        "over_gap": extract(mechanism, r"over gap ([0-9.]+)"),
        "large_over_gap": extract(mechanism, r"large-over gap ([0-9.]+)"),
        "nonoptimal_critical_over": extract(
            route_critical,
            r"non-optimal U-Net maps, critical minus global overestimate rate is ([0-9.]+)",
        ),
        "optimal_critical_over": extract(route_critical, r"optimal U-Net maps it is ([0-9.]+)"),
        "global_gap_spearman": extract(
            route_critical, r"optimality gap, best global Spearman is ([0-9.]+)"
        ),
        "critical_gap_spearman": extract(
            route_critical,
            r"optimality gap, best global Spearman is [0-9.]+ .*?best critical Spearman is ([0-9.]+)",
        ),
        "roughness_spearman": extract(mechanism, r"Roughness relationship is weak.*?Spearman ([0-9.]+)"),
        "design_direct_answer": extract(design_space, r"## Direct Answer\n\n(.*)"),
    }


def finding_block(file, label, finding, evidence, requirement, tradeoff):
    file.write(f"### {label}\n\n")
    file.write(f"- Finding: {finding}\n")
    file.write(f"- Supporting evidence: {evidence}\n")
    file.write(f"- Implied requirement: {requirement}\n")
    file.write(f"- Design trade-off: {tradeoff}\n\n")


def write_priority_table(file):
    rows = [
        (
            "Preserve useful route ordering",
            "high",
            "strong",
            "expanded nodes; U-Net minus MLP expansion gap",
            "Search may expand many unnecessary states even if distance MAE is acceptable.",
        ),
        (
            "Prioritize route-critical ordering",
            "high",
            "moderate",
            "route-critical ordering accuracy; expanded nodes",
            "Global ordering gains may miss the states that actually determine search behavior.",
        ),
        (
            "Preserve obstacle-aware route bias",
            "high",
            "moderate",
            "expanded nodes by structure class",
            "The method may regress toward Manhattan-like behavior on maze-like maps.",
        ),
        (
            "Maintain broad field reliability",
            "medium",
            "moderate",
            "global MAE; RMSE; expansion gap",
            "A search-specific heuristic may become locally clever but globally unstable.",
        ),
        (
            "Limit route-critical overestimation",
            "high",
            "strong",
            "cost gap; optimality rate; route-critical large-overestimate rate",
            "Optimal paths may be avoided because critical states look too expensive.",
        ),
        (
            "Avoid artificial heuristic barriers",
            "high",
            "strong",
            "cost gap; selected failure cases; large-overestimate rate",
            "Expansion gains may be bought by non-optimal paths through blocked-looking passages.",
        ),
        (
            "Use local passage-aware diagnostics",
            "medium",
            "moderate",
            "corridor/gap metrics; structure-stratified failures",
            "Global smoothness can falsely suggest a safe field while local passages fail.",
        ),
        (
            "Evaluate by structure class",
            "high",
            "strong",
            "maze_like; bottleneck; large_block; narrow_corridor metrics",
            "Aggregate gains may hide large_block or narrow_corridor regressions.",
        ),
        (
            "Track runtime when available",
            "low",
            "weak",
            "runtime",
            "Expanded-node gains may not translate into wall-clock gains.",
        ),
    ]
    file.write("| Requirement | Priority | Evidence strength | Affected metric | Design risk if ignored |\n")
    file.write("|---|---|---|---|---|\n")
    for row in rows:
        file.write("| " + " | ".join(row) + " |\n")
    file.write("\n")


def write_report(path, evidence):
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Algorithm Requirements For Future Learned-Heuristic A* Methods\n\n")
        file.write(
            "This report translates existing empirical findings into explicit requirements for a future A*-based learned-heuristic "
            "algorithm. It does not implement an algorithm, modify A*, retrain models, change loss functions, or modify model code.\n\n"
        )

        file.write("## 1. Motivation\n\n")
        file.write(
            "The current evidence shows that learned heuristics can reduce search effort, but also that learned fields can create "
            "non-optimal paths through localized overestimation. Requirements are needed before designing a new method so that the "
            "next phase optimizes the right properties: search efficiency, route guidance, calibration near critical cells, and "
            "structure-specific robustness. Without explicit requirements, a new method could improve aggregate expanded nodes while "
            "silently worsening optimality or failing on narrow corridors and large blocks.\n\n"
        )

        file.write("## 2. Empirical Findings -> Algorithm Requirements\n\n")
        finding_block(
            file,
            "A. Route Ordering Improves Expansion Reduction",
            "Maps where U-Net beats MLP have better relative route ordering.",
            f"Mechanism validation reports a relative ordering gap of {evidence['ordering_gap']}; U-Net also reduces expansions relative to Dijkstra by {evidence['dijkstra_reduction']}% and Manhattan by {evidence['manhattan_reduction']}%.",
            "Preserve useful route ordering.",
            "Ordering may improve efficiency even when absolute distance regression is imperfect, but ordering alone does not guarantee admissibility.",
        )
        finding_block(
            file,
            "B. Route-Critical Ordering Is More Informative Than Global Ordering",
            "Ordering near route-critical cells better separates U-Net-win behavior than global ordering alone.",
            f"Route-critical ordering advantage gap is {evidence['critical_ordering_gap']}, compared with global ordering advantage gap {evidence['global_ordering_gap']}.",
            "Prioritize ordering quality near route-critical cells.",
            "Focusing too narrowly on one reconstructed path may miss alternative optimal or near-optimal routes.",
        )
        finding_block(
            file,
            "C. Global Route Bias Helps In Maze-Like And Obstacle-Structured Maps",
            "U-Net is most competitive in maze-like maps, where spatial context can bias search toward useful route families.",
            "Research synthesis identifies maze_like as the most favorable controlled structure for U-Net and attributes successes to useful global route bias.",
            "Preserve obstacle-aware spatial guidance.",
            "Spatial context can help route selection, but the same context can produce harmful barriers in large_block and narrow_corridor maps.",
        )
        finding_block(
            file,
            "D. Global MAE Still Matters For Expansion Efficiency",
            "Expansion reduction is not explained only by route-critical metrics; broad field reliability remains useful.",
            f"The best global predictor of U-Net-minus-MLP expansion gap has Spearman {evidence['global_expansion_spearman']}, stronger than the best tested route-critical predictor at {evidence['critical_expansion_spearman']}.",
            "Maintain broad heuristic field reliability.",
            "Optimizing global MAE alone can miss search-specific ordering and local reliability requirements.",
        )
        finding_block(
            file,
            "E. Route-Critical Overestimation Predicts Optimality Failures",
            "Positive error near route-critical cells is more diagnostic of cost gap than global overestimation alone.",
            f"Critical minus global overestimate rate is {evidence['nonoptimal_critical_over']} on non-optimal U-Net maps and {evidence['optimal_critical_over']} on optimal maps; critical large-overestimate Spearman with cost gap is {evidence['critical_gap_spearman']} versus global {evidence['global_gap_spearman']}.",
            "Constrain positive error near optimal paths, bottlenecks, corridors, gaps, and necessary detours.",
            "Strong constraints may reduce heuristic aggressiveness and increase expansions.",
        )
        finding_block(
            file,
            "F. Artificial Heuristic Barriers Are Harmful",
            "U-Net failures are associated with localized positive error that makes necessary passages appear expensive.",
            f"Non-optimal maps show higher overestimation: overestimate gap {evidence['over_gap']}, large-overestimate gap {evidence['large_over_gap']}.",
            "Avoid learned heuristic fields that make necessary passages appear too expensive.",
            "Barrier avoidance may require local checks that add complexity or reduce the benefit of learned guidance.",
        )
        finding_block(
            file,
            "G. Global Roughness Is Too Weak To Explain Corridor Failures",
            "Map-wide smoothness does not strongly explain corridor-specific failures.",
            f"High-corridor roughness relationship is weak, with Spearman {evidence['roughness_spearman']}.",
            "Use local passage-aware diagnostics rather than relying only on global smoothness.",
            "Local diagnostics are harder to define robustly and may depend on map topology.",
        )

        file.write("## 3. Requirement Categories\n\n")
        file.write("### Efficiency Requirements\n\n")
        file.write("- Reduce expanded nodes.\n")
        file.write("- Preserve route ordering.\n")
        file.write("- Preserve global route bias.\n")
        file.write("- Maintain broad field quality.\n\n")
        file.write("### Reliability Requirements\n\n")
        file.write("- Preserve optimality when possible.\n")
        file.write("- Reduce cost gap.\n")
        file.write("- Avoid inadmissible behavior that changes path quality.\n\n")
        file.write("### Calibration Requirements\n\n")
        file.write("- Limit large overestimation.\n")
        file.write("- Limit route-critical overestimation.\n")
        file.write("- Avoid artificial barriers.\n\n")
        file.write("### Structure-Awareness Requirements\n\n")
        file.write("- Evaluate separately on maze_like, bottleneck, large_block, and narrow_corridor.\n")
        file.write("- Detect structure-specific failure modes.\n\n")
        file.write("### Evaluation Requirements\n\n")
        file.write("- Expanded nodes.\n")
        file.write("- Runtime if available.\n")
        file.write("- Optimality rate.\n")
        file.write("- Cost gap.\n")
        file.write("- U-Net minus MLP expansion gap.\n")
        file.write("- Route-critical ordering accuracy.\n")
        file.write("- Route-critical large-overestimate rate.\n")
        file.write("- Global MAE.\n")
        file.write("- Structure-stratified performance.\n\n")

        file.write("## 4. Priority Table\n\n")
        write_priority_table(file)

        file.write("## 5. Design Trade-offs\n\n")
        file.write(
            "Ordering vs admissibility: better ordering can reduce expansions, but a heuristic that orders states aggressively can still "
            "overestimate and change path quality. The design target should separate ordering quality from admissibility risk.\n\n"
        )
        file.write(
            "Route bias vs overestimation risk: global route bias is useful in maze-like maps, but the same learned spatial prior can "
            "create artificial barriers near passages. Bias should guide search without making necessary routes look unavailable.\n\n"
        )
        file.write(
            "Global MAE vs search-specific usefulness: global MAE helps explain expansion behavior, but search depends on relative ordering "
            "and local errors near critical cells. A low-MAE field can still be harmful if its errors concentrate on bottlenecks.\n\n"
        )
        file.write(
            "Spatial context vs artificial barriers: obstacle-aware context is the main advantage of U-Net-like fields, yet it is also "
            "the source of structure-specific failures. Future designs should preserve context while constraining local positive error.\n\n"
        )
        file.write(
            "Local reliability vs broad expansion efficiency: local constraints can protect optimality but may reduce heuristic aggressiveness. "
            "A future method should report both expanded-node savings and route-critical reliability, not one aggregate score.\n\n"
        )

        file.write("## 6. Hard Constraints For Future Algorithms\n\n")
        file.write("- Do not allow severe route-critical overestimation.\n")
        file.write("- Do not allow large artificial barriers near necessary passages.\n")
        file.write("- Do not accept expanded-node improvements that produce large optimality failures.\n")
        file.write("- Do not rely on aggregate gains that hide narrow_corridor or large_block failures.\n\n")

        file.write("## 7. Success Criteria For A Future Algorithm\n\n")
        file.write("- Reduces expanded nodes compared with Manhattan and preferably MLP table.\n")
        file.write("- Preserves or improves optimality rate relative to raw U-Net.\n")
        file.write("- Reduces route-critical large-overestimate rate.\n")
        file.write("- Improves or preserves route-critical ordering.\n")
        file.write("- Performs robustly across maze_like, bottleneck, large_block, and narrow_corridor.\n")
        file.write("- Does not only improve aggregate averages while worsening structure-specific failures.\n\n")

        file.write("## 8. Final Design Specification\n\n")
        file.write(
            "A future A*-based learned-heuristic algorithm should preserve useful learned route ordering and obstacle-aware route bias, "
            "while explicitly constraining route-critical overestimation and artificial heuristic barriers. It should optimize search "
            "efficiency and reliability separately, and must be evaluated with both global metrics and route-critical, structure-aware diagnostics.\n"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Derive algorithm requirements from existing learned-heuristic evidence.")
    parser.add_argument("--design-space", default="outputs/design_space_analysis/design_space_analysis.md")
    parser.add_argument("--research-synthesis", default="outputs/research_synthesis/research_synthesis.md")
    parser.add_argument("--mechanism-summary", default="outputs/mechanism_validation/mechanism_validation_summary.md")
    parser.add_argument("--route-critical-summary", default="outputs/route_critical_analysis/route_critical_summary.md")
    return parser.parse_args()


def main(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs", "algorithm_requirements")
    os.makedirs(output_dir, exist_ok=True)

    design_space = read_text(os.path.join(project_root, args.design_space))
    synthesis = read_text(os.path.join(project_root, args.research_synthesis))
    mechanism = read_text(os.path.join(project_root, args.mechanism_summary))
    route_critical = read_text(os.path.join(project_root, args.route_critical_summary))
    evidence = collect_evidence(design_space, synthesis, mechanism, route_critical)

    output_path = os.path.join(output_dir, "algorithm_requirements.md")
    write_report(output_path, evidence)
    print(f"Saved algorithm requirements to {output_path}")


if __name__ == "__main__":
    main(parse_args())
