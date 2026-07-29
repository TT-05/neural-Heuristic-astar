import math
import time


ORACLE_TOPK_METHODS = {
    "manhattan_oracle_top1_tiebreak": 1,
    "manhattan_oracle_top2_tiebreak": 2,
    "manhattan_oracle_top4_tiebreak": 4,
    "manhattan_oracle_top8_tiebreak": 8,
}


def manhattan_heuristic(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])


def true_distance_value(distance_grid, node):
    value = distance_grid[node[0]][node[1]]
    return float(value) if value >= 0 else math.inf


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def oracle_topk_astar_search(grid, start, goal, unet_heuristic, distance_grid, k):
    """
    Manhattan A* with an episode-based partial oracle inside equal-f tie sets.

    For each active minimum Manhattan-f layer, the currently open nodes are
    snapshotted once. Only the true-distance-best k snapshot nodes receive
    oracle ranks. Later arrivals with the same f are ordered by U-Net until the
    current snapshot is exhausted and a new episode is created.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows == 0 or cols == 0:
        return {"path": [], "cost": -1, "expanded": 0}

    for label, node in (("Start", start), ("Goal", goal)):
        r, c = node
        if r < 0 or r >= rows or c < 0 or c >= cols:
            raise ValueError(f"{label} is out of bounds.")
        if grid[r][c] == 1:
            raise ValueError(f"{label} cannot be on an obstacle.")

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    came_from = {}
    g_score = {start: 0}
    insertion_order = {start: 0}
    open_nodes = {start}
    closed = set()
    next_insertion = 1
    expanded = 0

    active_f = None
    active_snapshot = set()
    active_corrected = {}

    tie_episode_count = 0
    tie_snapshot_sizes = []
    oracle_corrected_nodes = 0
    oracle_corrected_expansions = 0
    later_arrivals_into_active_primary_f = 0

    def primary_f(node):
        return g_score[node] + manhattan_heuristic(node, goal)

    def create_episode(min_f):
        nonlocal active_f, active_snapshot, active_corrected
        nonlocal tie_episode_count, oracle_corrected_nodes

        snapshot = [node for node in open_nodes if primary_f(node) == min_f]
        snapshot.sort(
            key=lambda node: (
                true_distance_value(distance_grid, node),
                -g_score[node],
                insertion_order[node],
            )
        )
        active_f = min_f
        active_snapshot = set(snapshot)
        active_corrected = {node: rank for rank, node in enumerate(snapshot[:k])}
        tie_episode_count += 1
        tie_snapshot_sizes.append(len(snapshot))
        oracle_corrected_nodes += len(active_corrected)

    def choose_node():
        nonlocal active_f, active_snapshot, active_corrected
        min_f = min(primary_f(node) for node in open_nodes)
        if active_f != min_f or not active_snapshot.intersection(open_nodes):
            create_episode(min_f)
        candidates = [node for node in open_nodes if primary_f(node) == min_f]

        def priority(node):
            if node in active_corrected:
                return (0, active_corrected[node], -g_score[node], insertion_order[node])
            return (1, unet_heuristic(node, goal), -g_score[node], insertion_order[node])

        return min(candidates, key=priority)

    while open_nodes:
        current = choose_node()
        open_nodes.remove(current)
        active_snapshot.discard(current)
        closed.add(current)
        expanded += 1
        if current in active_corrected:
            oracle_corrected_expansions += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            return {
                "path": path,
                "cost": len(path) - 1,
                "expanded": expanded,
                "tie_diagnostics": {
                    "tie_episode_count": tie_episode_count,
                    "mean_tie_snapshot_size": sum(tie_snapshot_sizes) / len(tie_snapshot_sizes)
                    if tie_snapshot_sizes
                    else 0.0,
                    "max_tie_snapshot_size": max(tie_snapshot_sizes) if tie_snapshot_sizes else 0,
                    "oracle_corrected_nodes": oracle_corrected_nodes,
                    "oracle_corrected_expansion_fraction": oracle_corrected_expansions / expanded
                    if expanded
                    else 0.0,
                    "later_arrivals_into_active_primary_f": later_arrivals_into_active_primary_f,
                },
            }

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            nr, nc = neighbor
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if grid[nr][nc] == 1 or neighbor in closed:
                continue

            tentative_g = g_score[current] + 1
            if neighbor in g_score and tentative_g >= g_score[neighbor]:
                continue

            old_in_open = neighbor in open_nodes
            g_score[neighbor] = tentative_g
            came_from[neighbor] = current
            if neighbor not in insertion_order:
                insertion_order[neighbor] = next_insertion
                next_insertion += 1
            open_nodes.add(neighbor)

            if (
                active_f is not None
                and primary_f(neighbor) == active_f
                and neighbor not in active_snapshot
                and neighbor not in active_corrected
                and not old_in_open
            ):
                later_arrivals_into_active_primary_f += 1

    return {
        "path": [],
        "cost": -1,
        "expanded": expanded,
        "tie_diagnostics": {
            "tie_episode_count": tie_episode_count,
            "mean_tie_snapshot_size": sum(tie_snapshot_sizes) / len(tie_snapshot_sizes) if tie_snapshot_sizes else 0.0,
            "max_tie_snapshot_size": max(tie_snapshot_sizes) if tie_snapshot_sizes else 0,
            "oracle_corrected_nodes": oracle_corrected_nodes,
            "oracle_corrected_expansion_fraction": oracle_corrected_expansions / expanded if expanded else 0.0,
            "later_arrivals_into_active_primary_f": later_arrivals_into_active_primary_f,
        },
    }


def run_oracle_topk_search(grid, start, goal, unet_heuristic, distance_grid, k):
    start_time = time.perf_counter()
    result = oracle_topk_astar_search(grid, start, goal, unet_heuristic, distance_grid, k)
    return result, time.perf_counter() - start_time
