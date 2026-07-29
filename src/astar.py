# astar.py
# A* search algorithm.
import heapq


def astar_search(grid, start, goal, heuristic, diagnostics=None, secondary_heuristic=None, secondary_priority=None):
    """
    Perform A* search on the given grid from start to goal using the provided heuristic function.

    Parameters:
    grid (list of list of int): The input grid where 0 represents a free cell and 1 represents an obstacle.
    start (tuple): The starting coordinates (x, y).
    goal (tuple): The goal coordinates (x, y).
    heuristic (function): A function that takes two coordinates and returns the estimated cost to reach the goal.
    diagnostics (dict, optional): If provided, expanded-node diagnostics are appended here.
    secondary_heuristic (function, optional): If provided, A* uses lexicographic
        priority (g + heuristic, secondary_heuristic, tie_counter, g, node).
    secondary_priority (function, optional): If provided, called as
        secondary_priority(node, goal, g) and used instead of secondary_heuristic.

    Returns:
    dict: Search result with keys "path", "cost", and "expanded".
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    if rows == 0 or cols == 0:
        return {"path": [], "cost": -1, "expanded": 0}

    start_r, start_c = start
    goal_r, goal_c = goal

    if start_r < 0 or start_r >= rows or start_c < 0 or start_c >= cols:
        raise ValueError("Start is out of bounds.")

    if goal_r < 0 or goal_r >= rows or goal_c < 0 or goal_c >= cols:
        raise ValueError("Goal is out of bounds.")

    if grid[start_r][start_c] == 1:
        raise ValueError("Start cannot be on an obstacle.")

    if grid[goal_r][goal_c] == 1:
        raise ValueError("Goal cannot be on an obstacle.")

    open_set = []
    start_h = heuristic(start, goal)
    tie_counter = 0

    def secondary_score(node, node_g):
        if secondary_priority is not None:
            return secondary_priority(node, goal, node_g)
        return secondary_heuristic(node, goal)

    use_secondary_priority = secondary_heuristic is not None or secondary_priority is not None
    if not use_secondary_priority:
        heapq.heappush(open_set, (start_h, 0, start))  # (f, g, node)
    else:
        heapq.heappush(open_set, (start_h, secondary_score(start, 0), tie_counter, 0, start))
    came_from = {}
    g_score = {start: 0}
    expanded = 0

    if diagnostics is not None:
        diagnostics.setdefault("expanded_nodes", [])
        diagnostics.setdefault("max_expanded_logs", 25)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_set:
        if not use_secondary_priority:
            f, current_g, current = heapq.heappop(open_set)
        else:
            f, _, _, current_g, current = heapq.heappop(open_set)

        if current_g > g_score[current]:
            continue

        expanded += 1

        if diagnostics is not None and len(diagnostics["expanded_nodes"]) < diagnostics["max_expanded_logs"]:
            current_h = heuristic(current, goal)
            entry = {
                "node": current,
                "g": current_g,
                "h": current_h,
                "f": current_g + current_h,
            }
            if use_secondary_priority:
                entry["secondary_h"] = secondary_score(current, current_g)
            diagnostics["expanded_nodes"].append(entry)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return {
                "path": path,
                "cost": len(path) - 1,
                "expanded": expanded,
            }

        for dr, dc in directions:
            nr = current[0] + dr
            nc = current[1] + dc
            neighbor = (nr, nc)

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue

            if grid[nr][nc] == 1:
                continue

            tentative_g_score = current_g + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                came_from[neighbor] = current
                f_score = tentative_g_score + heuristic(neighbor, goal)
                if not use_secondary_priority:
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))
                else:
                    tie_counter += 1
                    heapq.heappush(
                        open_set,
                        (f_score, secondary_score(neighbor, tentative_g_score), tie_counter, tentative_g_score, neighbor),
                    )

    return {
        "path": [],
        "cost": -1,
        "expanded": expanded,
    }
