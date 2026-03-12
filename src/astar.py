# astar.py
# A* search algorithm.
import heapq
def astar_search(grid, start, goal, heuristic):
    """
    Perform A* search on the given grid from start to goal using the provided heuristic function.

    Parameters:
    grid (list of list of int): The input grid where 0 represents a free cell and 1 represents an obstacle.
    start (tuple): The starting coordinates (x, y).
    goal (tuple): The goal coordinates (x, y).
    heuristic (function): A function that takes two coordinates and returns the estimated cost to reach the goal.

    Returns:
    list of tuple: The path from start to goal as a list of coordinates. If no path is found, returns an empty list.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    if rows == 0 or cols == 0:
        return []

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
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_set:
        f, current_g, current = heapq.heappop(open_set)

        if current_g > g_score[current]:
            continue

        expanded += 1

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return {
                "path": path,
                "cost": len(path) - 1,
                "expanded": expanded
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
                heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

    return {
        "path": [],
        "cost": -1,
        "expanded": expanded
    }