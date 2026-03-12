# bfs_label.py
# Compute the distance from each cell to the goal using BFS.
from collections import deque
def compute_distance_to_goal(grid, goal):
    """
    Compute the distance from each cell in the grid to the goal using BFS.

    Parameters:
    grid (list of list of int): The input grid where 0 represents an open cell and 1 represents a wall.
    goal (tuple): The coordinates of the goal cell (x, y).
    Returns:
    list of list of int: A grid where each cell contains the distance to the goal. Cells that are walls will have a distance of -1.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    if rows == 0 or cols == 0:
        return []

    goal_r, goal_c = goal

    if goal_r < 0 or goal_r >= rows or goal_c < 0 or goal_c >= cols:
        raise ValueError("Goal is out of bounds.")

    if grid[goal_r][goal_c] == 1:
        raise ValueError("Goal cannot be on an obstacle.")

    distance_grid = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(-1)
        distance_grid.append(row)

    queue = deque()
    queue.append((goal_r, goal_c))
    distance_grid[goal_r][goal_c] = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()

        for dr, dc in directions:
            nr = r + dr
            nc = c + dc

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue

            if grid[nr][nc] == 1:
                continue

            if distance_grid[nr][nc] != -1:
                continue

            distance_grid[nr][nc] = distance_grid[r][c] + 1
            queue.append((nr, nc))

    return distance_grid