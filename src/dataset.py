# dataset.py
# Functions to compute distance grid and extract samples for training.
from bfs_label import compute_distance_to_goal
def extract_samples_from_distance_grid(grid, goal, distance_grid):
    """
    Extract samples from the distance grid for training a neural network.

    Parameters:
    grid (list of list of int): The input grid where 0 represents an open cell and 1 represents a wall.
    goal (tuple): The coordinates of the goal cell (x, y).
    distance_grid (list of list of int): A grid where each cell contains the distance to the goal. Cells that are walls will have a distance of -1.

    Returns:
    list of dictionaries: A list of samples, where each sample is a dictionary with keys "current", "goal", and "distance_to_goal".
                    - current: A tuple representing the coordinates of the cell (x, y).
                    - goal: A tuple representing the coordinates of the goal cell (x, y).
                    - distance_to_goal: The distance from the current cell to the goal as given in the distance_grid.
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
    
    if len(distance_grid) != rows or any(len(row) != cols for row in distance_grid):
        raise ValueError("distance_grid must have the same shape as grid.")

    samples = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 or distance_grid[r][c] == -1:
                continue

            sample = {"current": (r, c), "goal": goal, "distance_to_goal": distance_grid[r][c]}
            samples.append(sample)

    return samples

def build_samples_from_grid(grid, goal):
    '''Build samples for training from the input grid and goal.'''
    distance_grid = compute_distance_to_goal(grid, goal)
    return extract_samples_from_distance_grid(grid, goal, distance_grid)
