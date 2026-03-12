# gen_map.py
# Generate a grid according to given seed.

import random
def gen_map(width, height, seed=None, obstacle_rate=0.2):
    '''
    Generate a grid map with given width and height.
    The map will contain obstacles randomly placed according to the obstacle_rate. 
    If a seed is provided, the random generation will be deterministic

    ParametersL:
    width (int): The width of the grid.
    height (int): The height of the grid.
    seed (int, optional): The seed for random number generation. Defaults to None.
    obstacle_rate (float, optional): The probability of placing an obstacle in a cell. Defaults to 0.2.
    Returns:
    list of list of int: A 2D grid where 0 represents a free cell and 1 represents an obstacle.
    '''
    if seed is not None:
        random.seed(seed)

    grid = []
    for i in range(height):
        row = []
        for j in range(width):
            if random.random() < obstacle_rate:
                row.append(1)  # 1 represents an obstacle
            else:
                row.append(0)  # 0 represents a free cell
        grid.append(row)
    return grid
