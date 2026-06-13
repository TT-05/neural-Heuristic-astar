import random

from gen_map import gen_map


STRUCTURED_TYPES = ["narrow_corridor", "bottleneck", "maze_like", "large_block"]
ALL_MAP_TYPES = ["open_random"] + STRUCTURED_TYPES


def empty_grid(width, height, value=0):
    return [[value for _ in range(width)] for _ in range(height)]


def fill_rect(grid, top, left, bottom, right, value):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for r in range(max(0, top), min(rows, bottom)):
        for c in range(max(0, left), min(cols, right)):
            grid[r][c] = value


def carve_rect(grid, top, left, bottom, right):
    fill_rect(grid, top, left, bottom, right, 0)


def add_sparse_noise(grid, rng, obstacle_rate, protected_cells):
    # Keep noise weak: these maps are meant to preserve controlled structure.
    noise_rate = max(0.0, min(0.04, obstacle_rate * 0.08))
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for r in range(rows):
        for c in range(cols):
            if (r, c) in protected_cells:
                continue
            if grid[r][c] == 0 and rng.random() < noise_rate:
                grid[r][c] = 1


def generate_open_random(width, height, seed, obstacle_rate):
    return gen_map(width=width, height=height, seed=seed, obstacle_rate=obstacle_rate)


def generate_narrow_corridor(width, height, seed, obstacle_rate):
    rng = random.Random(seed)
    grid = empty_grid(width, height, value=1)
    corridor_row = rng.randint(max(2, height // 4), min(height - 3, 3 * height // 4))
    corridor_width = 1 if obstacle_rate >= 0.2 else 2
    left_room_right = max(4, width // 3)
    right_room_left = min(width - 4, 2 * width // 3)

    carve_rect(grid, 1, 1, height - 1, left_room_right, )
    carve_rect(grid, 1, right_room_left, height - 1, width - 1)
    carve_rect(grid, corridor_row, left_room_right, corridor_row + corridor_width, right_room_left)

    # Add short side alcoves so the corridor is not completely trivial.
    for c in range(left_room_right + 2, right_room_left - 1, 3):
        if rng.random() < 0.5:
            carve_rect(grid, max(1, corridor_row - 2), c, corridor_row, c + 1)
        else:
            carve_rect(grid, corridor_row + corridor_width, c, min(height - 1, corridor_row + corridor_width + 2), c + 1)

    protected = {(corridor_row, c) for c in range(left_room_right, right_room_left)}
    add_sparse_noise(grid, rng, obstacle_rate, protected)
    return grid


def generate_bottleneck(width, height, seed, obstacle_rate):
    rng = random.Random(seed)
    grid = empty_grid(width, height, value=0)
    wall_col = width // 2 + rng.choice([-1, 0, 1])
    gap_row = rng.randint(3, height - 4)
    gap_size = 1 if obstacle_rate >= 0.2 else 2

    for r in range(height):
        grid[r][wall_col] = 1
    for r in range(gap_row, min(height, gap_row + gap_size)):
        grid[r][wall_col] = 0

    if obstacle_rate >= 0.3:
        second_col = max(2, wall_col - 4)
        second_gap = rng.randint(3, height - 4)
        for r in range(2, height - 2):
            grid[r][second_col] = 1
        for r in range(second_gap, min(height - 1, second_gap + 2)):
            grid[r][second_col] = 0

    protected = {(r, wall_col) for r in range(gap_row, min(height, gap_row + gap_size))}
    add_sparse_noise(grid, rng, obstacle_rate, protected)
    return grid


def generate_maze_like(width, height, seed, obstacle_rate):
    rng = random.Random(seed)
    grid = empty_grid(width, height, value=0)

    for c in range(3, width - 3, 4):
        gap = rng.randint(1, height - 2)
        for r in range(1, height - 1):
            if abs(r - gap) <= 1:
                continue
            grid[r][c] = 1

    for r in range(4, height - 4, 5):
        gap = rng.randint(1, width - 2)
        for c in range(1, width - 1):
            if abs(c - gap) <= 1:
                continue
            if grid[r][c] == 0:
                grid[r][c] = 1

    if obstacle_rate >= 0.3:
        for _ in range(2):
            top = rng.randint(2, height - 5)
            left = rng.randint(2, width - 5)
            fill_rect(grid, top, left, top + 2, left + 2, 1)

    return grid


def generate_large_block(width, height, seed, obstacle_rate):
    rng = random.Random(seed)
    grid = empty_grid(width, height, value=0)
    block_count = 1 if obstacle_rate < 0.25 else 2

    for _ in range(block_count):
        block_h = rng.randint(max(3, height // 5), max(4, height // 3))
        block_w = rng.randint(max(3, width // 5), max(4, width // 3))
        top = rng.randint(2, max(2, height - block_h - 2))
        left = rng.randint(2, max(2, width - block_w - 2))
        fill_rect(grid, top, left, top + block_h, left + block_w, 1)

    if obstacle_rate >= 0.3:
        wall_row = rng.randint(4, height - 5)
        gap_col = rng.randint(2, width - 3)
        for c in range(1, width - 1):
            if abs(c - gap_col) <= 1:
                continue
            grid[wall_row][c] = 1

    return grid


def generate_structured_map(width, height, seed, obstacle_rate, structured_type):
    if structured_type == "open_random":
        return generate_open_random(width, height, seed, obstacle_rate)
    if structured_type == "narrow_corridor":
        return generate_narrow_corridor(width, height, seed, obstacle_rate)
    if structured_type == "bottleneck":
        return generate_bottleneck(width, height, seed, obstacle_rate)
    if structured_type == "maze_like":
        return generate_maze_like(width, height, seed, obstacle_rate)
    if structured_type == "large_block":
        return generate_large_block(width, height, seed, obstacle_rate)
    raise ValueError(f"Unknown structured map type: {structured_type}")


def parse_structured_types(text):
    if text == "all":
        return list(STRUCTURED_TYPES)
    selected = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in selected if item not in ALL_MAP_TYPES]
    if unknown:
        raise ValueError(f"Unknown structured types: {unknown}")
    return selected
