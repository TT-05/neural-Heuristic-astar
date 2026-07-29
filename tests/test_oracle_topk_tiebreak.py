import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from astar import astar_search
from bfs_label import compute_distance_to_goal
from oracle_topk_tiebreak import oracle_topk_astar_search


def true_distance_heuristic(distance_grid):
    def heuristic(node, unused_goal):
        value = distance_grid[node[0]][node[1]]
        return float(value) if value >= 0 else float("inf")

    return heuristic


class OracleTopKTieBreakTest(unittest.TestCase):
    def test_topk_methods_are_optimal_and_behaviorally_distinct(self):
        grid_text = [
            ".#....",
            "...#..",
            "..#.#.",
            ".#..#.",
            ".#....",
            "....#.",
        ]
        grid = [[1 if cell == "#" else 0 for cell in row] for row in grid_text]
        start = (0, 0)
        goal = (5, 5)
        distance_grid = compute_distance_to_goal(grid, goal)

        def deliberately_bad_unet(node, unused_goal):
            value = distance_grid[node[0]][node[1]]
            return -float(value) if value >= 0 else float("inf")

        results = {
            k: oracle_topk_astar_search(grid, start, goal, deliberately_bad_unet, distance_grid, k)
            for k in (1, 2, 4, 8)
        }
        full = astar_search(
            grid,
            start,
            goal,
            lambda node, target: abs(node[0] - target[0]) + abs(node[1] - target[1]),
            secondary_heuristic=true_distance_heuristic(distance_grid),
        )

        for result in results.values():
            self.assertEqual(result["cost"], distance_grid[start[0]][start[1]])
        self.assertEqual(full["cost"], distance_grid[start[0]][start[1]])
        self.assertNotEqual(results[1]["expanded"], results[2]["expanded"])
        self.assertNotEqual(results[1]["expanded"], full["expanded"])

    def test_later_same_f_arrivals_are_not_retroactively_corrected(self):
        grid = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        start = (0, 0)
        goal = (3, 3)
        distance_grid = compute_distance_to_goal(grid, goal)

        result = oracle_topk_astar_search(grid, start, goal, lambda node, unused_goal: 0.0, distance_grid, 1)

        diagnostics = result["tie_diagnostics"]
        self.assertGreater(diagnostics["later_arrivals_into_active_primary_f"], 0)
        self.assertGreater(diagnostics["tie_episode_count"], 0)
        self.assertEqual(result["cost"], distance_grid[start[0]][start[1]])

    def test_repeated_runs_are_deterministic(self):
        grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        start = (0, 0)
        goal = (2, 2)
        distance_grid = compute_distance_to_goal(grid, goal)

        first = oracle_topk_astar_search(grid, start, goal, lambda node, unused_goal: 0.0, distance_grid, 1)
        second = oracle_topk_astar_search(grid, start, goal, lambda node, unused_goal: 0.0, distance_grid, 1)

        self.assertEqual(first["path"], second["path"])
        self.assertEqual(first["expanded"], second["expanded"])

    def test_unreachable_case_returns_no_path(self):
        grid = [
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ]
        start = (0, 0)
        goal = (2, 2)
        distance_grid = compute_distance_to_goal(grid, goal)

        result = oracle_topk_astar_search(grid, start, goal, lambda node, unused_goal: 0.0, distance_grid, 1)

        self.assertEqual(distance_grid[start[0]][start[1]], -1)
        self.assertEqual(result["path"], [])
        self.assertEqual(result["cost"], -1)


if __name__ == "__main__":
    unittest.main()
