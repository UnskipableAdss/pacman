import random
import time
import unittest
from collections import deque

from model.game_model import (DOT_SCORE, DOWN, LEFT, LOST, RIGHT, STOP, UP,
                              GameModel)
from model.layouts import CLASSIC_19x19, TINY_7x5
from solver.astar_solver import (astar_next_action, astar_search,
                                 path_to_actions)


def bfs_dist(model, start, goal):
    """Reference shortest-path length (unit costs) to cross-check A*."""
    q, seen = deque([(start, 0)]), {start}
    while q:
        n, d = q.popleft()
        if n == goal:
            return d
        for nb in model.neighbors(n):
            if nb not in seen:
                seen.add(nb)
                q.append((nb, d + 1))
    return None


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.m = GameModel(CLASSIC_19x19, lives=3)

    def test_maze_is_connected(self):
        open_cells = set(self.m._nbrs)
        seen, stack = {self.m.pacman}, [self.m.pacman]
        while stack:
            for nb in self.m.neighbors(stack.pop()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        self.assertEqual(seen, open_cells)
        self.assertTrue(all(g in open_cells for g in self.m.ghosts))

    def test_clone_is_independent(self):
        c = self.m.clone()
        c.step(LEFT, [UP, UP, UP])
        c.dots.clear()
        self.assertEqual(self.m.pacman, self.m.pacman_start)
        self.assertGreater(self.m.remaining_dots, 0)
        self.assertEqual(self.m.score, 0)
        self.assertIs(c.walls, self.m.walls)         # static data shared
        self.assertIsNot(c.dots, self.m.dots)        # dynamic data copied
        self.assertIsNot(c.ghosts, self.m.ghosts)

    def test_eating_dot_scores(self):
        m = GameModel(TINY_7x5, lives=1)
        m.step(RIGHT, [STOP])
        self.assertEqual(m.score, DOT_SCORE)

    def test_swap_with_ghost_is_a_collision(self):
        m = GameModel(["#####", "#PG.#", "#####"], lives=1)
        m.step(RIGHT, [LEFT])          # would "pass through" without the check
        self.assertEqual(m.status, LOST)

    def test_illegal_action_keeps_pacman_in_place(self):
        m = self.m
        m.step(UP, [])                 # wall above start
        self.assertEqual(m.pacman, m.pacman_start)

    def test_respawn_with_lives_left(self):
        m = GameModel(["#####", "#PG.#", "#####"], lives=2)
        m.step(RIGHT, [STOP])
        self.assertEqual(m.lives, 1)
        self.assertEqual(m.pacman, m.pacman_start)
        self.assertFalse(m.is_terminal())

    def test_tunnel(self):
        m = self.m
        self.assertTrue(m.has_tunnel)
        self.assertIn((18, 9), m.neighbors((0, 9)))
        self.assertEqual(m.manhattan((0, 9), (18, 9)), 1)


class AStarTests(unittest.TestCase):
    def setUp(self):
        self.m = GameModel(CLASSIC_19x19)

    def test_path_is_valid_and_optimal_vs_bfs(self):
        rnd = random.Random(7)
        cells = list(self.m._nbrs)
        for _ in range(200):
            s, g = rnd.choice(cells), rnd.choice(cells)
            r = astar_search(self.m, s, [g])
            self.assertTrue(r.found)
            self.assertEqual(r.path[0], s)
            self.assertEqual(r.path[-1], g)
            for a, b in zip(r.path, r.path[1:]):
                self.assertIn(b, self.m.neighbors(a))
            self.assertEqual(r.cost, bfs_dist(self.m, s, g))

    def test_multi_goal_picks_nearest(self):
        goals = list(self.m.dots)
        r = astar_search(self.m, self.m.pacman, goals)
        best = min(bfs_dist(self.m, self.m.pacman, g) for g in goals)
        self.assertEqual(r.cost, best)

    def test_blocked_cells_respected(self):
        s, g = (1, 1), (17, 1)
        blocked = {(5, 1)}
        r = astar_search(self.m, s, [g], blocked=blocked)
        self.assertTrue(r.found)
        self.assertNotIn((5, 1), r.path)

    def test_no_path_returns_not_found(self):
        r = astar_search(self.m, (1, 1), [(5, 1)], blocked={(5, 1)})
        self.assertFalse(r.found)

    def test_danger_cost_causes_detour(self):
        s, g = (1, 3), (17, 3)
        plain = astar_search(self.m, s, [g])
        penalised = astar_search(self.m, s, [g], extra_cost={(9, 3): 50.0})
        self.assertNotIn((9, 3), penalised.path)
        self.assertIn((9, 3), plain.path)

    def test_path_to_actions_roundtrip(self):
        r = astar_search(self.m, (1, 1), [(17, 17)])
        pos = r.path[0]
        for act in path_to_actions(self.m, r.path):
            pos = self.m.successor(pos, act)
        self.assertEqual(pos, (17, 17))

    def test_astar_expands_fewer_nodes_than_blind_search(self):
        # BFS (h = 0) on the same query for comparison
        s, g = (1, 1), (17, 17)
        informed = astar_search(self.m, s, [g]).expanded
        total_cells = len(self.m._nbrs)
        self.assertLess(informed, total_cells)

    def test_full_game_medium_pacman_no_ghosts_clears_board(self):
        m = GameModel(CLASSIC_19x19)
        m.ghosts = []                           # no ghosts: pure dot clearing
        for _ in range(2000):
            if m.is_terminal():
                break
            m.step(astar_next_action(m))
        self.assertTrue(m.is_win())
        print(f"\n  cleared {m.score // DOT_SCORE} dots in {m.steps} steps")


class Benchmarks(unittest.TestCase):
    def test_clone_speed(self):
        m = GameModel(CLASSIC_19x19)
        n = 20000
        t = time.perf_counter()
        for _ in range(n):
            m.clone()
        dt = time.perf_counter() - t
        print(f"\n  {n} clones in {dt:.3f}s  ({n / dt:,.0f}/s)")


if __name__ == "__main__":
    unittest.main()
