import random
import unittest

from controller.difficulty import Difficulty, build_ai
from model.game_model import DOWN, LEFT, LOST, RIGHT, STOP, UP, GameModel
from model.layouts import CLASSIC_19x19
from solver.astar_solver import AStarSolver, astar_search
from solver.bfs_solver import BFSSolver, bfs_path
from solver.ghost_policies import (CoordinatedGhosts, GreedyChaseGhosts,
                                   RandomWalkGhosts)
from solver.minimax_solver import MinimaxSolver


def random_midgame(seed):
    rnd = random.Random(seed)
    m = GameModel(CLASSIC_19x19, lives=1)
    for _ in range(rnd.randint(3, 40)):
        if m.is_terminal():
            break
        m.step(rnd.choice(m.get_legal_actions(m.pacman)),
               [rnd.choice(m.get_legal_actions(g)) for g in m.ghosts])
    return m


class BFSTests(unittest.TestCase):
    def test_bfs_matches_astar_cost(self):
        m = GameModel(CLASSIC_19x19)
        rnd = random.Random(3)
        cells = m.open_cells()
        for _ in range(100):
            s, g = rnd.choice(cells), rnd.choice(cells)
            p, _ = bfs_path(m, s, [g])
            self.assertEqual(len(p) - 1, astar_search(m, s, [g]).cost)

    def test_bfs_and_astar_same_distance_to_nearest_dot(self):
        m = GameModel(CLASSIC_19x19)
        p, _ = bfs_path(m, m.pacman, m.dots)
        self.assertEqual(len(p) - 1, astar_search(m, m.pacman, m.dots).cost)


class MinimaxTests(unittest.TestCase):
    def test_alpha_beta_value_equals_plain_minimax(self):
        plain = MinimaxSolver(2, use_pruning=False)
        ab = MinimaxSolver(2, use_pruning=True)
        checked = 0
        for seed in range(12):
            m = random_midgame(seed)
            if m.is_terminal():
                continue
            plain.get_action(m.clone())
            ab.get_action(m.clone())
            self.assertEqual(plain.last_stats["value"], ab.last_stats["value"])
            self.assertLessEqual(ab.nodes, plain.nodes)
            checked += 1
        self.assertGreater(checked, 5)

    def test_does_not_walk_into_adjacent_ghost(self):
        # Ghost directly to the right: RIGHT is instant death, LEFT is not.
        m = GameModel(["#########", "#...PG..#", "#########"], lives=1)
        self.assertEqual(MinimaxSolver(3).get_action(m.clone()), LEFT)

    def test_takes_free_dot_when_safe(self):
        # Ghost far away on the other side of the dots: just eat.
        m = GameModel(["#########", "#P.....G#", "#########"], lives=1)
        self.assertEqual(MinimaxSolver(2).get_action(m.clone()), RIGHT)

    def test_survives_a_chaser_on_a_loop(self):
        # On a cycle an equal-speed chaser can never catch a Pac-Man that
        # keeps its distance, so a correct adversarial search must survive.
        ring = ["#######",
                "#P....#",
                "#.###.#",
                "#....G#",
                "#######"]
        m = GameModel(ring, lives=1)
        solver, ghosts = MinimaxSolver(4), CoordinatedGhosts(seed=0)
        for _ in range(80):
            if m.is_terminal():
                break
            m.step(solver.get_action(m.clone()), ghosts.get_actions(m.clone()))
        self.assertNotEqual(m.status, LOST)

    def test_does_not_mutate_input(self):
        m = random_midgame(5)
        before = m.state_key()
        MinimaxSolver(2).get_action(m)
        self.assertEqual(before, m.state_key())


class GhostPolicyTests(unittest.TestCase):
    def test_policies_return_legal_actions(self):
        for P in (RandomWalkGhosts, GreedyChaseGhosts, CoordinatedGhosts):
            m = GameModel(CLASSIC_19x19)
            pol = P(seed=1)
            for _ in range(60):
                acts = pol.get_actions(m.clone())
                self.assertEqual(len(acts), m.num_ghosts)
                for g, a in zip(m.ghosts, acts):
                    self.assertIn(a, m.get_legal_actions(g))
                m.step(STOP, acts)

    def test_chaser_gets_closer_to_pacman(self):
        m = GameModel(["#########", "#P.....G#", "#########"], lives=1)
        pol = CoordinatedGhosts(seed=1)
        for _ in range(3):
            acts = pol.get_actions(m.clone())
            m.step(STOP, acts)
        self.assertEqual(m.ghosts[0], (4, 1))


class HumanSolverTests(unittest.TestCase):
    def test_keeps_going_then_stops_at_wall(self):
        from solver.human_solver import HumanSolver
        m = GameModel(["#####", "#P..#", "#####"], lives=1)
        h = HumanSolver()
        h.set_direction(RIGHT)
        for expected_pos in [(2, 1), (3, 1), (3, 1)]:
            m.step(h.get_action(m.clone()), [])
            self.assertEqual(m.pacman, expected_pos)
        self.assertEqual(h.get_action(m.clone()), STOP)

    def test_queued_turn_is_taken_when_it_becomes_legal(self):
        from solver.human_solver import HumanSolver
        m = GameModel(["#####", "#P..#", "#.#.#", "#...#", "#####"], lives=1)
        h = HumanSolver()
        h.set_direction(RIGHT)
        m.step(h.get_action(m.clone()), [])          # (2,1)
        h.set_direction(DOWN)                        # illegal here: wall below
        m.step(h.get_action(m.clone()), [])          # keeps going right (3,1)
        self.assertEqual(m.pacman, (3, 1))
        m.step(h.get_action(m.clone()), [])          # DOWN now legal
        self.assertEqual(m.pacman, (3, 2))

    def test_build_ai_human_flag(self):
        from solver.human_solver import HumanSolver
        s, g = build_ai(Difficulty.HARD, human=True)
        self.assertIsInstance(s, HumanSolver)
        self.assertIsInstance(g, CoordinatedGhosts)


class IntegrationTests(unittest.TestCase):
    def test_every_difficulty_runs_a_game_without_error(self):
        for d in Difficulty:
            m = GameModel(CLASSIC_19x19, lives=1)
            solver, ghosts = build_ai(d, seed=0)
            for t in range(80):
                if m.is_terminal():
                    break
                a = solver.get_action(m.clone())
                m.step(a, ghosts.get_actions(m.clone()))


if __name__ == "__main__":
    unittest.main()
