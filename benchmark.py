"""
benchmark.py - headless experiments for the report (no pygame needed).

    python benchmark.py                # win rates for all three difficulties
    python benchmark.py --games 50
    python benchmark.py --pruning      # Minimax vs Alpha-Beta node counts
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

import config
from controller.difficulty import PROFILES, Difficulty, build_ai
from model.game_model import DOT_SCORE, GameModel
from model.layouts import DEFAULT_LAYOUT
from solver.minimax_solver import MinimaxSolver

MAX_TURNS = 1500


def play(difficulty: Difficulty, seed: int, lives: int = 1):
    model = GameModel(DEFAULT_LAYOUT, lives=lives)
    solver, ghosts = build_ai(difficulty, seed)
    period = PROFILES[difficulty].ghost_period
    think = []
    turn = 0
    while not model.is_terminal() and turn < MAX_TURNS:
        t0 = time.perf_counter()
        a = solver.get_action(model.clone())
        think.append((time.perf_counter() - t0) * 1000)
        turn += 1
        g = ghosts.get_actions(model.clone()) if turn % period == 0 else []
        model.step(a, g)
    return model, think


def compare_difficulties(n: int) -> None:
    print(f"{'level':8} {'win%':>6} {'avg dots':>9} {'avg turns':>10} "
          f"{'think ms (avg/max)':>20}")
    for d in Difficulty:
        wins, dots, turns, avg_t, max_t = 0, [], [], [], []
        for seed in range(n):
            m, think = play(d, seed)
            wins += m.is_win()
            dots.append(m.score // DOT_SCORE)
            turns.append(m.steps)
            avg_t.append(statistics.mean(think))
            max_t.append(max(think))
        print(f"{d.value:8} {100 * wins / n:6.0f} {statistics.mean(dots):9.1f} "
              f"{statistics.mean(turns):10.0f} "
              f"{statistics.mean(avg_t):10.1f} /{max(max_t):7.1f}")


def compare_pruning(depth: int, samples: int) -> None:
    """Same positions, same depth: plain Minimax vs Alpha-Beta."""
    rng = random.Random(1)
    # ghost_radius=99: every ghost is branched on, so the tree is full size.
    plain = MinimaxSolver(depth, ghost_radius=99, use_pruning=False)
    ab = MinimaxSolver(depth, ghost_radius=99, use_pruning=True)
    tot_p = tot_a = 0
    t_p = t_a = 0.0
    for _ in range(samples):
        m = GameModel(DEFAULT_LAYOUT, lives=1)
        for _ in range(rng.randint(5, 40)):            # random mid-game state
            if m.is_terminal():
                break
            m.step(rng.choice(m.get_legal_actions(m.pacman)),
                   [rng.choice(m.get_legal_actions(g)) for g in m.ghosts])
        if m.is_terminal():
            continue
        t = time.perf_counter(); plain.get_action(m.clone()); t_p += time.perf_counter() - t
        t = time.perf_counter(); ab.get_action(m.clone());    t_a += time.perf_counter() - t
        assert plain.last_stats["value"] == ab.last_stats["value"]
        tot_p += plain.nodes
        tot_a += ab.nodes
    print(f"depth={depth} rounds, {samples} positions (values identical)")
    print(f"  plain minimax : {tot_p:>10,} nodes  {t_p:6.2f}s")
    print(f"  alpha-beta    : {tot_a:>10,} nodes  {t_a:6.2f}s")
    print(f"  nodes saved   : {100 * (1 - tot_a / tot_p):.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--pruning", action="store_true")
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()
    if args.pruning:
        compare_pruning(args.depth, 15)
    else:
        compare_difficulties(args.games)
