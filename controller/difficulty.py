"""
controller/difficulty.py
------------------------
Maps a difficulty level to its (Pac-Man solver, ghost policy) pair.
Contains no pygame code, so the benchmark script can reuse it headlessly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

import config
from solver.astar_solver import AStarSolver
from solver.base_solver import GhostPolicy, Solver
from solver.bfs_solver import BFSSolver
from solver.human_solver import HumanSolver
from solver.ghost_policies import (CoordinatedGhosts, GreedyChaseGhosts,
                                   RandomWalkGhosts)
from solver.minimax_solver import MinimaxSolver


class Difficulty(Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


@dataclass(frozen=True)
class Profile:
    pacman_text: str
    ghost_text: str
    ghost_period: int        # ghosts act every N-th turn (1 = same speed)
    make_solver: Callable[[], Solver]
    make_ghosts: Callable[[Optional[int]], GhostPolicy]


PROFILES = {
    Difficulty.EASY: Profile(
        "Pac-Man: Breadth-First Search to the nearest dot",
        "Ghosts: random walk (move at half speed)", 2,
        BFSSolver, lambda seed: RandomWalkGhosts(seed)),
    Difficulty.MEDIUM: Profile(
        "Pac-Man: A* search, Manhattan heuristic",
        "Ghosts: greedy chase (move at half speed)", 2,
        AStarSolver, lambda seed: GreedyChaseGhosts(seed)),
    Difficulty.HARD: Profile(
        "Pac-Man: Minimax with Alpha-Beta pruning",
        "Ghosts: coordinated pincer, full speed", 1,
        lambda: MinimaxSolver(depth=config.MINIMAX_DEPTH),
        lambda seed: CoordinatedGhosts(seed)),
}


def build_ai(difficulty: Difficulty, seed: Optional[int] = None,
             human: bool = False) -> Tuple[Solver, GhostPolicy]:
    """(Pac-Man controller, ghost policy) for a difficulty.
    With human=True the player steers Pac-Man and only the ghosts use the AI."""
    p = PROFILES[difficulty]
    pacman = HumanSolver() if human else p.make_solver()
    return pacman, p.make_ghosts(seed)
