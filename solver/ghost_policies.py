"""
solver/ghost_policies.py
------------------------
Ghost behaviours, one per difficulty level.

    RandomWalkGhosts   (Easy)    random legal move
    GreedyChaseGhosts  (Medium)  greedy: minimise Manhattan distance to Pac-Man
    CoordinatedGhosts  (Hard)    each ghost gets its OWN target around Pac-Man

All three follow the arcade rule "a ghost never reverses unless it is in a
dead end", which stops pointless jittering.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from model.game_model import Action, GameModel, Position
from solver.base_solver import GhostPolicy
from solver.graph_utils import bfs_distances


def _opposite(a: Action) -> Action:
    return (-a[0], -a[1])


class _GhostMemory(GhostPolicy):
    """Shared helper: remembers each ghost's previous action."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self._last: Dict[int, Action] = {}

    def _options(self, model: GameModel, i: int) -> List[Action]:
        legal = model.get_legal_actions(model.ghosts[i])
        last = self._last.get(i)
        if last is not None and len(legal) > 1:
            forward = [a for a in legal if a != _opposite(last)]
            if forward:
                return forward
        return legal


class RandomWalkGhosts(_GhostMemory):
    """EASY - uniform random choice among the allowed moves."""

    name = "Random walk"

    def get_actions(self, model: GameModel) -> List[Action]:
        out = []
        for i in range(model.num_ghosts):
            a = self.rng.choice(self._options(model, i))
            self._last[i] = a
            out.append(a)
        return out


class GreedyChaseGhosts(_GhostMemory):
    """
    MEDIUM - one-step greedy (best-first) chase.

    Each ghost picks the neighbour with the smallest Manhattan distance to
    Pac-Man. It is fast (O(1) per ghost) but NOT complete: walls can trap a
    greedy ghost in a local minimum, which is exactly why this is "medium".
    """

    name = "Greedy chase (Manhattan)"

    def get_actions(self, model: GameModel) -> List[Action]:
        target = model.pacman
        out = []
        for i, pos in enumerate(model.ghosts):
            opts = self._options(model, i)
            scored = [(model.manhattan(model.successor(pos, a), target), a)
                      for a in opts]
            best = min(s for s, _ in scored)
            a = self.rng.choice([a for s, a in scored if s == best])
            self._last[i] = a
            out.append(a)
        return out


class CoordinatedGhosts(_GhostMemory):
    """
    HARD - coordinated pincer.

    Ghost 0 is the CHASER: it follows the true shortest path to Pac-Man (BFS).
    Every other ghost picks an AMBUSH cell 2..R steps from Pac-Man that is as
    far as possible from the targets already handed out, and that it can reach
    quickly. Spreading the targets makes the ghosts come from different sides,
    so Pac-Man cannot simply run away from one of them.
    """

    name = "Coordinated pincer (BFS)"
    AMBUSH_MIN, AMBUSH_MAX = 2, 5

    def get_actions(self, model: GameModel) -> List[Action]:
        from_pacman = bfs_distances(model, model.pacman)
        ring = [c for c, d in from_pacman.items()
                if self.AMBUSH_MIN <= d <= self.AMBUSH_MAX]

        targets: List[Position] = []
        out = []
        for i, pos in enumerate(model.ghosts):
            if i == 0 or not ring:
                target = model.pacman
            else:
                from_ghost = bfs_distances(model, pos)
                # spread between targets (x3) vs how far the ghost must travel
                target = max(
                    ring,
                    key=lambda c: 3 * min(model.manhattan(c, t) for t in targets)
                    - from_ghost.get(c, 99))
            targets.append(target)
            out.append(self._step_toward(model, i, target))
        return out

    def _step_toward(self, model: GameModel, i: int, target: Position) -> Action:
        dist = bfs_distances(model, target)           # distance-to-target map
        pos = model.ghosts[i]
        opts = self._options(model, i)
        scored = [(dist.get(model.successor(pos, a), 999), a) for a in opts]
        best = min(s for s, _ in scored)
        a = self.rng.choice([a for s, a in scored if s == best])
        self._last[i] = a
        return a
