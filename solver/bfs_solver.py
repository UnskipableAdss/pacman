"""
solver/bfs_solver.py
--------------------
Breadth-First Search for the EASY difficulty.

BFS explores the maze graph in rings of increasing distance from Pac-Man. With
unit edge costs the first dot it reaches is guaranteed to be the NEAREST dot
(shortest path in number of moves).

    Time  : O(V + E)       (each cell is enqueued at most once)
    Space : O(V)
    Complete: yes, Optimal: yes (unit step cost)

Unlike A* there is no heuristic, so BFS expands cells in every direction.
Comparing `expanded` for BFS vs A* is a nice experiment for the report.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

from model.game_model import Action, GameModel, Position, STOP
from solver.base_solver import Solver


def bfs_path(model: GameModel, start: Position, goals: Iterable[Position],
             blocked: Optional[Set[Position]] = None
             ) -> Tuple[List[Position], int]:
    """Shortest path start -> nearest goal. Returns (path, nodes_expanded);
    the path is [] if no goal is reachable."""
    goal_set = frozenset(goals)
    blocked = blocked or set()
    if start in goal_set:
        return [start], 0

    parent: Dict[Position, Position] = {}
    seen = {start}                       # "discovered" set: a cell is marked
    queue = deque([start])               # when ENQUEUED, so it is queued once
    expanded = 0

    while queue:
        node = queue.popleft()
        expanded += 1
        for nb in model.neighbors(node):
            if nb in seen or nb in blocked:
                continue
            seen.add(nb)
            parent[nb] = node
            # Goal test at discovery time is safe for BFS with unit costs:
            # the first discovery of a goal is along a shortest path.
            if nb in goal_set:
                path = [nb]
                while path[-1] in parent:
                    path.append(parent[path[-1]])
                path.reverse()
                return path, expanded
            queue.append(nb)
    return [], expanded


class BFSSolver(Solver):
    """EASY: walk the shortest path to the nearest remaining dot."""

    name = "BFS (nearest dot)"

    def get_action(self, model: GameModel) -> Action:
        # First try to path around the cells ghosts currently stand on; if that
        # cuts off every dot, fall back to ignoring ghosts.
        path, expanded = bfs_path(model, model.pacman, model.dots,
                                  blocked=set(model.ghosts))
        if not path:
            path, expanded = bfs_path(model, model.pacman, model.dots)
        self.last_stats = {"expanded": expanded,
                           "path_len": max(len(path) - 1, 0)}
        if len(path) < 2:
            return STOP
        return model.action_between(path[0], path[1]) or STOP
