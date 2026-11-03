"""
solver/astar_solver.py
----------------------
A* Search for the MEDIUM difficulty.

    f(n) = g(n) + h(n)

    g(n): exact cost of the cheapest path found so far from start to n
    h(n): heuristic estimate of the remaining cost from n to the nearest goal
          = Manhattan distance to the closest dot (tunnel-aware)

Why this heuristic is valid
---------------------------
* ADMISSIBLE: every move changes x or y by exactly 1 (or wraps through the
  tunnel, which model.manhattan() already accounts for), so no path can be
  shorter than the Manhattan distance. h never overestimates.
* CONSISTENT: h(n) <= cost(n, n') + h(n') because one move changes the
  Manhattan distance by at most 1 (= the step cost). Hence the first time a
  node is popped from the open list its g-value is already optimal, and no
  node ever needs to be re-expanded.
  With several goals, h = min over goals of Manhattan - the minimum of
  consistent heuristics is still consistent.

Complexity (V = open cells, E = edges, E <= 4V on a grid)
    Time  : O(E log V)  worst case (binary-heap open list)
    Space : O(V)

Optional `extra_cost` (ghost-danger penalties) only ADDS non-negative cost to
edges, so h stays admissible and the result is still optimal for the
penalised cost function.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Set

from model.game_model import Action, GameModel, Position, STOP
from solver.base_solver import Solver


class AStarResult(NamedTuple):
    path: List[Position]   # start ... goal (inclusive); [] if no path
    cost: float            # total g(goal); inf if no path
    expanded: int          # nodes popped & expanded (useful for the report)
    found: bool


def astar_search(model: GameModel,
                 start: Position,
                 goals: Iterable[Position],
                 blocked: Optional[Set[Position]] = None,
                 extra_cost: Optional[Mapping[Position, float]] = None,
                 max_expansions: Optional[int] = None) -> AStarResult:
    """
    Find a cheapest path from `start` to ANY cell in `goals`.

    Parameters
    ----------
    model      : supplies neighbors() and the tunnel-aware manhattan()
    start      : starting cell
    goals      : one or more target cells (e.g. all remaining dots)
    blocked    : cells that may not be entered (e.g. current ghost cells)
    extra_cost : additional non-negative cost for ENTERING a cell
                 (e.g. cells close to a ghost are "expensive")
    max_expansions : optional safety cap
    """
    goal_set = frozenset(goals)
    if not goal_set:
        return AStarResult([], float("inf"), 0, False)

    blocked = blocked or set()
    extra = extra_cost or {}

    # ---- heuristic h(n) --------------------------------------------------
    if len(goal_set) == 1:
        (only_goal,) = goal_set
        def h(n: Position) -> int:
            return model.manhattan(n, only_goal)
    else:
        def h(n: Position) -> int:
            # Closest goal by Manhattan distance (min of admissible = admissible)
            return min(model.manhattan(n, g) for g in goal_set)

    # ---- data structures -------------------------------------------------
    # Open list: min-heap ordered by (f, h, tie-breaker, node).
    #  - ties on f are broken in favour of smaller h (nodes closer to a goal),
    #    which makes A* dive towards the goal instead of fanning out
    #  - the counter stops Python from comparing positions on full ties
    tie = itertools.count()
    h_start = h(start)
    open_heap = [(h_start, h_start, next(tie), start)]
    g_score: Dict[Position, float] = {start: 0.0}   # best known cost-so-far
    parent: Dict[Position, Position] = {}           # for path reconstruction
    closed: Set[Position] = set()
    expanded = 0

    while open_heap:
        _, _, _, node = heapq.heappop(open_heap)

        # Lazy deletion: a cheaper copy of this node may already have been
        # expanded; the stale heap entry is simply skipped.
        if node in closed:
            continue

        # Goal test on POP (not on push) - this is what guarantees optimality.
        if node in goal_set:
            return AStarResult(_reconstruct(parent, node), g_score[node],
                               expanded, True)

        closed.add(node)
        expanded += 1
        if max_expansions is not None and expanded >= max_expansions:
            break

        g_node = g_score[node]
        for nb in model.neighbors(node):
            if nb in closed or nb in blocked:
                continue
            # Edge cost = 1 (one move) + optional danger penalty >= 0
            tentative_g = g_node + 1.0 + extra.get(nb, 0.0)
            # Relaxation: keep nb only if this route is cheaper than any before
            if tentative_g < g_score.get(nb, float("inf")):
                g_score[nb] = tentative_g
                parent[nb] = node
                h_nb = h(nb)
                heapq.heappush(open_heap,
                               (tentative_g + h_nb, h_nb, next(tie), nb))

    return AStarResult([], float("inf"), expanded, False)


def _reconstruct(parent: Dict[Position, Position], goal: Position) -> List[Position]:
    """Walk parent pointers goal -> start, then reverse. O(path length)."""
    path = [goal]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def path_to_actions(model: GameModel, path: List[Position]) -> List[Action]:
    """Convert a list of cells into the list of actions that walks it."""
    return [model.action_between(a, b) for a, b in zip(path, path[1:])]


# --------------------------------------------------------------------------
# How the Controller uses it (MEDIUM difficulty Pac-Man)
# --------------------------------------------------------------------------
GHOST_DANGER_RADIUS = 2
GHOST_PENALTY = 8.0


def build_danger_map(model: GameModel) -> Dict[Position, float]:
    """
    Penalty field around every ghost: the closer a cell is to a ghost, the more
    expensive it is to enter, so A* detours around ghosts when it can.
    """
    danger: Dict[Position, float] = {}
    r = GHOST_DANGER_RADIUS
    for gx, gy in model.ghosts:
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                d = abs(dx) + abs(dy)
                cell = (gx + dx, gy + dy)
                if d <= r and model.is_open(cell):        # open cell only
                    danger[cell] = danger.get(cell, 0.0) + (r + 1 - d) * GHOST_PENALTY
    return danger


def astar_next_action(model: GameModel) -> Action:
    """
    Pac-Man policy for Medium: A* to the nearest dot (multi-goal search),
    avoiding ghost cells entirely and discouraging cells near ghosts.
    Returns STOP if no safe path exists.
    """
    result = astar_search(
        model,
        start=model.pacman,
        goals=model.dots,
        blocked=set(model.ghosts),
        extra_cost=build_danger_map(model),
    )
    if not result.found or len(result.path) < 2:
        return STOP
    return model.action_between(result.path[0], result.path[1]) or STOP


class AStarSolver(Solver):
    """Strategy object used by the Controller on MEDIUM difficulty."""

    name = "A* (Manhattan heuristic)"

    def get_action(self, model: GameModel) -> Action:
        result = astar_search(
            model,
            start=model.pacman,
            goals=model.dots,
            blocked=set(model.ghosts),
            extra_cost=build_danger_map(model),
        )
        self.last_stats = {"expanded": result.expanded,
                           "path_len": max(len(result.path) - 1, 0)}
        if not result.found or len(result.path) < 2:
            return STOP
        return model.action_between(result.path[0], result.path[1]) or STOP
