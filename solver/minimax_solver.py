"""
solver/minimax_solver.py
------------------------
Minimax with Alpha-Beta pruning for the HARD difficulty.

Game tree
---------
One "round" = Pac-Man moves, then each (nearby) ghost moves in turn:

    MAX (Pac-Man) -> MIN (ghost a) -> MIN (ghost b) -> ... -> MAX -> ...

All ghosts are treated as ONE coalition that minimises Pac-Man's value, so the
search assumes the ghosts play their best coordinated trap. That is what makes
Pac-Man's decisions "survive coordinated ghost traps".

The tree is generated with GameModel.clone()/step(): every node is an
independent copy, so the live game is never touched.

Alpha-Beta
----------
alpha = best value MAX can already guarantee on the path to the root
beta  = best value MIN can already guarantee on the path to the root
If alpha >= beta the rest of a node's children cannot change the root decision
and are skipped. The returned root value is identical to plain Minimax; only
the number of visited nodes shrinks (best case O(b^(d/2)) instead of O(b^d)
with perfect move ordering).

Move ordering (makes pruning effective)
  MAX node: children sorted by the evaluation function, best first.
  MIN node: ghost moves sorted by distance to Pac-Man, closest first.

Practical optimisations
  * Ghosts farther than `ghost_radius` (true maze distance) are not branched
    on - they cannot reach Pac-Man inside the horizon.
  * All-pairs maze distances are precomputed once (O(1) lookups in eval).

Complexity: b = branching factor (<= 4 per agent), m = #active ghosts,
d = depth in rounds  ->  O((b^(m+1))^d) worst case, shrinking towards
O((b^(m+1))^(d/2)) with good ordering.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from model.game_model import (EVENT_CAUGHT, STOP, WON, Action, GameModel,
                              Position)
from solver.base_solver import Solver
from solver.graph_utils import all_pairs_distances

INF = float("inf")

# Penalty for a ghost at a given TRUE maze distance from Pac-Man.
DANGER = {1: 300.0, 2: 90.0, 3: 30.0, 4: 10.0}
TERRITORY_WEIGHT = 2.0
DOT_PULL = 1.5          # value lost per step of distance to the nearest dot


class MinimaxSolver(Solver):
    WIN = 10_000.0
    LOSS = -10_000.0

    def __init__(self, depth: int = 3, ghost_radius: int = 8,
                 use_pruning: bool = True) -> None:
        super().__init__()
        self.depth = depth
        self.ghost_radius = ghost_radius
        self.use_pruning = use_pruning
        self.name = ("Minimax + Alpha-Beta" if use_pruning
                     else "Minimax (no pruning)")
        self.nodes = 0
        self._dist: Dict[Position, Dict[Position, int]] = {}
        self._dist_key = None
        self._root_action: Action = STOP

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def get_action(self, model: GameModel) -> Action:
        self._ensure_distances(model)
        self.nodes = 0
        self._root_action = STOP
        value = self._max_value(model, self.depth, -INF, INF, root=True)
        self.last_stats = {"nodes": self.nodes, "depth": self.depth,
                           "value": value}
        return self._root_action

    def _ensure_distances(self, model: GameModel) -> None:
        # `walls` is shared by reference between clones, so identity is a
        # cheap "same maze?" test.
        if self._dist_key is not model.walls:
            self._dist = all_pairs_distances(model)
            self._dist_key = model.walls

    # ------------------------------------------------------------------
    # MAX node: Pac-Man chooses the action with the HIGHEST value
    # ------------------------------------------------------------------
    def _max_value(self, s: GameModel, depth: int, alpha: float, beta: float,
                   root: bool = False) -> float:
        self.nodes += 1
        if depth == 0:
            return self._evaluate(s)

        active = self._active_ghosts(s)
        best = -INF
        for action, child in self._pacman_children(s):
            if child.last_event == EVENT_CAUGHT:
                v = self.LOSS - depth            # sooner loss = worse
            elif child.status == WON:
                v = self.WIN + depth             # sooner win = better
            else:
                v = self._min_value(child, depth, 0, active, alpha, beta)

            if v > best:
                best = v
                if root:
                    self._root_action = action

            if self.use_pruning:
                if best >= beta:                 # MIN above will never allow this
                    break
                alpha = max(alpha, best)
        return best

    # ------------------------------------------------------------------
    # MIN node: ghost active[k] chooses the action with the LOWEST value
    # ------------------------------------------------------------------
    def _min_value(self, s: GameModel, depth: int, k: int, active: List[int],
                   alpha: float, beta: float) -> float:
        if k == len(active):                     # every ghost has moved:
            return self._max_value(s, depth - 1, alpha, beta)   # next round

        self.nodes += 1
        worst = INF
        for child in self._ghost_children(s, active[k]):
            if child.last_event == EVENT_CAUGHT:
                v = self.LOSS - depth
            else:
                v = self._min_value(child, depth, k + 1, active, alpha, beta)

            worst = min(worst, v)
            if self.use_pruning:
                if worst <= alpha:               # MAX above already has better
                    break
                beta = min(beta, worst)
        return worst

    # ------------------------------------------------------------------
    # Successor generation (+ move ordering)
    # ------------------------------------------------------------------
    def _pacman_children(self, s: GameModel) -> List[Tuple[Action, GameModel]]:
        kids = [(a, s.generate_successor(a)) for a in s.get_legal_actions(s.pacman)]
        if self.use_pruning:
            kids.sort(key=lambda ac: self._order_key(ac[1]), reverse=True)
        return kids

    def _order_key(self, child: GameModel) -> float:
        if child.last_event == EVENT_CAUGHT:
            return -INF
        if child.status == WON:
            return INF
        return self._evaluate(child, quick=True)   # cheap: ordering only

    def _ghost_children(self, s: GameModel, gi: int) -> List[GameModel]:
        acts: List[Action] = [STOP] * s.num_ghosts
        kids = []
        for a in s.get_legal_actions(s.ghosts[gi]):
            acts[gi] = a
            # Pac-Man stands still (STOP); only ghost `gi` moves.
            kids.append(s.generate_successor(STOP, acts))
        if self.use_pruning:
            d = self._dist[s.pacman]
            kids.sort(key=lambda c: d.get(c.ghosts[gi], 999)
                      if c.last_event != EVENT_CAUGHT else -1)
        return kids

    def _active_ghosts(self, s: GameModel) -> List[int]:
        d = self._dist[s.pacman]
        return [i for i, g in enumerate(s.ghosts)
                if d.get(g, 999) <= self.ghost_radius]

    # ------------------------------------------------------------------
    # Static evaluation at the search horizon (from Pac-Man's point of view)
    # ------------------------------------------------------------------
    def _evaluate(self, s: GameModel, quick: bool = False) -> float:
        """
        value = score
              - DOT_PULL * (maze distance to nearest dot) pull towards food
              - DANGER[distance to each ghost]            keep a safety margin
              - 40 if cornered in a dead end near a ghost
              + 2.0 * territory                           (skipped if quick)

        territory = number of cells Pac-Man can reach STRICTLY before every
        ghost (a Voronoi region on the maze graph). It is what lets a shallow
        search see a pincer coming: walking into a pocket whose exits the
        ghosts reach first shrinks the territory long before any ghost is
        within the search horizon.
        """
        d = self._dist[s.pacman]
        value = float(s.score)                   # 10 per dot already eaten

        if s.dots:
            value -= DOT_PULL * min(d.get(x, 99) for x in s.dots)

        nearest = 99
        for g in s.ghosts:
            gd = d.get(g, 99)
            nearest = min(nearest, gd)
            value -= DANGER.get(gd, 0.0)

        if nearest <= 5 and len(s.neighbors(s.pacman)) == 1:
            value -= 40.0

        if not quick and s.ghosts:
            ghost_maps = [self._dist[g] for g in s.ghosts]
            territory = 0
            for cell, dp in d.items():
                for gm in ghost_maps:
                    if gm[cell] <= dp:           # a ghost gets there first/tie
                        break
                else:
                    territory += 1
            value += TERRITORY_WEIGHT * territory
        return value
