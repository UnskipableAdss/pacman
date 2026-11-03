"""
model/game_model.py
-------------------
The MODEL of the MVC architecture.

GameModel holds the *entire* mathematical state of a Pac-Man game as a
state-space search problem:

    State  s = (pacman position, ghost positions, remaining dots, score, lives)
    Graph  G = (V, E): V = open grid cells, E = 4-neighbour moves (+ tunnel wrap)
    Actions  = {UP, DOWN, LEFT, RIGHT, STOP}
    Transition function  s' = step(s, pacman_action, ghost_actions)

It has NO dependency on Pygame, so a solver can call clone() thousands of
times per decision without touching the live game or the screen.

Design decision (performance): everything that never changes during play
(walls, the adjacency table, start positions, width/height) is built once
and SHARED BY REFERENCE between clones. Only the small mutable part
(dots, pacman, ghosts, score, lives, ...) is copied, so
clone() costs O(#dots + #ghosts) instead of O(#cells).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------
# Types & constants
# --------------------------------------------------------------------------
Position = Tuple[int, int]   # (x, y) = (column, row); origin is top-left
Action = Tuple[int, int]     # (dx, dy)

UP: Action = (0, -1)
DOWN: Action = (0, 1)
LEFT: Action = (-1, 0)
RIGHT: Action = (1, 0)
STOP: Action = (0, 0)
ACTIONS: Tuple[Action, ...] = (UP, DOWN, LEFT, RIGHT)

DOT_SCORE = 10

PLAYING, WON, LOST = "PLAYING", "WON", "LOST"
EVENT_DOT, EVENT_CAUGHT, EVENT_WIN = "dot", "caught", "win"


class GameModel:
    """Complete, cloneable game state + transition function."""

    __slots__ = (
        # ---- immutable, shared between clones ----
        "width", "height", "walls", "pacman_start", "ghost_starts",
        "has_tunnel", "_moves", "_nbrs",
        # ---- mutable, copied by clone() ----
        "dots", "pacman", "ghosts", "score", "lives", "steps",
        "status", "last_event",
    )

    # ----------------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------------
    def __init__(self, layout: Sequence[str], lives: int = 1) -> None:
        self.height = len(layout)
        self.width = len(layout[0])
        if any(len(row) != self.width for row in layout):
            raise ValueError("All layout rows must have the same length")

        walls: Set[Position] = set()
        dots: Set[Position] = set()
        ghost_starts: List[Position] = []
        pacman_start: Optional[Position] = None

        for y, row in enumerate(layout):
            for x, ch in enumerate(row):
                if ch == "#":
                    walls.add((x, y))
                elif ch == ".":
                    dots.add((x, y))
                elif ch == "P":
                    pacman_start = (x, y)
                elif ch == "G":
                    ghost_starts.append((x, y))
                # ' ' = open tile without a dot -> nothing to record

        if pacman_start is None:
            raise ValueError("Layout needs exactly one 'P'")

        # Immutable data (frozenset / tuple) -> safe to share between clones.
        self.walls: FrozenSet[Position] = frozenset(walls)
        self.pacman_start: Position = pacman_start
        self.ghost_starts: Tuple[Position, ...] = tuple(ghost_starts)
        self._build_graph()

        # Mutable game state
        self.dots: Set[Position] = dots
        self.pacman: Position = pacman_start
        self.ghosts: List[Position] = list(ghost_starts)
        self.score: int = 0
        self.lives: int = lives
        self.steps: int = 0
        self.status: str = PLAYING
        self.last_event: Optional[str] = None

    def _build_graph(self) -> None:
        """
        Pre-compute the adjacency table once: for every open cell, the list of
        (action, next_cell) pairs. Search code then gets neighbours in O(1)
        instead of re-testing walls and bounds at every expansion.

        Horizontal wrap-around is allowed only when the cell on the opposite
        edge is open (the tunnel). There is no vertical wrap.
        """
        moves: Dict[Position, Tuple[Tuple[Action, Position], ...]] = {}
        nbrs: Dict[Position, Tuple[Position, ...]] = {}
        has_tunnel = False

        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.walls:
                    continue
                out = []
                for a in ACTIONS:
                    nx, ny = x + a[0], y + a[1]
                    if not 0 <= ny < self.height:
                        continue
                    if not 0 <= nx < self.width:
                        nx %= self.width              # tunnel wrap
                        has_tunnel = True
                    if (nx, ny) in self.walls:
                        continue
                    out.append((a, (nx, ny)))
                moves[(x, y)] = tuple(out)
                nbrs[(x, y)] = tuple(p for _, p in out)

        self._moves = moves
        self._nbrs = nbrs
        self.has_tunnel = has_tunnel

    # ----------------------------------------------------------------------
    # clone() - the key method for state-space search
    # ----------------------------------------------------------------------
    def clone(self) -> "GameModel":
        """
        Return an independent copy of the current state.

        * Positions are tuples (immutable), so copying the containers
          (set / list) is already a full deep copy.
        * Static data is shared by reference (never mutated after __init__).
        * __new__ skips __init__, avoiding any re-parsing of the layout.
        Cost: O(#dots + #ghosts).
        """
        c = GameModel.__new__(GameModel)
        # shared, immutable
        c.width, c.height = self.width, self.height
        c.walls = self.walls
        c.pacman_start = self.pacman_start
        c.ghost_starts = self.ghost_starts
        c.has_tunnel = self.has_tunnel
        c._moves = self._moves
        c._nbrs = self._nbrs
        # copied, mutable
        c.dots = set(self.dots)
        c.pacman = self.pacman
        c.ghosts = list(self.ghosts)
        c.score = self.score
        c.lives = self.lives
        c.steps = self.steps
        c.status = self.status
        c.last_event = self.last_event
        return c

    # ----------------------------------------------------------------------
    # Graph queries (used by every solver)
    # ----------------------------------------------------------------------
    def is_wall(self, pos: Position) -> bool:
        return pos in self.walls

    def is_open(self, pos: Position) -> bool:
        """True if `pos` is a walkable cell (a vertex of the graph)."""
        return pos in self._nbrs

    def open_cells(self) -> Tuple[Position, ...]:
        """All vertices of the maze graph."""
        return tuple(self._nbrs)

    def neighbors(self, pos: Position) -> Tuple[Position, ...]:
        """Cells reachable from `pos` in one move. O(1) table lookup."""
        return self._nbrs[pos]

    def get_legal_actions(self, pos: Position) -> List[Action]:
        return [a for a, _ in self._moves[pos]]

    def successor(self, pos: Position, action: Action) -> Optional[Position]:
        """Cell reached by taking `action` from `pos`; None if it hits a wall."""
        if action == STOP:
            return pos
        for a, nxt in self._moves[pos]:
            if a == action:
                return nxt
        return None

    def action_between(self, a: Position, b: Position) -> Optional[Action]:
        """Inverse of successor(): the action that moves a -> b (handles tunnel)."""
        for act, nxt in self._moves[a]:
            if nxt == b:
                return act
        return None

    def manhattan(self, a: Position, b: Position) -> int:
        """
        Manhattan distance, tunnel-aware.
        With a wrap-around corridor the true grid distance can be SHORTER than
        |dx|+|dy|, so we take min(|dx|, width-|dx|). This keeps the A*
        heuristic admissible (never overestimates) on this graph.
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if self.has_tunnel:
            dx = min(dx, self.width - dx)
        return dx + dy

    # ----------------------------------------------------------------------
    # Transition function  s' = T(s, a_pacman, a_ghosts)
    # ----------------------------------------------------------------------
    def step(self, pacman_action: Action,
             ghost_actions: Optional[Sequence[Action]] = None) -> Optional[str]:
        """
        Advance the game by ONE turn, in place. Turn order:
            1. Pac-Man moves (and eats a dot if present)
            2. collision check
            3. win check
            4. all ghosts move simultaneously
            5. collision check

        Because Pac-Man and the ghosts move sequentially, "swapping places"
        (Pac-Man A->B while a ghost B->A) is detected at step 2: the ghost is
        still on B when Pac-Man arrives. No tunnelling-through-ghost bug.

        Returns the event that happened this turn (or None).
        """
        if self.status != PLAYING:
            return self.last_event
        self.last_event = None
        self.steps += 1

        # 1. Pac-Man moves; an illegal action leaves him in place.
        nxt = self.successor(self.pacman, pacman_action)
        if nxt is not None:
            self.pacman = nxt
        if self.pacman in self.dots:
            self.dots.remove(self.pacman)
            self.score += DOT_SCORE
            self.last_event = EVENT_DOT

        # 2. caught by a ghost standing on the destination?
        if self._pacman_hits_ghost():
            self._lose_life()
            return self.last_event

        # 3. cleared the board?
        if not self.dots:
            self.status = WON
            self.last_event = EVENT_WIN
            return self.last_event

        # 4. ghosts move (missing / illegal actions => ghost stays put)
        ghost_actions = ghost_actions or ()
        for i, g in enumerate(self.ghosts):
            act = ghost_actions[i] if i < len(ghost_actions) else STOP
            nxt = self.successor(g, act)
            if nxt is not None:
                self.ghosts[i] = nxt

        # 5. a ghost walked onto Pac-Man?
        if self._pacman_hits_ghost():
            self._lose_life()
        return self.last_event

    def generate_successor(self, pacman_action: Action,
                           ghost_actions: Optional[Sequence[Action]] = None
                           ) -> "GameModel":
        """Non-destructive step: returns the NEXT state, leaves self untouched."""
        nxt = self.clone()
        nxt.step(pacman_action, ghost_actions)
        return nxt

    def _pacman_hits_ghost(self) -> bool:
        return self.pacman in self.ghosts

    def _lose_life(self) -> None:
        self.lives -= 1
        self.last_event = EVENT_CAUGHT
        if self.lives <= 0:
            self.status = LOST
        else:                                   # respawn everyone
            self.pacman = self.pacman_start
            self.ghosts = list(self.ghost_starts)

    # ----------------------------------------------------------------------
    # State inspection helpers
    # ----------------------------------------------------------------------
    @property
    def remaining_dots(self) -> int:
        return len(self.dots)

    @property
    def num_ghosts(self) -> int:
        return len(self.ghosts)

    def is_win(self) -> bool:
        return self.status == WON

    def is_lose(self) -> bool:
        return self.status == LOST

    def is_terminal(self) -> bool:
        """Terminal-state test used by Minimax."""
        return self.status != PLAYING

    def state_key(self):
        """Hashable snapshot - usable as a key for memoisation / visited sets."""
        return (self.pacman, tuple(self.ghosts),
                frozenset(self.dots), self.lives)

    def to_ascii(self) -> str:
        """Text rendering for debugging, tests and report figures."""
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                p = (x, y)
                if p == self.pacman:
                    row.append("P")
                elif p in self.ghosts:
                    row.append("G")
                elif p in self.walls:
                    row.append("#")
                elif p in self.dots:
                    row.append(".")
                else:
                    row.append(" ")
            rows.append("".join(row))
        return "\n".join(rows)
