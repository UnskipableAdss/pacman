"""
solver/human_solver.py
----------------------
A "solver" whose decisions come from the keyboard instead of a search.

It implements the same Solver interface as BFS / A* / Minimax, so the
Controller does not need a special case in the game loop: it simply asks
get_action() every turn. The Controller feeds key presses in through
set_direction().

Steering follows the arcade rule:
  * a pressed direction is remembered ("queued");
  * Pac-Man turns as soon as that direction is legal (so you can press a turn
    slightly early);
  * otherwise he keeps going straight until he hits a wall, then stops.
"""

from __future__ import annotations

from model.game_model import STOP, Action, GameModel
from solver.base_solver import Solver


class HumanSolver(Solver):
    name = "Human (arrow keys / WASD)"

    def __init__(self) -> None:
        super().__init__()
        self.queued: Action = STOP      # the direction the player asked for
        self.current: Action = STOP     # the direction Pac-Man is moving in

    def set_direction(self, action: Action) -> None:
        self.queued = action

    def get_action(self, model: GameModel) -> Action:
        legal = model.get_legal_actions(model.pacman)
        if self.queued != STOP and self.queued in legal:
            self.current = self.queued              # turn as soon as possible
        elif self.current not in legal:
            self.current = STOP                     # ran into a wall
        return self.current
