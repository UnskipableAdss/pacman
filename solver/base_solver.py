"""
solver/base_solver.py
---------------------
Abstract interfaces (Strategy pattern). The Controller only ever talks to these
two types, so a difficulty level is just a (Solver, GhostPolicy) pair.

Both depend on the Model only - never on the View or the Controller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from model.game_model import Action, GameModel


class Solver(ABC):
    """Chooses Pac-Man's next action from a (cloned) game state."""

    name: str = "Solver"

    def __init__(self) -> None:
        # Free-form numbers for the HUD / the report (nodes expanded, depth...)
        self.last_stats: Dict[str, float] = {}

    @abstractmethod
    def get_action(self, model: GameModel) -> Action:
        """Return the action Pac-Man should take. MUST NOT mutate `model`
        (the Controller passes a clone anyway)."""


class GhostPolicy(ABC):
    """Chooses one action per ghost for the coming turn."""

    name: str = "GhostPolicy"

    @abstractmethod
    def get_actions(self, model: GameModel) -> List[Action]:
        """Return a list with one action per ghost (index-aligned with
        model.ghosts)."""
