"""
controller/game_controller.py
-----------------------------
The CONTROLLER of the MVC architecture.

Responsibilities
  * difficulty menu
  * the game loop (input events, fixed-timestep turns, rendering)
  * handing the AI a CLONE of the live state and applying the chosen action

It is the only module that knows about all three other layers.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import pygame

import config
from controller.difficulty import PROFILES, Difficulty, build_ai
from model.game_model import DOWN, LEFT, RIGHT, UP, GameModel
from model.layouts import DEFAULT_LAYOUT
from view.pygame_view import View

# Keyboard -> action, used only in "play it yourself" mode.
KEYMAP = {
    pygame.K_UP: UP, pygame.K_w: UP,
    pygame.K_DOWN: DOWN, pygame.K_s: DOWN,
    pygame.K_LEFT: LEFT, pygame.K_a: LEFT,
    pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT,
}


class GameController:
    def __init__(self, layout=DEFAULT_LAYOUT) -> None:
        self.layout = layout
        self.model = GameModel(layout, lives=config.LIVES)
        self.view = View(self.model.width, self.model.height)
        self.clock = pygame.time.Clock()

        self.difficulty: Difficulty = Difficulty.EASY
        self.human = False           # False = AI plays, True = the user plays
        self._quit = False           # window closed from a menu
        self.pacman_solver = None
        self.ghost_policy = None
        self.move_interval = config.MOVE_INTERVAL_MS
        self.paused = False
        self._turn = 0
        self._think_ms = 0.0

    # ------------------------------------------------------------------
    # Top-level flow: menu -> game -> menu ...
    # ------------------------------------------------------------------
    def run(self) -> None:
        while True:
            choice = self.select_difficulty()
            if choice is None:
                break
            self.difficulty = choice          # remembered for the mode menu
            human = self.select_mode()
            if human is None:
                if self._quit:
                    break
                continue                      # ESC: back to the difficulty menu
            self.start_game(choice, human)
            if not self._game_loop():
                break
        self.view.close()

    def select_mode(self) -> Optional[bool]:
        """Second menu. Returns True = user plays, False = AI plays,
        None = go back (or quit, if self._quit was set)."""
        options = [
            ("Watch the AI solve it",
             ["Pac-Man is steered by the algorithm you picked"]),
            ("Play it yourself",
             ["Steer with the arrow keys or WASD",
              "The ghosts use the AI of the chosen difficulty"]),
        ]
        selected = 1 if self.human else 0
        footer = "UP/DOWN or 1-2 to choose, ENTER to start, ESC to go back"
        heading = f"Difficulty: {self.difficulty.value}  -  choose a mode"
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._quit = True
                    return None
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        return None
                    if ev.key == pygame.K_UP:
                        selected = (selected - 1) % len(options)
                    elif ev.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(options)
                    elif ev.key in (pygame.K_1, pygame.K_2):
                        return ev.key == pygame.K_2
                    elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                        return selected == 1
            self.view.draw_menu(options, selected, heading, footer)
            self.clock.tick(30)

    def select_difficulty(self) -> Optional[Difficulty]:
        levels = list(Difficulty)
        options = [(d.value, [PROFILES[d].pacman_text, PROFILES[d].ghost_text])
                   for d in levels]
        selected = levels.index(self.difficulty)
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._quit = True
                    return None
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        self._quit = True
                        return None
                    if ev.key == pygame.K_UP:
                        selected = (selected - 1) % len(levels)
                    elif ev.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(levels)
                    elif ev.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        return levels[ev.key - pygame.K_1]
                    elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                        return levels[selected]
            self.view.draw_menu(options, selected)
            self.clock.tick(30)

    def start_game(self, difficulty: Difficulty, human: bool = False) -> None:
        self.difficulty = difficulty
        self.human = human
        self.model = GameModel(self.layout, lives=config.LIVES)
        self.pacman_solver, self.ghost_policy = build_ai(difficulty, human=human)
        self.paused = False
        self._turn = 0
        self._think_ms = 0.0

    # ------------------------------------------------------------------
    # Game loop. Returns False if the window was closed, True for "back to menu".
    # ------------------------------------------------------------------
    def _game_loop(self) -> bool:
        acc = 0.0
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return False
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        return True
                    if ev.key == pygame.K_r:
                        self.start_game(self.difficulty, self.human)
                        acc = 0.0
                    elif ev.key == pygame.K_p:
                        self.paused = not self.paused
                    elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.move_interval = max(30, self.move_interval * 0.8)
                    elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.move_interval = min(800, self.move_interval * 1.25)
                    elif self.human and ev.key in KEYMAP:
                        self.pacman_solver.set_direction(KEYMAP[ev.key])

            acc += self.clock.tick(config.FPS)
            if acc >= self.move_interval and not self.paused \
                    and not self.model.is_terminal():
                acc = 0.0           # no catch-up bursts if the AI was slow
                self._update()

            self.view.render(self.model, self._hud())

    def _update(self) -> None:
        """One game turn: ask the AI (on COPIES of the state), then apply."""
        t0 = time.perf_counter()
        pac_action = self.pacman_solver.get_action(self.model.clone())
        self._think_ms = (time.perf_counter() - t0) * 1000

        self._turn += 1
        period = PROFILES[self.difficulty].ghost_period
        if self._turn % period == 0:
            ghost_actions = self.ghost_policy.get_actions(self.model.clone())
        else:
            ghost_actions = []
        self.model.step(pac_action, ghost_actions)

    def _hud(self) -> Dict[str, str]:
        if self.human:
            return {
                "difficulty": self.difficulty.value.upper() + "  (YOU PLAY)",
                "algorithm": f"You  vs  {self.ghost_policy.name}",
                "stats": "Arrows / WASD to steer   P pause   R restart   ESC menu",
                "paused": self.paused,
            }
        stats = " ".join(f"{k}={v:.0f}" if isinstance(v, float) and abs(v) >= 100
                         else f"{k}={v}" for k, v in self.pacman_solver.last_stats.items()
                         if k != "value")
        return {
            "difficulty": self.difficulty.value.upper(),
            "algorithm": f"{self.pacman_solver.name}  vs  {self.ghost_policy.name}",
            "stats": f"{stats}  think={self._think_ms:.0f}ms",
            "paused": self.paused,
        }
