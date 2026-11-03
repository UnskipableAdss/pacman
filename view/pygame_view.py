"""
view/pygame_view.py
-------------------
The VIEW of the MVC architecture.

It only READS the model to draw a frame. It never calls step(), clone() or
anything that changes state, and it contains no game rules. The few extra
strings it displays (difficulty name, solver stats) are handed in by the
Controller as plain data.

The only state kept here is purely presentational: Pac-Man's facing direction
(for the mouth) is inferred from his last drawn position.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import pygame

import config
from model.game_model import GameModel, LOST, WON


class View:
    def __init__(self, cols: int, rows: int, cell_size: int = config.CELL_SIZE,
                 hud_height: int = config.HUD_HEIGHT) -> None:
        pygame.display.init()          # no mixer: this game has no audio
        pygame.font.init()
        self.cell = cell_size
        self.hud_h = hud_height
        self.width = cols * cell_size
        self.height = rows * cell_size + hud_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pac-Man AI Solver")
        self.font = pygame.font.SysFont("dejavusansmono,consolas,menlo,monospace", 16)
        self.small = pygame.font.SysFont("dejavusansmono,consolas,menlo,monospace", 13)
        self.big = pygame.font.SysFont("dejavusansmono,consolas,menlo,monospace", 34, bold=True)
        self._prev_pac: Optional[Tuple[int, int]] = None
        self._facing = (1, 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def render(self, model: GameModel, hud: Optional[Dict[str, str]] = None) -> None:
        self.screen.fill(config.BLACK)
        self._draw_walls(model)
        self._draw_dots(model)
        self._draw_pacman(model)
        self._draw_ghosts(model)
        self._draw_hud(model, hud or {})
        if model.status in (WON, LOST):
            self.draw_game_over(model)
        pygame.display.flip()

    def draw_menu(self, options: Sequence[Tuple[str, Sequence[str]]],
                  selected: int, heading: str = "Select difficulty",
                  footer: str = "UP/DOWN or 1-3 to choose, ENTER to start, ESC to quit"
                  ) -> None:
        """options = [(title, [description lines]), ...]"""
        self.screen.fill(config.BLACK)
        self._text("PAC-MAN  AI SOLVER", self.big, config.ACCENT, self.width // 2, 70, center=True)
        self._text(heading, self.font, config.TEXT_DIM, self.width // 2, 118, center=True)
        y = 170
        for i, (title, lines) in enumerate(options):
            active = i == selected
            box = pygame.Rect(30, y - 10, self.width - 60, 40 + 20 * len(lines))
            if active:
                pygame.draw.rect(self.screen, config.WALL, box, width=2, border_radius=6)
            color = config.ACCENT if active else config.TEXT
            self._text(f"{i + 1}. {title}", self.font, color, 48, y)
            for j, line in enumerate(lines):
                self._text(line, self.small, config.TEXT_DIM, 70, y + 26 + 18 * j)
            y += 60 + 20 * len(lines)
        self._text(footer, self.small, config.TEXT_DIM,
                   self.width // 2, self.height - 30, center=True)
        pygame.display.flip()

    def draw_game_over(self, model: GameModel) -> None:
        veil = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 170))
        self.screen.blit(veil, (0, 0))
        if model.status == WON:
            msg, col = "BOARD CLEARED", config.ACCENT
        else:
            msg, col = "PAC-MAN CAUGHT", config.GHOST_COLORS[0]
        cy = self.height // 2
        self._text(msg, self.big, col, self.width // 2, cy - 20, center=True)
        self._text(f"Score {model.score}   Turns {model.steps}", self.font,
                   config.TEXT, self.width // 2, cy + 20, center=True)
        self._text("R = restart   ESC = menu", self.small, config.TEXT_DIM,
                   self.width // 2, cy + 50, center=True)

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _rect(self, x: int, y: int) -> pygame.Rect:
        return pygame.Rect(x * self.cell, y * self.cell + self.hud_h,
                           self.cell, self.cell)

    def _center(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        r = self._rect(*pos)
        return r.centerx, r.centery

    def _draw_walls(self, model: GameModel) -> None:
        for (x, y) in model.walls:
            r = self._rect(x, y)
            pygame.draw.rect(self.screen, config.WALL, r.inflate(-2, -2))
            pygame.draw.rect(self.screen, config.WALL_EDGE, r.inflate(-2, -2), width=1)

    def _draw_dots(self, model: GameModel) -> None:
        radius = max(2, self.cell // 10)
        for pos in model.dots:
            pygame.draw.circle(self.screen, config.DOT, self._center(pos), radius)

    def _draw_pacman(self, model: GameModel) -> None:
        pos = model.pacman
        if self._prev_pac is not None and pos != self._prev_pac:
            dx, dy = pos[0] - self._prev_pac[0], pos[1] - self._prev_pac[1]
            if abs(dx) > 1:                    # tunnel wrap: direction flips
                dx = -dx
            self._facing = (max(-1, min(1, dx)), max(-1, min(1, dy)))
        self._prev_pac = pos

        cx, cy = self._center(pos)
        radius = self.cell // 2 - 3
        mouth = 0.05 + 0.30 * abs(math.sin(pygame.time.get_ticks() / 110.0))
        heading = math.atan2(self._facing[1], self._facing[0])
        pts = [(cx, cy)]
        steps = 24
        for i in range(steps + 1):
            a = heading + mouth + (2 * math.pi - 2 * mouth) * i / steps
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
        pygame.draw.polygon(self.screen, config.PACMAN, pts)

    def _draw_ghosts(self, model: GameModel) -> None:
        r = self.cell // 2 - 3
        for i, pos in enumerate(model.ghosts):
            cx, cy = self._center(pos)
            color = config.GHOST_COLORS[i % len(config.GHOST_COLORS)]
            pygame.draw.circle(self.screen, color, (cx, cy - 1), r)
            pygame.draw.rect(self.screen, color, (cx - r, cy - 1, 2 * r, r))
            for ex in (-r // 2, r // 2):                       # eyes
                pygame.draw.circle(self.screen, (255, 255, 255), (cx + ex, cy - 3), 4)
                pygame.draw.circle(self.screen, (20, 20, 160), (cx + ex, cy - 3), 2)

    def _draw_hud(self, model: GameModel, hud: Dict[str, str]) -> None:
        self._text(f"SCORE {model.score}", self.font, config.ACCENT, 12, 8)
        self._text(f"LIVES {model.lives}", self.font, config.TEXT, 190, 8)
        self._text(f"DOTS {model.remaining_dots}", self.font, config.TEXT, 300, 8)
        self._text(hud.get("difficulty", ""), self.font, config.ACCENT, self.width - 12, 8, right=True)
        self._text(hud.get("algorithm", ""), self.small, config.TEXT_DIM, 12, 29)
        stats = hud.get("stats", "")
        if hud.get("paused"):
            stats = "PAUSED  " + stats
        self._text(stats, self.small, config.TEXT_DIM, 12, 45)

    def _text(self, s: str, font: pygame.font.Font, color, x: int, y: int,
              center: bool = False, right: bool = False) -> None:
        surf = font.render(s, True, color)
        rect = surf.get_rect()
        if center:
            rect.midtop = (x, y)
        elif right:
            rect.topright = (x, y)
        else:
            rect.topleft = (x, y)
        self.screen.blit(surf, rect)
