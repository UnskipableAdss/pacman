"""Tunable constants shared by View and Controller (no game logic here)."""

CELL_SIZE = 32          # pixels per maze tile
HUD_HEIGHT = 64         # pixels reserved above the maze
FPS = 60                # render rate
MOVE_INTERVAL_MS = 140  # time between game turns (smaller = faster game)
LIVES = 3
MINIMAX_DEPTH = 3       # rounds of look-ahead for HARD

# Colours (R, G, B)
BLACK = (0, 0, 0)
WALL = (33, 33, 222)
WALL_EDGE = (90, 110, 255)
DOT = (255, 210, 170)
PACMAN = (255, 235, 0)
TEXT = (235, 235, 235)
TEXT_DIM = (140, 140, 160)
ACCENT = (255, 235, 0)
GHOST_COLORS = [(255, 40, 40), (255, 150, 220), (0, 230, 230), (255, 160, 40)]
