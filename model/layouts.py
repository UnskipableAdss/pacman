"""
model/layouts.py
----------------
Static maze definitions. A layout is a list of equal-length strings.

Legend
    '#'  wall
    '.'  dot (starts active)
    ' '  open tile without a dot (used for the tunnel mouths)
    'P'  Pac-Man start
    'G'  ghost start (one per ghost, left-to-right / top-to-bottom order)

Row 9 has open tiles at both screen edges: the model treats them as a
horizontal "tunnel" (wrap-around edge), exactly like classic Pac-Man.
"""

CLASSIC_19x19 = [
    "###################",
    "#........#........#",
    "#.##.###.#.###.##.#",
    "#.................#",
    "#.##.#.#####.#.##.#",
    "#....#...#...#....#",
    "####.###.#.###.####",
    "####.#.......#.####",
    "####.#.##.##.#.####",
    " ......#GGG#...... ",
    "####.#.#####.#.####",
    "####.#.......#.####",
    "####.#.#####.#.####",
    "#........#........#",
    "#.##.###.#.###.##.#",
    "#..#.....P.....#..#",
    "##.#.#.#####.#.#.##",
    "#....#...#...#....#",
    "###################",
]

# Tiny maze handy for unit tests and for drawing diagrams in the report.
TINY_7x5 = [
    "#######",
    "#P..G.#",
    "#.###.#",
    "#.....#",
    "#######",
]

DEFAULT_LAYOUT = CLASSIC_19x19
