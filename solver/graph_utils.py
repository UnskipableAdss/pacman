"""
solver/graph_utils.py
---------------------
Small graph helpers shared by the solvers and the ghost policies.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Set

from model.game_model import GameModel, Position


def bfs_distances(model: GameModel, source: Position,
                  blocked: Optional[Set[Position]] = None) -> Dict[Position, int]:
    """
    Single-source shortest path lengths (unit edge cost) by Breadth-First Search.

    Time O(V + E), space O(V). Cells that cannot be reached are simply absent
    from the returned dict.
    """
    dist: Dict[Position, int] = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        nd = dist[node] + 1
        for nb in model.neighbors(node):
            if nb not in dist and (blocked is None or nb not in blocked):
                dist[nb] = nd
                queue.append(nb)
    return dist


def all_pairs_distances(model: GameModel) -> Dict[Position, Dict[Position, int]]:
    """
    True maze distance between every pair of open cells: one BFS per vertex,
    O(V * (V + E)). For the 19x19 maze (~250 cells) that is ~250k operations,
    paid ONCE; afterwards any distance query inside Minimax is an O(1) lookup.
    """
    return {cell: bfs_distances(model, cell) for cell in model.open_cells()}
