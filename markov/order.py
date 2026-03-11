"""Part ordering and local adjacency helpers for Markov training."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Tuple

from .types import ShipPart

__all__ = [
    "choose_root",
    "order_ship_parts",
    "parts_touch",
    "order_ship_parts_from_graph",
]


def _distance(a: ShipPart, b: ShipPart) -> Tuple[int, int, int, str, int, int]:
    """Return a deterministic distance sort tuple from *a* to *b*"""

    dx = b.x - a.x
    dy = b.y - a.y
    return (abs(dx) + abs(dy), abs(dx), abs(dy), b.part_id, b.x, b.y)


def choose_root(parts: Sequence[ShipPart]) -> ShipPart:
    """Choose the geometrically central root part for one ship"""

    cx = sum(part.x for part in parts) / len(parts)
    cy = sum(part.y for part in parts) / len(parts)
    scored = sorted(parts, key=lambda p: ((p.x - cx) ** 2 + (p.y - cy) ** 2, p.part_id, p.x, p.y))
    return scored[0]


def order_ship_parts(
    parts: Sequence[ShipPart], anchor_window: int = 128
) -> List[Tuple[ShipPart, Optional[ShipPart]]]:
    """Order parts for Markov training using geometric nearest-anchor heuristics"""

    remaining_parts = list(parts)
    root_part = choose_root(remaining_parts)
    remaining_parts.remove(root_part)
    remaining_parts.sort(
        key=lambda part: (
            (part.x - root_part.x) ** 2 + (part.y - root_part.y) ** 2,
            abs(part.x - root_part.x) + abs(part.y - root_part.y),
            part.part_id,
            part.x,
            part.y,
        )
    )

    ordered_parts: List[Tuple[ShipPart, Optional[ShipPart]]] = [(root_part, None)]
    placed_parts = [root_part]
    for candidate_part in remaining_parts:
        # Keep anchor selection local to avoid long-distance anchor drift
        anchor_candidates = (
            placed_parts[-anchor_window:] if len(placed_parts) > anchor_window else placed_parts
        )
        anchor_part = min(anchor_candidates, key=lambda part: _distance(part, candidate_part))
        ordered_parts.append((candidate_part, anchor_part))
        placed_parts.append(candidate_part)
    return ordered_parts


def parts_touch(a: ShipPart, b: ShipPart, geometry_cache: Dict[str, object]) -> bool:
    """Return True when two placed parts share at least one touching side"""

    a_cells = a.footprint_cells(geometry_cache)
    b_cells = b.footprint_cells(geometry_cache)
    for ax, ay in a_cells:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (ax + dx, ay + dy) in b_cells:
                return True
    return False


def order_ship_parts_from_graph(
    parts: List[ShipPart],
    node_id_to_idx: Dict[int, int],
    edges: List[dict],
    anchor_window: int = 128,
) -> List[Tuple[ShipPart, Optional[ShipPart]]]:
    """Order graph-derived parts using BFS over touching structural edges"""

    if not parts:
        return []

    part_count = len(parts)
    adjacency: Dict[int, List[int]] = defaultdict(list)
    for edge in edges:
        source_idx = node_id_to_idx.get(edge["source"])
        target_idx = node_id_to_idx.get(edge["target"])
        if source_idx is not None and target_idx is not None:
            adjacency[source_idx].append(target_idx)
            adjacency[target_idx].append(source_idx)
    for node_idx in adjacency:
        adjacency[node_idx].sort()

    root_part = choose_root(parts)
    root_idx = parts.index(root_part)

    visited = [False] * part_count
    visited[root_idx] = True
    ordered_parts: List[Tuple[ShipPart, Optional[ShipPart]]] = [(root_part, None)]
    placed_parts: List[ShipPart] = [root_part]
    placed_at_index: Dict[int, int] = {root_idx: 0}
    bfs_queue: deque = deque([root_idx])

    # First pass follows structural connectivity from graph edges
    while bfs_queue:
        current_idx = bfs_queue.popleft()
        for neighbor_idx in adjacency[current_idx]:
            if visited[neighbor_idx]:
                continue
            visited[neighbor_idx] = True
            anchor_part = placed_parts[placed_at_index[current_idx]]
            placed_at_index[neighbor_idx] = len(placed_parts)
            placed_parts.append(parts[neighbor_idx])
            ordered_parts.append((parts[neighbor_idx], anchor_part))
            bfs_queue.append(neighbor_idx)

    # Second pass handles disconnected islands via geometric nearest-neighbor
    for remaining_idx in range(part_count):
        if visited[remaining_idx]:
            continue
        anchor_candidates = (
            placed_parts[-anchor_window:] if len(placed_parts) > anchor_window else placed_parts
        )
        anchor_part = min(anchor_candidates, key=lambda part: _distance(part, parts[remaining_idx]))
        placed_at_index[remaining_idx] = len(placed_parts)
        placed_parts.append(parts[remaining_idx])
        ordered_parts.append((parts[remaining_idx], anchor_part))
        visited[remaining_idx] = True

    return ordered_parts
