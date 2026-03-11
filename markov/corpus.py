"""Corpus adapters that normalize extracted data into ShipPart records."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from common.geometry import is_vanilla_part_id, load_vanilla_part_geometry

from .types import ShipPart

__all__ = ["iter_vanilla_parts_from_ship", "iter_vanilla_parts_from_graph"]


def _coerce_coord_pair(value: object) -> list[int] | None:
    """Return `[x, y]` when *value* is a 2-element numeric list."""

    if not isinstance(value, list) or len(value) != 2:
        return None
    return [int(value[0]), int(value[1])]


def _local_2x_to_global_grid(local_2x: object, center_2x: object) -> list[int] | None:
    """Convert centered `2x` coordinates into global grid coordinates."""

    local_pair = _coerce_coord_pair(local_2x)
    center_pair = _coerce_coord_pair(center_2x)
    if local_pair is None or center_pair is None:
        return None

    summed_x = local_pair[0] + center_pair[0]
    summed_y = local_pair[1] + center_pair[1]
    if summed_x % 2 != 0 or summed_y % 2 != 0:
        return None
    return [summed_x // 2, summed_y // 2]


def iter_vanilla_parts_from_ship(
    ship_data: dict,
    geometry_cache: Optional[Dict[str, object]] = None,
) -> List[ShipPart]:
    """Extract vanilla ShipPart records from one extracted ship JSON payload

    Args:
        ship_data: Extracted Cosmoteer ship JSON object
        geometry_cache: Optional shared geometry cache for repeated calls

    Returns:
        List of validated vanilla `ShipPart` records
    """

    geometry_cache = geometry_cache or load_vanilla_part_geometry()
    coord_transform = ship_data.get("coord_transform", {})
    center_2x = _coerce_coord_pair(coord_transform.get("center_2x")) if isinstance(coord_transform, dict) else None
    vanilla_parts: List[ShipPart] = []
    for part in ship_data.get("Parts", []):
        if not isinstance(part, dict):
            continue
        part_id = part.get("ID") or part.get("IDString")
        if not part_id or not is_vanilla_part_id(part_id):
            continue
        if part_id not in geometry_cache:
            continue
        if center_2x is None:
            continue
        location = _local_2x_to_global_grid(part.get("Location2x"), center_2x)
        if location is None:
            continue
        rotation = int(part.get("Rotation", 0)) % 4
        if rotation not in geometry_cache[part_id].rotations:
            continue
        vanilla_parts.append(
            ShipPart(
                part_id=part_id,
                rotation=rotation,
                x=int(location[0]),
                y=int(location[1]),
                flip_x=bool(part.get("FlipX", False)),
                flip_y=bool(part.get("FlipY", False)),
            )
        )
    return vanilla_parts


def iter_vanilla_parts_from_graph(
    graph_data: dict,
    geometry_cache: Optional[Dict[str, object]] = None,
) -> Tuple[List[ShipPart], Dict[int, int]]:
    """Extract vanilla ShipParts from one structural graph payload

    Returns:
        Tuple of `(parts, node_id_to_index)` where indexes align with `parts`
    """

    geometry_cache = geometry_cache or load_vanilla_part_geometry()
    coord_transform = graph_data.get("coord_transform", {})
    center_2x = _coerce_coord_pair(coord_transform.get("center_2x")) if isinstance(coord_transform, dict) else None
    nodes = graph_data.get("graphs", {}).get("A_structural_part_graph", {}).get("nodes", [])
    parts: List[ShipPart] = []
    node_id_to_index: Dict[int, int] = {}

    for node in nodes:
        part_id = node.get("part_id", "")
        if not is_vanilla_part_id(part_id) or part_id not in geometry_cache:
            continue
        rotation = int(node.get("rotation", 0)) % 4
        if rotation not in geometry_cache[part_id].rotations:
            continue
        if center_2x is None:
            continue
        location = _local_2x_to_global_grid(node.get("location_2x"), center_2x)
        if location is None:
            continue
        node_id = node["id"]
        node_id_to_index[node_id] = len(parts)
        parts.append(
            ShipPart(
                part_id=part_id,
                rotation=rotation,
                x=int(location[0]),
                y=int(location[1]),
            )
        )
    return parts, node_id_to_index
