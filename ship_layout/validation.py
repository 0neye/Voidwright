"""Shared placement validation helpers for generation and analysis."""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from typing import Dict

from .geometry import footprint_cells
from .types import Coord, PlacedPart

__all__ = [
    "is_mirror_placement",
    "is_primary_placement",
    "part_overlaps_occupied_cells",
    "placement_within_bounds",
]


def part_overlaps_occupied_cells(
    part: object,
    geometry_cache: Dict[str, object],
    occupied_cells: AbstractSet[Coord],
) -> bool:
    """Return True when a part placement overlaps any occupied cell."""

    placed_part = PlacedPart.from_object(part)
    return bool(footprint_cells(placed_part, geometry_cache) & occupied_cells)


def placement_within_bounds(
    part: object,
    geometry_cache: Dict[str, object],
    *,
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
) -> bool:
    """Return True when all footprint cells stay inside inclusive bounds."""

    placed_part = PlacedPart.from_object(part)
    return all(
        min_x <= world_x <= max_x and min_y <= world_y <= max_y
        for world_x, world_y in footprint_cells(placed_part, geometry_cache)
    )


def is_primary_placement(part: object, geometry_cache: Dict[str, object]) -> bool:
    """Return True when all footprint cells are on the primary left side."""

    placed_part = PlacedPart.from_object(part)
    return all(cell_x <= -1 for cell_x, _cell_y in footprint_cells(placed_part, geometry_cache))


def is_mirror_placement(part: object, geometry_cache: Dict[str, object]) -> bool:
    """Return True when all footprint cells are on the mirrored right side."""

    placed_part = PlacedPart.from_object(part)
    return all(cell_x >= 0 for cell_x, _cell_y in footprint_cells(placed_part, geometry_cache))


