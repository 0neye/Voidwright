"""Shared placement validation helpers for generation and analysis."""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from typing import Dict

from .geometry import footprint_cells
from .types import Coord, PlacedPart

__all__ = [
    "footprint_is_mirror_balanced",
    "is_anchor_eligible_mirror_primary",
    "is_mirror_placement",
    "mirror_cell_x",
    "mirror_cells_x",
    "occupied_cells_are_mirror_balanced",
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


def mirror_cell_x(world_cell: Coord) -> Coord:
    """Return one cell reflected across the mirror axis at x = -0.5."""

    cell_x, cell_y = world_cell
    return (-cell_x - 1, cell_y)


def mirror_cells_x(world_cells: AbstractSet[Coord]) -> frozenset[Coord]:
    """Return a footprint reflected across the mirror axis at x = -0.5."""

    return frozenset(mirror_cell_x(world_cell) for world_cell in world_cells)


def occupied_cells_are_mirror_balanced(world_cells: AbstractSet[Coord]) -> bool:
    """Return True when occupied cells are unchanged by horizontal reflection."""

    return frozenset(world_cells) == mirror_cells_x(world_cells)


def footprint_is_mirror_balanced(part: object, geometry_cache: Dict[str, object]) -> bool:
    """Return True when a part's occupied footprint is symmetric around x = -0.5."""

    placed_part = PlacedPart.from_object(part)
    return occupied_cells_are_mirror_balanced(footprint_cells(placed_part, geometry_cache))


def is_primary_placement(part: object, geometry_cache: Dict[str, object]) -> bool:
    """Return True when all footprint cells are on the primary left side."""

    placed_part = PlacedPart.from_object(part)
    return all(cell_x <= -1 for cell_x, _cell_y in footprint_cells(placed_part, geometry_cache))


def is_mirror_placement(part: object, geometry_cache: Dict[str, object]) -> bool:
    """Return True when all footprint cells are on the mirrored right side."""

    placed_part = PlacedPart.from_object(part)
    return all(cell_x >= 0 for cell_x, _cell_y in footprint_cells(placed_part, geometry_cache))


def is_anchor_eligible_mirror_primary(part: object, geometry_cache: Dict[str, object]) -> bool:
    """Return True when a placement can act as a mirror-mode primary anchor.

    A primary anchor can be either:
    - fully on the left side (`x <= -1`), or
    - centerline-straddling with a self-mirroring footprint
    """

    return is_primary_placement(part, geometry_cache) or footprint_is_mirror_balanced(part, geometry_cache)


