"""Shared heat-exchanger radius helpers.

The heat exchanger uses a fixed 1x-tile stencil:
- start with an 11x11 square centered on the exchanger tile
- remove 5 cells from each corner:
  - the corner cell itself
  - the two cells adjacent along the horizontal edge
  - the two cells adjacent along the vertical edge

That leaves 101 included tiles total, including the exchanger tile itself.

The visualization/graph pipeline works in a centered local ``2x`` frame where
tile origins can be odd. To avoid half-tile anchor drift, this module exposes
the stencil as tile-origin coordinates in that local ``2x`` frame.
"""

from __future__ import annotations

import functools
from typing import Any, Mapping

__all__ = [
    "HEAT_EXCHANGER_ABSORPTION_RADIUS_TILES",
    "HEAT_EXCHANGER_PART_ID",
    "footprint_tile_origins_2x",
    "heat_exchanger_radius_region_tile_origins_2x",
    "is_heat_exchanger",
    "tile_set_within_heat_exchanger_radius_2x",
]

HEAT_EXCHANGER_PART_ID = "cosmoteer.heat_exchanger"
HEAT_EXCHANGER_ABSORPTION_RADIUS_TILES = 5.0


def is_heat_exchanger(part_id: object) -> bool:
    """Return True when *part_id* is the vanilla heat exchanger."""

    return str(part_id).lower() == HEAT_EXCHANGER_PART_ID


def footprint_tile_origins_2x(node: Mapping[str, Any]) -> set[tuple[int, int]]:
    """Return occupied 1x tile origins for *node* in the centered local 2x frame."""

    location_2x = node.get("location_2x")
    footprint = node.get("footprint")
    if not isinstance(location_2x, (list, tuple)) or len(location_2x) != 2:
        return set()
    origin_x2 = int(location_2x[0])
    origin_y2 = int(location_2x[1])
    if not isinstance(footprint, dict):
        return {(origin_x2, origin_y2)}
    width_tiles = int(footprint.get("width", 0))
    height_tiles = int(footprint.get("height", 0))
    if width_tiles <= 0 or height_tiles <= 0:
        return {(origin_x2, origin_y2)}
    return {
        (origin_x2 + 2 * col, origin_y2 + 2 * row)
        for row in range(height_tiles)
        for col in range(width_tiles)
    }


@functools.lru_cache(maxsize=None)
def heat_exchanger_radius_region_tile_origins_2x(
    exchanger_tile_origin_2x: tuple[int, int],
    radius_tiles: float = HEAT_EXCHANGER_ABSORPTION_RADIUS_TILES,
) -> frozenset[tuple[int, int]]:
    """Return the exact heat-exchanger stencil as local 2x tile origins.

    The result is cached keyed on ``(exchanger_tile_origin_2x, radius_tiles)``.
    """

    radius = int(max(0.0, float(radius_tiles)))
    base_x2, base_y2 = exchanger_tile_origin_2x
    region: set[tuple[int, int]] = set()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            abs_dx = abs(dx)
            abs_dy = abs(dy)
            in_corner_cutout = (
                (abs_dx == radius and abs_dy >= radius - 2)
                or (abs_dy == radius and abs_dx >= radius - 2)
            )
            if in_corner_cutout:
                continue
            region.add((base_x2 + 2 * dx, base_y2 + 2 * dy))
    return frozenset(region)


def tile_set_within_heat_exchanger_radius_2x(
    exchanger_tile_origins_2x: set[tuple[int, int]],
    candidate_tile_origins_2x: set[tuple[int, int]],
    radius_tiles: float = HEAT_EXCHANGER_ABSORPTION_RADIUS_TILES,
) -> bool:
    """Return True when any candidate tile origin is inside the exact exchanger stencil."""

    if not exchanger_tile_origins_2x or not candidate_tile_origins_2x:
        return False
    included_tile_origins_2x: set[tuple[int, int]] = set()
    for exchanger_tile_origin_2x in exchanger_tile_origins_2x:
        included_tile_origins_2x.update(
            heat_exchanger_radius_region_tile_origins_2x(
                exchanger_tile_origin_2x,
                radius_tiles,
            )
        )
    return bool(included_tile_origins_2x & candidate_tile_origins_2x)
