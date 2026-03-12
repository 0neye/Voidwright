"""Left-right mirror symmetry helpers for ship generation.

Axis convention
---------------
The mirror axis sits at x = -0.5, i.e. between grid columns -1 and 0.

  - PRIMARY placements:  all footprint cells have x <= -1  (left half)
  - MIRROR placements:   all footprint cells have x >= 0   (right half)
  - CENTERLINE parts:    may straddle both halves when their occupied footprint
                         is mirror-balanced across the axis
  - Mirror of footprint cell (cx, cy) is (-cx - 1, cy)
  - Mirror part origin:  mirror_x = -origin_x - W
      where W = part width for its rotation
  - y-coordinate is unchanged.

Rotation convention
-------------------
Cosmoteer uses 0 = default, 1 = 90 degrees CW, 2 = 180 degrees, 3 = 270
degrees CW. Most mirrored parts still use the historic handedness swap
`(4 - r) % 4`, while triangle half-cells keep their saved rotation and rely on
FlipX for the reflected local geometry.

Limitations (first-pass)
------------------------
- Only left-right (vertical-axis) symmetry is implemented.
- The axis is fixed between columns -1 and 0. Ships are therefore centered on
  this axis line; their combined bounding box spans x = [-N, N-1] for some N.
- Triangle half-cells rely on FlipX-aware geometry to preserve their rotation.
"""

from __future__ import annotations

from typing import Dict

from ship_layout.validation import (
    footprint_is_mirror_balanced as _footprint_is_mirror_balanced,
    is_anchor_eligible_mirror_primary as _is_anchor_eligible_mirror_primary,
    is_primary_placement as _is_primary_placement,
    mirror_cells_x as _mirror_cells_x,
    occupied_cells_are_mirror_balanced as _occupied_cells_are_mirror_balanced,
)

__all__ = [
    "MIRROR_ROTATION",
    "PART_MIRROR_ROTATION_OVERRIDES",
    "mirror_rotation",
    "mirror_flip_x",
    "mirror_part",
    "primary_root_x",
    "is_anchor_eligible_mirror_primary",
    "is_primary_placement",
    "footprint_is_mirror_balanced",
    "verify_mirror_footprint",
]

# mirror_rotation[r] gives the rotation to use for the horizontally mirrored
# copy of a part that was placed with rotation r.
MIRROR_ROTATION: Dict[int, int] = {0: 0, 1: 3, 2: 2, 3: 1}

PART_MIRROR_ROTATION_OVERRIDES: Dict[str, Dict[int, int]] = {
    "cosmoteer.armor_wedge": {0: 1, 1: 0, 2: 3, 3: 2},
    "cosmoteer.structure_wedge": {0: 1, 1: 0, 2: 3, 3: 2},
    "cosmoteer.armor_structure_hybrid_1x1": {0: 1, 1: 0, 2: 3, 3: 2},
    "cosmoteer.armor_tri": {0: 0, 1: 1, 2: 2, 3: 3},
    "cosmoteer.structure_tri": {0: 0, 1: 1, 2: 2, 3: 3},
    "cosmoteer.armor_structure_hybrid_tri": {0: 0, 1: 1, 2: 2, 3: 3},
}


def mirror_rotation(r: int, part_id: str | None = None) -> int:
    """Return the mirrored rotation for rotation *r* (left-right flip)."""

    normalized = r % 4
    if part_id is not None:
        part_override = PART_MIRROR_ROTATION_OVERRIDES.get(part_id)
        if part_override is not None:
            return part_override[normalized]
    return MIRROR_ROTATION[normalized]


def mirror_flip_x(part_id: str, flip_x: bool) -> bool:
    """Return the mirrored FlipX state for one part placement."""

    del part_id
    return not flip_x


def mirror_part(part, geometry_cache: dict):
    """Return a new ShipPart that is the left-right mirror of *part*.

    The mirror axis is at x = -0.5.

    Returns ``None`` if:
    - *part_id* is not in *geometry_cache*
    - the computed mirror rotation is not in the part's rotation table
      (very unlikely for vanilla parts; fallback to original rotation if so)

    The returned part has:
      x      = -origin_x - W    (W = part width for its rotation)
      y      = origin_y         (unchanged)
      rotation = mirrored rotation for this part
    """

    # Import from shared types to avoid model runtime import cycles
    from .types import ShipPart

    geom = geometry_cache.get(part.part_id)
    if geom is None:
        return None
    rot_geom = geom.rotations.get(part.rotation)
    if rot_geom is None:
        return None

    width = rot_geom.width
    mirrored_x = -part.x - width
    mirrored_rotation = mirror_rotation(part.rotation, part.part_id)

    if mirrored_rotation not in geom.rotations:
        mirrored_rotation = part.rotation

    return ShipPart(
        part_id=part.part_id,
        rotation=mirrored_rotation,
        x=mirrored_x,
        y=part.y,
        flip_x=mirror_flip_x(part.part_id, part.flip_x),
        flip_y=part.flip_y,
    )


def primary_root_x(part_id: str, rotation: int, geometry_cache: dict) -> int:
    """Return the x-origin that places a root part flush against the axis.

    "Flush against the axis" means the rightmost footprint cell is at x = -1,
    so its mirror's leftmost cell lands at x = 0.

    For a part of width W: origin_x = -W -> rightmost cell = -W + W - 1 = -1.
    When possible, this helper prefers a centerline-straddling root origin whose
    footprint mirrors onto itself (for example 2-wide parts at x = -1).
    """

    geom = geometry_cache.get(part_id)
    if geom is None:
        return -1
    rot_geom = geom.rotations.get(rotation)
    if rot_geom is None:
        return -1
    # Try centered placements first so mirror mode can emit one self-mirroring
    # part instead of forcing left-only roots when geometry allows it.
    local_cells = set(rot_geom.footprint_tiles)
    for candidate_x in range(-rot_geom.width, 1):
        candidate_cells = {(candidate_x + local_x, local_y) for local_x, local_y in local_cells}
        if _occupied_cells_are_mirror_balanced(candidate_cells):
            return candidate_x
    return -rot_geom.width


def is_primary_placement(part, geometry_cache: dict) -> bool:
    """Return True when all footprint cells of *part* are on the primary side."""

    return _is_primary_placement(part, geometry_cache)


def is_anchor_eligible_mirror_primary(part, geometry_cache: dict) -> bool:
    """Return True when a part can be used as a primary-side mirror anchor."""

    return _is_anchor_eligible_mirror_primary(part, geometry_cache)


def footprint_is_mirror_balanced(part, geometry_cache: dict) -> bool:
    """Return True when a part footprint is symmetric around x = -0.5."""

    return _footprint_is_mirror_balanced(part, geometry_cache)


def verify_mirror_footprint(part, mirror, geometry_cache: dict) -> bool:
    """Confirm that *mirror* equals the horizontal reflection of *part*."""

    original_cells = part.footprint_cells(geometry_cache)
    expected_mirror = _mirror_cells_x(original_cells)
    actual_mirror = mirror.footprint_cells(geometry_cache)
    return expected_mirror == actual_mirror
