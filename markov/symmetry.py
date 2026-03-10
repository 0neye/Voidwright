"""Left-right mirror symmetry helpers for ship generation.

Axis convention
---------------
The mirror axis sits at x = -0.5, i.e. between grid columns -1 and 0.

  - PRIMARY placements:  all footprint cells have x ≤ -1  (left half)
  - MIRROR placements:   all footprint cells have x ≥  0  (right half)
  - Mirror of footprint cell (cx, cy) is (-cx - 1, cy)
  - Mirror part origin:  mirror_x = -origin_x - W
      where W = part width for its rotation
  - y-coordinate is unchanged.

No part can sit "on" the axis (no integer x satisfies x = -0.5), so there are
no centerline-straddling parts to handle. Parts at x = -1 (1-cell wide) mirror
to x = 0, giving a two-cell-wide symmetric band around the centerline.

Rotation convention
-------------------
Cosmoteer uses 0 = default, 1 = 90° CW, 2 = 180°, 3 = 270° CW.
Reflecting across the vertical axis reverses the handedness of CW rotations:

    mirror_rotation(r) = (4 - r) % 4
    i.e. 0 ↔ 0, 1 ↔ 3, 2 ↔ 2

This is consistent with vanilla footprint geometry: a 2×1 part at rotation 0
occupies tiles (0,0) and (1,0); its mirror footprint is identical (horizontal
strip), which is rotation 0 of the same part. For asymmetric parts like wedges
(1×1 but directional), the rotation flip changes their visual orientation so the
mirrored part faces the correct way.

Limitations (first-pass)
------------------------
- Only left-right (vertical-axis) symmetry is implemented.
- The axis is fixed between columns -1 and 0. Ships are therefore centred on
  this axis line; their combined bounding box spans x = [−N, N−1] for some N.
- Rotations are transformed with (4-r)%4. For the vast majority of vanilla
  parts this produces the correct mirrored geometry. A small number of parts
  whose footprint is inherently asymmetric under horizontal reflection (if any
  exist in the vanilla set) would need a per-part override table; none are
  currently known to require it.
"""

from __future__ import annotations

from typing import Dict

# mirror_rotation[r] gives the rotation to use for the horizontally mirrored
# copy of a part that was placed with rotation r.
MIRROR_ROTATION: Dict[int, int] = {0: 0, 1: 3, 2: 2, 3: 1}

# Some directional 1x1 wedge sprites use a different visual handedness than the
# general footprint-based rotation rule. The corpus of mirror-built vanilla
# ships consistently maps these wedges with 0<->1 and 2<->3 on reflection.
PART_MIRROR_ROTATION_OVERRIDES: Dict[str, Dict[int, int]] = {
    "cosmoteer.armor_wedge": {0: 1, 1: 0, 2: 3, 3: 2},
    "cosmoteer.structure_wedge": {0: 1, 1: 0, 2: 3, 3: 2},
    "cosmoteer.armor_structure_hybrid_1x1": {0: 1, 1: 0, 2: 3, 3: 2},
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
      x      = -origin_x - W    (W = width of the part for its rotation)
      y      = origin_y          (unchanged)
      rotation = (4 - rotation) % 4
    """

    # Import here to avoid circular imports; ShipPart lives in model.
    from .model import ShipPart

    geom = geometry_cache.get(part.part_id)
    if geom is None:
        return None
    rot_geom = geom.rotations.get(part.rotation)
    if rot_geom is None:
        return None

    width = rot_geom.width
    mirrored_x = -part.x - width
    mirrored_rotation = mirror_rotation(part.rotation, part.part_id)

    # Guard: if computed mirror rotation has no geometry, fall back to original.
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
    """

    geom = geometry_cache.get(part_id)
    if geom is None:
        return -1
    rot_geom = geom.rotations.get(rotation)
    if rot_geom is None:
        return -1
    return -rot_geom.width


def is_primary_placement(part, geometry_cache: dict) -> bool:
    """Return True when all footprint cells of *part* are on the primary side."""

    return all(cell_x <= -1 for cell_x, _cell_y in part.footprint_cells(geometry_cache))


def verify_mirror_footprint(part, mirror, geometry_cache: dict) -> bool:
    """Confirm that *mirror* equals the horizontal reflection of *part*."""

    original_cells = part.footprint_cells(geometry_cache)
    expected_mirror = frozenset((-cell_x - 1, cell_y) for cell_x, cell_y in original_cells)
    actual_mirror = mirror.footprint_cells(geometry_cache)
    return expected_mirror == actual_mirror
