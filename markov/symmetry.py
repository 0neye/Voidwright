"""Backward-compatibility re-exports. Prefer ship_layout.symmetry for new code.

Mirror computation has moved to ``ship_layout.symmetry``.  This module stays
as a shim so that existing callers and tests continue to work unchanged.

The one behavioral difference is that ``mirror_part`` here returns a
``ShipPart`` (which has a ``footprint_cells`` method) while
``ship_layout.symmetry.mirror_part`` returns the lighter ``PlacedPart``.
"""

from __future__ import annotations

from ship_layout.symmetry import (
    MIRROR_ROTATION,
    PART_MIRROR_ROTATION_OVERRIDES,
    mirror_flip_x,
    mirror_rotation,
    primary_root_x,
    verify_mirror_footprint,
)
from ship_layout.symmetry import mirror_part as _sl_mirror_part
from ship_layout.validation import (
    footprint_is_mirror_balanced,
    is_anchor_eligible_mirror_primary,
    is_primary_placement,
)

__all__ = [
    "MIRROR_ROTATION",
    "PART_MIRROR_ROTATION_OVERRIDES",
    "footprint_is_mirror_balanced",
    "is_anchor_eligible_mirror_primary",
    "is_primary_placement",
    "mirror_flip_x",
    "mirror_part",
    "mirror_rotation",
    "primary_root_x",
    "verify_mirror_footprint",
]


def mirror_part(part, geometry_cache: dict):
    """Return a ShipPart that is the left-right mirror of *part*.

    Thin shim around ``ship_layout.symmetry.mirror_part`` that wraps the result
    in ``ShipPart`` so that callers using ``result.footprint_cells(...)`` keep
    working without modification.
    """

    from .types import ShipPart

    result = _sl_mirror_part(part, geometry_cache)
    if result is None:
        return None
    return ShipPart(
        part_id=result.part_id,
        rotation=result.rotation,
        x=result.x,
        y=result.y,
        flip_x=result.flip_x,
        flip_y=result.flip_y,
    )
