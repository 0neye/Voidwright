"""Shared layout validation and connectivity utilities."""

from .connectivity import (
    part_attachment_cells,
    parts_overlap,
    parts_structurally_touch,
    shared_attachment_sides,
)
from .geometry import attachment_cells, attachment_segments_2x, footprint_cells, overhang_cells
from .symmetry import (
    MIRROR_ROTATION,
    PART_MIRROR_ROTATION_OVERRIDES,
    mirror_flip_x,
    mirror_part,
    mirror_rotation,
    primary_root_x,
    verify_mirror_footprint,
)
from .types import Coord, Coord2x, PlacedPart, Segment2x
from .validation import (
    is_mirror_placement,
    is_primary_placement,
    part_overlaps_occupied_cells,
    placement_within_bounds,
)
from .validator import PlacementValidator, ValidationResult

__all__ = [
    "Coord",
    "Coord2x",
    "MIRROR_ROTATION",
    "PART_MIRROR_ROTATION_OVERRIDES",
    "PlacedPart",
    "PlacementValidator",
    "Segment2x",
    "ValidationResult",
    "attachment_cells",
    "attachment_segments_2x",
    "footprint_cells",
    "is_mirror_placement",
    "is_primary_placement",
    "mirror_flip_x",
    "mirror_part",
    "mirror_rotation",
    "overhang_cells",
    "part_attachment_cells",
    "part_overlaps_occupied_cells",
    "parts_overlap",
    "parts_structurally_touch",
    "placement_within_bounds",
    "primary_root_x",
    "shared_attachment_sides",
    "verify_mirror_footprint",
]
