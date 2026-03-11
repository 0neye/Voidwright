"""Shared layout validation and connectivity utilities."""

from .connectivity import (
    part_attachment_cells,
    parts_overlap,
    parts_structurally_touch,
    shared_attachment_sides,
)
from .geometry import attachment_cells, attachment_segments_2x, footprint_cells, overhang_cells
from .types import Coord, Coord2x, PlacedPart, Segment2x
from .validation import (
    is_mirror_placement,
    is_primary_placement,
    part_overlaps_occupied_cells,
    placement_within_bounds,
)

__all__ = [
    "Coord",
    "Coord2x",
    "PlacedPart",
    "Segment2x",
    "attachment_cells",
    "attachment_segments_2x",
    "footprint_cells",
    "is_mirror_placement",
    "is_primary_placement",
    "overhang_cells",
    "part_attachment_cells",
    "part_overlaps_occupied_cells",
    "parts_overlap",
    "parts_structurally_touch",
    "placement_within_bounds",
    "shared_attachment_sides",
]
