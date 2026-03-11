"""Shared structural connectivity checks across preprocessing and Markov."""

from __future__ import annotations

from typing import Dict, Set

from .geometry import attachment_cells, attachment_segments_2x, footprint_cells
from .types import PlacedPart, Segment2x

__all__ = [
    "part_attachment_cells",
    "parts_overlap",
    "parts_structurally_touch",
    "shared_attachment_sides",
]


def part_attachment_cells(part: object, geometry_cache: Dict[str, object]):
    """Return structural attachment cells for one part placement."""

    placed_part = PlacedPart.from_object(part)
    return attachment_cells(placed_part, geometry_cache)


def parts_overlap(part_a: object, part_b: object, geometry_cache: Dict[str, object]) -> bool:
    """Return True when two parts overlap in footprint occupancy cells."""

    placed_part_a = PlacedPart.from_object(part_a)
    placed_part_b = PlacedPart.from_object(part_b)
    try:
        return bool(
            footprint_cells(placed_part_a, geometry_cache)
            & footprint_cells(placed_part_b, geometry_cache)
        )
    except KeyError:
        # Unknown geometry should not crash shared validation callers
        return False


def shared_attachment_sides(part_a: object, part_b: object, geometry_cache: Dict[str, object]) -> Set[Segment2x]:
    """Return shared attachable side segments between two placed parts."""

    placed_part_a = PlacedPart.from_object(part_a)
    placed_part_b = PlacedPart.from_object(part_b)
    try:
        a_segments = attachment_segments_2x(placed_part_a, geometry_cache)
        b_segments = attachment_segments_2x(placed_part_b, geometry_cache)
    except KeyError:
        # Unknown geometry should be treated as non-structural contact
        return set()
    return a_segments & b_segments


def parts_structurally_touch(part_a: object, part_b: object, geometry_cache: Dict[str, object]) -> bool:
    """Return True when two parts share at least one attachable flat side."""

    return bool(shared_attachment_sides(part_a, part_b, geometry_cache))


