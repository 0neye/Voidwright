"""2x-space geometry helpers for ship structure and validation."""

from __future__ import annotations

import logging
from typing import Dict, Set

from common.geometry import polygon_vertices_to_2x, resolve_geometry_part_id_and_rotation

from .types import Coord, Coord2x, PlacedPart, Segment2x

__all__ = [
    "attachment_cells",
    "attachment_segments_2x",
    "footprint_cells",
    "overhang_cells",
]

_logger = logging.getLogger(__name__)


def _resolve_geometry(
    part: PlacedPart,
    geometry_cache: Dict[str, object],
) -> tuple[object, object, int]:
    """Resolve part ID plus rotation into geometry-cache lookup records."""

    resolved_part_id, resolved_rotation = resolve_geometry_part_id_and_rotation(
        part.part_id,
        part.rotation,
    )
    geometry = geometry_cache[resolved_part_id]
    local_rotation = geometry.rotations.get(resolved_rotation)
    if local_rotation is None:
        _logger.warning(
            "Part %r has no geometry for rotation %d; falling back to rotation 0 — "
            "attachment geometry may be incorrect",
            resolved_part_id,
            resolved_rotation,
        )
        local_rotation = geometry.rotations.get(0) or next(iter(geometry.rotations.values()))
    return geometry, local_rotation, resolved_rotation


def _rotate_cell_around_bbox(local_cell: Coord, rotation: int, width: int, height: int) -> Coord:
    """Rotate one local cell around the part bbox using Cosmoteer CW rotations."""

    cell_x, cell_y = local_cell
    normalized_rotation = rotation % 4
    if normalized_rotation == 0:
        return (cell_x, cell_y)
    if normalized_rotation == 1:
        return (height - 1 - cell_y, cell_x)
    if normalized_rotation == 2:
        return (width - 1 - cell_x, height - 1 - cell_y)
    return (cell_y, width - 1 - cell_x)


def _flip_cell_within_bbox(
    local_cell: Coord,
    width: int,
    height: int,
    *,
    flip_x: bool,
    flip_y: bool,
) -> Coord:
    """Reflect one local cell across the part-local bbox when flips are set."""

    cell_x, cell_y = local_cell
    if flip_x:
        cell_x = width - 1 - cell_x
    if flip_y:
        cell_y = height - 1 - cell_y
    return (cell_x, cell_y)


def _flip_cells_within_bbox(
    local_cells: Set[Coord],
    width: int,
    height: int,
    *,
    flip_x: bool,
    flip_y: bool,
) -> Set[Coord]:
    """Reflect a local cell set across the part-local bbox when flips are set."""

    if not (flip_x or flip_y):
        return set(local_cells)
    return {
        _flip_cell_within_bbox(local_cell, width, height, flip_x=flip_x, flip_y=flip_y)
        for local_cell in local_cells
    }


def _flip_vertices_2x_within_bbox(
    vertices_2x: tuple[Coord2x, ...],
    width: int,
    height: int,
    *,
    flip_x: bool,
    flip_y: bool,
) -> tuple[Coord2x, ...]:
    """Reflect local 2x polygon vertices across the part-local bbox."""

    if not (flip_x or flip_y):
        return vertices_2x

    width_2x = width * 2
    height_2x = height * 2
    flipped_vertices: list[Coord2x] = []
    for vertex_x, vertex_y in vertices_2x:
        if flip_x:
            vertex_x = width_2x - vertex_x
        if flip_y:
            vertex_y = height_2x - vertex_y
        flipped_vertices.append((vertex_x, vertex_y))
    return tuple(flipped_vertices)


def _local_physical_rect_cells(geometry: object, resolved_rotation: int) -> set[Coord]:
    """Return local attachment-body cells derived from rotated physical_rect."""

    if geometry.physical_rect is None:
        return set()

    base_rotation = geometry.rotations.get(0) or next(iter(geometry.rotations.values()))
    base_width = int(base_rotation.width)
    base_height = int(base_rotation.height)
    base_cells = {
        (geometry.physical_rect.x + delta_x, geometry.physical_rect.y + delta_y)
        for delta_x in range(geometry.physical_rect.width)
        for delta_y in range(geometry.physical_rect.height)
    }
    return {
        _rotate_cell_around_bbox(cell, resolved_rotation, base_width, base_height)
        for cell in base_cells
    }


def _cell_boundary_segments_from_local_cells_2x(local_cells: Set[Coord]) -> Set[Segment2x]:
    """Return 2x boundary segments for a local-cell set using edge cancellation."""

    local_segments: Set[Segment2x] = set()
    for local_cell in local_cells:
        for boundary_segment in _cell_boundary_segments_2x(local_cell):
            if boundary_segment in local_segments:
                local_segments.remove(boundary_segment)
            else:
                local_segments.add(boundary_segment)
    return local_segments


def _normalize_segment(segment_start: Coord2x, segment_end: Coord2x) -> Segment2x:
    """Return one segment in deterministic endpoint order."""

    return (segment_start, segment_end) if segment_start <= segment_end else (segment_end, segment_start)


def _split_axis_aligned_segment(segment_start: Coord2x, segment_end: Coord2x) -> Set[Segment2x]:
    """Split one axis-aligned segment into unit 2x edge segments."""

    normalized_start, normalized_end = _normalize_segment(segment_start, segment_end)
    start_x, start_y = normalized_start
    end_x, end_y = normalized_end

    split_segments: Set[Segment2x] = set()
    if start_x == end_x:
        # Vertical segment: walk in 2x tile-edge steps to keep comparisons exact
        step = 2
        for segment_start_y in range(start_y, end_y, step):
            split_segments.add(
                _normalize_segment(
                    (start_x, segment_start_y),
                    (start_x, segment_start_y + step),
                )
            )
        return split_segments

    if start_y == end_y:
        # Horizontal segment: walk in 2x tile-edge steps to keep comparisons exact
        step = 2
        for segment_start_x in range(start_x, end_x, step):
            split_segments.add(
                _normalize_segment(
                    (segment_start_x, start_y),
                    (segment_start_x + step, start_y),
                )
            )
        return split_segments

    return split_segments


def _cell_boundary_segments_2x(local_cell: Coord) -> tuple[Segment2x, Segment2x, Segment2x, Segment2x]:
    """Return local 2x boundary segments for one footprint cell."""

    cell_x, cell_y = local_cell
    x0 = cell_x * 2
    x1 = x0 + 2
    y0 = cell_y * 2
    y1 = y0 + 2
    return (
        _normalize_segment((x0, y0), (x1, y0)),
        _normalize_segment((x1, y0), (x1, y1)),
        _normalize_segment((x0, y1), (x1, y1)),
        _normalize_segment((x0, y0), (x0, y1)),
    )


def _rect_boundary_segments_2x(x2: int, y2: int, width2: int, height2: int) -> Set[Segment2x]:
    """Return axis-aligned 2x boundary segments for one rectangle."""

    x_end = x2 + width2
    y_end = y2 + height2
    full_edges = (
        ((x2, y2), (x_end, y2)),
        ((x_end, y2), (x_end, y_end)),
        ((x2, y_end), (x_end, y_end)),
        ((x2, y2), (x2, y_end)),
    )
    split_segments: Set[Segment2x] = set()
    for edge_start, edge_end in full_edges:
        split_segments.update(_split_axis_aligned_segment(edge_start, edge_end))
    return split_segments


def _polygon_boundary_segments_2x(vertices_2x: tuple[Coord2x, ...]) -> Set[Segment2x]:
    """Return axis-aligned 2x boundary segments from a polygon loop."""

    segments: Set[Segment2x] = set()
    if len(vertices_2x) < 2:
        return segments

    wrapped_vertices = vertices_2x + (vertices_2x[0],)
    for idx in range(len(vertices_2x)):
        start = wrapped_vertices[idx]
        end = wrapped_vertices[idx + 1]
        # Structural touching should only count flat side contacts, not diagonals
        if start[0] != end[0] and start[1] != end[1]:
            continue
        if start == end:
            continue
        segments.update(_split_axis_aligned_segment(start, end))
    return segments


def _translate_segment(segment: Segment2x, offset_2x: Coord2x) -> Segment2x:
    """Translate one local 2x segment into world-space coordinates."""

    (start_x, start_y), (end_x, end_y) = segment
    offset_x, offset_y = offset_2x
    return _normalize_segment((start_x + offset_x, start_y + offset_y), (end_x + offset_x, end_y + offset_y))


def footprint_cells(part: PlacedPart, geometry_cache: Dict[str, object]) -> frozenset[Coord]:
    """Return world footprint cells for one part placement."""

    _geometry, local_rotation, _resolved_rotation = _resolve_geometry(part, geometry_cache)
    local_footprint_cells = _flip_cells_within_bbox(
        set(local_rotation.footprint_tiles),
        local_rotation.width,
        local_rotation.height,
        flip_x=part.flip_x,
        flip_y=part.flip_y,
    )
    return frozenset((part.x + local_x, part.y + local_y) for local_x, local_y in local_footprint_cells)


def attachment_cells(part: PlacedPart, geometry_cache: Dict[str, object]) -> frozenset[Coord]:
    """Return world cells that can structurally attach to neighboring parts."""

    geometry, local_rotation, resolved_rotation = _resolve_geometry(part, geometry_cache)
    local_footprint_cells = _flip_cells_within_bbox(
        set(local_rotation.footprint_tiles),
        local_rotation.width,
        local_rotation.height,
        flip_x=part.flip_x,
        flip_y=part.flip_y,
    )

    # For parts with explicit physical_rect metadata, use only the core body as
    # structural attachment area and treat the rest as decorative overhang.
    if geometry.physical_rect is not None:
        core_cells = _flip_cells_within_bbox(
            _local_physical_rect_cells(geometry, resolved_rotation),
            local_rotation.width,
            local_rotation.height,
            flip_x=part.flip_x,
            flip_y=part.flip_y,
        )
        local_attachment_cells = core_cells & local_footprint_cells
    else:
        local_attachment_cells = local_footprint_cells

    return frozenset((part.x + local_x, part.y + local_y) for local_x, local_y in local_attachment_cells)


def overhang_cells(part: PlacedPart, geometry_cache: Dict[str, object]) -> frozenset[Coord]:
    """Return world footprint cells that should not count as structural body."""

    return footprint_cells(part, geometry_cache) - attachment_cells(part, geometry_cache)


def attachment_segments_2x(part: PlacedPart, geometry_cache: Dict[str, object]) -> Set[Segment2x]:
    """Return world-space 2x attachable boundary segments for one placement."""

    geometry, local_rotation, resolved_rotation = _resolve_geometry(part, geometry_cache)
    world_origin_2x: Coord2x = (part.x * 2, part.y * 2)

    # Polygon edges are the most accurate collision boundary for wedges/tris.
    local_polygon_vertices_2x = _flip_vertices_2x_within_bbox(
        polygon_vertices_to_2x(local_rotation.polygon_vertices),
        local_rotation.width,
        local_rotation.height,
        flip_x=part.flip_x,
        flip_y=part.flip_y,
    )
    if local_polygon_vertices_2x:
        return {
            _translate_segment(local_segment, world_origin_2x)
            for local_segment in _polygon_boundary_segments_2x(local_polygon_vertices_2x)
        }

    # Physical rect gives the hull boundary for overhang-heavy parts.
    if geometry.physical_rect is not None:
        local_body_cells = _flip_cells_within_bbox(
            _local_physical_rect_cells(geometry, resolved_rotation),
            local_rotation.width,
            local_rotation.height,
            flip_x=part.flip_x,
            flip_y=part.flip_y,
        )
        return {
            _translate_segment(local_segment, world_origin_2x)
            for local_segment in _cell_boundary_segments_from_local_cells_2x(local_body_cells)
        }

    # Fallback to tile footprint boundary for parts with no richer metadata.
    local_segments = _cell_boundary_segments_from_local_cells_2x(
        _flip_cells_within_bbox(
            set(local_rotation.footprint_tiles),
            local_rotation.width,
            local_rotation.height,
            flip_x=part.flip_x,
            flip_y=part.flip_y,
        )
    )
    return {_translate_segment(local_segment, world_origin_2x) for local_segment in local_segments}

