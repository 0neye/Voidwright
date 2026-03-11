"""Frame rendering for generation visualization videos."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from common.geometry import load_vanilla_part_geometry, resolve_geometry_part_id_and_rotation

from .events import VisualizationEvent, VisualizationPart
from .icons import PartIconLibrary

__all__ = ["render_visualization_frames"]


CELL_PADDING = 1
HEADER_HEIGHT = 260
SUMMARY_HOLD_FRAMES = 2

_FONT_SIZE_TITLE = 50
_FONT_SIZE_STATUS = 45


def _get_font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _require_pillow():
    from PIL import Image, ImageDraw

    return Image, ImageDraw


def _part_bounds(part: VisualizationPart, geometry_cache: dict) -> tuple[int, int, int, int]:
    resolved_part_id, resolved_rotation = resolve_geometry_part_id_and_rotation(part.part_id, part.rotation)
    geometry = geometry_cache.get(resolved_part_id)
    if geometry is None:
        # Unknown/non-vanilla parts can appear in rejected seed events
        # Keep them visible without depending on vanilla geometry metadata
        return (part.x, part.y, part.x, part.y)
    rotated_geometry = geometry.rotations.get(resolved_rotation) or next(iter(geometry.rotations.values()))
    return (
        part.x,
        part.y,
        part.x + rotated_geometry.width - 1,
        part.y + rotated_geometry.height - 1,
    )


def _collect_canvas_bounds(events: Iterable[VisualizationEvent]) -> tuple[int, int, int, int]:
    geometry_cache = load_vanilla_part_geometry()
    bounds: list[tuple[int, int, int, int]] = []
    for event in events:
        if event.part is None:
            continue
        bounds.append(_part_bounds(event.part, geometry_cache))
    if not bounds:
        return (-1, -1, 1, 1)
    min_x = min(bound[0] for bound in bounds)
    min_y = min(bound[1] for bound in bounds)
    max_x = max(bound[2] for bound in bounds)
    max_y = max(bound[3] for bound in bounds)
    return min_x, min_y, max_x, max_y


def _draw_grid(draw, width: int, height: int, *, cell_size: int, header_height: int) -> None:
    grid_color = (64, 73, 88, 255)
    for pixel_x in range(0, width + 1, cell_size):
        draw.line((pixel_x, header_height, pixel_x, height), fill=grid_color, width=1)
    for pixel_y in range(header_height, height + 1, cell_size):
        draw.line((0, pixel_y, width, pixel_y), fill=grid_color, width=1)


def _paste_part(
    canvas,
    icon_library: PartIconLibrary,
    part: VisualizationPart,
    *,
    origin_x: int,
    origin_y: int,
    tint: tuple[int, int, int, int] | None = None,
) -> None:
    Image, _ImageDraw = _require_pillow()
    icon = icon_library.get_icon(
        part.part_id,
        part.rotation,
        flip_x=part.flip_x,
        flip_y=part.flip_y,
    )
    if tint is not None:
        tint_overlay = Image.new("RGBA", icon.size, tint)
        icon = Image.blend(icon, tint_overlay, alpha=0.45)
    pixel_x = origin_x + part.x * icon_library.cell_size
    pixel_y = origin_y + part.y * icon_library.cell_size
    canvas.alpha_composite(icon, (pixel_x, pixel_y))


def _status_lines(event: VisualizationEvent, rejected_counts: Counter[str]) -> list[str]:
    if event.kind == "sample_started":
        return ["Generation started"]
    if event.kind == "part_placed" and event.part is not None:
        return [
            event.message or "Accepted placement",
            f"{event.part.part_id} @ ({event.part.x}, {event.part.y}) rot={event.part.rotation}",
        ]
    if event.kind == "attempt_rejected":
        reason = str(event.metadata.get("reason", "rejected"))
        if event.part is None:
            return [
                event.message or f"Rejected attempt: {reason}",
                f"Rejected counts so far: {dict(rejected_counts)}",
            ]
        return [
            event.message or f"Rejected attempt: {reason}",
            f"{event.part.part_id} @ ({event.part.x}, {event.part.y}) rot={event.part.rotation}",
        ]
    if event.kind == "sample_finished":
        stats = event.metadata.get("stats", {})
        return [
            event.message or "Generation finished",
            f"stop={event.metadata.get('stop_reason', 'unknown')} parts={stats.get('parts_generated', 0)} attempts={stats.get('attempts', 0)}",
        ]
    return [event.message or event.kind]


def _draw_header(draw, event: VisualizationEvent, lines: list[str], width: int) -> None:
    draw.rectangle((0, 0, width, HEADER_HEIGHT), fill=(19, 24, 33, 255))
    draw.text((16, 12), f"sample-{event.sample_index:03d} | {event.kind}", fill=(238, 243, 255, 255), font=_get_font(_FONT_SIZE_TITLE))
    line_y = 75
    for line in lines[:3]:
        draw.text((16, line_y), line, fill=(200, 212, 230, 255), font=_get_font(_FONT_SIZE_STATUS))
        line_y += 58


def render_visualization_frames(
    events: list[VisualizationEvent],
    *,
    icon_library: PartIconLibrary,
) -> list[object]:
    """Render one Pillow image per recorded event, plus a short final hold."""

    Image, ImageDraw = _require_pillow()

    min_x, min_y, max_x, max_y = _collect_canvas_bounds(events)
    grid_width = max_x - min_x + 1 + (CELL_PADDING * 2)
    grid_height = max_y - min_y + 1 + (CELL_PADDING * 2)
    image_width = grid_width * icon_library.cell_size
    image_height = HEADER_HEIGHT + (grid_height * icon_library.cell_size)
    origin_x = (CELL_PADDING - min_x) * icon_library.cell_size
    origin_y = HEADER_HEIGHT + ((CELL_PADDING - min_y) * icon_library.cell_size)

    accepted_parts: list[VisualizationPart] = []
    rejected_counts: Counter[str] = Counter()
    rendered_frames: list[object] = []

    for event in events:
        canvas = Image.new("RGBA", (image_width, image_height), (28, 33, 43, 255))
        draw = ImageDraw.Draw(canvas)
        _draw_grid(draw, image_width, image_height, cell_size=icon_library.cell_size, header_height=HEADER_HEIGHT)

        for accepted_part in accepted_parts:
            _paste_part(canvas, icon_library, accepted_part, origin_x=origin_x, origin_y=origin_y)

        if event.kind == "part_placed" and event.part is not None:
            accepted_parts.append(event.part)
            _paste_part(
                canvas,
                icon_library,
                event.part,
                origin_x=origin_x,
                origin_y=origin_y,
                tint=(70, 190, 110, 255),
            )
        elif event.kind == "attempt_rejected":
            reason = str(event.metadata.get("reason", "rejected"))
            rejected_counts[reason] += 1
            if event.part is not None:
                _paste_part(
                    canvas,
                    icon_library,
                    event.part,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    tint=(200, 85, 90, 255),
                )

        lines = _status_lines(event, rejected_counts)
        _draw_header(draw, event, lines, image_width)
        rendered_frames.append(canvas.convert("RGB"))

    if rendered_frames:
        for _ in range(SUMMARY_HOLD_FRAMES):
            rendered_frames.append(rendered_frames[-1].copy())

    return rendered_frames
