"""Shared rendering utilities for static ship visualization backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from common.cosmoteer import parse_ship_png
from graph_expansion.structural import enrich_graph
from preprocessing.graphs import process_ship
from preprocessing.relative_coords import apply_relative_coords_transform
from visualizer.icons import PartIconLibrary, _require_pillow

__all__ = [
    "BACKGROUND",
    "GRID_COLOR",
    "PADDING_2X",
    "SUBCELL_SIZE",
    "bounds_2x",
    "draw_grid",
    "draw_zone_legend",
    "flip_lookup",
    "font",
    "load_ship_for_visualization",
    "paste_tinted",
]

SUBCELL_SIZE = 24       # one 2x-cell unit == 24 px; one tile == 48 px
PADDING_2X = 4          # canvas padding in 2x-cell units on each side
BACKGROUND = (22, 25, 32, 255)
GRID_COLOR = (55, 63, 78, 255)


def font(size: int):
    """Return a PIL font at *size* points, with a safe fallback for older Pillow."""
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def flip_lookup(ship_data: dict[str, Any]) -> dict[tuple[int, int], tuple[bool, bool]]:
    """Build a ``location_2x -> (flip_x, flip_y)`` map from extracted part records."""
    result: dict[tuple[int, int], tuple[bool, bool]] = {}
    for raw_part in ship_data.get("Parts", []):
        if not isinstance(raw_part, dict):
            continue
        loc2x = raw_part.get("Location2x")
        if isinstance(loc2x, list) and len(loc2x) == 2:
            result[(int(loc2x[0]), int(loc2x[1]))] = (
                bool(raw_part.get("FlipX", False)),
                bool(raw_part.get("FlipY", False)),
            )
    return result


def bounds_2x(nodes: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    """Return ``(min_x, min_y, max_x, max_y)`` bounding box in 2x-cell units.

    The returned *max_x* / *max_y* are exclusive (one 2x-cell past the last
    occupied column/row) so callers can compute width/height directly.
    """
    min_x, min_y = 10**9, 10**9
    max_x, max_y = -(10**9), -(10**9)
    for node in nodes:
        location_2x = node.get("location_2x")
        footprint = node.get("footprint") or {}
        if not isinstance(location_2x, (list, tuple)) or len(location_2x) != 2:
            continue
        x2, y2 = int(location_2x[0]), int(location_2x[1])
        width = int(footprint.get("width", 1) or 1)
        height = int(footprint.get("height", 1) or 1)
        min_x = min(min_x, x2)
        min_y = min(min_y, y2)
        max_x = max(max_x, x2 + 2 * width)
        max_y = max(max_y, y2 + 2 * height)
    if min_x > max_x:
        return (-2, -2, 2, 2)
    return min_x, min_y, max_x, max_y


def draw_grid(
    draw,
    width_px: int,
    height_px: int,
    *,
    header_height: int,
    subcell_size: int = SUBCELL_SIZE,
) -> None:
    """Draw a faint grid of 2x-cell lines over the ship canvas."""
    for x in range(0, width_px + 1, subcell_size):
        draw.line((x, header_height, x, height_px), fill=GRID_COLOR, width=1)
    for y in range(header_height, height_px + 1, subcell_size):
        draw.line((0, y, width_px, y), fill=GRID_COLOR, width=1)


def paste_tinted(
    canvas,
    icon_library: PartIconLibrary,
    node: dict[str, Any],
    *,
    origin_x: int,
    origin_y: int,
    tint: tuple[int, int, int, int],
    flip_map: dict[tuple[int, int], tuple[bool, bool]],
    subcell_size: int = SUBCELL_SIZE,
) -> None:
    """Composite a tinted part icon onto *canvas* at its 2x-cell location."""
    Image, _ = _require_pillow()
    x2, y2 = map(int, node["location_2x"])
    flip_x, flip_y = flip_map.get((x2, y2), (False, False))
    icon = icon_library.get_icon(
        str(node["part_id"]),
        int(node.get("rotation", 0)),
        flip_x=flip_x,
        flip_y=flip_y,
        toggle_values=node.get("toggle_values"),
    )
    tint_overlay = Image.new("RGBA", icon.size, tint)
    icon = Image.blend(icon, tint_overlay, alpha=0.55)
    pixel_x = origin_x + x2 * subcell_size
    pixel_y = origin_y + y2 * subcell_size
    canvas.alpha_composite(icon, (pixel_x, pixel_y))


def draw_zone_legend(
    draw,
    width_px: int,
    zone_counts: dict[str, int],
    zone_names: list[str],
    color_fn,
) -> None:
    """Draw a horizontal swatch legend for zone backends."""
    swatch_size = 14
    x, y = 16, 86
    for zone_name in zone_names:
        count = zone_counts.get(zone_name, 0)
        if count == 0:
            continue
        color = color_fn(zone_name)
        draw.rectangle((x, y, x + swatch_size, y + swatch_size), fill=color)
        label = f"{zone_name} ({count})"
        draw.text((x + swatch_size + 4, y), label, fill=(210, 220, 240, 255), font=font(14))
        x += len(label) * 8 + swatch_size + 18
        if x > width_px - 120:
            x = 16
            y += 22


def load_ship_for_visualization(
    ship_png_path: Path,
    work_dir: Path,
) -> tuple[dict[str, Any], dict[tuple[int, int], tuple[bool, bool]]]:
    """Parse, preprocess, and fully expand one ship PNG for visualization.

    Runs the full structural graph expansion pipeline on *ship_png_path* and
    returns the enriched graph payload together with a flip map.

    Parameters
    ----------
    ship_png_path:
        Path to a ``.ship.png`` file.
    work_dir:
        Scratch directory for intermediate JSON files.  Created if absent.

    Returns
    -------
    expanded_data
        Fully enriched graph dict (all structural passes applied).
    flip_map
        ``location_2x -> (flip_x, flip_y)`` map built from the extracted payload.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    ship_data = apply_relative_coords_transform(parse_ship_png(ship_png_path))
    extracted_json_path = work_dir / (ship_png_path.stem + ".json")
    extracted_json_path.write_bytes(
        orjson.dumps(ship_data, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )
    _flip_map = flip_lookup(ship_data)
    graph_data = process_ship(extracted_json_path)
    expanded_data = enrich_graph(graph_data)
    return expanded_data, _flip_map
