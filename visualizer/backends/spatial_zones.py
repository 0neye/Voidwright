"""Static visualization backend for cardinal spatial zones."""

from __future__ import annotations

import argparse
import colorsys
import math
from pathlib import Path
from typing import Any

from graph_expansion.passes.spatial_zones import ZONE_NAMES
from visualizer.backends.base import StaticVisualizationBackend
from visualizer.icons import PartIconLibrary, _require_pillow
from visualizer.static_render import (
    BACKGROUND,
    PADDING_2X,
    SUBCELL_SIZE,
    bounds_2x,
    draw_grid,
    draw_zone_legend,
    font,
    paste_tinted,
)

__all__ = ["SpatialZonesBackend"]

_HEADER_HEIGHT = 150


def _zone_color(zone_name: str) -> tuple[int, int, int, int]:
    idx = ZONE_NAMES.index(zone_name) if zone_name in ZONE_NAMES else 0
    hue = idx / len(ZONE_NAMES)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255), 255)


def _draw_zone_wedges(draw, origin_x: int, origin_y: int, width_px: int, height_px: int) -> None:
    """Draw faint radial lines at cardinal zone boundaries (offset 22.5° from axes)."""
    color = (80, 90, 110, 200)
    radius = max(width_px, height_px) * 2
    for i in range(8):
        angle = math.radians(-22.5 + i * 45.0)
        ex = origin_x + int(radius * math.cos(angle))
        ey = origin_y + int(radius * math.sin(angle))
        draw.line((origin_x, origin_y, ex, ey), fill=color, width=1)


class SpatialZonesBackend(StaticVisualizationBackend):
    """Render parts tinted by their cardinal spatial zone assignment."""

    name = "spatial-zones"
    default_output_dir = "out/visualizations/spatial-zones"

    def register_parser(self, parser: argparse.ArgumentParser) -> None:
        pass  # no backend-specific arguments

    def render_ship(
        self,
        ship_name: str,
        expanded_data: dict[str, Any],
        flip_map: dict[tuple[int, int], tuple[bool, bool]],
        output_dir: Path,
        icon_library: PartIconLibrary,
        args: argparse.Namespace,
    ) -> Path:
        Image, ImageDraw = _require_pillow()

        nodes = expanded_data["graphs"]["A_structural_part_graph"]["nodes"]

        # Build zones_by_part_id from cross_edges with kind=="zone_member".
        # source is zone name (str like "zone_e"), target is part_id (int).
        # Parts straddling a boundary receive multiple edges; collect all.
        expansion_graph = expanded_data["graphs"].get("X_expansion_structural", {})
        zones_by_part_id: dict[int, list[str]] = {}
        for edge in expansion_graph.get("cross_edges", []):
            if edge.get("kind") == "zone_member":
                part_id = int(edge["target"])
                zone_name = str(edge["source"])
                zones_by_part_id.setdefault(part_id, []).append(zone_name)

        min_x2, min_y2, max_x2, max_y2 = bounds_2x(nodes)
        width_px = (max_x2 - min_x2 + 2 * PADDING_2X) * SUBCELL_SIZE
        height_px = _HEADER_HEIGHT + (max_y2 - min_y2 + 2 * PADDING_2X) * SUBCELL_SIZE
        origin_x = (PADDING_2X - min_x2) * SUBCELL_SIZE
        origin_y = _HEADER_HEIGHT + (PADDING_2X - min_y2) * SUBCELL_SIZE

        canvas = Image.new("RGBA", (width_px, height_px), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        draw_grid(draw, width_px, height_px, header_height=_HEADER_HEIGHT)
        _draw_zone_wedges(draw, origin_x, origin_y, width_px, height_px)

        fallback_zone = ZONE_NAMES[0]
        zone_colors = {z: _zone_color(z) for z in ZONE_NAMES}
        zone_counts: dict[str, int] = {z: 0 for z in ZONE_NAMES}
        for node in sorted(nodes, key=lambda n: int(n["id"])):
            node_id = int(node["id"])
            zones = zones_by_part_id.get(node_id) or [fallback_zone]
            for z in zones:
                zone_counts[z] += 1
            colors = [zone_colors[z] for z in zones]
            n = len(colors)
            tint = (
                sum(c[0] for c in colors) // n,
                sum(c[1] for c in colors) // n,
                sum(c[2] for c in colors) // n,
                sum(c[3] for c in colors) // n,
            )
            paste_tinted(
                canvas, icon_library, node,
                origin_x=origin_x, origin_y=origin_y,
                tint=tint, flip_map=flip_map,
            )

        draw.rectangle((0, 0, width_px, _HEADER_HEIGHT), fill=(16, 18, 24, 255))
        base_name = ship_name.removesuffix(".ship.png").removesuffix(".json")
        title = f"{base_name} — spatial zones (cardinal)"
        subtitle = (
            f"parts={len(nodes)}  zones_populated={sum(1 for c in zone_counts.values() if c > 0)}/8  "
            f"boundaries at 22.5° offset from axes"
        )
        draw.text((16, 12), title, fill=(240, 244, 255, 255), font=font(28))
        draw.text((16, 54), subtitle, fill=(192, 205, 230, 255), font=font(18))
        draw_zone_legend(draw, width_px, zone_counts, ZONE_NAMES, _zone_color)

        output_path = output_dir / f"{base_name}-spatial-zones.png"
        canvas.convert("RGB").save(output_path)
        return output_path
