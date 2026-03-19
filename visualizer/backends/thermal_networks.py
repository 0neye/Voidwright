"""Static visualization backend for thermal networks."""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
from typing import Any

from visualizer.backends.base import StaticVisualizationBackend
from visualizer.icons import PartIconLibrary, _require_pillow
from visualizer.static_render import (
    BACKGROUND,
    PADDING_2X,
    SUBCELL_SIZE,
    bounds_2x,
    draw_grid,
    font,
    paste_tinted,
)

__all__ = ["ThermalNetworksBackend"]

_HEADER_HEIGHT = 140
_ISOLATED_TINT = (90, 90, 90, 255)


def _network_color(network_index: int) -> tuple[int, int, int, int]:
    # Golden-ratio hue spread, warm saturation to evoke heat
    hue = (network_index * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255), 255)


class ThermalNetworksBackend(StaticVisualizationBackend):
    """Render parts tinted by their thermal network membership.

    Parts sharing a thermal connection chain are the same color.
    Parts with no thermal connections are rendered in gray.
    """

    name = "thermal-networks"
    default_output_dir = "out/visualizations/thermal-networks"

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

        # Build network_by_part_id from cross_edges with kind=="thermal_member".
        # source is "thermal_network_N" (N is int), target is part_id (int).
        expansion_graph = expanded_data["graphs"].get("X_expansion_structural", {})
        network_by_part_id: dict[int, int] = {}
        for edge in expansion_graph.get("cross_edges", []):
            if edge.get("kind") == "thermal_member":
                part_id = int(edge["target"])
                source = str(edge["source"])  # e.g. "thermal_network_3"
                network_index = int(source.split("_")[-1])
                network_by_part_id[part_id] = network_index

        network_count = len(set(network_by_part_id.values()))
        connected_count = len(network_by_part_id)

        min_x2, min_y2, max_x2, max_y2 = bounds_2x(nodes)
        width_px = (max_x2 - min_x2 + 2 * PADDING_2X) * SUBCELL_SIZE
        height_px = _HEADER_HEIGHT + (max_y2 - min_y2 + 2 * PADDING_2X) * SUBCELL_SIZE
        origin_x = (PADDING_2X - min_x2) * SUBCELL_SIZE
        origin_y = _HEADER_HEIGHT + (PADDING_2X - min_y2) * SUBCELL_SIZE

        canvas = Image.new("RGBA", (width_px, height_px), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        draw_grid(draw, width_px, height_px, header_height=_HEADER_HEIGHT)

        for node in sorted(nodes, key=lambda n: int(n["id"])):
            network_index = network_by_part_id.get(int(node["id"]))
            tint = _ISOLATED_TINT if network_index is None else _network_color(network_index)
            paste_tinted(
                canvas, icon_library, node,
                origin_x=origin_x, origin_y=origin_y,
                tint=tint, flip_map=flip_map,
            )

        draw.rectangle((0, 0, width_px, _HEADER_HEIGHT), fill=(16, 18, 24, 255))
        base_name = ship_name.removesuffix(".ship.png").removesuffix(".json")
        title = f"{base_name} — thermal networks"
        subtitle = (
            f"networks={network_count}  connected_parts={connected_count}/{len(nodes)}  "
            f"gray=thermally isolated"
        )
        draw.text((16, 12), title, fill=(240, 244, 255, 255), font=font(28))
        draw.text((16, 54), subtitle, fill=(192, 205, 230, 255), font=font(20))
        draw.text(
            (16, 90),
            "Blueprint sprites from game Data/ships/terran/*/blueprints.png; colors indicate ThermalNetworksPass output.",
            fill=(170, 180, 205, 255),
            font=font(18),
        )

        output_path = output_dir / f"{base_name}-thermal-networks.png"
        canvas.convert("RGB").save(output_path)
        return output_path
