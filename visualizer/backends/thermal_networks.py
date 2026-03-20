"""Static visualization backend for thermal networks."""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
from typing import Any

from common.heat_exchanger import (
    HEAT_EXCHANGER_ABSORPTION_RADIUS_TILES,
    footprint_tile_origins_2x,
    heat_exchanger_radius_region_tile_origins_2x,
    is_heat_exchanger,
)
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
_ISOLATED_TINT = (128, 132, 140, 255)
_ISOLATED_EXCHANGER_STROKE = (150, 160, 176, 180)


def _network_color(network_index: int) -> tuple[int, int, int, int]:
    # Golden-ratio hue spread with softer saturation/value for readability.
    hue = (network_index * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.42, 0.86)
    return (int(r * 255), int(g * 255), int(b * 255), 255)


def _parse_network_index(node_id: object) -> int:
    """Extract the numeric suffix N from a virtual node ID like ``thermal_network_N``."""
    return int(str(node_id).split("_")[-1])


def _tile_boundary_segments_1x(
    tiles: set[tuple[int, int]],
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    """Return perimeter segments for a tile mask expressed as local 2x tile origins."""

    segments: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for tile_x, tile_y in tiles:
        tile_segments = (
            ((tile_x, tile_y), (tile_x + 2, tile_y)),
            ((tile_x + 2, tile_y), (tile_x + 2, tile_y + 2)),
            ((tile_x, tile_y + 2), (tile_x + 2, tile_y + 2)),
            ((tile_x, tile_y), (tile_x, tile_y + 2)),
        )
        for start, end in tile_segments:
            normalized = (start, end) if start <= end else (end, start)
            if normalized in segments:
                segments.remove(normalized)
            else:
                segments.add(normalized)
    return segments


def _draw_heat_exchanger_radius_overlays(
    draw,
    nodes: list[dict[str, Any]],
    network_by_part_id: dict[int, int],
    *,
    origin_x: int,
    origin_y: int,
) -> int:
    """Draw absorption-radius overlays for heat exchangers and return count."""

    overlays_drawn = 0

    for node in sorted(nodes, key=lambda n: int(n["id"])):
        if not is_heat_exchanger(str(node.get("part_id", ""))):
            continue

        exchanger_tiles = footprint_tile_origins_2x(node)
        if not exchanger_tiles:
            continue
        radius_tiles: set[tuple[int, int]] = set()
        for exchanger_tile in exchanger_tiles:
            radius_tiles.update(
                heat_exchanger_radius_region_tile_origins_2x(
                    exchanger_tile,
                    HEAT_EXCHANGER_ABSORPTION_RADIUS_TILES,
                )
            )

        network_index = network_by_part_id.get(int(node["id"]))
        if network_index is None:
            stroke = _ISOLATED_EXCHANGER_STROKE
        else:
            nr, ng, nb, _ = _network_color(network_index)
            stroke = (nr, ng, nb, 200)

        for (start_x, start_y), (end_x, end_y) in _tile_boundary_segments_1x(radius_tiles):
            draw.line(
                (
                    origin_x + start_x * SUBCELL_SIZE,
                    origin_y + start_y * SUBCELL_SIZE,
                    origin_x + end_x * SUBCELL_SIZE,
                    origin_y + end_y * SUBCELL_SIZE,
                ),
                fill=stroke,
                width=max(1, SUBCELL_SIZE // 3),
            )
        overlays_drawn += 1

    return overlays_drawn


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
        # Multi-network leaf members appear in multiple edges; resolve to the
        # largest network (by member_count), breaking ties by smallest index.
        expansion_graph = expanded_data["graphs"].get("X_expansion_structural", {})

        network_size: dict[int, int] = {}
        for vnode in expansion_graph.get("nodes", []):
            if vnode.get("kind") == "thermal_network":
                network_size[_parse_network_index(vnode["id"])] = int(vnode.get("member_count", 0))

        part_networks: dict[int, list[int]] = {}
        for edge in expansion_graph.get("cross_edges", []):
            if edge.get("kind") == "thermal_member":
                part_id = int(edge["target"])
                network_index = _parse_network_index(edge["source"])  # e.g. "thermal_network_3"
                part_networks.setdefault(part_id, []).append(network_index)

        network_by_part_id: dict[int, int] = {
            part_id: max(indices, key=lambda i: (network_size.get(i, 0), -i))
            for part_id, indices in part_networks.items()
        }

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
        heat_exchanger_count = _draw_heat_exchanger_radius_overlays(
            draw,
            nodes,
            network_by_part_id,
            origin_x=origin_x,
            origin_y=origin_y,
        )

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
            f"Heat exchanger absorption radius shown as 1x-space outlines ({HEAT_EXCHANGER_ABSORPTION_RADIUS_TILES:g}m), count={heat_exchanger_count}.",
            fill=(170, 180, 205, 255),
            font=font(18),
        )

        output_path = output_dir / f"{base_name}-thermal-networks.png"
        canvas.convert("RGB").save(output_path)
        return output_path
