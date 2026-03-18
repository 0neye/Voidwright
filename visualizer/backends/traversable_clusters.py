"""Static visualization backend for traversable clusters."""

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

__all__ = ["TraversableClustersBackend"]

_HEADER_HEIGHT = 140
_NON_CLUSTER_TINT = (90, 90, 90, 255)


def _cluster_color(cluster_index: int) -> tuple[int, int, int, int]:
    hue = (cluster_index * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255), 255)


class TraversableClustersBackend(StaticVisualizationBackend):
    """Render parts tinted by their traversable cluster membership.

    Parts not belonging to any cluster are rendered in gray.
    """

    name = "traversable-clusters"
    default_output_dir = "out/visualizations/traversable-clusters"

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

        # Build cluster_by_part_id from cross_edges with kind=="super_member".
        # source is "traversable_cluster_N" (N is int), target is part_id (int).
        expansion_graph = expanded_data["graphs"].get("X_expansion_structural", {})
        cluster_by_part_id: dict[int, int] = {}
        for edge in expansion_graph.get("cross_edges", []):
            if edge.get("kind") == "super_member":
                part_id = int(edge["target"])
                source = str(edge["source"])  # e.g. "traversable_cluster_3"
                cluster_index = int(source.split("_")[-1])
                cluster_by_part_id[part_id] = cluster_index

        cluster_count = len(set(cluster_by_part_id.values()))

        min_x2, min_y2, max_x2, max_y2 = bounds_2x(nodes)
        width_px = (max_x2 - min_x2 + 2 * PADDING_2X) * SUBCELL_SIZE
        height_px = _HEADER_HEIGHT + (max_y2 - min_y2 + 2 * PADDING_2X) * SUBCELL_SIZE
        origin_x = (PADDING_2X - min_x2) * SUBCELL_SIZE
        origin_y = _HEADER_HEIGHT + (PADDING_2X - min_y2) * SUBCELL_SIZE

        canvas = Image.new("RGBA", (width_px, height_px), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        draw_grid(draw, width_px, height_px, header_height=_HEADER_HEIGHT)

        clustered_nodes = 0
        for node in sorted(nodes, key=lambda n: int(n["id"])):
            cluster_index = cluster_by_part_id.get(int(node["id"]))
            if cluster_index is None:
                tint = _NON_CLUSTER_TINT
            else:
                tint = _cluster_color(cluster_index)
                clustered_nodes += 1
            paste_tinted(
                canvas, icon_library, node,
                origin_x=origin_x, origin_y=origin_y,
                tint=tint, flip_map=flip_map,
            )

        draw.rectangle((0, 0, width_px, _HEADER_HEIGHT), fill=(16, 18, 24, 255))
        base_name = ship_name.removesuffix(".ship.png").removesuffix(".json")
        title = f"{base_name} — traversable clusters"
        subtitle = (
            f"clusters={cluster_count}  clustered_parts={clustered_nodes}/{len(nodes)}  "
            f"gray=not in any traversable cluster"
        )
        draw.text((16, 12), title, fill=(240, 244, 255, 255), font=font(28))
        draw.text((16, 54), subtitle, fill=(192, 205, 230, 255), font=font(20))
        draw.text(
            (16, 90),
            "Blueprint sprites from game Data/ships/terran/*/blueprints.png; colors indicate TraversableClustersPass output.",
            fill=(170, 180, 205, 255),
            font=font(18),
        )

        output_path = output_dir / f"{base_name}-traversable-clusters.png"
        canvas.convert("RGB").save(output_path)
        return output_path
