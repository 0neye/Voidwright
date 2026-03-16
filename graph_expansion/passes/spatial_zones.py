"""Spatial zone computation and emission pass.

This pass assigns each structural part node to one of eight directional
zones based on the angle of its footprint centroid in the 2x coordinate
frame. The resulting virtual zone nodes and membership cross-edges let
downstream models learn mirror symmetry and other spatial patterns as
co-occurrence structure across zone pairs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

from graph_expansion.context import ExpansionContext
from graph_expansion.passes.base import ExpansionPass

__all__ = ["SpatialZonesPass", "ZONE_NAMES"]

ZONE_NAMES: List[str] = [
    "zone_e",
    "zone_ne",
    "zone_n",
    "zone_nw",
    "zone_w",
    "zone_sw",
    "zone_s",
    "zone_se",
]

_EXPANSION_GRAPH_NAME = "X_expansion_structural"


def _compute_zone_for_node(node: Mapping[str, Any]) -> str:
    """Compute the spatial zone label for a structural node.

    The zone is derived from the angle of the part's footprint centroid in
    the 2x coordinate system relative to the origin at (0, 0).
    """

    # Extract required structural metadata from the node
    node_id = node.get("id")
    if not isinstance(node_id, int):
        raise ValueError("structural node is missing an integer 'id'")

    location_2x = node.get("location_2x")
    if not isinstance(location_2x, Sequence) or len(location_2x) != 2:
        return ZONE_NAMES[0]
    lx, ly = int(location_2x[0]), int(location_2x[1])

    footprint = node.get("footprint")
    if not isinstance(footprint, Mapping):
        width, height = 1, 1
    else:
        width = int(footprint.get("width", 1)) or 1
        height = int(footprint.get("height", 1)) or 1

    rotation = int(node.get("rotation", 0)) % 4

    # Rotation-adjusted effective dimensions
    if rotation % 2 == 0:
        effective_width = width
        effective_height = height
    else:
        effective_width = height
        effective_height = width

    # Centroid in 2x coordinates. Coordinates are already relative to the
    # origin at (0, 0) in the structural graph frame.
    centroid_x = lx + (effective_width - 1)
    centroid_y = ly + (effective_height - 1)

    # Compute angle and map it into one of eight equal sectors.
    dx = centroid_x
    dy = centroid_y
    angle = math.atan2(dy, dx)
    sector_idx = int(round(angle * 4 / math.pi)) % 8
    return ZONE_NAMES[sector_idx]


class SpatialZonesPass(ExpansionPass):
    """Emit spatial zone virtual nodes and membership cross-edges."""

    name = "spatial_zones"
    version = 1
    requires = ("base_indexes",)
    provides = ("zone_by_part_id",)

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Compute spatial zones and emit virtual nodes and edges.

        The pass:

        - Assigns each structural part node to exactly one zone based on
          its centroid angle
        - Stores the per-part zone mapping in the context annotations
        - Emits one virtual node per non-empty zone into the structural
          expansion graph
        - Emits ``zone_member`` cross-edges from each zone node to its
          member structural nodes
        """

        # Reuse structural node cache from the base index pass when
        # available, falling back to a direct graph read otherwise.
        structural_nodes = context.caches.get("structural_nodes")
        if structural_nodes is None:
            structural_graph = context.get_source_graph("A_structural_part_graph")
            structural_nodes = list(structural_graph.get("nodes", []))

        # Build mapping from part id to zone label and from zone label
        # to member part ids.
        zone_by_part_id: Dict[int, str] = {}
        members_by_zone: Dict[str, List[int]] = {zone_name: [] for zone_name in ZONE_NAMES}

        for node in structural_nodes:
            node_id = node.get("id")
            if not isinstance(node_id, int):
                continue

            # Compute the zone label for this node
            zone_name = _compute_zone_for_node(node)
            zone_by_part_id[node_id] = zone_name
            members_by_zone[zone_name].append(node_id)

        context.set_annotation("zone_by_part_id", zone_by_part_id)

        # Materialize zone virtual nodes and cross-edges only for zones
        # that have at least one structural member. Zones are emitted in
        # deterministic order.
        expansion_graph = context.ensure_emitted_graph(_EXPANSION_GRAPH_NAME)
        nodes: List[MutableMapping[str, Any]] = expansion_graph["nodes"]
        cross_edges: List[MutableMapping[str, Any]] = expansion_graph["cross_edges"]

        zone_nodes: List[Dict[str, Any]] = []
        zone_member_edges: List[Dict[str, Any]] = []

        for zone_name in ZONE_NAMES:
            member_ids = members_by_zone.get(zone_name)
            if not member_ids:
                continue

            # Emit one virtual node per populated zone
            zone_nodes.append(
                {
                    "id": zone_name,
                    "kind": "spatial_zone",
                    "zone_label": zone_name,
                }
            )

            # Emit cross-edges from the zone node to each member part
            for member_id in member_ids:
                zone_member_edges.append(
                    {
                        "source": zone_name,
                        "source_graph": _EXPANSION_GRAPH_NAME,
                        "target": member_id,
                        "target_graph": "A_structural_part_graph",
                        "kind": "zone_member",
                    }
                )

        nodes.extend(zone_nodes)
        cross_edges.extend(zone_member_edges)

        # Update the expansion-graph summary with spatial zone counts.
        summary = expansion_graph.setdefault("summary", {})
        summary.setdefault("spatial_zone_nodes", 0)
        summary.setdefault("zone_member_edges", 0)
        summary["spatial_zone_nodes"] += len(zone_nodes)
        summary["zone_member_edges"] += len(zone_member_edges)

        return {
            "spatial_zone_nodes": len(zone_nodes),
            "zone_member_edges": len(zone_member_edges),
        }

