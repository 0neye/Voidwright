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

from graph_expansion.context import EXPANSION_GRAPH_NAME, STRUCTURAL_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base import ExpansionPass

__all__ = ["SpatialZonesPass", "SpatialZonesRotatedPass", "ZONE_NAMES", "ZONE_NAMES_ROTATED"]


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

# Interstitial zones rotated 22.5° from ZONE_NAMES so that the dividing lines
# fall on the cardinal and semi-cardinal directions (0°, 45°, 90°, …) instead
# of between them.  Each name follows the standard 16-point compass convention.
ZONE_NAMES_ROTATED: List[str] = [
    "zone_ene",
    "zone_nne",
    "zone_nnw",
    "zone_wnw",
    "zone_wsw",
    "zone_ssw",
    "zone_sse",
    "zone_ese",
]


def _compute_zones_impl(node: Mapping[str, Any], zone_names: List[str], angle_offset: float) -> List[str]:
    """Map a structural node to one or more zone labels from *zone_names*.

    Every 2x cell in the part's footprint is checked independently.  If cells
    fall into more than one of the eight equal sectors the node is assigned to
    all of them, so parts that straddle a zone dividing line receive membership
    edges in every zone they touch.  The returned list is ordered by ascending
    sector index (i.e. the canonical order of *zone_names*).

    *angle_offset* (radians) is subtracted from each cell's angle before
    quantising, allowing callers to rotate the zone boundaries without
    duplicating the coordinate arithmetic.
    """

    node_id = node.get("id")
    if not isinstance(node_id, int):
        raise ValueError("structural node is missing an integer 'id'")

    location_2x = node.get("location_2x")
    if not isinstance(location_2x, Sequence) or len(location_2x) != 2:
        return [zone_names[0]]
    lx, ly = int(location_2x[0]), int(location_2x[1])

    footprint = node.get("footprint")
    if not isinstance(footprint, Mapping):
        width, height = 1, 1
    else:
        width = int(footprint.get("width", 1)) or 1
        height = int(footprint.get("height", 1)) or 1

    # footprint.width/height are stored in rotation-specific form by
    # preprocessing (infer_meta returns rotation-specific RotationGeometry
    # dimensions), so no rotation-based swap is needed here.

    # --- Fast path: 1×1 part ---
    if width == 1 and height == 1:
        angle = math.atan2(ly, lx) - angle_offset
        return [zone_names[int(round(angle * 4 / math.pi)) % 8]]

    rx, ry = lx + 2 * (width - 1), ly + 2 * (height - 1)  # far corner (inclusive, 2x stride)

    # --- Corner-check shortcut (origin outside the footprint) ---
    # Zone boundaries are rays that emanate from the origin.  When the origin
    # lies outside the footprint such a ray can only split the rectangle into
    # two non-empty parts by crossing two distinct edges; any edge crossing
    # places the corners on either side of the ray in *different* zones.
    # Therefore: if all four corners share the same zone the entire footprint
    # is in that zone and we can skip the full cell scan.
    #
    # When the origin is inside the footprint we cannot use this shortcut
    # because all zone boundaries radiate from it, potentially touching every
    # cell.
    origin_inside = (lx <= 0 <= rx) and (ly <= 0 <= ry)
    if not origin_inside:
        corners = ((lx, ly), (rx, ly), (lx, ry), (rx, ry))
        corner_sectors = {
            int(round((math.atan2(cy, cx) - angle_offset) * 4 / math.pi)) % 8
            for cx, cy in corners
        }
        if len(corner_sectors) == 1:
            return [zone_names[next(iter(corner_sectors))]]

    # --- Full cell scan (straddling or origin-inside) ---
    # Each tile occupies 2 units in the 2x frame, so step by 2.
    sector_indices: set[int] = set()
    for cy in range(ly, ly + 2 * height, 2):
        for cx in range(lx, lx + 2 * width, 2):
            angle = math.atan2(cy, cx) - angle_offset
            sector_indices.add(int(round(angle * 4 / math.pi)) % 8)
        if len(sector_indices) == 8:
            return list(zone_names)

    return [zone_names[i] for i in sorted(sector_indices)]


def _compute_zones_for_node(node: Mapping[str, Any]) -> List[str]:
    """Compute the spatial zone labels for a structural node (cardinal axes)."""
    return _compute_zones_impl(node, ZONE_NAMES, 0.0)


def _compute_rotated_zones_for_node(node: Mapping[str, Any]) -> List[str]:
    """Compute the rotated spatial zone labels (dividing lines on cardinal axes)."""
    return _compute_zones_impl(node, ZONE_NAMES_ROTATED, math.pi / 8)


class SpatialZonesPass(ExpansionPass):
    """Emit spatial zone virtual nodes and membership cross-edges."""

    name = "spatial_zones"
    version = 2
    requires = ("base_indexes",)
    provides = ("zone_by_part_id",)

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Compute spatial zones and emit virtual nodes and edges.

        The pass:

        - Assigns each structural part node to one or more zones based on
          which zones its footprint cells touch (parts that straddle a zone
          dividing line are assigned to all zones they overlap)
        - Stores the per-part zone list in the context annotations
        - Emits one virtual node per non-empty zone into the structural
          expansion graph
        - Emits ``zone_member`` cross-edges from each zone node to its
          member structural nodes
        """

        # Reuse structural node cache from the base index pass when
        # available, falling back to a direct graph read otherwise.
        structural_nodes = context.caches.get("structural_nodes")
        if structural_nodes is None:
            structural_graph = context.get_source_graph(STRUCTURAL_GRAPH_NAME)
            structural_nodes = list(structural_graph.get("nodes", []))

        # Build mapping from part id to zone label list and from zone label
        # to member part ids.
        zone_by_part_id: Dict[int, List[str]] = {}
        members_by_zone: Dict[str, List[int]] = {zone_name: [] for zone_name in ZONE_NAMES}

        for node in structural_nodes:
            node_id = node.get("id")
            if not isinstance(node_id, int):
                continue

            # Compute the zone labels for this node (may be more than one for
            # parts whose footprint straddles a zone boundary).
            zone_names_for_node = _compute_zones_for_node(node)
            zone_by_part_id[node_id] = zone_names_for_node
            for zone_name in zone_names_for_node:
                members_by_zone[zone_name].append(node_id)

        context.set_annotation("zone_by_part_id", zone_by_part_id)

        # Materialize zone virtual nodes and cross-edges only for zones
        # that have at least one structural member. Zones are emitted in
        # deterministic order.
        expansion_graph = context.ensure_emitted_graph(EXPANSION_GRAPH_NAME)
        nodes: List[MutableMapping[str, Any]] = expansion_graph["nodes"]
        cross_edges: List[MutableMapping[str, Any]] = expansion_graph["cross_edges"]

        zone_nodes: List[Dict[str, Any]] = []
        zone_member_edges: List[Dict[str, Any]] = []

        for zone_name in ZONE_NAMES:
            member_ids = members_by_zone[zone_name]
            if not member_ids:
                continue

            # Emit one virtual node per populated zone
            zone_nodes.append(
                {
                    "id": zone_name,
                    "kind": "spatial_zone",
                    "zone_label": zone_name,
                    "member_count": len(member_ids),
                }
            )

            # Emit cross-edges from the zone node to each member part
            for member_id in member_ids:
                zone_member_edges.append(
                    {
                        "source": zone_name,
                        "source_graph": EXPANSION_GRAPH_NAME,
                        "target": member_id,
                        "target_graph": STRUCTURAL_GRAPH_NAME,
                        "kind": "zone_member",
                    }
                )

        nodes.extend(zone_nodes)
        cross_edges.extend(zone_member_edges)

        context.increment_summary(
            EXPANSION_GRAPH_NAME,
            spatial_zone_nodes=len(zone_nodes),
            zone_member_edges=len(zone_member_edges),
        )

        return {
            "spatial_zone_nodes": len(zone_nodes),
            "zone_member_edges": len(zone_member_edges),
        }


class SpatialZonesRotatedPass(ExpansionPass):
    """Emit rotated spatial zone virtual nodes and membership cross-edges.

    Identical to :class:`SpatialZonesPass` but with the sector dividing lines
    rotated 22.5° so that they fall on the cardinal and semi-cardinal compass
    directions (0°, 45°, 90°, …) rather than between them.  The resulting zone
    centres are the interstitial 16-point compass directions (ENE, NNE, NNW, …).

    Zone node IDs use the ``zone_ene`` / ``zone_nne`` / … naming convention.
    Cross-edges carry ``kind = "zone_member_rotated"`` to distinguish them from
    the unrotated set.
    """

    name = "spatial_zones_rotated"
    version = 2
    requires = ("base_indexes",)
    provides = ("rotated_zone_by_part_id",)

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Compute rotated spatial zones and emit virtual nodes and edges."""

        structural_nodes = context.caches.get("structural_nodes")
        if structural_nodes is None:
            structural_graph = context.get_source_graph(STRUCTURAL_GRAPH_NAME)
            structural_nodes = list(structural_graph.get("nodes", []))

        rotated_zone_by_part_id: Dict[int, List[str]] = {}
        members_by_zone: Dict[str, List[int]] = {zone_name: [] for zone_name in ZONE_NAMES_ROTATED}

        for node in structural_nodes:
            node_id = node.get("id")
            if not isinstance(node_id, int):
                continue

            zone_names_for_node = _compute_rotated_zones_for_node(node)
            rotated_zone_by_part_id[node_id] = zone_names_for_node
            for zone_name in zone_names_for_node:
                members_by_zone[zone_name].append(node_id)

        context.set_annotation("rotated_zone_by_part_id", rotated_zone_by_part_id)

        expansion_graph = context.ensure_emitted_graph(EXPANSION_GRAPH_NAME)
        nodes: List[MutableMapping[str, Any]] = expansion_graph["nodes"]
        cross_edges: List[MutableMapping[str, Any]] = expansion_graph["cross_edges"]

        zone_nodes: List[Dict[str, Any]] = []
        zone_member_edges: List[Dict[str, Any]] = []

        for zone_name in ZONE_NAMES_ROTATED:
            member_ids = members_by_zone[zone_name]
            if not member_ids:
                continue

            zone_nodes.append(
                {
                    "id": zone_name,
                    "kind": "spatial_zone_rotated",
                    "zone_label": zone_name,
                    "member_count": len(member_ids),
                }
            )

            for member_id in member_ids:
                zone_member_edges.append(
                    {
                        "source": zone_name,
                        "source_graph": EXPANSION_GRAPH_NAME,
                        "target": member_id,
                        "target_graph": STRUCTURAL_GRAPH_NAME,
                        "kind": "zone_member_rotated",
                    }
                )

        nodes.extend(zone_nodes)
        cross_edges.extend(zone_member_edges)

        context.increment_summary(
            EXPANSION_GRAPH_NAME,
            spatial_zone_rotated_nodes=len(zone_nodes),
            zone_member_rotated_edges=len(zone_member_edges),
        )

        return {
            "spatial_zone_rotated_nodes": len(zone_nodes),
            "zone_member_rotated_edges": len(zone_member_edges),
        }
