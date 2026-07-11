"""Global ship-info node and virtual-node linker pass.

This pass emits the single global ship-info node (containing the top-level
``ship`` metadata) and then connects it to every other virtual node in the
expansion graph via ``global_virtual_member`` cross-edges.

It must run last so that all virtual nodes from prior passes are present when
it builds the linker edges.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping

from graph_expansion.context import EXPANSION_GRAPH_NAME, STRUCTURAL_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base import ExpansionPass

__all__ = ["GlobalVirtualLinkerPass", "GLOBAL_SHIP_NODE_ID"]

GLOBAL_SHIP_NODE_ID = "global_ship"


class GlobalVirtualLinkerPass(ExpansionPass):
    """Emit the global ship-info node and connect it to every other virtual node."""

    name = "global_virtual_linker"
    version = 4
    requires = ("base_indexes",)

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Emit the ``global_ship`` node and ``global_virtual_member`` edges."""

        ship_info = context.source.get("ship", {}) or {}

        # Compute ship-level structural summary for the global conditioning node.
        structural_nodes = context.caches.get("structural_nodes")
        if structural_nodes is None:
            structural_graph = context.get_source_graph(STRUCTURAL_GRAPH_NAME)
            structural_nodes = list(structural_graph.get("nodes", []))

        total_parts = len(structural_nodes)
        occupied_cells = sum(
            n.get("footprint", {}).get("cell_count", 0) for n in structural_nodes
        )
        xs = [n["location_2x"][0] for n in structural_nodes if n.get("location_2x")]
        ys = [n["location_2x"][1] for n in structural_nodes if n.get("location_2x")]
        footprint_w_2x = float(max(xs) - min(xs)) if len(xs) >= 2 else 0.0
        footprint_h_2x = float(max(ys) - min(ys)) if len(ys) >= 2 else 0.0

        expansion_graph = context.ensure_emitted_graph(EXPANSION_GRAPH_NAME)
        nodes: List[MutableMapping[str, Any]] = expansion_graph["nodes"]
        cross_edges: List[MutableMapping[str, Any]] = expansion_graph["cross_edges"]

        # Snapshot the virtual nodes emitted by prior passes before appending
        # the global node, so we can link to exactly those nodes without a
        # self-edge filter.
        prior_virtual_nodes = list(nodes)

        # Count virtual node kinds emitted by prior passes for subsystem summary.
        kind_counts: Dict[str, int] = {}
        for vn in prior_virtual_nodes:
            k = vn.get("kind", "")
            kind_counts[k] = kind_counts.get(k, 0) + 1

        global_node: Dict[str, Any] = {
            "id": GLOBAL_SHIP_NODE_ID,
            "kind": "global_ship_info",
            "ship": ship_info,
            "total_parts": total_parts,
            "occupied_cells": occupied_cells,
            "footprint_w_2x": footprint_w_2x,
            "footprint_h_2x": footprint_h_2x,
            "cluster_count": kind_counts.get("traversable_cluster", 0),
            "thermal_count": kind_counts.get("thermal_network", 0),
            "zone_count": kind_counts.get("spatial_zone", 0),
            "zone_rot_count": kind_counts.get("spatial_zone_rotated", 0),
            "weapon_grp_count": kind_counts.get("weapon_group", 0),
        }
        nodes.append(global_node)

        new_edges = []
        for node in prior_virtual_nodes:
            if node.get("member_count", 1) <= 0:
                continue
            virtual_id = node["id"]
            new_edges.append({
                "source": GLOBAL_SHIP_NODE_ID,
                "source_graph": EXPANSION_GRAPH_NAME,
                "target": virtual_id,
                "target_graph": EXPANSION_GRAPH_NAME,
                "kind": "global_virtual_member",
            })
            new_edges.append({
                "source": virtual_id,
                "source_graph": EXPANSION_GRAPH_NAME,
                "target": GLOBAL_SHIP_NODE_ID,
                "target_graph": EXPANSION_GRAPH_NAME,
                "kind": "global_virtual_member",
            })

        cross_edges.extend(new_edges)

        edge_count = len(new_edges)
        context.increment_summary(
            EXPANSION_GRAPH_NAME,
            global_ship_nodes=1,
            global_virtual_member_edges=edge_count,
        )

        return {"global_nodes": 1, "global_virtual_member_edges": edge_count}
