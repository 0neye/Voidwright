"""Global ship-info virtual node pass.

This pass emits a single global ship-info node containing the top-level
``ship`` metadata and connects it to each structural part node via
``global_member`` cross-edges.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping

from graph_expansion.context import EXPANSION_GRAPH_NAME, STRUCTURAL_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base import ExpansionPass

__all__ = ["GlobalShipInfoPass", "GLOBAL_SHIP_NODE_ID"]

GLOBAL_SHIP_NODE_ID = "global_ship"


class GlobalShipInfoPass(ExpansionPass):
    """Emit the global ship-info node and membership cross-edges."""

    name = "global_ship_info"
    version = 1
    requires = ("base_indexes",)

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Emit the global ship-info node and cross-edges."""

        ship_info = context.source.get("ship", {}) or {}

        structural_nodes = context.caches.get("structural_nodes")
        if structural_nodes is None:
            structural_graph = context.get_source_graph(STRUCTURAL_GRAPH_NAME)
            structural_nodes = list(structural_graph.get("nodes", []))

        expansion_graph = context.ensure_emitted_graph(EXPANSION_GRAPH_NAME)
        nodes: List[MutableMapping[str, Any]] = expansion_graph["nodes"]
        cross_edges: List[MutableMapping[str, Any]] = expansion_graph["cross_edges"]

        global_node: Dict[str, Any] = {
            "id": GLOBAL_SHIP_NODE_ID,
            "kind": "global_ship_info",
            "ship": ship_info,
        }
        global_edges: List[Dict[str, Any]] = [
            {
                "source": GLOBAL_SHIP_NODE_ID,
                "source_graph": EXPANSION_GRAPH_NAME,
                "target": node["id"],
                "target_graph": STRUCTURAL_GRAPH_NAME,
                "kind": "global_member",
            }
            for node in structural_nodes
        ]

        nodes.append(global_node)
        cross_edges.extend(global_edges)

        context.increment_summary(
            EXPANSION_GRAPH_NAME,
            global_ship_nodes=1,
            global_member_edges=len(global_edges),
        )

        return {
            "global_nodes": 1,
            "global_member_edges": len(global_edges),
        }
