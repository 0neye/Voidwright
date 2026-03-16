"""Global ship-info virtual node pass.

This pass emits a single global ship-info node containing the top-level
``ship`` metadata and connects it to each structural part node via
``global_member`` cross-edges.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping

from graph_expansion.context import ExpansionContext
from graph_expansion.passes.base import ExpansionPass

__all__ = ["GlobalShipInfoPass"]

_EXPANSION_GRAPH_NAME = "X_expansion_structural"


class GlobalShipInfoPass(ExpansionPass):
    """Emit the global ship-info node and membership cross-edges."""

    name = "global_ship_info"
    version = 1
    requires = ("base_indexes",)
    provides: tuple[str, ...] = ()

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Emit the global ship-info node and cross-edges."""

        ship_info = context.source.get("ship", {}) or {}

        structural_graph = context.get_source_graph("A_structural_part_graph")
        structural_nodes = list(structural_graph.get("nodes", []))

        expansion_graph = context.ensure_emitted_graph(_EXPANSION_GRAPH_NAME)
        nodes: List[MutableMapping[str, Any]] = expansion_graph["nodes"]
        cross_edges: List[MutableMapping[str, Any]] = expansion_graph["cross_edges"]

        global_node: Dict[str, Any] = {
            "id": "global_ship",
            "kind": "global_ship_info",
            "ship": ship_info,
        }
        global_edges: List[Dict[str, Any]] = [
            {
                "source": "global_ship",
                "source_graph": _EXPANSION_GRAPH_NAME,
                "target": node["id"],
                "target_graph": "A_structural_part_graph",
                "kind": "global_member",
            }
            for node in structural_nodes
        ]

        nodes.insert(0, global_node)
        cross_edges[0:0] = global_edges

        summary = expansion_graph.setdefault("summary", {})
        summary.setdefault("global_ship_nodes", 0)
        summary.setdefault("global_member_edges", 0)
        summary["global_ship_nodes"] += 1
        summary["global_member_edges"] += len(global_edges)

        return {
            "global_nodes": 1,
            "global_member_edges": len(global_edges),
        }

