"""Base index construction pass for graph expansion.

This pass builds common lookup structures from the structural part
graph so later passes can reuse them without re-scanning the source
payload.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping

from graph_expansion.context import STRUCTURAL_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base import ExpansionPass

__all__ = ["BaseIndexesPass"]


class BaseIndexesPass(ExpansionPass):
    """Construct reusable indexes for the structural part graph."""

    name = "base_indexes"
    version = 1
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = (
        "node_by_id",
        "structural_nodes",
        "structural_edges",
        "walkable_part_ids",
        "door_edges",
        "touching_edges",
    )

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Build and cache structural graph indexes."""

        structural_graph = context.get_source_graph(STRUCTURAL_GRAPH_NAME)
        nodes = list(structural_graph.get("nodes", []))
        edges = list(structural_graph.get("edges", []))

        node_by_id: Dict[int, MutableMapping[str, Any]] = {}
        walkable_part_ids: set[int] = set()

        for node in nodes:
            node_id = node.get("id")
            if isinstance(node_id, int):
                node_by_id[node_id] = node
                if node.get("walkable_cells_2x"):
                    walkable_part_ids.add(node_id)

        door_edges = [edge for edge in edges if edge.get("kind") == "door"]
        touching_edges = [edge for edge in edges if edge.get("kind") == "touching"]

        context.caches["structural_nodes"] = nodes
        context.caches["structural_edges"] = edges
        context.caches["node_by_id"] = node_by_id
        context.caches["walkable_part_ids"] = walkable_part_ids
        context.caches["door_edges"] = door_edges
        context.caches["touching_edges"] = touching_edges

        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "walkable_parts": len(walkable_part_ids),
            "door_edges": len(door_edges),
        }

