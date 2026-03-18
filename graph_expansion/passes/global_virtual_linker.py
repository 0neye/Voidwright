"""Global virtual-node linker pass.

This pass connects the ``global_ship`` node to every other virtual node
emitted into the expansion graph by prior passes, using
``global_virtual_member`` cross-edges.

It must run after all virtual-node-emitting passes so that the expansion
graph's node list is complete when this pass executes.
"""

from __future__ import annotations

from typing import Any, List, Mapping, MutableMapping

from graph_expansion.context import EXPANSION_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base import ExpansionPass

__all__ = ["GlobalVirtualLinkerPass"]

_GLOBAL_SHIP_ID = "global_ship"


class GlobalVirtualLinkerPass(ExpansionPass):
    """Connect ``global_ship`` to every other virtual node in the expansion graph."""

    name = "global_virtual_linker"
    version = 1
    requires = ("global_ship_info",)

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Emit ``global_virtual_member`` edges from ``global_ship`` to all other virtual nodes."""

        expansion_graph = context.ensure_emitted_graph(EXPANSION_GRAPH_NAME)
        nodes: List[MutableMapping[str, Any]] = expansion_graph["nodes"]
        cross_edges: List[MutableMapping[str, Any]] = expansion_graph["cross_edges"]

        virtual_node_ids = [
            node["id"] for node in nodes if node["id"] != _GLOBAL_SHIP_ID
        ]

        new_edges = [
            {
                "source": _GLOBAL_SHIP_ID,
                "source_graph": EXPANSION_GRAPH_NAME,
                "target": node_id,
                "target_graph": EXPANSION_GRAPH_NAME,
                "kind": "global_virtual_member",
            }
            for node_id in virtual_node_ids
        ]

        cross_edges.extend(new_edges)

        edge_count = len(new_edges)
        context.increment_summary(
            EXPANSION_GRAPH_NAME,
            global_virtual_member_edges=edge_count,
        )

        return {"global_virtual_member_edges": edge_count}
