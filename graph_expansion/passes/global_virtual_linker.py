"""Global ship-info node and virtual-node linker pass.

This pass emits the single global ship-info node (containing the top-level
``ship`` metadata) and then connects it to every other virtual node in the
expansion graph via ``global_virtual_member`` cross-edges.

It must run last so that all virtual nodes from prior passes are present when
it builds the linker edges.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping

from graph_expansion.context import EXPANSION_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base import ExpansionPass

__all__ = ["GlobalVirtualLinkerPass", "GLOBAL_SHIP_NODE_ID"]

GLOBAL_SHIP_NODE_ID = "global_ship"


class GlobalVirtualLinkerPass(ExpansionPass):
    """Emit the global ship-info node and connect it to every other virtual node."""

    name = "global_virtual_linker"
    version = 2
    requires = ("base_indexes",)

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Emit the ``global_ship`` node and ``global_virtual_member`` edges."""

        ship_info = context.source.get("ship", {}) or {}

        expansion_graph = context.ensure_emitted_graph(EXPANSION_GRAPH_NAME)
        nodes: List[MutableMapping[str, Any]] = expansion_graph["nodes"]
        cross_edges: List[MutableMapping[str, Any]] = expansion_graph["cross_edges"]

        # Snapshot the virtual nodes emitted by prior passes before appending
        # the global node, so we can link to exactly those nodes without a
        # self-edge filter.
        prior_virtual_nodes = list(nodes)

        global_node: Dict[str, Any] = {
            "id": GLOBAL_SHIP_NODE_ID,
            "kind": "global_ship_info",
            "ship": ship_info,
        }
        nodes.append(global_node)

        new_edges = [
            {
                "source": GLOBAL_SHIP_NODE_ID,
                "source_graph": EXPANSION_GRAPH_NAME,
                "target": node["id"],
                "target_graph": EXPANSION_GRAPH_NAME,
                "kind": "global_virtual_member",
            }
            for node in prior_virtual_nodes
            if node.get("member_count", 1) > 0
        ]

        cross_edges.extend(new_edges)

        edge_count = len(new_edges)
        context.increment_summary(
            EXPANSION_GRAPH_NAME,
            global_ship_nodes=1,
            global_virtual_member_edges=edge_count,
        )

        return {"global_nodes": 1, "global_virtual_member_edges": edge_count}
