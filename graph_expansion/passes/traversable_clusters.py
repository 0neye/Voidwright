"""Traversable cluster computation and emission pass.

This module contains both the expansion pass that materializes
traversable-cluster super-nodes and the underlying connectivity
helpers. The connectivity rules mirror the original structural expansion logic:

- any two walkable parts joined by a door edge are in the same cluster
- corridor and moving-walkway parts that share or are adjacent in the
  2x coordinate frame are in the same cluster
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Set

from graph_expansion.context import EXPANSION_GRAPH_NAME, STRUCTURAL_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base import ExpansionPass

__all__ = [
    "TraversableClustersPass",
    "is_corridor_like",
    "build_traversable_clusters",
]

_CORRIDOR_LIKE_SUBSTRINGS: tuple[str, ...] = ("corridor", "walkway")


def is_corridor_like(part_id: str) -> bool:
    """Return True when *part_id* identifies a corridor or moving walkway."""

    lower_id = part_id.lower()
    return any(token in lower_id for token in _CORRIDOR_LIKE_SUBSTRINGS)


def build_traversable_clusters(nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> List[List[int]]:
    """Group node IDs into traversable clusters.

    Two parts join the same cluster when either condition holds:

    1. Door edge: they are connected by an edge with ``kind == "door"``
       in the structural part graph (applies to any two walkable parts).
    2. Corridor-like adjacency: both parts are corridor or moving-
       walkway parts and at least one walkable cell of each part is
       adjacent in the 2x coordinate frame (differs by 2 in exactly one
       axis).

    Args:
        nodes: Structural part graph nodes.
        edges: Structural part graph edges.

    Returns:
        Sorted list of sorted member-ID lists, one per cluster.
    """

    parts_with_walkable: Set[int] = {
        node["id"] for node in nodes if node.get("walkable_cells_2x")
    }
    if not parts_with_walkable:
        return []

    parent: Dict[int, int] = {node_id: node_id for node_id in parts_with_walkable}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    # Rule 1: door edges connect any two walkable parts regardless of type.
    for edge in edges:
        if edge.get("kind") != "door":
            continue
        src = edge.get("source")
        tgt = edge.get("target")
        if src in parts_with_walkable and tgt in parts_with_walkable:
            union(int(src), int(tgt))

    # Rule 2: corridor-like parts merge when their walkable cells are adjacent.
    corridor_nodes: List[Mapping[str, Any]] = [
        node
        for node in nodes
        if node.get("id") in parts_with_walkable and is_corridor_like(node.get("part_id", ""))
    ]
    cell_to_corridor_parts: Dict[tuple[int, int], Set[int]] = {}
    for node in corridor_nodes:
        node_id = int(node["id"])
        for cell in node.get("walkable_cells_2x", []):
            key = (int(cell[0]), int(cell[1]))
            cell_to_corridor_parts.setdefault(key, set()).add(node_id)

    for (cx, cy), part_ids in cell_to_corridor_parts.items():
        # Merge corridor parts that share the same walkable cell.
        part_ids_list = sorted(part_ids)
        for index in range(1, len(part_ids_list)):
            union(part_ids_list[0], part_ids_list[index])
        # Merge corridor parts whose walkable cells are adjacent.
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            neighbor_parts = cell_to_corridor_parts.get((cx + dx, cy + dy))
            if neighbor_parts:
                for pid_a in part_ids:
                    for pid_b in neighbor_parts:
                        if pid_a != pid_b:
                            union(pid_a, pid_b)

    clusters: Dict[int, List[int]] = {}
    for node_id in parts_with_walkable:
        root = find(node_id)
        clusters.setdefault(root, []).append(int(node_id))

    sorted_clusters: List[List[int]] = [sorted(member_ids) for member_ids in clusters.values()]
    return sorted(sorted_clusters)


class TraversableClustersPass(ExpansionPass):
    """Emit traversable-cluster super-nodes and membership cross-edges."""

    name = "traversable_clusters"
    version = 1
    requires = ("base_indexes",)
    provides = ("traversable_clusters", "cluster_by_part_id")

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Compute traversable clusters and emit virtual nodes and edges."""

        structural_nodes = context.caches.get("structural_nodes")
        structural_edges = context.caches.get("structural_edges")
        if structural_nodes is None or structural_edges is None:
            structural_graph = context.get_source_graph(STRUCTURAL_GRAPH_NAME)
            structural_nodes = list(structural_graph.get("nodes", []))
            structural_edges = list(structural_graph.get("edges", []))

        clusters = build_traversable_clusters(structural_nodes, structural_edges)
        context.set_annotation("traversable_clusters", clusters)

        cluster_by_part_id: Dict[int, int] = {}
        for cluster_index, member_ids in enumerate(clusters):
            for member_id in member_ids:
                cluster_by_part_id[member_id] = cluster_index
        context.set_annotation("cluster_by_part_id", cluster_by_part_id)

        expansion_graph = context.ensure_emitted_graph(EXPANSION_GRAPH_NAME)
        nodes: List[MutableMapping[str, Any]] = expansion_graph["nodes"]
        cross_edges: List[MutableMapping[str, Any]] = expansion_graph["cross_edges"]

        cluster_nodes: List[Dict[str, Any]] = []
        cluster_cross_edges: List[Dict[str, Any]] = []
        for cluster_index, member_ids in enumerate(clusters):
            cluster_id = f"traversable_cluster_{cluster_index}"
            cluster_nodes.append(
                {
                    "id": cluster_id,
                    "kind": "traversable_cluster",
                    "member_count": len(member_ids),
                }
            )
            for member_id in member_ids:
                cluster_cross_edges.append(
                    {
                        "source": cluster_id,
                        "source_graph": EXPANSION_GRAPH_NAME,
                        "target": member_id,
                        "target_graph": STRUCTURAL_GRAPH_NAME,
                        "kind": "super_member",
                    }
                )

        nodes.extend(cluster_nodes)
        cross_edges.extend(cluster_cross_edges)

        context.increment_summary(
            EXPANSION_GRAPH_NAME,
            traversable_clusters=len(clusters),
            super_member_edges=len(cluster_cross_edges),
        )

        return {
            "cluster_count": len(clusters),
            "super_member_edges": len(cluster_cross_edges),
        }

