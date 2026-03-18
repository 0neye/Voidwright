"""Traversable cluster computation and emission pass.

This module contains both the expansion pass that materializes
traversable-cluster super-nodes and the underlying connectivity
helpers. Traversable clusters are intentionally conservative:

- any two walkable parts joined by a door edge are in the same cluster
- corridor, moving-walkway, and conveyor parts that share or are adjacent in
  the 2x coordinate frame are in the same cluster

Structural touching alone does not merge clusters. Open room-to-corridor
contacts are represented later by travel/accessibility edges rather than by
collapsing cluster membership.
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

_CORRIDOR_LIKE_SUBSTRINGS: tuple[str, ...] = ("corridor", "walkway", "conveyor")

# Clusters with a combined walkable-cell footprint at or below this 2x threshold
# *and* no door edges are considered trivially isolated and are not emitted.
# 16 2x-cells corresponds to a 4-tile (2x2) footprint in canonical coordinates.
_SMALL_CLUSTER_2X_CELL_THRESHOLD: int = 16


def is_corridor_like(part_id: str) -> bool:
    """Return True when *part_id* identifies a corridor, walkway, or conveyor."""

    lower_id = part_id.lower()
    return any(token in lower_id for token in _CORRIDOR_LIKE_SUBSTRINGS)


def build_traversable_clusters(nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> List[List[int]]:
    """Group node IDs into traversable clusters.

    Two parts join the same cluster when either condition holds:

    1. Door edge: they are connected by an edge with ``kind == "door"``
       in the structural part graph (applies to any two walkable parts).
    2. Corridor-like adjacency: both parts are corridor, moving-
       walkway, or conveyor parts and at least one walkable cell of each part is
       adjacent in the 2x coordinate frame (differs by 2 in exactly one
       axis).

    Structural touching is not enough on its own. That contact is preserved for
    later accessibility/support passes, but it does not collapse clusters.

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

    node_by_id = {int(node["id"]): node for node in nodes if node.get("id") in parts_with_walkable}

    # Rule 2: corridor-like parts merge when their walkable cells are adjacent.
    corridor_nodes: List[Mapping[str, Any]] = [
        node for node in node_by_id.values() if is_corridor_like(str(node.get("part_id", "")))
    ]
    cell_to_corridor_parts: Dict[tuple[int, int], Set[int]] = {}
    for node in corridor_nodes:
        node_id = int(node["id"])
        for cell in node.get("walkable_cells_2x", []):
            key = (int(cell[0]), int(cell[1]))
            cell_to_corridor_parts.setdefault(key, set()).add(node_id)

    for (cx, cy), part_ids in cell_to_corridor_parts.items():
        neighboring_sets = [
            cell_to_corridor_parts.get((cx + 2, cy), set()),
            cell_to_corridor_parts.get((cx - 2, cy), set()),
            cell_to_corridor_parts.get((cx, cy + 2), set()),
            cell_to_corridor_parts.get((cx, cy - 2), set()),
            part_ids,  # shared cell also counts
        ]
        for neighbor_ids in neighboring_sets:
            for a in part_ids:
                for b in neighbor_ids:
                    if a != b:
                        union(a, b)

    clusters: Dict[int, List[int]] = {}
    for node_id in parts_with_walkable:
        root = find(node_id)
        clusters.setdefault(root, []).append(int(node_id))

    sorted_clusters: List[List[int]] = [sorted(member_ids) for member_ids in clusters.values()]
    return sorted(sorted_clusters)


class TraversableClustersPass(ExpansionPass):
    """Emit traversable-cluster super-nodes and membership cross-edges."""

    name = "traversable_clusters"
    version = 3
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

        # Filter out trivially isolated clusters: small footprint with no door access.
        # Reuse the door-edge list cached by BaseIndexesPass when available.
        door_edges = context.caches.get("door_edges") or [
            e for e in structural_edges if e.get("kind") == "door"
        ]
        door_part_ids: Set[int] = set()
        for edge in door_edges:
            src, tgt = edge.get("source"), edge.get("target")
            if src is not None:
                door_part_ids.add(int(src))
            if tgt is not None:
                door_part_ids.add(int(tgt))

        node_2x_cell_count: Dict[int, int] = {
            int(n["id"]): len(n.get("walkable_cells_2x") or [])
            for n in structural_nodes
            if n.get("walkable_cells_2x")
        }

        kept_clusters: List[List[int]] = []
        filtered_count: int = 0
        for member_ids in clusters:
            if len(member_ids) == 1:
                filtered_count += 1
                continue
            total_cells = sum(node_2x_cell_count.get(mid, 0) for mid in member_ids)
            has_door = any(mid in door_part_ids for mid in member_ids)
            if total_cells <= _SMALL_CLUSTER_2X_CELL_THRESHOLD and not has_door:
                filtered_count += 1
                continue
            kept_clusters.append(member_ids)
        clusters = kept_clusters

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
            "filtered_small_clusters": filtered_count,
        }

