"""ThermalNetworksPass — builds thermal connectivity networks from port geometry.

This pass identifies thermal connections between structural part nodes by
matching thermal port geometry in ship space. Parts whose thermal ports face
each other across a single tile boundary are connected; connected components
form named thermal-network virtual nodes in the expansion graph.

Port geometry is loaded from ``common.geometry`` via the ``thermal_ports``
attribute on ``RotationGeometry``.
Ports marked ``overclock_conditional=True`` are only active when the owning
part node has ``overclocked=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Set, Tuple

from common.geometry import load_vanilla_part_geometry, resolve_geometry_part_id_and_rotation
from graph_expansion.context import EXPANSION_GRAPH_NAME, STRUCTURAL_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base import ExpansionPass
from graph_expansion.passes.travel_support import _OPPOSITE_DIRECTION

__all__ = ["ThermalNetworksPass"]

# Maps direction → the facing delta (dx, dy) in 2x-space units.
# One tile step = 2 units in 2x-space.
_DIRECTION_DELTA: Dict[str, Tuple[int, int]] = {
    "Up":    (0, -2),
    "Down":  (0,  2),
    "Left":  (-2, 0),
    "Right": ( 2, 0),
}


@dataclass(frozen=True)
class _ActivePort:
    """One resolved thermal port in ship-space 2x coordinates.

    Attributes:
        node_id: Integer ID of the owning structural part node.
        direction: Facing direction string (``"Up"``, ``"Down"``, etc.).
    """

    node_id: int
    direction: str


def _build_port_index(
    context: ExpansionContext,
) -> Tuple[Dict[Tuple[int, int, str], List[_ActivePort]], int]:
    """Build a dict keyed by (ship_x, ship_y, direction) → list of _ActivePort.

    For each structural node the function looks up rotation-specific
    ``thermal_ports`` from the vanilla geometry cache.  Ports with
    ``overclock_conditional=True`` are only included when the node carries
    ``overclocked=True``.  Nodes without geometry entries are silently skipped.

    Args:
        context: Expansion context for the current source artifact.

    Returns:
        A tuple of ``(port_map, parts_with_ports)`` where *port_map* maps
        ``(ship_x, ship_y, direction)`` to the list of active ports at that
        position, and *parts_with_ports* counts nodes that contributed at
        least one active port.
    """

    geometry_cache = load_vanilla_part_geometry()
    structural_nodes: List[Mapping[str, Any]] = context.caches.get("structural_nodes") or []
    port_map: Dict[Tuple[int, int, str], List[_ActivePort]] = {}
    parts_with_ports = 0

    for node in structural_nodes:
        node_id = node["id"]
        part_id = node.get("part_id", "")
        rotation = int(node.get("rotation", 0))
        overclocked = bool(node.get("overclocked", False))
        location_2x = node.get("location_2x")

        if not isinstance(location_2x, list) or len(location_2x) != 2:
            continue

        geo_part_id, geo_rotation = resolve_geometry_part_id_and_rotation(part_id, rotation)
        vanilla_geo = geometry_cache.get(geo_part_id)
        if vanilla_geo is None:
            continue

        rot_geo = vanilla_geo.rotation_geometry(geo_rotation)
        thermal_ports = rot_geo.thermal_ports
        if not thermal_ports:
            continue

        has_active_port = False
        for port in thermal_ports:
            if port.overclock_conditional and not overclocked:
                continue  # skip inactive overclock-only ports

            # Port position in 2x-space: port.location is a tile-space offset
            # from the part origin.  One tile = 2 units in 2x-space.
            port_2x_x = location_2x[0] + int(port.location[0]) * 2
            port_2x_y = location_2x[1] + int(port.location[1]) * 2
            direction = port.direction
            key = (port_2x_x, port_2x_y, direction)
            port_map.setdefault(key, []).append(
                _ActivePort(node_id=int(node_id), direction=direction)
            )
            has_active_port = True

        if has_active_port:
            parts_with_ports += 1

    return port_map, parts_with_ports


def _find_thermal_edges(
    port_map: Dict[Tuple[int, int, str], List[_ActivePort]],
) -> List[Tuple[int, int]]:
    """Return deduplicated (node_id_a, node_id_b) thermal connection pairs.

    Two ports connect when port A at position (x, y) facing direction D is
    matched by a complementary port at the adjacent tile in the D direction,
    i.e. at position (x + dx*2, y + dy*2) facing the opposite direction.

    Args:
        port_map: Mapping from ``(ship_x, ship_y, direction)`` to active ports.

    Returns:
        Sorted list of unique ``(min_id, max_id)`` node-ID pairs.
    """

    edges: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()

    for (px, py, direction), ports_here in port_map.items():
        dx, dy = _DIRECTION_DELTA[direction]
        target_key = (px + dx, py + dy, _OPPOSITE_DIRECTION[direction])
        ports_there = port_map.get(target_key)
        if not ports_there:
            continue
        for pa in ports_here:
            for pb in ports_there:
                if pa.node_id == pb.node_id:
                    continue
                pair = (min(pa.node_id, pb.node_id), max(pa.node_id, pb.node_id))
                if pair not in seen:
                    seen.add(pair)
                    edges.append(pair)

    return sorted(edges)


def _union_find_clusters(
    node_ids: List[int],
    edges: List[Tuple[int, int]],
) -> List[List[int]]:
    """Group *node_ids* into connected components using union-find.

    Args:
        node_ids: All node IDs that participate in at least one thermal edge.
        edges: Pairs of connected node IDs.

    Returns:
        Sorted list of sorted member-ID lists, one per connected component.
    """

    parent: Dict[int, int] = {n: n for n in node_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: Dict[int, List[int]] = {}
    for n in node_ids:
        root = find(n)
        groups.setdefault(root, []).append(n)

    return [sorted(members) for members in sorted(groups.values())]


class ThermalNetworksPass(ExpansionPass):
    """Emit thermal-network virtual nodes and membership cross-edges.

    For each connected component of thermally linked parts this pass emits one
    ``thermal_network_N`` virtual node and ``thermal_member`` cross-edges to
    each member in the structural part graph.

    Parts without thermal port geometry, or parts whose only ports are
    ``overclock_conditional`` while the part is not overclocked, are completely
    omitted from thermal networks (they receive no isolated node).
    """

    name = "thermal_networks"
    version = 1
    requires = ("base_indexes",)
    provides = ("thermal_networks", "thermal_network_by_part_id")

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Compute thermal networks and emit virtual nodes and edges.

        Args:
            context: Expansion context for the current source artifact.

        Returns:
            Mapping with counts for ports, edges, networks, and member sizes.
        """

        port_map, parts_with_ports = context.get_or_build_cache(
            "thermal_port_map",
            lambda: _build_port_index(context),
        )

        thermal_edges = _find_thermal_edges(port_map)

        # Collect all node IDs that participate in at least one thermal edge.
        connected_ids: Set[int] = set()
        for a, b in thermal_edges:
            connected_ids.add(a)
            connected_ids.add(b)

        if not connected_ids:
            context.set_annotation("thermal_networks", [])
            context.set_annotation("thermal_network_by_part_id", {})
            return {
                "parts_with_ports": parts_with_ports,
                "thermal_edges": 0,
                "networks": 0,
                "network_sizes": [],
            }

        clusters = _union_find_clusters(sorted(connected_ids), thermal_edges)

        # Build annotations and emit virtual nodes and cross-edges in one pass.
        network_by_part_id: Dict[int, str] = {}
        expansion_graph = context.ensure_emitted_graph(EXPANSION_GRAPH_NAME)
        nodes: List[Dict[str, Any]] = expansion_graph["nodes"]
        cross_edges: List[Dict[str, Any]] = expansion_graph["cross_edges"]

        thermal_nodes: List[Dict[str, Any]] = []
        thermal_member_edges: List[Dict[str, Any]] = []

        for cluster_index, members in enumerate(clusters):
            network_id = f"thermal_network_{cluster_index}"
            for member_id in members:
                network_by_part_id[member_id] = network_id
            thermal_nodes.append(
                {
                    "id": network_id,
                    "kind": "thermal_network",
                    "member_count": len(members),
                }
            )
            for member_id in members:
                thermal_member_edges.append(
                    {
                        "source": network_id,
                        "source_graph": EXPANSION_GRAPH_NAME,
                        "target": member_id,
                        "target_graph": STRUCTURAL_GRAPH_NAME,
                        "kind": "thermal_member",
                    }
                )

        context.set_annotation("thermal_networks", clusters)
        context.set_annotation("thermal_network_by_part_id", network_by_part_id)

        nodes.extend(thermal_nodes)
        cross_edges.extend(thermal_member_edges)

        context.increment_summary(
            EXPANSION_GRAPH_NAME,
            thermal_network_nodes=len(thermal_nodes),
            thermal_member_edges=len(thermal_member_edges),
        )

        return {
            "parts_with_ports": parts_with_ports,
            "thermal_edges": len(thermal_edges),
            "networks": len(clusters),
            "network_sizes": [len(m) for m in clusters],
        }
