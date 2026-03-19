"""ThermalNetworksPass — builds thermal connectivity networks from port geometry.

This pass identifies thermal connections between structural part nodes by
matching thermal port geometry in ship space. Parts whose thermal ports face
each other across a single tile boundary are connected; connected components
form named thermal-network virtual nodes in the expansion graph.

Port geometry is loaded from ``common.geometry`` via the ``thermal_ports``
attribute on ``RotationGeometry``.
Ports marked ``overclock_conditional=True`` are only active when the owning
part node has ``overclocked=True``.

Engine room special case
------------------------
Overclocked engine rooms force every directly connected thruster to be
overclocked as well (this is already reflected in the ``overclocked`` field
written by preprocessing; a future generator verification step should confirm
this invariant).

Additionally, an overclocked engine room acts as a heat conduit for any
physically adjacent thruster: if any 2x footprint cell of the thruster is
tile-adjacent (one tile = 2 units in 2x-space) to a cell of the overclocked
engine room, an implicit thermal edge is added between the two, regardless of
whether their explicit thermal ports align.  This means thrusters that are not
connected via dedicated heat pipes are still pulled into the engine room's
thermal network simply by touching it.

Heat exchanger radius special case
----------------------------------
After direct thermal connectivity is constructed, each connected heat exchanger
acts as an inclusion source for overclocked parts inside its absorption radius.
Overclocked parts that are not already in that thermal network are attached to
it when any occupied tile is within the heat exchanger's configured radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Set, Tuple

from common.geometry import load_vanilla_part_geometry, resolve_geometry_part_id_and_rotation
from graph_expansion.context import EXPANSION_GRAPH_NAME, STRUCTURAL_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base import ExpansionPass
from graph_expansion.passes.travel_support import _OPPOSITE_DIRECTION, is_engine_room, is_thruster

__all__ = ["ThermalNetworksPass"]

# Maps direction → the facing delta (dx, dy) in 2x-space units.
# One tile step = 2 units in 2x-space.
_DIRECTION_DELTA: Dict[str, Tuple[int, int]] = {
    "Up":    (0, -2),
    "Down":  (0,  2),
    "Left":  (-2, 0),
    "Right": ( 2, 0),
}

# Heat exchanger absorption rules are defined in:
# Data/ships/terran/heat_exchanger/heat_exchanger.rules
#   Region { Type = EdgeDistance; Distance = 5 }
# Distances here are in tile units.  One tile = 2 units in this pass's 2x-space.
_HEAT_EXCHANGER_ABSORPTION_RADIUS_TILES = 5.0
_HEAT_EXCHANGER_PART_ID = "cosmoteer.heat_exchanger"


@dataclass(frozen=True)
class _ActivePort:
    """One resolved thermal port in ship-space 2x coordinates.

    Attributes:
        node_id: Integer ID of the owning structural part node.
        direction: Facing direction string (``"Up"``, ``"Down"``, etc.).
    """

    node_id: int
    direction: str


def _is_heat_exchanger(part_id: str) -> bool:
    return part_id.lower() == _HEAT_EXCHANGER_PART_ID


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


def _footprint_cells_2x(node: Mapping[str, Any]) -> Set[Tuple[int, int]]:
    """Return the set of 2x-space grid cells occupied by *node*.

    Uses ``location_2x`` and ``footprint`` width/height stored on the node by
    preprocessing.  Returns an empty set when required attributes are absent or
    malformed.
    """

    location_2x = node.get("location_2x")
    footprint = node.get("footprint")
    if not isinstance(location_2x, (list, tuple)) or len(location_2x) != 2:
        return set()
    lx, ly = int(location_2x[0]), int(location_2x[1])
    if not isinstance(footprint, dict):
        # Fall back to a single 1x1 occupied tile at the part origin.
        # This keeps heuristics available for synthetic/minimal test nodes.
        return {(lx, ly)}
    w = int(footprint.get("width", 0))
    h = int(footprint.get("height", 0))
    if w <= 0 or h <= 0:
        return {(lx, ly)}
    return {(lx + 2 * col, ly + 2 * row) for row in range(h) for col in range(w)}


def _cells_within_distance_2x(
    cells_a: Set[Tuple[int, int]],
    cells_b: Set[Tuple[int, int]],
    max_distance_2x: float,
) -> bool:
    """Return True when any cell pair is within *max_distance_2x* (euclidean)."""

    if not cells_a or not cells_b:
        return False
    max_sq = max_distance_2x * max_distance_2x
    for ax, ay in cells_a:
        for bx, by in cells_b:
            dx = float(ax - bx)
            dy = float(ay - by)
            if (dx * dx) + (dy * dy) <= max_sq:
                return True
    return False


def _build_engine_room_thruster_edges(
    context: ExpansionContext,
) -> List[Tuple[int, int]]:
    """Return implicit thermal edges between overclocked engine rooms and adjacent thrusters.

    An overclocked engine room acts as a heat conduit: any thruster that shares a
    tile boundary with it receives an implicit thermal edge without requiring
    explicit port-to-port alignment.

    Args:
        context: Expansion context for the current source artifact.

    Returns:
        Sorted list of unique ``(min_id, max_id)`` node-ID pairs.
    """

    structural_nodes: List[Mapping[str, Any]] = context.caches.get("structural_nodes") or []
    node_by_id: Dict[int, Mapping[str, Any]] = context.caches.get("node_by_id") or {}

    # Single pass: build cell → node_id index and collect overclocked engine rooms
    # with their pre-computed footprint cells to avoid recomputing them later.
    cell_to_node: Dict[Tuple[int, int], int] = {}
    overclocked_ers: List[Tuple[int, Set[Tuple[int, int]]]] = []  # (node_id, cells)

    for node in structural_nodes:
        node_id = node.get("id")
        if not isinstance(node_id, int):
            continue
        cells = _footprint_cells_2x(node)
        for cell in cells:
            cell_to_node[cell] = node_id
        if is_engine_room(node.get("part_id", "")) and node.get("overclocked"):
            overclocked_ers.append((node_id, cells))

    edges: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()

    for node_id, er_cells in overclocked_ers:
        if not er_cells:
            continue

        for cx, cy in er_cells:
            for ddx, ddy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                neighbor_cell = (cx + ddx, cy + ddy)
                if neighbor_cell in er_cells:
                    continue  # still within the engine room footprint
                neighbor_id = cell_to_node.get(neighbor_cell)
                if neighbor_id is None or neighbor_id == node_id:
                    continue
                neighbor_node = node_by_id.get(neighbor_id)
                if neighbor_node is None:
                    continue
                if not is_thruster(neighbor_node.get("part_id", "")):
                    continue
                pair = (min(node_id, neighbor_id), max(node_id, neighbor_id))
                if pair not in seen:
                    seen.add(pair)
                    edges.append(pair)

    return sorted(edges)


def _build_heat_exchanger_radius_edges(
    context: ExpansionContext,
    clusters: List[List[int]],
) -> List[Tuple[int, int]]:
    """Return edges that pull overclocked parts into connected heat-exchanger networks.

    For each already-connected thermal network, every heat exchanger in that
    network can connect to overclocked parts that are not yet network members
    when the shortest occupied-cell distance is within the exchanger's
    absorption radius.
    """

    node_by_id: Dict[int, Mapping[str, Any]] = context.caches.get("node_by_id") or {}
    cells_by_id: Dict[int, Set[Tuple[int, int]]] = {
        node_id: _footprint_cells_2x(node)
        for node_id, node in node_by_id.items()
    }

    candidates_overclocked: List[int] = [
        node_id
        for node_id, node in node_by_id.items()
        if node.get("overclocked")
    ]

    max_distance_2x = _HEAT_EXCHANGER_ABSORPTION_RADIUS_TILES * 2.0
    edges: Set[Tuple[int, int]] = set()

    for members in clusters:
        member_set = set(members)
        exchanger_ids = [
            node_id
            for node_id in members
            if _is_heat_exchanger((node_by_id.get(node_id) or {}).get("part_id", ""))
        ]
        if not exchanger_ids:
            continue

        for exchanger_id in exchanger_ids:
            exchanger_cells = cells_by_id.get(exchanger_id) or set()
            if not exchanger_cells:
                continue

            for candidate_id in candidates_overclocked:
                if candidate_id in member_set:
                    continue
                candidate_cells = cells_by_id.get(candidate_id) or set()
                if not candidate_cells:
                    continue
                if not _cells_within_distance_2x(
                    exchanger_cells,
                    candidate_cells,
                    max_distance_2x,
                ):
                    continue
                edges.add((min(exchanger_id, candidate_id), max(exchanger_id, candidate_id)))

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

    Engine room special case: an overclocked engine room implicitly forms a
    thermal edge with every physically adjacent thruster (any cell of the
    thruster is tile-adjacent to any cell of the engine room), regardless of
    explicit thermal port alignment.  See module docstring for the full rules.
    """

    name = "thermal_networks"
    version = 3
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
        er_thruster_edges = _build_engine_room_thruster_edges(context)

        # Merge port-matched and engine-room-proximity edges, deduplicating in case
        # both mechanisms would independently produce the same pair.
        all_edges = sorted(set(thermal_edges) | set(er_thruster_edges))

        # Collect all node IDs that participate in at least one thermal edge.
        connected_ids: Set[int] = set()
        for a, b in all_edges:
            connected_ids.add(a)
            connected_ids.add(b)

        if not connected_ids:
            context.set_annotation("thermal_networks", [])
            context.set_annotation("thermal_network_by_part_id", {})
            return {
                "parts_with_ports": parts_with_ports,
                "thermal_edges": 0,
                "engine_room_thruster_edges": 0,
                "heat_exchanger_radius_edges": 0,
                "networks": 0,
                "network_sizes": [],
            }

        initial_clusters = _union_find_clusters(sorted(connected_ids), all_edges)
        heat_exchanger_radius_edges = _build_heat_exchanger_radius_edges(context, initial_clusters)
        all_edges = sorted(set(all_edges) | set(heat_exchanger_radius_edges))

        connected_ids = {node for edge in all_edges for node in edge}

        clusters = _union_find_clusters(sorted(connected_ids), all_edges)

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
            "engine_room_thruster_edges": len(er_thruster_edges),
            "heat_exchanger_radius_edges": len(heat_exchanger_radius_edges),
            "networks": len(clusters),
            "network_sizes": [len(m) for m in clusters],
        }
