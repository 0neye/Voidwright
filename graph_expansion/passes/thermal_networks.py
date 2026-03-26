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

Overclocked-part thermal-conduit restriction
---------------------------------------------
Overclocked parts must not relay thermal connectivity to arbitrary non-overclocked
parts.  Two restrictions apply:

1. A port-matched edge between two overclocked parts is suppressed unless one
   side is an engine room and the other is a thruster.  Engine rooms are not
   broadly exempt; only the engine-room ↔ thruster pairing is allowed.

2. A port-matched edge between one overclocked part and one non-overclocked part
   is only permitted when the non-overclocked part is a *thermal conduit* — a
   dedicated heat-relay part such as a radiator, heat exchanger, heat pipe, or
   resonance beam turret (see ``travel_support.is_thermal_conduit``).  Parts like
   railguns whose thermal ports exist for weapon-assembly purposes must not bridge
   into the overclocked thermal network via accidental port alignment.

Resonance beam turret (thermal lance) special case
---------------------------------------------------
Resonance beam turrets act as first-class thermal conduits — like heat pipes,
they can connect to overclocked parts via their (limited) physical thermal ports
and relay heat freely through the network.  They are recognised by
``travel_support.is_thermal_conduit`` and are exempt from restriction 2 above.

An overclocked part can join a thermal network by being port-connected to a
thermal conduit, or by falling within the absorption radius of a connected heat
exchanger.

Thermal canister missile launcher special case
----------------------------------------------
A missile launcher set to thermal canister mode (``toggle_values["missile_type"]
== 4``) participates in the thermal network as a backbone part.  Its thermal
ports are treated as active conduit ports: port-based edges are built the same
way as for any other conduit, and ``_is_backbone`` returns ``True`` for the
node.  In all other missile-type modes the launcher's thermal ports are
suppressed entirely — the node cannot join any thermal network.

Railgun assembly special case
------------------------------
Railgun components (loaders, launchers, accelerators) form a single logical
weapon unit when stacked end-to-end along their barrel axis.  The game's
thermal port data only covers side connections; this module synthesises
virtual thermal edges between any two railgun parts whose footprints are
tile-adjacent along the barrel axis (Y-axis for rotation 0/2, X-axis for
rotation 1/3).  Port-based railgun-to-railgun connections remain subject to
the same overclocked heat-pipe restriction as any other part.

Thermal-network membership gating
---------------------------------
Thermal clusters intentionally include only:

1. Backbone parts (non-overclocked thermal conduits and thermal-canister
   missile launchers), and
2. Overclocked parts.

Non-overclocked non-backbone parts (including railgun loaders/accelerators)
are excluded from thermal clusters even if virtual railgun assembly edges exist.

Railgun assembly membership promotion
-------------------------------------
When any railgun component in a barrel-connected railgun assembly is
overclocked (typically the launcher), every railgun component in that assembly
is promoted into thermal-network membership. This keeps the full railgun unit
coherent without admitting unrelated non-overclocked non-backbone parts.

Two-phase thermal clustering
----------------------------
Thermal networks are built in two phases to prevent overclocked parts from
acting as bridges between separate conduit networks:

**Phase 1 (backbone)**: union-find runs only on edges where *both* endpoints
are non-overclocked thermal conduits.  This establishes the "spine" of each
thermal system independently of any OC attachments.

**Phase 2 (leaf attachment)**: remaining nodes (OC parts, railgun assemblies,
thrusters, etc.) are first grouped by non-backbone edges among themselves.
Each sub-group is then assigned to whichever backbone cluster it first touches
via a cross-edge (first-assignment wins, sorted-edge order).  A sub-group that
touches multiple backbone clusters is assigned to only one of them, preserving
the separation of the underlying conduit spines.

**Multi-network leaf exception**: any non-backbone sub-group (OC parts, railgun
assemblies, thrusters, etc.) that touches more than one backbone cluster is
added to *all* of them instead of just one.  Each node in the sub-group is a
leaf member in every network it touches and does not act as a conduit between
those networks — the backbone clusters remain independent.  As a result the
``thermal_networks`` annotation and the cross-edge list may contain the same
node IDs in multiple cluster entries.  The ``thermal_network_by_part_id``
annotation maps each node ID to the list of all network IDs it belongs to (a
list of length > 1 for nodes that span multiple networks).

Sub-groups with no backbone attachment form their own isolated clusters.

Heat exchanger radius special case
-----------------------------------
After two-phase clustering, each heat exchanger expands its cluster by pulling
in nearby *overclocked non-conduit* parts that are not yet assigned to any
cluster.  Thermal conduit parts (heat pipes, thermal batteries, dilation pumps,
etc.) are *excluded* from radius inclusion — they join networks only via direct
port connections.  Parts already in another cluster are also excluded to prevent
the radius from merging two separate thermal systems.

Radius computation is performed in 1x tile space from the heat exchanger
part's geometric center (middle of its footprint), not its top-left tile.
Candidates are included when any occupied candidate tile is within the
configured radius from that center point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Set, Tuple

from common.heat_exchanger import (
    HEAT_EXCHANGER_ABSORPTION_RADIUS_TILES,
    footprint_tile_origins_2x,
    is_heat_exchanger,
    tile_set_within_heat_exchanger_radius_2x,
)
from common.geometry import load_vanilla_part_geometry, resolve_geometry_part_id_and_rotation
from graph_expansion.context import EXPANSION_GRAPH_NAME, STRUCTURAL_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base import ExpansionPass
from graph_expansion.passes.travel_support import _OPPOSITE_DIRECTION, is_engine_room, is_missile_weapon, is_railgun, is_thermal_conduit, is_thermal_missile_launcher, is_thruster

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
@dataclass(frozen=True)
class _ActivePort:
    """One resolved thermal port in ship-space 2x coordinates.

    Attributes:
        node_id: Integer ID of the owning structural part node.
        direction: Facing direction string (``"Up"``, ``"Down"``, etc.).
        overclocked: Whether the owning part is overclocked.
        part_id: Part identifier of the owning node (used for exception checks).
    """

    node_id: int
    direction: str
    overclocked: bool
    part_id: str
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

        # Missile launchers expose thermal ports only when configured for thermal
        # canister mode (missile_type == 4).  In any other mode the launcher has
        # no thermal connectivity and its ports must not appear in the index.
        if is_missile_weapon(part_id) and not is_thermal_missile_launcher(node):
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
                _ActivePort(
                    node_id=int(node_id),
                    direction=direction,
                    overclocked=overclocked,
                    part_id=part_id,
                )
            )
            has_active_port = True

        if has_active_port:
            parts_with_ports += 1

    return port_map, parts_with_ports


def _can_form_thermal_edge(pa: _ActivePort, pb: _ActivePort) -> bool:
    """Return True when a port-matched edge between *pa* and *pb* is permitted.

    Three cases:

    1. Neither side overclocked → always allowed.
    2. Both sides overclocked → only allowed when one side is an engine room
       and the other is a thruster.  Railgun components are NOT exempt here;
       their barrel-axis connectivity is handled separately by
       ``_build_railgun_assembly_edges``.
    3. Exactly one side overclocked → the non-overclocked side must be a
       *thermal conduit* (heat pipe, radiator, heat exchanger, resonance beam
       turret, etc. — see ``travel_support.is_thermal_conduit``).  This
       prevents railguns and other weapons whose thermal ports exist for weapon-
       assembly purposes from accidentally bridging into the ship's overclocked
       thermal network.
    """
    if not pa.overclocked and not pb.overclocked:
        return True  # neither side overclocked — always allowed

    if pa.overclocked and pb.overclocked:
        # Both overclocked: only engine-room ↔ thruster is exempt.
        er_side = is_engine_room(pa.part_id) or is_engine_room(pb.part_id)
        thr_side = is_thruster(pa.part_id) or is_thruster(pb.part_id)
        return er_side and thr_side

    # Exactly one side is overclocked — non-OC side must be a thermal conduit.
    oc_port, non_oc_port = (pa, pb) if pa.overclocked else (pb, pa)
    return is_thermal_conduit(non_oc_port.part_id)


def _find_thermal_edges(
    port_map: Dict[Tuple[int, int, str], List[_ActivePort]],
) -> List[Tuple[int, int]]:
    """Return deduplicated (node_id_a, node_id_b) thermal connection pairs.

    Two ports connect when port A at position (x, y) facing direction D is
    matched by a complementary port at the adjacent tile in the D direction,
    i.e. at position (x + dx*2, y + dy*2) facing the opposite direction.

    Additionally, edges between two overclocked parts are suppressed unless
    both are exempt (engine rooms or railgun components) — see
    ``_can_form_thermal_edge`` for the full rule.

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
                if not _can_form_thermal_edge(pa, pb):
                    continue
                pair = (min(pa.node_id, pb.node_id), max(pa.node_id, pb.node_id))
                if pair not in seen:
                    seen.add(pair)
                    edges.append(pair)

    return sorted(edges)


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
        cells = footprint_tile_origins_2x(node)
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


def _build_two_phase_clusters(
    all_edges: List[Tuple[int, int]],
    node_by_id: Dict[int, Mapping[str, Any]],
) -> List[List[int]]:
    """Cluster thermally connected nodes using a backbone-first, leaf-attachment approach.

    Phase 1 unions non-overclocked thermal conduit nodes using only
    conduit↔conduit edges, establishing independent thermal backbone clusters.
    Phase 2 groups non-backbone nodes (OC parts, railgun assemblies, etc.) by
    their mutual edges, then assigns each sub-group to the *largest* backbone
    cluster it touches via any cross-edge; ties are broken by smallest cluster
    index.  A sub-group that bridges two backbone clusters is still attached to
    only one of them, preserving conduit-network separation.  Sub-groups with no
    backbone attachment form their own isolated clusters.

    **Multi-network leaf exception**: any non-backbone sub-group that touches
    more than one backbone cluster is added to *all* of them.  Each member is
    a leaf in every touched network and does not bridge them.  The returned
    cluster lists may therefore be non-disjoint (the same node ID can appear
    in more than one cluster).

    Args:
        all_edges: Sorted list of ``(min_id, max_id)`` node-ID pairs.
        node_by_id: Mapping of node ID to structural node data.

    Returns:
        Sorted list of sorted member-ID lists, one per cluster.  Lists are
        disjoint except for railgun assembly nodes that span multiple networks.
    """

    def _is_backbone(node_id: int) -> bool:
        node = node_by_id.get(node_id, {})
        if node.get("overclocked", False):
            return False
        part_id = str(node.get("part_id", ""))
        return is_thermal_conduit(part_id) or is_thermal_missile_launcher(node)

    all_node_ids = sorted({nid for edge in all_edges for nid in edge})
    backbone_ids: Set[int] = {nid for nid in all_node_ids if _is_backbone(nid)}
    non_backbone_ids: Set[int] = set(all_node_ids) - backbone_ids

    # Phase 1: cluster non-OC thermal conduits on conduit↔conduit edges only.
    backbone_edges = [(a, b) for a, b in all_edges if a in backbone_ids and b in backbone_ids]
    backbone_clusters = _union_find_clusters(sorted(backbone_ids), backbone_edges)
    backbone_cluster_by_node: Dict[int, int] = {
        m: idx for idx, members in enumerate(backbone_clusters) for m in members
    }

    # Phase 2a: cluster non-backbone nodes on non-backbone↔non-backbone edges.
    nb_edges = [(a, b) for a, b in all_edges if a in non_backbone_ids and b in non_backbone_ids]
    nb_clusters = _union_find_clusters(sorted(non_backbone_ids), nb_edges)
    nb_cluster_by_node: Dict[int, int] = {
        m: idx for idx, members in enumerate(nb_clusters) for m in members
    }

    # Phase 2b: collect all backbone clusters each non-backbone sub-group touches,
    # then assign it to the largest one (tie-break: smallest cluster index).
    nb_group_bb_candidates: Dict[int, Set[int]] = {}
    for a, b in all_edges:
        a_bb = a in backbone_ids
        b_bb = b in backbone_ids
        if a_bb == b_bb:
            continue  # not a cross-edge
        bb_node = a if a_bb else b
        nb_node = b if a_bb else a
        nb_group = nb_cluster_by_node[nb_node]
        nb_group_bb_candidates.setdefault(nb_group, set()).add(backbone_cluster_by_node[bb_node])

    # Build final cluster member lists.
    result: Dict[int, List[int]] = {
        idx: list(members) for idx, members in enumerate(backbone_clusters)
    }
    for nb_group_idx, nb_members in enumerate(nb_clusters):
        bb_candidates = nb_group_bb_candidates.get(nb_group_idx, set())
        if len(bb_candidates) > 1:
            # Sub-group touches multiple backbone clusters: add to ALL of them as leaves.
            # Each member participates in every touched network without merging them.
            for bb_cluster in sorted(bb_candidates):
                result[bb_cluster].extend(nb_members)
        elif bb_candidates:
            result[next(iter(bb_candidates))].extend(nb_members)
        else:
            own_key = len(backbone_clusters) + nb_group_idx
            result[own_key] = list(nb_members)

    return [sorted(members) for members in sorted(result.values())]


def _apply_heat_exchanger_radius_to_clusters(
    context: ExpansionContext,
    clusters: List[List[int]],
) -> Tuple[List[List[int]], int]:
    """Expand clusters by pulling nearby unattached overclocked non-conduit parts.

    For each cluster that contains at least one heat exchanger, any overclocked
    non-conduit part that is *not yet assigned to any cluster* and whose footprint
    overlaps the exchanger's absorption-radius stencil is added to that cluster.

    Thermal conduit parts are explicitly excluded from radius inclusion (rule 1):
    parts like thermal batteries and dilation pumps must join networks only via
    direct port connections, never by proximity.  Parts already in another cluster
    are also excluded to prevent a heat exchanger from merging two separate thermal
    systems (rule 2).

    Each candidate is assigned to the *largest* cluster (by member count at the
    start of the radius phase) whose heat exchanger covers it; ties are broken by
    smallest cluster index.  A candidate is never assigned to more than one
    cluster.

    Args:
        context: Expansion context for the current source artifact.
        clusters: Current cluster list from two-phase clustering.

    Returns:
        A tuple ``(updated_clusters, nodes_added)`` where *updated_clusters* is
        the extended cluster list and *nodes_added* is the count of candidates
        that were pulled in by radius.
    """

    node_by_id: Dict[int, Mapping[str, Any]] = context.caches.get("node_by_id") or {}

    already_assigned: Set[int] = {m for cluster in clusters for m in cluster}

    # Candidates: overclocked, not a thermal conduit, not yet in any cluster.
    remaining: List[int] = [
        nid
        for nid, node in node_by_id.items()
        if node.get("overclocked")
        and not is_thermal_conduit(str(node.get("part_id", "")))
        and nid not in already_assigned
    ]

    if not remaining:
        return clusters, 0

    updated: List[List[int]] = [list(c) for c in clusters]

    # Build per-cluster exchanger lists once (only clusters that have exchangers).
    clusters_with_exchangers: List[Tuple[int, List[int]]] = [
        (cluster_idx, exchanger_ids)
        for cluster_idx, members in enumerate(updated)
        if (exchanger_ids := [
            nid
            for nid in members
            if is_heat_exchanger((node_by_id.get(nid) or {}).get("part_id", ""))
        ])
    ]

    if not clusters_with_exchangers:
        return clusters, 0

    # Compute tile footprints only for the nodes that will actually be queried.
    needed_ids: Set[int] = set(remaining) | {
        exc_id
        for _, exchanger_ids in clusters_with_exchangers
        for exc_id in exchanger_ids
    }
    tiles_by_id: Dict[int, Set[Tuple[int, int]]] = {
        nid: footprint_tile_origins_2x(node_by_id[nid])
        for nid in needed_ids
        if nid in node_by_id
    }

    # Snapshot initial sizes for tie-breaking (computed before any radius additions).
    initial_sizes: List[int] = [len(c) for c in updated]

    nodes_added = 0
    for candidate_id in remaining:
        candidate_tiles = tiles_by_id.get(candidate_id) or set()
        if not candidate_tiles:
            continue
        eligible: List[int] = [
            cluster_idx
            for cluster_idx, exchanger_ids in clusters_with_exchangers
            if any(
                tile_set_within_heat_exchanger_radius_2x(
                    tiles_by_id.get(exc_id) or set(),
                    candidate_tiles,
                    HEAT_EXCHANGER_ABSORPTION_RADIUS_TILES,
                )
                for exc_id in exchanger_ids
            )
        ]
        if not eligible:
            continue
        best = max(eligible, key=lambda idx: (initial_sizes[idx], -idx))
        updated[best].append(candidate_id)
        nodes_added += 1

    return [sorted(c) for c in updated], nodes_added


def _is_thermal_member_candidate(node: Mapping[str, Any]) -> bool:
    """Return True when *node* is eligible for thermal-network membership."""

    if node.get("overclocked", False):
        return True
    part_id = str(node.get("part_id", ""))
    return is_thermal_conduit(part_id) or is_thermal_missile_launcher(node)


def _railgun_promoted_member_ids(
    railgun_assembly_edges: List[Tuple[int, int]],
    node_by_id: Mapping[int, Mapping[str, Any]],
) -> Set[int]:
    """Return railgun node IDs promoted by an overclocked assembly member.

    Any connected component in the railgun assembly graph that contains at
    least one overclocked node is promoted as a whole.
    """

    railgun_node_ids = {
        node_id
        for node_id, node in node_by_id.items()
        if is_railgun(str(node.get("part_id", "")))
    }
    if not railgun_node_ids:
        return set()

    components = _union_find_clusters(list(railgun_node_ids), railgun_assembly_edges)
    promoted: Set[int] = set()
    for members in components:
        if any(node_by_id[member_id].get("overclocked", False) for member_id in members):
            promoted.update(members)
    return promoted


# Barrel-axis adjacency deltas (2x-space) for each rotation.
# Rotation 0/2: barrel runs along the Y-axis; rotation 1/3: along the X-axis.
_RAILGUN_BARREL_DELTAS: Dict[int, Tuple[Tuple[int, int], ...]] = {
    0: ((0, -2), (0, 2)),
    1: ((-2, 0), (2, 0)),
    2: ((0, -2), (0, 2)),
    3: ((-2, 0), (2, 0)),
}


def _build_railgun_assembly_edges(
    context: ExpansionContext,
) -> List[Tuple[int, int]]:
    """Return virtual thermal edges for railgun parts stacked along their barrel axis.

    The game's thermal port data covers only side connections; this function
    synthesises edges between any two railgun parts (loader, launcher,
    accelerator) whose 2x footprint cells are tile-adjacent along the barrel
    axis.  These edges apply regardless of overclocked status so that the
    entire assembly always forms a single thermal unit.

    Args:
        context: Expansion context for the current source artifact.

    Returns:
        Sorted list of unique ``(min_id, max_id)`` node-ID pairs.
    """

    node_by_id: Dict[int, Mapping[str, Any]] = context.caches.get("node_by_id") or {}

    railgun_nodes = [
        node for node in node_by_id.values()
        if is_railgun(str(node.get("part_id", "")))
    ]
    if not railgun_nodes:
        return []

    # Build a 2x-cell → node_id index restricted to railgun parts.
    railgun_cell_to_node: Dict[Tuple[int, int], int] = {}
    for node in railgun_nodes:
        node_id = int(node["id"])
        for cell in footprint_tile_origins_2x(node):
            railgun_cell_to_node[cell] = node_id

    edges: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()

    for node in railgun_nodes:
        node_id = int(node["id"])
        rotation = int(node.get("rotation", 0)) % 4
        barrel_deltas = _RAILGUN_BARREL_DELTAS[rotation]
        own_cells = footprint_tile_origins_2x(node)

        for cx, cy in own_cells:
            for ddx, ddy in barrel_deltas:
                neighbor_cell = (cx + ddx, cy + ddy)
                if neighbor_cell in own_cells:
                    continue  # still within own footprint
                neighbor_id = railgun_cell_to_node.get(neighbor_cell)
                if neighbor_id is None or neighbor_id == node_id:
                    continue
                neighbor_node = node_by_id[neighbor_id]
                neighbor_rotation = int(neighbor_node.get("rotation", 0)) % 4
                if rotation % 2 != neighbor_rotation % 2:
                    continue  # different barrel axis — not part of the same assembly
                pair = (min(node_id, neighbor_id), max(node_id, neighbor_id))
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

    Clustering is two-phase: first the non-overclocked thermal conduit backbone
    is clustered (conduit↔conduit edges only), then OC parts and other non-
    backbone nodes are attached as leaves without merging backbone clusters.
    This prevents an overclocked intermediary from bridging two separate conduit
    networks into one thermal system.

    Heat exchanger radius expansion only adds overclocked non-conduit parts not
    yet in any cluster.  Thermal conduits (heat pipes, thermal batteries, etc.)
    are excluded from radius inclusion and must join via direct port connections.

    Railgun assembly: barrel-stacked railgun components (any rotation) still
    receive virtual thermal edges. Non-overclocked non-backbone railgun parts
    are excluded unless they belong to an assembly with at least one
    overclocked railgun member, in which case the full assembly is included.

    Multi-network leaf membership: any non-backbone sub-group (OC parts, railgun
    assemblies, thrusters, etc.) that touches more than one backbone cluster is
    added as a leaf member of *all* those networks.  The backbone networks remain
    independent — non-backbone parts do not merge them.  As a result the same
    node may receive ``thermal_member`` cross-edges to more than one
    ``thermal_network_N`` virtual node.  The ``thermal_network_by_part_id``
    annotation maps each node ID to a list of network ID strings; nodes spanning
    multiple networks have lists longer than one element.

    Engine room special case: an overclocked engine room implicitly forms a
    thermal edge with every physically adjacent thruster (any cell of the
    thruster is tile-adjacent to any cell of the engine room), regardless of
    explicit thermal port alignment.  For port-based edges, the engine room is
    only exempt from the OC-OC suppression rule when the other port also belongs
    to a thruster; all other OC-OC port pairs involving an engine room are
    suppressed.  See module docstring for the full rules.

    Thermal canister missile launcher special case: a missile launcher whose
    ``toggle_values["missile_type"]`` equals 4 (thermal canister mode) exposes
    its thermal ports and participates in the thermal network as a backbone
    part — exactly like a thermal conduit.  In any other mode the launcher has
    no thermal ports and is fully excluded from all thermal networks.
    """

    name = "thermal_networks"
    version = 12
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
        railgun_assembly_edges = _build_railgun_assembly_edges(context)
        thermal_edge_set = set(thermal_edges)
        er_thruster_edge_set = set(er_thruster_edges)
        railgun_assembly_edge_set = set(railgun_assembly_edges)

        # Merge all edge sources, deduplicating in case multiple mechanisms
        # independently produce the same pair.
        all_edges = sorted(thermal_edge_set | er_thruster_edge_set | railgun_assembly_edge_set)
        node_by_id: Dict[int, Mapping[str, Any]] = context.caches.get("node_by_id") or {}
        railgun_promoted_ids = _railgun_promoted_member_ids(railgun_assembly_edges, node_by_id)
        eligible_node_ids = {
            node_id
            for node_id, node in node_by_id.items()
            if _is_thermal_member_candidate(node)
        } | railgun_promoted_ids
        filtered_edges = [
            (a, b)
            for a, b in all_edges
            if a in eligible_node_ids and b in eligible_node_ids
        ]
        filtered_edge_set = set(filtered_edges)

        # Collect all node IDs that participate in at least one thermal edge.
        connected_ids: Set[int] = set()
        for a, b in filtered_edges:
            connected_ids.add(a)
            connected_ids.add(b)

        if not connected_ids:
            context.set_annotation("thermal_networks", [])
            context.set_annotation("thermal_network_by_part_id", {})
            return {
                "parts_with_ports": parts_with_ports,
                "thermal_edges": 0,
                "engine_room_thruster_edges": 0,
                "railgun_assembly_edges": 0,
                "heat_exchanger_radius_edges": 0,
                "networks": 0,
                "network_sizes": [],
            }

        # Two-phase clustering: backbone (non-OC conduits) first, then attach leaves.
        clusters = _build_two_phase_clusters(filtered_edges, node_by_id)

        # Expand clusters with nearby unattached overclocked non-conduit parts.
        clusters, heat_exchanger_radius_count = _apply_heat_exchanger_radius_to_clusters(
            context, clusters
        )

        # Build annotations and emit virtual nodes and cross-edges in one pass.
        # thermal_network_by_part_id maps node_id -> list of network IDs.  Most
        # nodes belong to exactly one network (list length 1), but railgun assembly
        # nodes that span multiple thermal networks appear in each network's list.
        network_by_part_id: Dict[int, List[str]] = {}
        expansion_graph = context.ensure_emitted_graph(EXPANSION_GRAPH_NAME)
        nodes: List[Dict[str, Any]] = expansion_graph["nodes"]
        cross_edges: List[Dict[str, Any]] = expansion_graph["cross_edges"]

        thermal_nodes: List[Dict[str, Any]] = []
        thermal_member_edges: List[Dict[str, Any]] = []

        for cluster_index, members in enumerate(clusters):
            network_id = f"thermal_network_{cluster_index}"
            backbone_count = 0
            overclocked_count = 0
            for member_id in members:
                network_by_part_id.setdefault(member_id, []).append(network_id)
                thermal_member_edges.append(
                    {
                        "source": network_id,
                        "source_graph": EXPANSION_GRAPH_NAME,
                        "target": member_id,
                        "target_graph": STRUCTURAL_GRAPH_NAME,
                        "kind": "thermal_member",
                    }
                )
                node = node_by_id.get(member_id, {})
                if node.get("overclocked", False):
                    overclocked_count += 1
                elif is_thermal_conduit(str(node.get("part_id", ""))) or is_thermal_missile_launcher(node):
                    backbone_count += 1
            thermal_nodes.append(
                {
                    "id": network_id,
                    "kind": "thermal_network",
                    "member_count": len(members),
                    "backbone_count": backbone_count,
                    "overclocked_count": overclocked_count,
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
            "thermal_edges": len(thermal_edge_set & filtered_edge_set),
            "engine_room_thruster_edges": len(er_thruster_edge_set & filtered_edge_set),
            "railgun_assembly_edges": len(railgun_assembly_edge_set & filtered_edge_set),
            "heat_exchanger_radius_edges": heat_exchanger_radius_count,
            "networks": len(clusters),
            "network_sizes": [len(m) for m in clusters],
        }
