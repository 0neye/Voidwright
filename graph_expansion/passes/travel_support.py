"""Shared travel-graph helpers for crew access and core support passes.

This module centralizes movement semantics, part-role classification, and cached
weighted traversable-cluster graph construction so Layer 1 and Layer 2 passes do
not depend on one another.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Set

from common.geometry import load_vanilla_part_geometry, resolve_geometry_part_id_and_rotation
from graph_expansion.context import ExpansionContext
from graph_expansion.passes.traversable_clusters import is_corridor_like

__all__ = [
    "ClusterTravelGraph",
    "Coord2x",
    "PartTravelProfile",
    "State",
    "build_part_travel_profiles",
    "build_touching_adjacency",
    "cluster_graph",
    "detect_part_role",
    "distances_to_target",
    "factory_support_mode",
    "find_proxy_part",
    "is_ammo_weapon",
    "is_energy_weapon",
    "is_engine_room",
    "is_generic_storage",
    "is_missile_weapon",
    "is_power_storage",
    "is_railgun",
    "is_shield",
    "is_thermal_conduit",
    "is_thruster",
    "layer1_part_sets",
    "min_distance_for_part",
]

Coord2x = tuple[int, int]
State = tuple[int, int, int]

_CREW_ROOM_SUBSTRINGS: tuple[str, ...] = ("crew_quarters", "quarters")
_REACTOR_SUBSTRINGS: tuple[str, ...] = ("reactor",)
_FACTORY_SUBSTRINGS: tuple[str, ...] = ("factory",)
_GENERIC_STORAGE_SUBSTRINGS: tuple[str, ...] = ("storage",)
_POWER_STORAGE_SUBSTRINGS: tuple[str, ...] = ("power_storage", "capacitor")
_SHIELD_SUBSTRINGS: tuple[str, ...] = ("shield_gen",)
_ENGINE_ROOM_SUBSTRINGS: tuple[str, ...] = ("engine_room",)
_THRUSTER_SUBSTRINGS: tuple[str, ...] = ("thruster",)
_ENERGY_WEAPON_SUBSTRINGS: tuple[str, ...] = (
    "laser_blaster",
    "disruptor",
    "ion_beam_emitter",
    "resonance_beam",
)
_RAILGUN_SUBSTRINGS: tuple[str, ...] = ("railgun",)
_THERMAL_CONDUIT_SUBSTRINGS: tuple[str, ...] = (
    "heat_exchanger",
    "heat_pipe",
    "radiator",
    "resonance_beam",
    "thermal_amplification",
    "thermal_battery",
    "thermal_dilation",
)
_AMMO_WEAPON_SUBSTRINGS: tuple[str, ...] = (
    "cannon",
    "flak_cannon",
    "chaingun",
    "point_defense",
    "railgun",
)
_MISSILE_WEAPON_SUBSTRINGS: tuple[str, ...] = ("missile_launcher",)
_FACTORY_AMMO_SOURCE_SUBSTRINGS: tuple[str, ...] = ("factory_ammo",)
_FACTORY_MISSILE_SOURCE_SUBSTRINGS: tuple[str, ...] = (
    "factory_emp",
    "factory_he",
    "factory_mine",
    "factory_nuke",
    "factory_thermal",
)
_CARDINAL_DELTAS_2X: tuple[tuple[int, int], ...] = ((2, 0), (-2, 0), (0, 2), (0, -2))
_DIRECTION_BY_DELTA_2X: dict[tuple[int, int], str] = {
    (2, 0): "Right",
    (-2, 0): "Left",
    (0, 2): "Down",
    (0, -2): "Up",
}
_OPPOSITE_DIRECTION: dict[str, str] = {
    "Up": "Down",
    "Down": "Up",
    "Left": "Right",
    "Right": "Left",
}


@dataclass(frozen=True)
class PartTravelProfile:
    """Rotation-aware travel profile for one structural part node."""

    node_id: int
    part_id: str
    rotation: int
    walkable_cells: frozenset[Coord2x]
    local_tile_by_world_cell: Mapping[Coord2x, tuple[int, int]]
    corridor_like: bool
    blocked_travel_cell_directions: Mapping[tuple[int, int], frozenset[str]]
    default_speed: float
    directional_speeds: Mapping[str, float] | None

    def move_cost(self, direction: str) -> float:
        speed = None
        if self.directional_speeds:
            speed = self.directional_speeds.get(direction)
        if speed is None:
            speed = self.default_speed
        if speed <= 0:
            return 1.0
        return 1.0 / float(speed)

    def is_direction_blocked(self, world_cell: Coord2x, direction: str) -> bool:
        local_tile = self.local_tile_by_world_cell.get(world_cell)
        if local_tile is None:
            return False
        return direction in self.blocked_travel_cell_directions.get(local_tile, frozenset())


@dataclass(frozen=True)
class ClusterTravelGraph:
    """Directed weighted travel graph for one traversable cluster."""

    cluster_index: int
    states_by_part_id: Mapping[int, tuple[State, ...]]
    reverse_adjacency: Mapping[State, tuple[tuple[State, float], ...]]


def detect_part_role(part_id: str) -> str | None:
    lower_id = part_id.lower()
    if any(token in lower_id for token in _CREW_ROOM_SUBSTRINGS):
        return "crew_room"
    if any(token in lower_id for token in _REACTOR_SUBSTRINGS):
        return "reactor"
    if any(token in lower_id for token in _FACTORY_SUBSTRINGS):
        return "factory"
    return None


def is_generic_storage(part_id: str) -> bool:
    lower_id = part_id.lower()
    return any(token in lower_id for token in _GENERIC_STORAGE_SUBSTRINGS) and not any(
        token in lower_id for token in _POWER_STORAGE_SUBSTRINGS
    )


def is_power_storage(part_id: str) -> bool:
    lower_id = part_id.lower()
    return any(token in lower_id for token in _POWER_STORAGE_SUBSTRINGS)


def is_shield(part_id: str) -> bool:
    lower_id = part_id.lower()
    return any(token in lower_id for token in _SHIELD_SUBSTRINGS)


def is_engine_room(part_id: str) -> bool:
    lower_id = part_id.lower()
    return any(token in lower_id for token in _ENGINE_ROOM_SUBSTRINGS)


def is_railgun(part_id: str) -> bool:
    lower_id = part_id.lower()
    return any(token in lower_id for token in _RAILGUN_SUBSTRINGS)


def is_thermal_conduit(part_id: str) -> bool:
    """Return True when *part_id* is a dedicated thermal conduit.

    Thermal conduits are non-overclocked parts whose primary role is to relay
    or absorb heat in the ship's thermal network: heat pipes, radiators, heat
    exchangers, resonance beam turrets (thermal lances), and thermal pumps /
    batteries.  They are treated as first-class thermal participants and are
    the only non-overclocked parts permitted to form edges with overclocked
    parts via port matching.
    """
    lower_id = part_id.lower()
    return any(token in lower_id for token in _THERMAL_CONDUIT_SUBSTRINGS)


def is_thruster(part_id: str) -> bool:
    lower_id = part_id.lower()
    return any(token in lower_id for token in _THRUSTER_SUBSTRINGS) and not is_engine_room(lower_id)


def is_energy_weapon(part_id: str) -> bool:
    lower_id = part_id.lower()
    return any(token in lower_id for token in _ENERGY_WEAPON_SUBSTRINGS)


def is_ammo_weapon(part_id: str) -> bool:
    lower_id = part_id.lower()
    return any(token in lower_id for token in _AMMO_WEAPON_SUBSTRINGS)


def is_missile_weapon(part_id: str) -> bool:
    lower_id = part_id.lower()
    return any(token in lower_id for token in _MISSILE_WEAPON_SUBSTRINGS)


def factory_support_mode(part_id: str) -> str | None:
    lower_id = part_id.lower()
    if any(token in lower_id for token in _FACTORY_AMMO_SOURCE_SUBSTRINGS):
        return "ammo"
    if any(token in lower_id for token in _FACTORY_MISSILE_SOURCE_SUBSTRINGS):
        return "missile"
    if any(token in lower_id for token in _FACTORY_SUBSTRINGS):
        return "storage_only"
    return None


def _coerce_cell_2x(cell: object) -> Coord2x | None:
    if not isinstance(cell, Sequence) or len(cell) != 2:
        return None
    try:
        return (int(cell[0]), int(cell[1]))
    except (TypeError, ValueError):
        return None


def _state_for(part_id: int, cell: Coord2x) -> State:
    return (int(part_id), int(cell[0]), int(cell[1]))


def _direction_from_cells(source_cell: Coord2x, target_cell: Coord2x) -> str | None:
    delta = (target_cell[0] - source_cell[0], target_cell[1] - source_cell[1])
    return _DIRECTION_BY_DELTA_2X.get(delta)


def build_part_travel_profiles(context: ExpansionContext) -> Dict[int, PartTravelProfile]:
    node_by_id: Mapping[int, Mapping[str, Any]] = context.caches.get("node_by_id") or {}
    geometry_cache = load_vanilla_part_geometry()
    profiles: Dict[int, PartTravelProfile] = {}

    for node_id, node in node_by_id.items():
        part_id = str(node.get("part_id", ""))
        walkable_cells = frozenset(
            cell for cell in (_coerce_cell_2x(cell) for cell in node.get("walkable_cells_2x", []) or [])
            if cell is not None
        )
        if not walkable_cells:
            continue

        rotation = int(node.get("rotation", 0)) % 4
        location_2x = _coerce_cell_2x(node.get("location_2x"))
        local_tile_by_world_cell: Dict[Coord2x, tuple[int, int]] = {}
        if location_2x is not None:
            base_x, base_y = location_2x
            for world_x, world_y in walkable_cells:
                if (world_x - base_x) % 2 == 0 and (world_y - base_y) % 2 == 0:
                    local_tile_by_world_cell[(world_x, world_y)] = (
                        (world_x - base_x) // 2,
                        (world_y - base_y) // 2,
                    )

        default_speed = 1.0
        directional_speeds: Mapping[str, float] | None = None
        blocked_travel_cell_directions: Mapping[tuple[int, int], frozenset[str]] = {}

        geometry_part_id, geometry_rotation = resolve_geometry_part_id_and_rotation(part_id, rotation)
        vanilla_geometry = geometry_cache.get(geometry_part_id)
        if vanilla_geometry is not None:
            rotation_geometry = vanilla_geometry.rotation_geometry(geometry_rotation)
            blocked_travel_cell_directions = rotation_geometry.blocked_travel_cell_directions
            directional_speeds = rotation_geometry.crew_speed_by_direction or vanilla_geometry.crew_speed_by_direction
            speed = vanilla_geometry.crew_speed_for_direction(geometry_rotation, "Up")
            if speed is None and vanilla_geometry.crew_speed_factor is not None:
                speed = vanilla_geometry.crew_speed_factor
            if speed is not None and speed > 0:
                default_speed = float(speed)
        profiles[node_id] = PartTravelProfile(
            node_id=node_id,
            part_id=part_id,
            rotation=rotation,
            walkable_cells=walkable_cells,
            local_tile_by_world_cell=local_tile_by_world_cell,
            corridor_like=is_corridor_like(part_id),
            blocked_travel_cell_directions=blocked_travel_cell_directions,
            default_speed=default_speed,
            directional_speeds=directional_speeds,
        )

    return profiles


def _add_reverse_edge(
    reverse_edges: MutableMapping[State, list[tuple[State, float]]],
    destination_state: State,
    source_state: State,
    cost: float,
) -> None:
    reverse_edges.setdefault(destination_state, []).append((source_state, float(cost)))


def _build_cluster_travel_graph(
    *,
    cluster_index: int,
    member_ids: Sequence[int],
    part_profiles: Mapping[int, PartTravelProfile],
    door_edges: Sequence[Mapping[str, Any]],
) -> ClusterTravelGraph:
    states_by_part_id: Dict[int, tuple[State, ...]] = {}
    reverse_edges: Dict[State, list[tuple[State, float]]] = {}

    cluster_member_ids = {int(part_id) for part_id in member_ids}
    for part_id in sorted(cluster_member_ids):
        profile = part_profiles.get(part_id)
        if profile is None:
            continue
        states = tuple(_state_for(part_id, cell) for cell in sorted(profile.walkable_cells))
        states_by_part_id[part_id] = states
        for state in states:
            reverse_edges.setdefault(state, [])

    for part_id in sorted(states_by_part_id):
        profile = part_profiles[part_id]
        walkable_cells = profile.walkable_cells
        for source_cell in sorted(walkable_cells):
            source_state = _state_for(part_id, source_cell)
            for delta_x, delta_y in _CARDINAL_DELTAS_2X:
                target_cell = (source_cell[0] + delta_x, source_cell[1] + delta_y)
                if target_cell not in walkable_cells:
                    continue
                direction = _direction_from_cells(source_cell, target_cell)
                if direction is None:
                    continue
                opposite_direction = _OPPOSITE_DIRECTION[direction]
                if profile.is_direction_blocked(source_cell, direction) or profile.is_direction_blocked(
                    target_cell,
                    opposite_direction,
                ):
                    continue
                target_state = _state_for(part_id, target_cell)
                _add_reverse_edge(reverse_edges, target_state, source_state, profile.move_cost(direction))

    corridor_states_by_cell: Dict[Coord2x, list[tuple[int, State]]] = {}
    for part_id in sorted(states_by_part_id):
        profile = part_profiles[part_id]
        if not profile.corridor_like:
            continue
        for state in states_by_part_id[part_id]:
            cell = (state[1], state[2])
            corridor_states_by_cell.setdefault(cell, []).append((part_id, state))

    emitted_directed_edges: Set[tuple[State, State]] = set()

    for cell, owners in corridor_states_by_cell.items():
        for index, (part_id_a, state_a) in enumerate(owners):
            for part_id_b, state_b in owners[index + 1 :]:
                if part_id_a == part_id_b:
                    continue
                if (state_a, state_b) not in emitted_directed_edges:
                    _add_reverse_edge(reverse_edges, state_b, state_a, 0.0)
                    emitted_directed_edges.add((state_a, state_b))
                if (state_b, state_a) not in emitted_directed_edges:
                    _add_reverse_edge(reverse_edges, state_a, state_b, 0.0)
                    emitted_directed_edges.add((state_b, state_a))

        for delta_x, delta_y in _CARDINAL_DELTAS_2X:
            neighbor_cell = (cell[0] + delta_x, cell[1] + delta_y)
            neighbor_owners = corridor_states_by_cell.get(neighbor_cell)
            if not neighbor_owners:
                continue
            direction = _direction_from_cells(cell, neighbor_cell)
            if direction is None:
                continue
            for source_part_id, source_state in owners:
                for target_part_id, target_state in neighbor_owners:
                    if source_part_id == target_part_id:
                        continue
                    directed_edge = (source_state, target_state)
                    if directed_edge in emitted_directed_edges:
                        continue
                    target_profile = part_profiles[target_part_id]
                    _add_reverse_edge(
                        reverse_edges,
                        target_state,
                        source_state,
                        target_profile.move_cost(direction),
                    )
                    emitted_directed_edges.add(directed_edge)

    for edge in door_edges:
        source_part_id = edge.get("source")
        target_part_id = edge.get("target")
        if not isinstance(source_part_id, int) or not isinstance(target_part_id, int):
            continue
        if source_part_id not in cluster_member_ids or target_part_id not in cluster_member_ids:
            continue
        source_profile = part_profiles.get(source_part_id)
        target_profile = part_profiles.get(target_part_id)
        if source_profile is None or target_profile is None:
            continue

        source_cell = _coerce_cell_2x(edge.get("source_cell_2x"))
        target_cell = _coerce_cell_2x(edge.get("target_cell_2x"))
        if source_cell is None or target_cell is None:
            continue
        if source_cell not in source_profile.walkable_cells or target_cell not in target_profile.walkable_cells:
            continue

        forward_direction = _direction_from_cells(source_cell, target_cell)
        reverse_direction = _direction_from_cells(target_cell, source_cell)
        if forward_direction is None or reverse_direction is None:
            continue

        source_state = _state_for(source_part_id, source_cell)
        target_state = _state_for(target_part_id, target_cell)
        _add_reverse_edge(reverse_edges, target_state, source_state, target_profile.move_cost(forward_direction))
        _add_reverse_edge(reverse_edges, source_state, target_state, source_profile.move_cost(reverse_direction))

    normalized_reverse_edges = {
        state: tuple(sorted(neighbors, key=lambda item: (item[0][0], item[0][1], item[0][2], item[1])))
        for state, neighbors in reverse_edges.items()
    }
    return ClusterTravelGraph(
        cluster_index=cluster_index,
        states_by_part_id=states_by_part_id,
        reverse_adjacency=normalized_reverse_edges,
    )


def _reverse_dijkstra(
    reverse_adjacency: Mapping[State, Sequence[tuple[State, float]]],
    target_states: Sequence[State],
) -> Dict[State, float]:
    distances: Dict[State, float] = {}
    heap: list[tuple[float, int, int, int]] = []
    for target_state in sorted(target_states):
        distances[target_state] = 0.0
        heapq.heappush(heap, (0.0, target_state[0], target_state[1], target_state[2]))

    while heap:
        distance, part_id, cell_x, cell_y = heapq.heappop(heap)
        current_state = (part_id, cell_x, cell_y)
        if distance != distances.get(current_state):
            continue
        for previous_state, edge_cost in reverse_adjacency.get(current_state, ()):  # incoming original edge
            next_distance = distance + float(edge_cost)
            prior_best = distances.get(previous_state)
            if prior_best is None or next_distance < prior_best:
                distances[previous_state] = next_distance
                heapq.heappush(
                    heap,
                    (next_distance, previous_state[0], previous_state[1], previous_state[2]),
                )
    return distances


def min_distance_for_part(distances: Mapping[State, float], part_states: Sequence[State]) -> float | None:
    candidates = [distances[state] for state in part_states if state in distances]
    if not candidates:
        return None
    return min(candidates)


def build_touching_adjacency(touching_edges: Sequence[Mapping[str, Any]]) -> Dict[int, tuple[int, ...]]:
    adjacency: Dict[int, Set[int]] = {}
    for edge in touching_edges:
        source = edge.get("source")
        target = edge.get("target")
        if not isinstance(source, int) or not isinstance(target, int):
            continue
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    return {node_id: tuple(sorted(neighbors)) for node_id, neighbors in adjacency.items()}


def find_proxy_part(
    *,
    crew_part_id: int,
    target_clusters: Set[int],
    touching_adjacency: Mapping[int, Sequence[int]],
    cluster_by_part_id: Mapping[int, int],
    walkable_part_ids: Set[int],
) -> tuple[int, int, int] | None:
    for neighbor_part_id in touching_adjacency.get(crew_part_id, ()):  # deterministic tuple
        if neighbor_part_id not in walkable_part_ids:
            continue
        cluster_index = cluster_by_part_id.get(neighbor_part_id)
        if cluster_index in target_clusters:
            return neighbor_part_id, 1, int(cluster_index)
    return None


def layer1_part_sets(
    node_by_id: Mapping[int, Mapping[str, Any]],
    walkable_part_ids: Set[int],
    cluster_by_part_id: Mapping[int, int],
) -> tuple[list[int], dict[int, list[int]], dict[int, list[int]]]:
    crew_room_ids: List[int] = []
    reactor_ids_by_cluster: Dict[int, List[int]] = {}
    factory_ids_by_cluster: Dict[int, List[int]] = {}
    for part_id in sorted(node_by_id):
        role = detect_part_role(str(node_by_id[part_id].get("part_id", "")))
        if role is None or part_id not in walkable_part_ids:
            continue
        cluster_index = cluster_by_part_id.get(part_id)
        if cluster_index is None:
            continue
        if role == "crew_room":
            crew_room_ids.append(part_id)
        elif role == "reactor":
            reactor_ids_by_cluster.setdefault(int(cluster_index), []).append(part_id)
        elif role == "factory":
            factory_ids_by_cluster.setdefault(int(cluster_index), []).append(part_id)
    for target_ids in reactor_ids_by_cluster.values():
        target_ids.sort()
    for target_ids in factory_ids_by_cluster.values():
        target_ids.sort()
    return crew_room_ids, reactor_ids_by_cluster, factory_ids_by_cluster


def cluster_graph(
    context: ExpansionContext,
    *,
    cluster_index: int,
    traversable_clusters: Sequence[Sequence[int]],
    part_profiles: Mapping[int, PartTravelProfile],
    door_edges: Sequence[Mapping[str, Any]],
) -> ClusterTravelGraph:
    return context.get_or_build_cache(
        f"crew_access_cluster_graph_{cluster_index}",
        lambda: _build_cluster_travel_graph(
            cluster_index=cluster_index,
            member_ids=traversable_clusters[cluster_index],
            part_profiles=part_profiles,
            door_edges=door_edges,
        ),
    )


def distances_to_target(
    context: ExpansionContext,
    *,
    cluster_index: int,
    target_part_id: int,
    traversable_clusters: Sequence[Sequence[int]],
    part_profiles: Mapping[int, PartTravelProfile],
    door_edges: Sequence[Mapping[str, Any]],
) -> Mapping[State, float]:
    dijkstra_cache: Dict[tuple[int, int], Dict[State, float]] = context.get_or_build_cache(
        "crew_access_reverse_dijkstra",
        lambda: {},
    )
    cache_key = (cluster_index, target_part_id)
    if cache_key in dijkstra_cache:
        return dijkstra_cache[cache_key]
    graph = cluster_graph(
        context,
        cluster_index=cluster_index,
        traversable_clusters=traversable_clusters,
        part_profiles=part_profiles,
        door_edges=door_edges,
    )
    target_states = graph.states_by_part_id.get(target_part_id, ())
    dijkstra_cache[cache_key] = _reverse_dijkstra(graph.reverse_adjacency, target_states)
    return dijkstra_cache[cache_key]
