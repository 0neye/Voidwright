"""Core-support Layer 2 expansion pass.

This pass emits downstream reactor/factory support edges inside each traversable
cluster. It reuses the weighted travel graph utilities from
``graph_expansion.passes.crew_access_layer1`` so travel semantics stay defined
in one place while the pass implementation lives in its own file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Set

from graph_expansion.context import EXPANSION_GRAPH_NAME, STRUCTURAL_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base import ExpansionPass
from graph_expansion.passes.travel_support import (
    PartTravelProfile,
    build_part_travel_profiles,
    build_touching_adjacency,
    cluster_graph,
    distances_to_target,
    factory_support_mode,
    is_ammo_weapon,
    is_energy_weapon,
    is_engine_room,
    is_generic_storage,
    is_missile_weapon,
    is_power_storage,
    is_shield,
    is_thruster,
    layer1_part_sets,
    min_distance_for_part,
)

__all__ = ["Layer2CoreSupportPass"]


def _support_edge(
    source: int, target: int, kind: str, travel_distance: float, cluster_index: int
) -> Dict[str, Any]:
    return {
        "source": source,
        "source_graph": STRUCTURAL_GRAPH_NAME,
        "target": target,
        "target_graph": STRUCTURAL_GRAPH_NAME,
        "kind": kind,
        "travel_distance": travel_distance,
        "distance_unit": "movement_cost",
        "path_model": "dijkstra_cardinal_cell_entry_cost_v1",
        "cluster_id": f"traversable_cluster_{cluster_index}",
    }


class Layer2CoreSupportPass(ExpansionPass):
    """Emit Layer 2 reactor/factory downstream support edges."""

    name = "core_support_layer2"
    version = 1
    requires = ("base_indexes", "traversable_clusters", "crew_access_layer1")
    provides = ()

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Compute and emit reactor/factory support relations inside each cluster."""

        node_by_id: Mapping[int, Mapping[str, Any]] = context.caches.get("node_by_id") or {}
        door_edges: Sequence[Mapping[str, Any]] = context.caches.get("door_edges") or []
        touching_edges: Sequence[Mapping[str, Any]] = context.caches.get("touching_edges") or []
        traversable_clusters: Sequence[Sequence[int]] = context.get_annotation("traversable_clusters") or []
        cluster_by_part_id: Mapping[int, int] = context.get_annotation("cluster_by_part_id") or {}
        if not traversable_clusters:
            return {
                "reactor_support_edges": 0,
                "factory_support_edges": 0,
            }

        part_profiles: Mapping[int, PartTravelProfile] = context.get_or_build_cache(
            "crew_access_part_profiles",
            lambda: build_part_travel_profiles(context),
        )
        walkable_part_ids: Set[int] = set(part_profiles)
        touching_adjacency: Mapping[int, Sequence[int]] = context.get_or_build_cache(
            "crew_access_touching_adjacency",
            lambda: build_touching_adjacency(touching_edges),
        )
        _, reactor_ids_by_cluster, factory_ids_by_cluster = layer1_part_sets(
            node_by_id,
            walkable_part_ids,
            cluster_by_part_id,
        )

        power_storage_ids_by_cluster: Dict[int, List[int]] = {}
        shield_ids_by_cluster: Dict[int, List[int]] = {}
        engine_room_ids_by_cluster: Dict[int, List[int]] = {}
        thruster_ids_by_cluster: Dict[int, List[int]] = {}
        energy_weapon_ids_by_cluster: Dict[int, List[int]] = {}
        generic_storage_ids_by_cluster: Dict[int, List[int]] = {}
        ammo_weapon_ids_by_cluster: Dict[int, List[int]] = {}
        missile_weapon_ids_by_cluster: Dict[int, List[int]] = {}
        factory_mode_by_id: Dict[int, str] = {}

        for part_id in sorted(node_by_id):
            if part_id not in walkable_part_ids:
                continue
            cluster_index = cluster_by_part_id.get(part_id)
            if cluster_index is None:
                continue
            part_name = str(node_by_id[part_id].get("part_id", ""))
            if is_power_storage(part_name):
                power_storage_ids_by_cluster.setdefault(int(cluster_index), []).append(part_id)
            if is_shield(part_name):
                shield_ids_by_cluster.setdefault(int(cluster_index), []).append(part_id)
            if is_engine_room(part_name):
                engine_room_ids_by_cluster.setdefault(int(cluster_index), []).append(part_id)
            if is_thruster(part_name):
                thruster_ids_by_cluster.setdefault(int(cluster_index), []).append(part_id)
            if is_energy_weapon(part_name):
                energy_weapon_ids_by_cluster.setdefault(int(cluster_index), []).append(part_id)
            if is_generic_storage(part_name):
                generic_storage_ids_by_cluster.setdefault(int(cluster_index), []).append(part_id)
            if is_ammo_weapon(part_name):
                ammo_weapon_ids_by_cluster.setdefault(int(cluster_index), []).append(part_id)
            if is_missile_weapon(part_name):
                missile_weapon_ids_by_cluster.setdefault(int(cluster_index), []).append(part_id)
            factory_mode = factory_support_mode(part_name)
            if factory_mode is not None:
                factory_mode_by_id[part_id] = factory_mode

        for mapping in (
            power_storage_ids_by_cluster,
            shield_ids_by_cluster,
            engine_room_ids_by_cluster,
            thruster_ids_by_cluster,
            energy_weapon_ids_by_cluster,
            generic_storage_ids_by_cluster,
            ammo_weapon_ids_by_cluster,
            missile_weapon_ids_by_cluster,
        ):
            for target_ids in mapping.values():
                target_ids.sort()

        cross_edges_out: List[Dict[str, Any]] = []
        reactor_edge_count = 0
        factory_edge_count = 0

        for cluster_index, reactor_ids in sorted(reactor_ids_by_cluster.items()):
            graph = cluster_graph(
                context,
                cluster_index=cluster_index,
                traversable_clusters=traversable_clusters,
                part_profiles=part_profiles,
                door_edges=door_edges,
            )
            power_storage_targets = power_storage_ids_by_cluster.get(cluster_index, [])
            shield_targets = shield_ids_by_cluster.get(cluster_index, [])
            engine_room_targets = engine_room_ids_by_cluster.get(cluster_index, [])
            thruster_targets = thruster_ids_by_cluster.get(cluster_index, [])
            energy_weapon_targets = energy_weapon_ids_by_cluster.get(cluster_index, [])
            for reactor_part_id in reactor_ids:
                source_states = graph.states_by_part_id.get(reactor_part_id, ())
                if not source_states:
                    continue

                accessible_engine_room_ids: Set[int] = set()
                target_specs = (
                    ("reactor_supports_power_storage", power_storage_targets),
                    ("reactor_supports_shield", shield_targets),
                    ("reactor_supports_engine_room", engine_room_targets),
                    ("reactor_supports_energy_weapon", energy_weapon_targets),
                )
                for edge_kind, target_ids in target_specs:
                    for target_part_id in target_ids:
                        if target_part_id == reactor_part_id:
                            continue
                        distance_map = distances_to_target(
                            context,
                            cluster_index=cluster_index,
                            target_part_id=target_part_id,
                            traversable_clusters=traversable_clusters,
                            part_profiles=part_profiles,
                            door_edges=door_edges,
                        )
                        travel_distance = min_distance_for_part(distance_map, source_states)
                        if travel_distance is None:
                            continue
                        if edge_kind == "reactor_supports_engine_room":
                            accessible_engine_room_ids.add(target_part_id)
                        cross_edges_out.append(
                            _support_edge(reactor_part_id, target_part_id, edge_kind, travel_distance, cluster_index)
                        )
                        reactor_edge_count += 1

                for target_part_id in thruster_targets:
                    if target_part_id == reactor_part_id:
                        continue
                    touching_neighbors = set(touching_adjacency.get(target_part_id, ()))
                    if touching_neighbors & accessible_engine_room_ids:
                        continue
                    distance_map = distances_to_target(
                        context,
                        cluster_index=cluster_index,
                        target_part_id=target_part_id,
                        traversable_clusters=traversable_clusters,
                        part_profiles=part_profiles,
                        door_edges=door_edges,
                    )
                    travel_distance = min_distance_for_part(distance_map, source_states)
                    if travel_distance is None:
                        continue
                    cross_edges_out.append(
                        _support_edge(reactor_part_id, target_part_id, "reactor_supports_thruster", travel_distance, cluster_index)
                    )
                    reactor_edge_count += 1

        for cluster_index, factory_ids in sorted(factory_ids_by_cluster.items()):
            graph = cluster_graph(
                context,
                cluster_index=cluster_index,
                traversable_clusters=traversable_clusters,
                part_profiles=part_profiles,
                door_edges=door_edges,
            )
            storage_targets = generic_storage_ids_by_cluster.get(cluster_index, [])
            ammo_targets = ammo_weapon_ids_by_cluster.get(cluster_index, [])
            missile_targets = missile_weapon_ids_by_cluster.get(cluster_index, [])
            for factory_part_id in factory_ids:
                source_states = graph.states_by_part_id.get(factory_part_id, ())
                if not source_states:
                    continue
                factory_mode = factory_mode_by_id.get(factory_part_id, "storage_only")
                target_specs = [("factory_supports_storage", storage_targets)]
                if factory_mode == "ammo":
                    target_specs.append(("factory_supports_ammo_weapon", ammo_targets))
                elif factory_mode == "missile":
                    target_specs.append(("factory_supports_missile_weapon", missile_targets))
                for edge_kind, target_ids in target_specs:
                    for target_part_id in target_ids:
                        if target_part_id == factory_part_id:
                            continue
                        distance_map = distances_to_target(
                            context,
                            cluster_index=cluster_index,
                            target_part_id=target_part_id,
                            traversable_clusters=traversable_clusters,
                            part_profiles=part_profiles,
                            door_edges=door_edges,
                        )
                        travel_distance = min_distance_for_part(distance_map, source_states)
                        if travel_distance is None:
                            continue
                        cross_edges_out.append(
                            _support_edge(factory_part_id, target_part_id, edge_kind, travel_distance, cluster_index)
                        )
                        factory_edge_count += 1

        cross_edges_out.sort(
            key=lambda edge: (
                int(edge["source"]),
                str(edge["kind"]),
                int(edge["target"]),
            )
        )

        expansion_graph = context.ensure_emitted_graph(EXPANSION_GRAPH_NAME)
        cross_edges: List[MutableMapping[str, Any]] = expansion_graph["cross_edges"]
        cross_edges.extend(cross_edges_out)
        context.increment_summary(
            EXPANSION_GRAPH_NAME,
            core_support_edges=len(cross_edges_out),
            reactor_support_edges=reactor_edge_count,
            factory_support_edges=factory_edge_count,
        )
        return {
            "reactor_support_edges": reactor_edge_count,
            "factory_support_edges": factory_edge_count,
        }
