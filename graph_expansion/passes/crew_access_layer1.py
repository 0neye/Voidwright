"""Crew-access Layer 1 expansion pass.

This pass emits hierarchical crew-access cross-edges from crew rooms to
reactors and factories. Travel distance is computed with weighted Dijkstra over
exact walkable 2x cells plus explicit inter-part transitions.

Movement semantics intentionally stay outside preprocessing graph JSON for now.
The structural graph provides compact walkable cells and door portal endpoints,
while richer travel metadata such as blocked intra-part directions and
rotation-aware speed modifiers is loaded on demand from ``common.geometry``.

Classic ship exports sometimes contain crew rooms that are effectively isolated
from the walkable traversal graph because legacy door placement rules were more
permissive. Proxy discovery via structural touching edges remains available in
shared helper code, but Layer 1 currently keeps it disabled so crew-access
edges always stay within the crew room's own traversable cluster. If no
in-cluster target exists, the crew room is skipped.
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
    find_proxy_part,
    layer1_part_sets,
    min_distance_for_part,
)

__all__ = ["Layer1CrewAccessPass"]


_ENABLE_CREW_ROOM_PROXY_FALLBACK = False


class Layer1CrewAccessPass(ExpansionPass):
    """Emit Layer 1 hierarchical crew-access edges."""

    name = "crew_access_layer1"
    version = 2
    requires = ("base_indexes", "traversable_clusters")
    provides = ()

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Compute and emit crew->reactor / crew->factory access relations."""

        node_by_id: Mapping[int, Mapping[str, Any]] = context.caches.get("node_by_id") or {}
        door_edges: Sequence[Mapping[str, Any]] = context.caches.get("door_edges") or []
        touching_edges: Sequence[Mapping[str, Any]] = context.caches.get("touching_edges") or []
        traversable_clusters: Sequence[Sequence[int]] = context.get_annotation("traversable_clusters") or []
        cluster_by_part_id: Mapping[int, int] = context.get_annotation("cluster_by_part_id") or {}

        if not traversable_clusters:
            return {
                "crew_rooms": 0,
                "crew_access_reactor_edges": 0,
                "crew_access_factory_edges": 0,
            }

        part_profiles: Mapping[int, PartTravelProfile] = context.get_or_build_cache(
            "crew_access_part_profiles",
            lambda: build_part_travel_profiles(context),
        )
        walkable_part_ids: Set[int] = set(part_profiles)
        touching_adjacency = (
            context.get_or_build_cache(
                "crew_access_touching_adjacency",
                lambda: build_touching_adjacency(touching_edges),
            )
            if _ENABLE_CREW_ROOM_PROXY_FALLBACK
            else {}
        )

        crew_room_ids, reactor_ids_by_cluster, factory_ids_by_cluster = layer1_part_sets(
            node_by_id,
            walkable_part_ids,
            cluster_by_part_id,
        )

        cross_edges_out: List[Dict[str, Any]] = []
        reactor_edge_count = 0
        factory_edge_count = 0

        target_kind_specs = (
            ("crew_access_factory", factory_ids_by_cluster),
            ("crew_access_reactor", reactor_ids_by_cluster),
        )
        for crew_room_id in crew_room_ids:
            source_cluster_index = cluster_by_part_id.get(crew_room_id)
            if source_cluster_index is None:
                continue
            for edge_kind, target_ids_by_cluster in target_kind_specs:
                path_cluster_index = int(source_cluster_index)
                source_part_id_for_path = crew_room_id
                via_proxy = False
                proxy_part_id: int | None = None
                proxy_touching_hops: int | None = None
                candidate_target_ids = target_ids_by_cluster.get(path_cluster_index, [])

                if not candidate_target_ids:
                    if not _ENABLE_CREW_ROOM_PROXY_FALLBACK:
                        continue
                    target_clusters = set(target_ids_by_cluster)
                    if not target_clusters:
                        continue
                    proxy = find_proxy_part(
                        crew_part_id=crew_room_id,
                        target_clusters=target_clusters,
                        touching_adjacency=touching_adjacency,
                        cluster_by_part_id=cluster_by_part_id,
                        walkable_part_ids=walkable_part_ids,
                    )
                    if proxy is None:
                        continue
                    proxy_part_id, proxy_touching_hops, path_cluster_index = proxy
                    candidate_target_ids = target_ids_by_cluster.get(path_cluster_index, [])
                    if not candidate_target_ids:
                        continue
                    source_part_id_for_path = proxy_part_id
                    via_proxy = True

                graph = cluster_graph(
                    context,
                    cluster_index=path_cluster_index,
                    traversable_clusters=traversable_clusters,
                    part_profiles=part_profiles,
                    door_edges=door_edges,
                )
                source_states = graph.states_by_part_id.get(source_part_id_for_path, ())
                if not source_states:
                    continue

                for target_part_id in candidate_target_ids:
                    distance_map = distances_to_target(
                        context,
                        cluster_index=path_cluster_index,
                        target_part_id=target_part_id,
                        traversable_clusters=traversable_clusters,
                        part_profiles=part_profiles,
                        door_edges=door_edges,
                    )
                    travel_distance = min_distance_for_part(distance_map, source_states)
                    if travel_distance is None:
                        continue
                    cross_edges_out.append(
                        {
                            "source": crew_room_id,
                            "source_graph": STRUCTURAL_GRAPH_NAME,
                            "target": target_part_id,
                            "target_graph": STRUCTURAL_GRAPH_NAME,
                            "kind": edge_kind,
                            "travel_distance": travel_distance,
                            "distance_unit": "movement_cost",
                            "path_model": "dijkstra_cardinal_cell_entry_cost_v1",
                            "cluster_id": f"traversable_cluster_{path_cluster_index}",
                            "via_proxy": via_proxy,
                            "proxy_part_id": proxy_part_id,
                            "proxy_touching_hops": proxy_touching_hops,
                        }
                    )
                    if edge_kind == "crew_access_reactor":
                        reactor_edge_count += 1
                    else:
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
            crew_access_edges=len(cross_edges_out),
            crew_access_reactor_edges=reactor_edge_count,
            crew_access_factory_edges=factory_edge_count,
        )
        return {
            "crew_rooms": len(crew_room_ids),
            "crew_access_reactor_edges": reactor_edge_count,
            "crew_access_factory_edges": factory_edge_count,
        }

