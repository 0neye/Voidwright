"""Targeted tests for graph_expansion framework primitives and passes."""

from __future__ import annotations

from typing import Any

from graph_expansion.context import EXPANSION_GRAPH_NAME, STRUCTURAL_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base_indexes import BaseIndexesPass
from graph_expansion.passes.global_virtual_linker import GLOBAL_SHIP_NODE_ID
from graph_expansion.passes.hull_perimeter import HullPerimeterPass
from graph_expansion.passes.spatial_zones import SpatialZonesPass, SpatialZonesRotatedPass, ZONE_NAMES, ZONE_NAMES_ROTATED
from graph_expansion.passes.traversable_clusters import TraversableClustersPass
from graph_expansion.passes.crew_access_layer1 import Layer1CrewAccessPass, _ENABLE_CREW_ROOM_PROXY_FALLBACK
from graph_expansion.passes.core_support_layer2 import Layer2CoreSupportPass
from graph_expansion.passes.weapon_groups import WeaponGroupsPass, WEAPON_TYPE_SUBSTRINGS
from graph_expansion.passes.global_virtual_linker import GlobalVirtualLinkerPass

__all__: list[str] = []


_EXPANSION_GRAPH_NAME = EXPANSION_GRAPH_NAME
_STRUCTURAL_GRAPH_NAME = STRUCTURAL_GRAPH_NAME
_CORRIDOR_ID = "cosmoteer.corridor"
_GENERIC_ID = "cosmoteer.reactor_small"


def make_node(
    node_id: int,
    walkable_cells: list[list[int]] | None = None,
    part_id: str = "",
) -> dict[str, Any]:
    """Build a minimal structural node dict."""

    node: dict[str, Any] = {"id": node_id, "part_id": part_id}
    if walkable_cells is not None:
        node["walkable_cells_2x"] = walkable_cells
    return node


def make_edge(source: int, target: int, kind: str) -> dict[str, Any]:
    """Build a minimal structural edge dict."""

    return {"source": source, "target": target, "kind": kind}



def make_graph_payload(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]] | None = None,
    *,
    ship: dict[str, Any] | None = None,
    extra_graphs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal graph JSON payload."""

    graphs = {
        _STRUCTURAL_GRAPH_NAME: {
            "nodes": nodes,
            "edges": edges if edges is not None else [],
        }
    }
    if extra_graphs is not None:
        graphs.update(extra_graphs)

    payload: dict[str, Any] = {"graphs": graphs}
    if ship is not None:
        payload["ship"] = ship
    return payload



def test_expansion_context_get_or_build_cache_builds_once() -> None:
    """get_or_build_cache should call its builder at most once per key."""

    context = ExpansionContext(make_graph_payload([]), expansion_name="structural", expansion_version=2)
    build_calls: list[str] = []

    def build_value() -> list[str]:
        build_calls.append("built")
        return ["value"]

    first = context.get_or_build_cache("demo", build_value)
    second = context.get_or_build_cache("demo", build_value)

    assert first == ["value"]
    assert second is first
    assert build_calls == ["built"]



def test_expansion_context_ensure_emitted_graph_initializes_and_reuses_defaults() -> None:
    """ensure_emitted_graph should create and then reuse the same graph container."""

    context = ExpansionContext(make_graph_payload([]), expansion_name="structural", expansion_version=2)

    graph = context.ensure_emitted_graph(_EXPANSION_GRAPH_NAME)
    graph["nodes"].append({"id": "n0"})

    same_graph = context.ensure_emitted_graph(_EXPANSION_GRAPH_NAME)

    assert same_graph is graph
    assert same_graph == {
        "nodes": [{"id": "n0"}],
        "edges": [],
        "cross_edges": [],
        "summary": {},
    }



def test_expansion_context_finalize_sorts_graphs_added_preserves_pass_order_and_does_not_mutate_source() -> None:
    """finalize should keep source graphs intact and emit deterministic metadata."""

    payload = make_graph_payload(
        [make_node(0)],
        extra_graphs={"Z_existing": {"nodes": [], "edges": []}},
    )
    original_graph_keys = list(payload["graphs"].keys())

    context = ExpansionContext(payload, expansion_name="structural", expansion_version=2)
    context.ensure_emitted_graph("Z_pass_graph")
    context.ensure_emitted_graph("A_pass_graph")
    context.add_pass_report("second_pass", 2, {"ok": True})
    context.add_pass_report("first_pass", 1, {"ok": True})

    enriched = context.finalize()

    assert list(payload["graphs"].keys()) == original_graph_keys
    assert list(enriched["graphs"].keys()) == [
        _STRUCTURAL_GRAPH_NAME,
        "Z_existing",
        "A_pass_graph",
        "Z_pass_graph",
    ]
    assert enriched["expansion"]["graphs_added"] == ["A_pass_graph", "Z_pass_graph"]
    assert enriched["expansion"]["passes"] == [
        {"name": "second_pass", "version": 2},
        {"name": "first_pass", "version": 1},
    ]



def test_base_indexes_pass_populates_expected_caches_and_summary() -> None:
    """BaseIndexesPass should cache core structural lookups exactly once."""

    nodes = [
        make_node(0, [[0, 0]], _CORRIDOR_ID),
        make_node(1, part_id=_GENERIC_ID),
        make_node(2, [[10, 10]], _GENERIC_ID),
    ]
    edges = [
        make_edge(0, 1, "door"),
        make_edge(1, 2, "touching"),
        make_edge(2, 0, "support"),
    ]
    context = ExpansionContext(make_graph_payload(nodes, edges), expansion_name="structural", expansion_version=2)

    summary = BaseIndexesPass().run(context)

    assert summary == {
        "node_count": 3,
        "edge_count": 3,
        "walkable_parts": 2,
        "door_edges": 1,
    }
    assert context.caches["structural_nodes"] == nodes
    assert context.caches["structural_edges"] == edges
    assert context.caches["node_by_id"] == {0: nodes[0], 1: nodes[1], 2: nodes[2]}
    assert context.caches["walkable_part_ids"] == {0, 2}
    assert context.caches["door_edges"] == [edges[0]]
    assert context.caches["touching_edges"] == [edges[1]]



def test_global_virtual_linker_emits_node_and_no_edges_when_run_alone() -> None:
    """GlobalVirtualLinkerPass run alone (no prior virtual passes) emits the
    global_ship node and zero linker edges."""

    nodes = [make_node(10, part_id=_GENERIC_ID), make_node(20, part_id=_CORRIDOR_ID)]
    ship = {"Name": "TestShip", "Crew": 4}
    context = ExpansionContext(
        make_graph_payload(nodes, ship=ship),
        expansion_name="structural",
        expansion_version=2,
    )
    BaseIndexesPass().run(context)

    summary = GlobalVirtualLinkerPass().run(context)
    expansion_graph = context.emitted_graphs[_EXPANSION_GRAPH_NAME]

    assert summary == {"global_nodes": 1, "global_virtual_member_edges": 0}
    global_node = expansion_graph["nodes"][0]
    assert global_node["id"] == GLOBAL_SHIP_NODE_ID
    assert global_node["kind"] == "global_ship_info"
    assert global_node["ship"] == ship
    assert global_node["total_parts"] == 2
    assert global_node["occupied_cells"] == 0  # make_node nodes have no footprint
    assert global_node["cluster_count"] == 0
    assert global_node["thermal_count"] == 0
    assert expansion_graph["cross_edges"] == []
    assert expansion_graph["summary"] == {"global_ship_nodes": 1, "global_virtual_member_edges": 0}



def test_traversable_clusters_pass_records_annotations_and_emits_exact_clusters() -> None:
    """TraversableClustersPass should annotate and emit exact cluster membership."""

    nodes = [
        make_node(0, [[0, 0]], _CORRIDOR_ID),
        make_node(1, [[2, 0]], _CORRIDOR_ID),
        make_node(2, [[100, 100]], _GENERIC_ID),
        make_node(3, part_id=_GENERIC_ID),
    ]
    edges = [make_edge(2, 0, "door"), make_edge(1, 3, "door"), make_edge(0, 2, "touching")]
    context = ExpansionContext(make_graph_payload(nodes, edges), expansion_name="structural", expansion_version=2)
    BaseIndexesPass().run(context)

    summary = TraversableClustersPass().run(context)
    expansion_graph = context.emitted_graphs[_EXPANSION_GRAPH_NAME]

    assert context.get_annotation("traversable_clusters") == [[0, 1, 2]]
    assert context.get_annotation("cluster_by_part_id") == {0: 0, 1: 0, 2: 0}
    assert summary == {"cluster_count": 1, "super_member_edges": 3, "filtered_small_clusters": 0}
    cluster_node = expansion_graph["nodes"][0]
    assert cluster_node["id"] == "traversable_cluster_0"
    assert cluster_node["kind"] == "traversable_cluster"
    assert cluster_node["member_count"] == 3
    # All 3 members (0, 1, 2) appear in door edges → door_count == 3
    assert cluster_node["door_count"] == 3
    # Nodes 0, 1, 2 each have 1 walkable cell
    assert cluster_node["walkable_cells_2x"] == 3
    # make_node nodes have no location_2x → centroid defaults to 0.0
    assert cluster_node["centroid_x"] == 0.0
    assert cluster_node["centroid_y"] == 0.0
    assert expansion_graph["cross_edges"] == [
        {
            "source": "traversable_cluster_0",
            "source_graph": _EXPANSION_GRAPH_NAME,
            "target": 0,
            "target_graph": _STRUCTURAL_GRAPH_NAME,
            "kind": "super_member",
        },
        {
            "source": "traversable_cluster_0",
            "source_graph": _EXPANSION_GRAPH_NAME,
            "target": 1,
            "target_graph": _STRUCTURAL_GRAPH_NAME,
            "kind": "super_member",
        },
        {
            "source": "traversable_cluster_0",
            "source_graph": _EXPANSION_GRAPH_NAME,
            "target": 2,
            "target_graph": _STRUCTURAL_GRAPH_NAME,
            "kind": "super_member",
        },
    ]
    assert expansion_graph["summary"] == {
        "traversable_clusters": 1,
        "super_member_edges": 3,
    }


# ---------------------------------------------------------------------------
# Layer1CrewAccessPass
# ---------------------------------------------------------------------------


def make_door_edge_with_cells(
    source: int,
    target: int,
    source_cell_2x: list[int],
    target_cell_2x: list[int],
) -> dict[str, Any]:
    """Build a structural door edge including portal cell coordinates."""

    return {
        "source": source,
        "target": target,
        "kind": "door",
        "source_cell_2x": source_cell_2x,
        "target_cell_2x": target_cell_2x,
    }


def test_layer1_crew_access_simple_distances_reactor_and_factory() -> None:
    """Crew access should compute weighted travel distance with deterministic output."""

    nodes = [
        make_full_node(0, [0, 0], part_id="mod.crew_quarters", walkable_cells=[[0, 0]]),
        make_full_node(1, [2, 0], part_id="cosmoteer.corridor", walkable_cells=[[2, 0], [4, 0], [4, 2]]),
        make_full_node(2, [6, 0], part_id="mod.reactor_test", walkable_cells=[[6, 0]]),
        make_full_node(3, [6, 2], part_id="mod.factory_test", walkable_cells=[[6, 2]]),
    ]
    edges = [
        make_door_edge_with_cells(0, 1, [0, 0], [2, 0]),
        make_door_edge_with_cells(1, 2, [4, 0], [6, 0]),
        make_door_edge_with_cells(1, 3, [4, 2], [6, 2]),
    ]
    context = ExpansionContext(make_graph_payload(nodes, edges), expansion_name="structural", expansion_version=4)
    BaseIndexesPass().run(context)
    TraversableClustersPass().run(context)

    summary = Layer1CrewAccessPass().run(context)
    assert summary["crew_rooms"] == 1
    assert summary["crew_access_reactor_edges"] == 1
    assert summary["crew_access_factory_edges"] == 1

    cross_edges = context.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
    crew_edges = [e for e in cross_edges if e["kind"].startswith("crew_access_")]
    assert {(e["kind"], e["target"]) for e in crew_edges} == {
        ("crew_access_reactor", 2),
        ("crew_access_factory", 3),
    }

    reactor_edge = next(e for e in crew_edges if e["kind"] == "crew_access_reactor")
    factory_edge = next(e for e in crew_edges if e["kind"] == "crew_access_factory")

    assert reactor_edge["travel_distance"] == 3.0
    assert factory_edge["travel_distance"] == 4.0
    assert reactor_edge["distance_unit"] == "movement_cost"
    assert reactor_edge["path_model"] == "dijkstra_cardinal_cell_entry_cost_v1"
    assert reactor_edge["cluster_id"] == "traversable_cluster_0"
    assert reactor_edge["via_proxy"] is False
    assert reactor_edge["proxy_part_id"] is None
    assert reactor_edge["proxy_touching_hops"] is None


def test_layer1_crew_access_conveyor_direction_changes_weighted_cost() -> None:
    """Directional conveyor speed should affect weighted travel distance."""

    def run_for_rotation(rotation: int) -> float:
        nodes = [
            make_full_node(0, [0, 0], part_id="mod.crew_quarters", walkable_cells=[[0, 0]]),
            make_full_node(
                1,
                [2, 0],
                rotation=rotation,
                part_id="cosmoteer.conveyor",
                walkable_cells=[[2, 0]],
            ),
            make_full_node(2, [4, 0], part_id="mod.reactor_test", walkable_cells=[[4, 0]]),
        ]
        edges = [
            make_door_edge_with_cells(0, 1, [0, 0], [2, 0]),
            make_door_edge_with_cells(1, 2, [2, 0], [4, 0]),
        ]
        context = ExpansionContext(
            make_graph_payload(nodes, edges),
            expansion_name="structural",
            expansion_version=4,
        )
        BaseIndexesPass().run(context)
        TraversableClustersPass().run(context)
        Layer1CrewAccessPass().run(context)
        edge = next(
            e
            for e in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
            if e["kind"] == "crew_access_reactor"
        )
        return float(edge["travel_distance"])

    assert run_for_rotation(1) < run_for_rotation(0)


def test_layer1_crew_access_blocked_internal_direction_prevents_path() -> None:
    """Per-rotation blocked travel directions must prevent illegal in-part movement."""

    nodes = [
        make_full_node(0, [-2, 2], part_id="mod.crew_quarters", walkable_cells=[[-2, 2]]),
        make_full_node(
            1,
            [0, 0],
            part_id="cosmoteer.control_room_med",
            walkable_cells=[[2, 2], [2, 4]],
        ),
        make_full_node(2, [4, 4], part_id="mod.reactor_test", walkable_cells=[[4, 4]]),
    ]
    edges = [
        make_door_edge_with_cells(0, 1, [-2, 2], [2, 2]),
        make_door_edge_with_cells(1, 2, [2, 4], [4, 4]),
    ]
    context = ExpansionContext(make_graph_payload(nodes, edges), expansion_name="structural", expansion_version=4)
    BaseIndexesPass().run(context)
    TraversableClustersPass().run(context)

    Layer1CrewAccessPass().run(context)
    cross_edges = context.emitted_graphs.get(_EXPANSION_GRAPH_NAME, {}).get("cross_edges", [])
    assert [e for e in cross_edges if e.get("kind") == "crew_access_reactor"] == []


def test_layer1_crew_access_proxy_fallback_recovers_cross_cluster_touching_path() -> None:
    """A direct touching walkable proxy should recover cross-cluster crew access."""

    nodes = [
        make_full_node(10, [0, 0], part_id="mod.crew_quarters", walkable_cells=[[0, 0]]),
        make_full_node(11, [2, 0], part_id="cosmoteer.corridor", walkable_cells=[[2, 0], [4, 0]]),
        make_full_node(12, [6, 0], part_id="mod.reactor_test", walkable_cells=[[6, 0]]),
    ]
    edges = [
        make_edge(10, 11, "touching"),
        make_door_edge_with_cells(11, 12, [4, 0], [6, 0]),
    ]

    context = ExpansionContext(make_graph_payload(nodes, edges), expansion_name="structural", expansion_version=4)
    BaseIndexesPass().run(context)
    TraversableClustersPass().run(context)

    summary = Layer1CrewAccessPass().run(context)
    cross_edges = context.emitted_graphs.get(_EXPANSION_GRAPH_NAME, {}).get("cross_edges", [])
    reactor_edges = [e for e in cross_edges if e.get("kind") == "crew_access_reactor"]

    if _ENABLE_CREW_ROOM_PROXY_FALLBACK:
        assert summary["crew_access_reactor_edges"] == 1
        assert len(reactor_edges) == 1
        assert reactor_edges[0]["source"] == 10
        assert reactor_edges[0]["target"] == 12
        assert reactor_edges[0]["via_proxy"] is True
        assert reactor_edges[0]["proxy_part_id"] == 11
        assert reactor_edges[0]["proxy_touching_hops"] == 1
    else:
        assert summary["crew_access_reactor_edges"] == 0
        assert len(reactor_edges) == 0

def test_layer1_crew_access_proxy_fallback_failure_skips_crew_room() -> None:
    """When proxy discovery fails, isolated crew rooms must not emit edges."""

    nodes = [
        make_full_node(20, [0, 0], part_id="mod.crew_quarters", walkable_cells=[[0, 0]]),
        make_full_node(21, [2, 0], part_id="cosmoteer.armor", walkable_cells=None),
        make_full_node(22, [10, 0], part_id="mod.reactor_test", walkable_cells=[[10, 0]]),
    ]
    edges = [make_edge(20, 21, "touching")]
    context = ExpansionContext(make_graph_payload(nodes, edges), expansion_name="structural", expansion_version=4)
    BaseIndexesPass().run(context)
    TraversableClustersPass().run(context)

    Layer1CrewAccessPass().run(context)
    cross_edges = context.emitted_graphs.get(_EXPANSION_GRAPH_NAME, {}).get("cross_edges", [])
    assert [
        e
        for e in cross_edges
        if e.get("source") == 20 and str(e.get("kind", "")).startswith("crew_access_")
    ] == []


def test_layer1_crew_access_proxy_fallback_does_not_follow_touching_chains() -> None:
    """Proxy fallback must stay local instead of traversing arbitrary touching chains."""

    nodes = [
        make_full_node(30, [0, 0], part_id="mod.crew_quarters", walkable_cells=[[0, 0]]),
        make_full_node(31, [2, 0], part_id="cosmoteer.armor", walkable_cells=None),
        make_full_node(32, [4, 0], part_id="cosmoteer.corridor", walkable_cells=[[4, 0], [6, 0]]),
        make_full_node(33, [8, 0], part_id="mod.reactor_test", walkable_cells=[[8, 0]]),
    ]
    edges = [
        make_edge(30, 31, "touching"),
        make_edge(31, 32, "touching"),
        make_door_edge_with_cells(32, 33, [6, 0], [8, 0]),
    ]
    context = ExpansionContext(make_graph_payload(nodes, edges), expansion_name="structural", expansion_version=4)
    BaseIndexesPass().run(context)
    TraversableClustersPass().run(context)

    Layer1CrewAccessPass().run(context)
    cross_edges = context.emitted_graphs.get(_EXPANSION_GRAPH_NAME, {}).get("cross_edges", [])
    assert [
        e
        for e in cross_edges
        if e.get("source") == 30 and str(e.get("kind", "")).startswith("crew_access_")
    ] == []


def test_layer1_crew_access_edges_emitted_in_deterministic_order() -> None:
    """Crew-access edges must be deterministic regardless of input node/edge order."""

    nodes = [
        make_full_node(5, [0, 0], part_id="mod.crew_quarters", walkable_cells=[[0, 0]]),
        make_full_node(2, [2, 0], part_id="cosmoteer.corridor", walkable_cells=[[2, 0], [4, 0], [2, 2], [4, 2]]),
        make_full_node(9, [6, 0], part_id="mod.reactor_test", walkable_cells=[[6, 0]]),
        make_full_node(7, [6, 2], part_id="mod.factory_test", walkable_cells=[[6, 2]]),
        make_full_node(3, [0, 2], part_id="mod.crew_quarters_large", walkable_cells=[[0, 2]]),
    ]
    edges = [
        make_door_edge_with_cells(2, 9, [4, 0], [6, 0]),
        make_door_edge_with_cells(2, 7, [4, 2], [6, 2]),
        make_door_edge_with_cells(3, 2, [0, 2], [2, 2]),
        make_door_edge_with_cells(5, 2, [0, 0], [2, 0]),
    ]

    context_a = ExpansionContext(
        make_graph_payload(list(reversed(nodes)), list(reversed(edges))),
        expansion_name="structural",
        expansion_version=4,
    )
    BaseIndexesPass().run(context_a)
    TraversableClustersPass().run(context_a)
    Layer1CrewAccessPass().run(context_a)
    edges_a = [
        e
        for e in context_a.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
        if e["kind"].startswith("crew_access_")
    ]

    context_b = ExpansionContext(make_graph_payload(nodes, edges), expansion_name="structural", expansion_version=4)
    BaseIndexesPass().run(context_b)
    TraversableClustersPass().run(context_b)
    Layer1CrewAccessPass().run(context_b)
    edges_b = [
        e
        for e in context_b.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
        if e["kind"].startswith("crew_access_")
    ]

    def edge_key(e: Mapping[str, Any]) -> tuple[int, str, int]:
        return (int(e["source"]), str(e["kind"]), int(e["target"]))

    assert [edge_key(e) for e in edges_a] == [edge_key(e) for e in edges_b]
    assert [edge_key(e) for e in edges_b] == sorted(edge_key(e) for e in edges_b)


# ---------------------------------------------------------------------------
# Layer2CoreSupportPass
# ---------------------------------------------------------------------------


def _run_layer2(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[ExpansionContext, dict[str, Any]]:
    """Run the Layer 1 + Layer 2 support passes on a minimal payload."""

    context = ExpansionContext(
        make_graph_payload(nodes, edges),
        expansion_name="structural",
        expansion_version=5,
    )
    BaseIndexesPass().run(context)
    TraversableClustersPass().run(context)
    Layer1CrewAccessPass().run(context)
    summary = Layer2CoreSupportPass().run(context)
    return context, summary


def test_layer2_reactor_support_edges_cover_power_storage_shield_engine_room_thruster_and_energy_weapon() -> None:
    """Reactors should connect to cluster-local infrastructure and skip engine-room-adjacent thrusters."""

    nodes = [
        make_full_node(1, [0, 0], part_id="mod.reactor_test", walkable_cells=[[0, 0]]),
        make_full_node(9, [2, 0], part_id="cosmoteer.corridor", walkable_cells=[[2, 0], [4, 0], [6, 0], [8, 0], [10, 0], [12, 0], [14, 0], [16, 0]]),
        make_full_node(2, [18, 0], part_id="mod.power_storage", walkable_cells=[[18, 0]]),
        make_full_node(3, [20, 0], part_id="mod.shield_gen", walkable_cells=[[20, 0]]),
        make_full_node(5, [22, 0], part_id="mod.engine_room", walkable_cells=[[22, 0]]),
        make_full_node(6, [24, 0], part_id="mod.thruster_med", walkable_cells=[[24, 0]]),
        make_full_node(7, [26, 0], part_id="mod.thruster_boost", walkable_cells=[[26, 0]]),
        make_full_node(4, [28, 0], part_id="mod.laser_blaster", walkable_cells=[[28, 0]]),
    ]
    edges = [
        make_door_edge_with_cells(1, 9, [0, 0], [2, 0]),
        make_door_edge_with_cells(9, 2, [16, 0], [18, 0]),
        make_door_edge_with_cells(2, 3, [18, 0], [20, 0]),
        make_door_edge_with_cells(3, 5, [20, 0], [22, 0]),
        make_door_edge_with_cells(5, 6, [22, 0], [24, 0]),
        make_door_edge_with_cells(6, 7, [24, 0], [26, 0]),
        make_door_edge_with_cells(7, 4, [26, 0], [28, 0]),
        make_edge(5, 6, "touching"),
    ]

    context, summary = _run_layer2(nodes, edges)
    assert summary["reactor_support_edges"] == 5
    reactor_edges = [
        e
        for e in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
        if e["kind"].startswith("reactor_supports_")
    ]
    assert {(e["kind"], e["target"]) for e in reactor_edges} == {
        ("reactor_supports_power_storage", 2),
        ("reactor_supports_shield", 3),
        ("reactor_supports_engine_room", 5),
        ("reactor_supports_thruster", 7),
        ("reactor_supports_energy_weapon", 4),
    }
    assert all(e["travel_distance"] > 0 for e in reactor_edges)


def test_layer2_factory_support_edges_respect_factory_mode() -> None:
    """Ammo factories should target ammo weapons, missile factories missile weapons, and all factories storage."""

    nodes = [
        make_full_node(10, [0, 0], part_id="cosmoteer.factory_ammo", walkable_cells=[[0, 0]]),
        make_full_node(11, [0, 2], part_id="cosmoteer.factory_he", walkable_cells=[[0, 2]]),
        make_full_node(12, [8, 0], part_id="mod.storage", walkable_cells=[[8, 0]]),
        make_full_node(13, [8, 2], part_id="cosmoteer.cannon_med", walkable_cells=[[8, 2]]),
        make_full_node(14, [8, 4], part_id="cosmoteer.missile_launcher", walkable_cells=[[8, 4]]),
        make_full_node(15, [0, 4], part_id="cosmoteer.factory_steel", walkable_cells=[[0, 4]]),
        make_full_node(16, [2, 0], part_id="cosmoteer.corridor", walkable_cells=[[2, 0], [4, 0], [6, 0]]),
        make_full_node(17, [2, 2], part_id="cosmoteer.corridor", walkable_cells=[[2, 2], [4, 2], [6, 2]]),
        make_full_node(18, [2, 4], part_id="cosmoteer.corridor", walkable_cells=[[2, 4], [4, 4], [6, 4]]),
    ]
    edges = [
        make_door_edge_with_cells(10, 16, [0, 0], [2, 0]),
        make_door_edge_with_cells(11, 17, [0, 2], [2, 2]),
        make_door_edge_with_cells(15, 18, [0, 4], [2, 4]),
        make_door_edge_with_cells(16, 12, [6, 0], [8, 0]),
        make_door_edge_with_cells(17, 13, [6, 2], [8, 2]),
        make_door_edge_with_cells(18, 14, [6, 4], [8, 4]),
        make_door_edge_with_cells(16, 17, [4, 0], [4, 2]),
        make_door_edge_with_cells(17, 18, [4, 2], [4, 4]),
    ]

    context, summary = _run_layer2(nodes, edges)
    assert summary["factory_support_edges"] == 5
    factory_edges = [
        e
        for e in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
        if e["kind"].startswith("factory_supports_")
    ]
    assert {(e["source"], e["kind"], e["target"]) for e in factory_edges} == {
        (10, "factory_supports_storage", 12),
        (10, "factory_supports_ammo_weapon", 13),
        (11, "factory_supports_storage", 12),
        (11, "factory_supports_missile_weapon", 14),
        (15, "factory_supports_storage", 12),
    }


def test_layer2_support_edges_are_deterministic() -> None:
    """Layer 2 support edges must not depend on input ordering."""

    nodes = [
        make_full_node(1, [0, 0], part_id="mod.reactor_test", walkable_cells=[[0, 0]]),
        make_full_node(2, [2, 0], part_id="cosmoteer.power_storage", walkable_cells=[[2, 0]]),
        make_full_node(3, [4, 0], part_id="cosmoteer.shield_gen_small", walkable_cells=[[4, 0]]),
        make_full_node(4, [0, 2], part_id="cosmoteer.factory_ammo", walkable_cells=[[0, 2]]),
        make_full_node(5, [2, 2], part_id="cosmoteer.storage_3x2", walkable_cells=[[2, 2]]),
        make_full_node(6, [4, 2], part_id="cosmoteer.cannon_med", walkable_cells=[[4, 2]]),
    ]
    edges = [
        make_door_edge_with_cells(1, 2, [0, 0], [2, 0]),
        make_door_edge_with_cells(2, 3, [2, 0], [4, 0]),
        make_door_edge_with_cells(4, 5, [0, 2], [2, 2]),
        make_door_edge_with_cells(5, 6, [2, 2], [4, 2]),
        make_door_edge_with_cells(2, 5, [2, 0], [2, 2]),
    ]

    context_a, _ = _run_layer2(list(reversed(nodes)), list(reversed(edges)))
    context_b, _ = _run_layer2(nodes, edges)

    def edge_key(e: Mapping[str, Any]) -> tuple[int, str, int]:
        return (int(e["source"]), str(e["kind"]), int(e["target"]))

    edges_a = [
        e for e in context_a.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
        if e["kind"].startswith("reactor_supports_") or e["kind"].startswith("factory_supports_")
    ]
    edges_b = [
        e for e in context_b.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
        if e["kind"].startswith("reactor_supports_") or e["kind"].startswith("factory_supports_")
    ]
    assert [edge_key(e) for e in edges_a] == [edge_key(e) for e in edges_b]
    assert [edge_key(e) for e in edges_b] == sorted(edge_key(e) for e in edges_b)


# ---------------------------------------------------------------------------
# Helpers for passes that require location_2x and footprint
# ---------------------------------------------------------------------------


def make_full_node(
    node_id: int,
    location_2x: list[int],
    *,
    width: int = 1,
    height: int = 1,
    rotation: int = 0,
    part_id: str = "cosmoteer.reactor_small",
    walkable_cells: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Build a structural node that includes location_2x and footprint metadata."""

    node: dict[str, Any] = {
        "id": node_id,
        "part_id": part_id,
        "location_2x": location_2x,
        "rotation": rotation,
        "footprint": {"width": width, "height": height, "cell_count": width * height},
    }
    if walkable_cells is not None:
        node["walkable_cells_2x"] = walkable_cells
    return node


def _run_hull_perimeter(nodes: list[dict[str, Any]]) -> tuple[ExpansionContext, dict[str, Any]]:
    """Run BaseIndexesPass then HullPerimeterPass and return the context and summary."""

    context = ExpansionContext(
        make_graph_payload(nodes), expansion_name="structural", expansion_version=3
    )
    BaseIndexesPass().run(context)
    summary = HullPerimeterPass().run(context)
    return context, summary


def _run_spatial_zones(nodes: list[dict[str, Any]]) -> tuple[ExpansionContext, dict[str, Any]]:
    """Run BaseIndexesPass then SpatialZonesPass and return the context and summary."""

    context = ExpansionContext(
        make_graph_payload(nodes), expansion_name="structural", expansion_version=3
    )
    BaseIndexesPass().run(context)
    summary = SpatialZonesPass().run(context)
    return context, summary


def _run_weapon_groups(nodes: list[dict[str, Any]]) -> tuple[ExpansionContext, dict[str, Any]]:
    """Run BaseIndexesPass then WeaponGroupsPass and return the context and summary."""

    context = ExpansionContext(
        make_graph_payload(nodes), expansion_name="structural", expansion_version=3
    )
    BaseIndexesPass().run(context)
    summary = WeaponGroupsPass().run(context)
    return context, summary


# ---------------------------------------------------------------------------
# HullPerimeterPass
# ---------------------------------------------------------------------------


def test_hull_perimeter_single_node_is_perimeter() -> None:
    """A lone part with no occupied neighbors must be classified as perimeter."""

    nodes = [make_full_node(0, [0, 0])]
    context, summary = _run_hull_perimeter(nodes)
    assert context.get_annotation("hull_role_by_part_id") == {0: "perimeter"}
    assert summary == {"hull_perimeter_parts": 1, "interior_parts": 0}


def test_hull_perimeter_inner_part_of_3x3_grid_is_interior() -> None:
    """In a fully packed 3×3 grid of 1×1 parts the center node is interior."""

    # 3×3 grid: parts at 2x positions [0,0],[2,0],[4,0] / [0,2],[2,2],[4,2] / [0,4],[2,4],[4,4]
    coords = [(col * 2, row * 2) for row in range(3) for col in range(3)]
    nodes = [make_full_node(i, list(c)) for i, c in enumerate(coords)]
    # Node 4 is at (2,2) — center of the grid.
    context, summary = _run_hull_perimeter(nodes)
    role = context.get_annotation("hull_role_by_part_id")

    assert role[4] == "interior"
    # All eight surrounding nodes must be perimeter.
    for nid in range(9):
        if nid != 4:
            assert role[nid] == "perimeter", f"expected node {nid} to be perimeter"
    assert summary["hull_perimeter_parts"] == 8
    assert summary["interior_parts"] == 1


def test_hull_perimeter_emits_both_virtual_nodes() -> None:
    """HullPerimeterPass must always emit both hull_perimeter and interior virtual nodes."""

    nodes = [make_full_node(0, [0, 0])]
    context, _ = _run_hull_perimeter(nodes)
    emitted_ids = {n["id"] for n in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]}
    assert "hull_perimeter" in emitted_ids
    assert "interior" in emitted_ids


def test_hull_perimeter_virtual_node_kinds() -> None:
    """hull_perimeter node must have kind 'hull_perimeter'; interior node kind 'hull_interior'."""

    nodes = [make_full_node(0, [0, 0])]
    context, _ = _run_hull_perimeter(nodes)
    kind_by_id = {
        n["id"]: n["kind"]
        for n in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]
    }
    assert kind_by_id["hull_perimeter"] == "hull_perimeter"
    assert kind_by_id["interior"] == "hull_interior"


def test_hull_perimeter_cross_edge_kinds() -> None:
    """Perimeter members use hull_member edges; interior members use interior_member edges."""

    # 3-part row: node 0 and 2 are perimeter, node 1 is sandwiched in x but exposed in y.
    # Use a 3-part row: [0,0], [2,0], [4,0] — all perimeter (none sandwiched in both axes).
    # To get an interior node we need a 3×3 grid. Reuse: center node 4.
    coords = [(col * 2, row * 2) for row in range(3) for col in range(3)]
    nodes = [make_full_node(i, list(c)) for i, c in enumerate(coords)]
    context, _ = _run_hull_perimeter(nodes)
    cross_edges = context.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]

    perimeter_targets = {e["target"] for e in cross_edges if e["kind"] == "hull_member"}
    interior_targets = {e["target"] for e in cross_edges if e["kind"] == "interior_member"}
    assert 4 in interior_targets
    assert 4 not in perimeter_targets
    assert perimeter_targets == {0, 1, 2, 3, 5, 6, 7, 8}


def test_hull_perimeter_node_missing_location_2x_treated_as_interior() -> None:
    """Nodes without location_2x or footprint must not crash and default to interior."""

    bare_node: dict[str, Any] = {"id": 0, "part_id": "cosmoteer.armor"}
    context, summary = _run_hull_perimeter([bare_node])
    role = context.get_annotation("hull_role_by_part_id")
    assert role.get(0) == "interior"
    assert summary["interior_parts"] == 1


def test_hull_perimeter_summary_updates_expansion_graph_summary() -> None:
    """hull_perimeter_parts and interior_parts keys must appear in the graph summary."""

    nodes = [make_full_node(0, [0, 0]), make_full_node(1, [2, 0])]
    context, _ = _run_hull_perimeter(nodes)
    graph_summary = context.emitted_graphs[_EXPANSION_GRAPH_NAME]["summary"]
    assert "hull_perimeter_parts" in graph_summary
    assert "interior_parts" in graph_summary
    assert graph_summary["hull_perimeter_parts"] + graph_summary["interior_parts"] == 2


def test_hull_perimeter_rotation_swaps_footprint_dimensions() -> None:
    """Preprocessing stores rotation-specific dimensions; the pass reads them directly."""

    # 1×2 part at [0,0] rotation=0: preprocessing stores width=1, height=2.
    # Cells: (0,0) and (0,2). Neighbor to the right of (0,0) is (2,0) — unoccupied → perimeter.
    node = make_full_node(0, [0, 0], width=1, height=2, rotation=0)
    context, summary = _run_hull_perimeter([node])
    assert context.get_annotation("hull_role_by_part_id")[0] == "perimeter"

    # Same physical part rotated 90° (rotation=1): preprocessing stores width=2, height=1.
    # Cells: (0,0) and (2,0). No dimension swap is applied by the pass — the stored
    # values are already rotation-specific.
    node_r = make_full_node(1, [0, 0], width=2, height=1, rotation=1)
    context_r, _ = _run_hull_perimeter([node_r])
    assert context_r.get_annotation("hull_role_by_part_id")[1] == "perimeter"


# ---------------------------------------------------------------------------
# SpatialZonesPass
# ---------------------------------------------------------------------------


def test_spatial_zones_all_eight_directions() -> None:
    """Each compass direction must produce the correct zone for a 1×1 part."""

    # (location_2x, expected_zone) for a 1×1 part well within a single sector.
    cases: list[tuple[list[int], str]] = [
        ([4, 0], "zone_e"),
        ([4, 4], "zone_ne"),
        ([0, 4], "zone_n"),
        ([-4, 4], "zone_nw"),
        ([-4, 0], "zone_w"),
        ([-4, -4], "zone_sw"),
        ([0, -4], "zone_s"),
        ([4, -4], "zone_se"),
    ]
    for node_id, (loc, expected_zone) in enumerate(cases):
        nodes = [make_full_node(node_id, loc)]
        context, _ = _run_spatial_zones(nodes)
        annotation = context.get_annotation("zone_by_part_id")
        assert annotation[node_id] == [expected_zone], (
            f"location_2x={loc} expected [{expected_zone}], got {annotation[node_id]}"
        )


def test_spatial_zones_only_populated_zones_emitted() -> None:
    """Zone virtual nodes must not be emitted for zones with no members."""

    # Two parts both in zone_e.
    nodes = [make_full_node(0, [4, 0]), make_full_node(1, [8, 0])]
    context, summary = _run_spatial_zones(nodes)
    emitted_ids = {n["id"] for n in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]}
    assert emitted_ids == {"zone_e"}
    assert summary["spatial_zone_nodes"] == 1
    assert summary["zone_member_edges"] == 2


def test_spatial_zones_emitted_in_zone_names_order() -> None:
    """Zone nodes must be emitted in the canonical ZONE_NAMES order."""

    # Populate zone_se and zone_e (out of ZONE_NAMES order).
    nodes = [make_full_node(0, [4, -4]), make_full_node(1, [4, 0])]
    context, _ = _run_spatial_zones(nodes)
    emitted = [
        n["id"]
        for n in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]
    ]
    # zone_e is index 0, zone_se is index 7 — so zone_e must come first.
    assert emitted.index("zone_e") < emitted.index("zone_se")


def test_spatial_zones_annotation_maps_all_nodes() -> None:
    """zone_by_part_id must contain an entry for every structural node."""

    nodes = [
        make_full_node(0, [4, 0]),
        make_full_node(1, [0, 4]),
        make_full_node(2, [-4, 0]),
    ]
    context, _ = _run_spatial_zones(nodes)
    annotation = context.get_annotation("zone_by_part_id")
    assert set(annotation.keys()) == {0, 1, 2}
    assert annotation[0] == ["zone_e"]
    assert annotation[1] == ["zone_n"]
    assert annotation[2] == ["zone_w"]


def test_spatial_zones_cross_edges_have_zone_member_kind() -> None:
    """All cross-edges emitted by SpatialZonesPass must have kind 'zone_member'."""

    nodes = [make_full_node(0, [4, 0]), make_full_node(1, [0, 4])]
    context, _ = _run_spatial_zones(nodes)
    cross_edges = context.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
    assert all(e["kind"] == "zone_member" for e in cross_edges)


def test_spatial_zones_mirror_pair_lands_in_opposite_zones() -> None:
    """A part at +x and its mirror at -x must be in zone_e and zone_w respectively."""

    # Mirrors across x = -0.5 in 1x == x = -1 in 2x.
    # Part at location_2x=[4,0] → centroid (4,0) → zone_e.
    # Mirror part at location_2x=[-6,0] → centroid (-6,0) → zone_w.
    nodes = [make_full_node(0, [4, 0]), make_full_node(1, [-6, 0])]
    context, _ = _run_spatial_zones(nodes)
    annotation = context.get_annotation("zone_by_part_id")
    assert annotation[0] == ["zone_e"]
    assert annotation[1] == ["zone_w"]


def test_spatial_zones_node_missing_location_2x_falls_back_to_zone_e() -> None:
    """A node without location_2x must not raise and must fall back to zone_e."""

    bare: dict[str, Any] = {"id": 0, "part_id": "cosmoteer.armor"}
    context, summary = _run_spatial_zones([bare])
    annotation = context.get_annotation("zone_by_part_id")
    assert annotation[0] == ["zone_e"]
    assert summary["spatial_zone_nodes"] == 1


def test_spatial_zones_large_part_footprint_cells() -> None:
    """Zone assignment iterates all footprint cells; straddling parts land in multiple zones."""

    # 3×1 part at location_2x=[0,0]: cells (0,0), (1,0), (2,0) — all at y=0 → zone_e only.
    node = make_full_node(0, [0, 0], width=3, height=1)
    context, _ = _run_spatial_zones([node])
    assert context.get_annotation("zone_by_part_id")[0] == ["zone_e"]

    # Same part rotated 90°: preprocessing stores already-rotated dims width=1, height=3.
    # Cells: (0,0) → zone_e; (0,1) → zone_n; (0,2) → zone_n.
    # The part straddles the E/N boundary so it is assigned to both zones.
    node_r = make_full_node(1, [0, 0], width=1, height=3, rotation=1)
    context_r, _ = _run_spatial_zones([node_r])
    assert context_r.get_annotation("zone_by_part_id")[1] == ["zone_e", "zone_n"]


# ---------------------------------------------------------------------------
# SpatialZonesRotatedPass
# ---------------------------------------------------------------------------


def _run_spatial_zones_rotated(nodes: list[dict[str, Any]]) -> tuple[ExpansionContext, dict[str, Any]]:
    """Run BaseIndexesPass then SpatialZonesRotatedPass and return the context and summary."""

    context = ExpansionContext(
        make_graph_payload(nodes), expansion_name="structural", expansion_version=3
    )
    BaseIndexesPass().run(context)
    summary = SpatialZonesRotatedPass().run(context)
    return context, summary


def test_spatial_zones_rotated_all_eight_directions() -> None:
    """Each interstitial compass direction must produce the correct rotated zone for a 1×1 part."""

    # (location_2x, expected_zone) for a 1×1 part (centroid == location_2x).
    # Rotated zones are centred at 22.5°, 67.5°, 112.5°, … so we test points
    # that lie clearly within each sector.
    cases: list[tuple[list[int], str]] = [
        ([4, 2], "zone_ene"),   # ~26.6° — between E and NE
        ([2, 4], "zone_nne"),   # ~63.4° — between NE and N
        ([-2, 4], "zone_nnw"),  # ~116.6° — between N and NW
        ([-4, 2], "zone_wnw"),  # ~153.4° — between NW and W
        ([-4, -2], "zone_wsw"), # ~206.6° — between W and SW
        ([-2, -4], "zone_ssw"), # ~243.4° — between SW and S
        ([2, -4], "zone_sse"),  # ~296.6° — between S and SE
        ([4, -2], "zone_ese"),  # ~333.4° — between SE and E
    ]
    for node_id, (loc, expected_zone) in enumerate(cases):
        nodes = [make_full_node(node_id, loc)]
        context, _ = _run_spatial_zones_rotated(nodes)
        annotation = context.get_annotation("rotated_zone_by_part_id")
        assert annotation[node_id] == [expected_zone], (
            f"location_2x={loc} expected [{expected_zone}], got {annotation[node_id]}"
        )


def test_spatial_zones_rotated_only_populated_zones_emitted() -> None:
    """Rotated zone virtual nodes must not be emitted for zones with no members."""

    nodes = [make_full_node(0, [4, 2]), make_full_node(1, [8, 3])]
    context, summary = _run_spatial_zones_rotated(nodes)
    emitted_ids = {n["id"] for n in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]}
    assert emitted_ids == {"zone_ene"}
    assert summary["spatial_zone_rotated_nodes"] == 1
    assert summary["zone_member_rotated_edges"] == 2


def test_spatial_zones_rotated_emitted_in_zone_names_rotated_order() -> None:
    """Rotated zone nodes must be emitted in ZONE_NAMES_ROTATED order."""

    # Populate zone_ese (last) and zone_ene (first) out of order.
    nodes = [make_full_node(0, [4, -2]), make_full_node(1, [4, 2])]
    context, _ = _run_spatial_zones_rotated(nodes)
    emitted = [
        n["id"]
        for n in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]
    ]
    assert emitted.index("zone_ene") < emitted.index("zone_ese")


def test_spatial_zones_rotated_annotation_maps_all_nodes() -> None:
    """rotated_zone_by_part_id must contain an entry for every structural node."""

    nodes = [
        make_full_node(0, [4, 2]),
        make_full_node(1, [-4, 2]),
        make_full_node(2, [-4, -2]),
    ]
    context, _ = _run_spatial_zones_rotated(nodes)
    annotation = context.get_annotation("rotated_zone_by_part_id")
    assert set(annotation.keys()) == {0, 1, 2}
    assert annotation[0] == ["zone_ene"]
    assert annotation[1] == ["zone_wnw"]
    assert annotation[2] == ["zone_wsw"]


def test_spatial_zones_rotated_cross_edges_have_zone_member_rotated_kind() -> None:
    """All cross-edges from SpatialZonesRotatedPass must have kind 'zone_member_rotated'."""

    nodes = [make_full_node(0, [4, 2]), make_full_node(1, [-4, 2])]
    context, _ = _run_spatial_zones_rotated(nodes)
    cross_edges = context.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
    assert all(e["kind"] == "zone_member_rotated" for e in cross_edges)


def test_spatial_zones_rotated_node_missing_location_2x_falls_back_to_zone_ene() -> None:
    """A node without location_2x must not raise and must fall back to zone_ene."""

    bare: dict[str, Any] = {"id": 0, "part_id": "cosmoteer.armor"}
    context, summary = _run_spatial_zones_rotated([bare])
    annotation = context.get_annotation("rotated_zone_by_part_id")
    assert annotation[0] == ["zone_ene"]
    assert summary["spatial_zone_rotated_nodes"] == 1


def test_spatial_zones_rotated_distinct_from_unrotated() -> None:
    """Running both passes on the same context must not produce duplicate zone node IDs."""

    nodes = [make_full_node(0, [4, 0]), make_full_node(1, [0, 4])]
    context = ExpansionContext(
        make_graph_payload(nodes), expansion_name="structural", expansion_version=3
    )
    BaseIndexesPass().run(context)
    SpatialZonesPass().run(context)
    SpatialZonesRotatedPass().run(context)
    emitted_ids = [n["id"] for n in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]]
    # No duplicate IDs — rotated names (zone_ene, …) differ from cardinal names (zone_e, …)
    assert len(emitted_ids) == len(set(emitted_ids))
    # Both sets present
    assert any(nid in ZONE_NAMES for nid in emitted_ids)
    assert any(nid in ZONE_NAMES_ROTATED for nid in emitted_ids)


def test_spatial_zones_straddling_part_gets_multiple_zone_edges() -> None:
    """A part whose footprint crosses a zone boundary must appear in all touched zones."""

    # 1×3 part at [0,0]: cells (0,0)→zone_e, (0,1)→zone_n, (0,2)→zone_n.
    # Straddles the E/N boundary → must be a member of both zone_e and zone_n.
    node = make_full_node(0, [0, 0], width=1, height=3)
    context, summary = _run_spatial_zones([node])

    annotation = context.get_annotation("zone_by_part_id")
    assert annotation[0] == ["zone_e", "zone_n"]

    emitted_ids = {n["id"] for n in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]}
    assert "zone_e" in emitted_ids
    assert "zone_n" in emitted_ids
    assert summary["spatial_zone_nodes"] == 2

    cross_edges = context.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
    edge_sources = [e["source"] for e in cross_edges]
    assert edge_sources.count("zone_e") == 1
    assert edge_sources.count("zone_n") == 1
    assert summary["zone_member_edges"] == 2


# ---------------------------------------------------------------------------
# WeaponGroupsPass
# ---------------------------------------------------------------------------


def test_weapon_groups_cannon_part_detected() -> None:
    """A part with 'cannon' in its ID must produce a weapon_group_cannon node."""

    nodes = [make_full_node(0, [0, 0], part_id="cosmoteer.cannon_med")]
    context, summary = _run_weapon_groups(nodes)
    emitted_ids = {n["id"] for n in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]}
    assert "weapon_group_cannon" in emitted_ids
    assert summary == {"weapon_group_nodes": 1, "weapon_member_edges": 1}


def test_weapon_groups_non_weapon_part_emits_nothing() -> None:
    """A non-weapon part must not produce any weapon_group nodes or edges."""

    nodes = [make_full_node(0, [0, 0], part_id="cosmoteer.reactor_small")]
    context, summary = _run_weapon_groups(nodes)
    emitted_ids = {n["id"] for n in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]}
    assert not any(nid.startswith("weapon_group_") for nid in emitted_ids)
    assert summary == {"weapon_group_nodes": 0, "weapon_member_edges": 0}


def test_weapon_groups_multiple_types_emit_separate_group_nodes() -> None:
    """Distinct weapon types must produce distinct virtual group nodes."""

    nodes = [
        make_full_node(0, [0, 0], part_id="cosmoteer.cannon_med"),
        make_full_node(1, [2, 0], part_id="cosmoteer.railgun_launcher"),
    ]
    context, summary = _run_weapon_groups(nodes)
    emitted_ids = {n["id"] for n in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]}
    assert "weapon_group_cannon" in emitted_ids
    assert "weapon_group_railgun" in emitted_ids
    assert summary["weapon_group_nodes"] == 2
    assert summary["weapon_member_edges"] == 2


def test_weapon_groups_first_match_wins_for_ambiguous_part_id() -> None:
    """When multiple substrings match, the first one in WEAPON_TYPE_SUBSTRINGS wins."""

    # 'cannon' comes before 'railgun' in the list.
    # Construct a synthetic part ID that contains both.
    nodes = [make_full_node(0, [0, 0], part_id="mod.railgun_cannon_hybrid")]
    context, _ = _run_weapon_groups(nodes)
    annotation = context.get_annotation("weapon_group_by_part_id")
    # 'cannon' is earlier in WEAPON_TYPE_SUBSTRINGS than 'railgun'.
    cannon_idx = WEAPON_TYPE_SUBSTRINGS.index("cannon")
    railgun_idx = WEAPON_TYPE_SUBSTRINGS.index("railgun")
    assert cannon_idx < railgun_idx
    assert annotation[0] == "cannon"


def test_weapon_groups_group_nodes_emitted_in_substring_priority_order() -> None:
    """Weapon group nodes must appear in WEAPON_TYPE_SUBSTRINGS order, not part order."""

    # Add railgun part first, then cannon.
    nodes = [
        make_full_node(0, [0, 0], part_id="cosmoteer.railgun_launcher"),
        make_full_node(1, [2, 0], part_id="cosmoteer.cannon_med"),
    ]
    context, _ = _run_weapon_groups(nodes)
    emitted_ids = [n["id"] for n in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]]
    cannon_pos = emitted_ids.index("weapon_group_cannon")
    railgun_pos = emitted_ids.index("weapon_group_railgun")
    assert cannon_pos < railgun_pos


def test_weapon_groups_member_count_field_matches_members() -> None:
    """The member_count field on a weapon_group node must equal its actual membership."""

    nodes = [
        make_full_node(0, [0, 0], part_id="cosmoteer.cannon_med"),
        make_full_node(1, [2, 0], part_id="cosmoteer.cannon_heavy"),
        make_full_node(2, [4, 0], part_id="cosmoteer.cannon_turret"),
    ]
    context, _ = _run_weapon_groups(nodes)
    cannon_node = next(
        n for n in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]
        if n["id"] == "weapon_group_cannon"
    )
    assert cannon_node["member_count"] == 3


def test_weapon_groups_cross_edges_in_ascending_id_order() -> None:
    """weapon_member cross-edges must target node IDs in ascending order."""

    # IDs 5 and 3: we expect edges to 3 then 5.
    nodes = [
        make_full_node(5, [0, 0], part_id="cosmoteer.cannon_heavy"),
        make_full_node(3, [2, 0], part_id="cosmoteer.cannon_med"),
    ]
    context, _ = _run_weapon_groups(nodes)
    targets = [
        e["target"]
        for e in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
        if e["kind"] == "weapon_member"
    ]
    assert targets == sorted(targets)


def test_weapon_groups_annotation_maps_node_id_to_type() -> None:
    """weapon_group_by_part_id must map integer node IDs to their weapon type string."""

    nodes = [
        make_full_node(7, [0, 0], part_id="cosmoteer.railgun_launcher"),
        make_full_node(2, [2, 0], part_id="cosmoteer.cannon_med"),
    ]
    context, _ = _run_weapon_groups(nodes)
    annotation = context.get_annotation("weapon_group_by_part_id")
    assert annotation[7] == "railgun"
    assert annotation[2] == "cannon"


def test_weapon_groups_summary_in_expansion_graph_summary() -> None:
    """weapon_group_nodes and weapon_member_edges must appear in the graph summary."""

    nodes = [make_full_node(0, [0, 0], part_id="cosmoteer.disruptor")]
    context, _ = _run_weapon_groups(nodes)
    graph_summary = context.emitted_graphs[_EXPANSION_GRAPH_NAME]["summary"]
    assert graph_summary["weapon_group_nodes"] == 1
    assert graph_summary["weapon_member_edges"] == 1


def test_weapon_groups_weapon_node_has_weapon_type_field() -> None:
    """Each weapon_group virtual node must carry a 'weapon_type' field."""

    nodes = [make_full_node(0, [0, 0], part_id="cosmoteer.missile_launcher_small")]
    context, _ = _run_weapon_groups(nodes)
    missile_node = next(
        n for n in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]
        if n["id"] == "weapon_group_missile_launcher"
    )
    assert missile_node["weapon_type"] == "missile_launcher"
    assert missile_node["kind"] == "weapon_group"


# ---------------------------------------------------------------------------
# GlobalVirtualLinkerPass
# ---------------------------------------------------------------------------


def _run_global_virtual_linker(
    nodes: list[dict],
    edges: list[dict] | None = None,
) -> tuple[ExpansionContext, dict]:
    """Run BaseIndexes + TraversableClusters + HullPerimeter + GlobalVirtualLinker."""

    context = ExpansionContext(
        make_graph_payload(nodes, edges),
        expansion_name="structural",
        expansion_version=2,
    )
    BaseIndexesPass().run(context)
    TraversableClustersPass().run(context)
    HullPerimeterPass().run(context)
    summary = GlobalVirtualLinkerPass().run(context)
    return context, summary


def test_global_virtual_linker_connects_to_all_virtual_nodes() -> None:
    """GlobalVirtualLinkerPass should emit one edge per non-global virtual node."""

    nodes = [
        make_node(0, [[0, 0]], _CORRIDOR_ID),
        make_node(1, [[2, 0]], _CORRIDOR_ID),
    ]
    edges = [make_edge(0, 1, "door")]
    context, summary = _run_global_virtual_linker(nodes, edges)

    expansion_nodes = context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]
    # Only non-global nodes with at least one member should receive a linker edge.
    virtual_ids = {
        n["id"] for n in expansion_nodes
        if n["id"] != "global_ship" and n.get("member_count", 1) > 0
    }
    assert len(virtual_ids) > 0, "expected at least one other virtual node"

    linker_edges = [
        e for e in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
        if e.get("kind") == "global_virtual_member"
    ]
    linked_targets = {e["target"] for e in linker_edges}

    assert linked_targets == virtual_ids
    assert all(e["source"] == "global_ship" for e in linker_edges)
    assert all(e["source_graph"] == _EXPANSION_GRAPH_NAME for e in linker_edges)
    assert all(e["target_graph"] == _EXPANSION_GRAPH_NAME for e in linker_edges)
    assert summary == {"global_nodes": 1, "global_virtual_member_edges": len(virtual_ids)}


def test_global_virtual_linker_skips_empty_virtual_nodes() -> None:
    """GlobalVirtualLinkerPass must not link to virtual nodes with member_count == 0.

    A single-part ship has no interior parts, so HullPerimeterPass emits an
    ``interior`` node with ``member_count=0``.  That placeholder must not
    receive a ``global_virtual_member`` edge.
    """

    nodes = [make_node(0, [[0, 0]], _CORRIDOR_ID)]
    context, _ = _run_global_virtual_linker(nodes)

    expansion_nodes = context.emitted_graphs[_EXPANSION_GRAPH_NAME]["nodes"]

    # make_node produces no location_2x/footprint, so the part falls back to
    # interior classification — hull_perimeter gets member_count == 0.
    hull_node = next(
        (n for n in expansion_nodes if n.get("kind") == "hull_perimeter"),
        None,
    )
    assert hull_node is not None, "HullPerimeterPass should always emit hull_perimeter node"
    assert hull_node["member_count"] == 0

    # Confirm no linker edge targets the empty placeholder.
    linker_edges = [
        e for e in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
        if e.get("kind") == "global_virtual_member"
    ]
    linked_targets = {e["target"] for e in linker_edges}
    assert hull_node["id"] not in linked_targets


def test_global_virtual_linker_no_self_edge() -> None:
    """GlobalVirtualLinkerPass must not emit an edge from global_ship to itself."""

    nodes = [make_node(0, [[0, 0]], _CORRIDOR_ID)]
    context, _ = _run_global_virtual_linker(nodes)

    self_edges = [
        e for e in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
        if e.get("kind") == "global_virtual_member" and e["target"] == "global_ship"
    ]
    assert self_edges == []


def test_global_virtual_linker_summary_increments_graph_summary() -> None:
    """GlobalVirtualLinkerPass should increment global_virtual_member_edges in the graph summary."""

    nodes = [make_node(0, [[0, 0]], _CORRIDOR_ID)]
    context, _ = _run_global_virtual_linker(nodes)

    graph_summary = context.emitted_graphs[_EXPANSION_GRAPH_NAME]["summary"]
    linker_edges = [
        e for e in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
        if e.get("kind") == "global_virtual_member"
    ]
    assert graph_summary["global_virtual_member_edges"] == len(linker_edges)


def test_global_virtual_linker_empty_expansion_graph() -> None:
    """GlobalVirtualLinkerPass with no prior virtual passes emits no linker edges."""

    nodes = [make_node(0, part_id=_GENERIC_ID)]
    context = ExpansionContext(
        make_graph_payload(nodes),
        expansion_name="structural",
        expansion_version=2,
    )
    BaseIndexesPass().run(context)
    # Deliberately skip other virtual passes so global_ship has no peers.
    summary = GlobalVirtualLinkerPass().run(context)

    linker_edges = [
        e for e in context.emitted_graphs[_EXPANSION_GRAPH_NAME]["cross_edges"]
        if e.get("kind") == "global_virtual_member"
    ]
    assert linker_edges == []
    assert summary == {"global_nodes": 1, "global_virtual_member_edges": 0}
