"""Targeted tests for graph_expansion framework primitives and passes."""

from __future__ import annotations

from typing import Any

from graph_expansion.context import ExpansionContext
from graph_expansion.passes.base_indexes import BaseIndexesPass
from graph_expansion.passes.global_ship_info import GlobalShipInfoPass
from graph_expansion.passes.traversable_clusters import TraversableClustersPass

__all__: list[str] = []


_EXPANSION_GRAPH_NAME = "X_expansion_structural"
_STRUCTURAL_GRAPH_NAME = "A_structural_part_graph"
_CORRIDOR_ID = "cosmoteer.corridor"
_GENERIC_ID = "cosmoteer.reactor_small"


def make_node(
    node_id: int,
    walkable_cells: list[list[int]] | None = None,
    part_id: str = "",
    *,
    is_corridor_like: bool = False,
) -> dict[str, Any]:
    """Build a minimal structural node dict."""

    node: dict[str, Any] = {"id": node_id, "part_id": part_id}
    if walkable_cells is not None:
        node["walkable_cells_2x"] = walkable_cells
    if is_corridor_like:
        node["is_corridor_like"] = True
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
        make_node(0, [[0, 0]], _CORRIDOR_ID, is_corridor_like=True),
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
    assert context.caches["corridor_like_part_ids"] == {0}
    assert context.caches["door_edges"] == [edges[0]]
    assert context.caches["touching_edges"] == [edges[1]]



def test_global_ship_info_pass_emits_exact_node_edges_and_summary() -> None:
    """GlobalShipInfoPass should emit one node and one edge per structural node."""

    nodes = [make_node(10, part_id=_GENERIC_ID), make_node(20, part_id=_CORRIDOR_ID)]
    ship = {"Name": "TestShip", "Crew": 4}
    context = ExpansionContext(
        make_graph_payload(nodes, ship=ship),
        expansion_name="structural",
        expansion_version=2,
    )

    summary = GlobalShipInfoPass().run(context)
    expansion_graph = context.emitted_graphs[_EXPANSION_GRAPH_NAME]

    assert summary == {"global_nodes": 1, "global_member_edges": 2}
    assert expansion_graph["nodes"] == [
        {"id": "global_ship", "kind": "global_ship_info", "ship": ship}
    ]
    assert expansion_graph["cross_edges"] == [
        {
            "source": "global_ship",
            "source_graph": _EXPANSION_GRAPH_NAME,
            "target": 10,
            "target_graph": _STRUCTURAL_GRAPH_NAME,
            "kind": "global_member",
        },
        {
            "source": "global_ship",
            "source_graph": _EXPANSION_GRAPH_NAME,
            "target": 20,
            "target_graph": _STRUCTURAL_GRAPH_NAME,
            "kind": "global_member",
        },
    ]
    assert expansion_graph["summary"] == {
        "global_ship_nodes": 1,
        "global_member_edges": 2,
    }



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
    assert summary == {"cluster_count": 1, "super_member_edges": 3}
    assert expansion_graph["nodes"] == [
        {"id": "traversable_cluster_0", "kind": "traversable_cluster", "member_count": 3}
    ]
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
