"""Tests for the graph_expansion package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.cosmoteer import parse_ship_png
from graph_expansion.cli import build_parser
from graph_expansion.cli import main as graph_expansion_main
from graph_expansion.passes.crew_access_layer1 import _ENABLE_CREW_ROOM_PROXY_FALLBACK
from graph_expansion.structural import (
    EXPANSION_VERSION,
    build_traversable_clusters as _build_traversable_clusters,
    enrich_graph as _enrich_graph,
    expand_dir,
    is_corridor_like as _is_corridor_like,
)
from preprocessing.graphs import process_ship
from preprocessing.relative_coords import apply_relative_coords_transform

__all__: list[str] = []

_CORRIDOR_ID = "cosmoteer.corridor"
_WALKWAY_ID = "mod.moving_walkway_1x1"
_CONVEYOR_ID = "cosmoteer.conveyor"
_GENERIC_ID = "cosmoteer.reactor_small"
_CREW_QUARTERS_ID = "cosmoteer.crew_quarters_med"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_node(
    node_id: int,
    walkable_cells: list[list[int]] | None = None,
    part_id: str = "",
) -> dict:
    """Build a minimal structural node dict."""
    node: dict = {"id": node_id, "part_id": part_id}
    if walkable_cells is not None:
        node["walkable_cells_2x"] = walkable_cells
    return node


def make_door_edge(source: int, target: int) -> dict:
    """Build a door edge between two part node IDs."""
    return {"source": source, "target": target, "kind": "door"}


def make_touching_edge(source: int, target: int) -> dict:
    """Build a structural touching edge (not a door)."""
    return {"source": source, "target": target, "kind": "touching"}


def make_graph_data(
    nodes: list[dict],
    ship: dict | None = None,
    edges: list[dict] | None = None,
) -> dict:
    """Build a minimal graph JSON payload matching the preprocessing output schema."""
    data: dict = {
        "graphs": {
            "A_structural_part_graph": {
                "nodes": nodes,
                "edges": edges if edges is not None else [],
            }
        }
    }
    if ship is not None:
        data["ship"] = ship
    return data


def write_graph_json(
    path: Path,
    nodes: list[dict],
    edges: list[dict] | None = None,
) -> None:
    """Write a minimal graph JSON file."""
    path.write_text(json.dumps(make_graph_data(nodes, edges=edges)) + "\n", encoding="utf-8")


def _load_traversable_tester_graph(tmp_path: Path) -> dict:
    """Parse the checked-in Traversable Tester PNG and build its structural graph."""

    ship_payload = apply_relative_coords_transform(
        parse_ship_png(Path(__file__).resolve().parent / "data" / "traversable_tester.ship.png")
    )
    source_path = tmp_path / "traversable_tester.ship.json"
    source_path.write_text(json.dumps(ship_payload) + "\n", encoding="utf-8")
    return process_ship(source_path)


def _load_thermal_tester_graph(tmp_path: Path) -> dict:
    """Parse the checked-in Thermal Tester PNG and build its structural graph."""

    ship_payload = apply_relative_coords_transform(
        parse_ship_png(Path(__file__).resolve().parent / "data" / "thermal_tester.ship.png")
    )
    source_path = tmp_path / "thermal_tester.ship.json"
    source_path.write_text(json.dumps(ship_payload) + "\n", encoding="utf-8")
    return process_ship(source_path)


# ---------------------------------------------------------------------------
# _is_corridor_like
# ---------------------------------------------------------------------------


def test_is_corridor_like_matches_corridor() -> None:
    assert _is_corridor_like("cosmoteer.corridor") is True


def test_is_corridor_like_matches_walkway() -> None:
    assert _is_corridor_like("mod.moving_walkway_1x1") is True


def test_is_corridor_like_matches_conveyor() -> None:
    assert _is_corridor_like("cosmoteer.conveyor") is True


def test_is_corridor_like_rejects_generic_part() -> None:
    assert _is_corridor_like("cosmoteer.reactor_small") is False


def test_is_corridor_like_rejects_empty_string() -> None:
    assert _is_corridor_like("") is False


def test_is_corridor_like_is_case_insensitive() -> None:
    assert _is_corridor_like("cosmoteer.CORRIDOR") is True


# ---------------------------------------------------------------------------
# _build_traversable_clusters — baseline / empty cases
# ---------------------------------------------------------------------------


def test_clusters_empty_nodes() -> None:
    assert _build_traversable_clusters([], []) == []


def test_clusters_no_walkable_cells() -> None:
    nodes = [make_node(0, part_id=_GENERIC_ID), make_node(1, part_id=_GENERIC_ID)]
    assert _build_traversable_clusters(nodes, []) == []


def test_clusters_single_walkable_part() -> None:
    assert _build_traversable_clusters([make_node(0, [[0, 0]], _GENERIC_ID)], []) == [[0]]


# ---------------------------------------------------------------------------
# _build_traversable_clusters — door-based connectivity
# ---------------------------------------------------------------------------


def test_clusters_door_edge_merges_any_two_walkable_parts() -> None:
    nodes = [
        make_node(0, [[0, 0]], _GENERIC_ID),
        make_node(1, [[100, 100]], _GENERIC_ID),
    ]
    edges = [make_door_edge(0, 1)]
    assert _build_traversable_clusters(nodes, edges) == [[0, 1]]


def test_clusters_touching_edge_does_not_merge() -> None:
    # Only door edges create traversal connectivity; structural touching edges do not.
    nodes = [
        make_node(0, [[0, 0]], _GENERIC_ID),
        make_node(1, [[2, 0]], _GENERIC_ID),
    ]
    edges = [make_touching_edge(0, 1)]
    assert _build_traversable_clusters(nodes, edges) == [[0], [1]]


def test_clusters_door_edge_ignored_when_source_has_no_walkable_cells() -> None:
    # Node 1 has no walkable cells so the door cannot form a cluster.
    nodes = [make_node(0, [[0, 0]], _GENERIC_ID), make_node(1, part_id=_GENERIC_ID)]
    edges = [make_door_edge(0, 1)]
    assert _build_traversable_clusters(nodes, edges) == [[0]]


def test_clusters_multiple_door_edges_chain_parts_together() -> None:
    # 0 --door--> 1 --door--> 2; all three should be in one cluster.
    nodes = [
        make_node(0, [[0, 0]], _GENERIC_ID),
        make_node(1, [[50, 50]], _GENERIC_ID),
        make_node(2, [[100, 100]], _GENERIC_ID),
    ]
    edges = [make_door_edge(0, 1), make_door_edge(1, 2)]
    assert _build_traversable_clusters(nodes, edges) == [[0, 1, 2]]


# ---------------------------------------------------------------------------
# _build_traversable_clusters — corridor-like adjacency
# ---------------------------------------------------------------------------


def test_clusters_two_corridors_adjacent_in_x_merge() -> None:
    nodes = [
        make_node(0, [[0, 0]], _CORRIDOR_ID),
        make_node(1, [[2, 0]], _CORRIDOR_ID),
    ]
    assert _build_traversable_clusters(nodes, []) == [[0, 1]]


def test_clusters_two_corridors_adjacent_in_y_merge() -> None:
    nodes = [
        make_node(0, [[0, 0]], _CORRIDOR_ID),
        make_node(1, [[0, 2]], _CORRIDOR_ID),
    ]
    assert _build_traversable_clusters(nodes, []) == [[0, 1]]


def test_clusters_two_walkway_parts_adjacent_merge() -> None:
    nodes = [
        make_node(0, [[0, 0]], _WALKWAY_ID),
        make_node(1, [[2, 0]], _WALKWAY_ID),
    ]
    assert _build_traversable_clusters(nodes, []) == [[0, 1]]


def test_clusters_two_conveyors_adjacent_merge() -> None:
    nodes = [
        make_node(0, [[0, 0]], _CONVEYOR_ID),
        make_node(1, [[2, 0]], _CONVEYOR_ID),
    ]
    assert _build_traversable_clusters(nodes, []) == [[0, 1]]


def test_clusters_corridor_and_walkway_adjacent_merge() -> None:
    nodes = [
        make_node(0, [[0, 0]], _CORRIDOR_ID),
        make_node(1, [[2, 0]], _WALKWAY_ID),
    ]
    assert _build_traversable_clusters(nodes, []) == [[0, 1]]


def test_clusters_corridor_and_conveyor_adjacent_merge() -> None:
    nodes = [
        make_node(0, [[0, 0]], _CORRIDOR_ID),
        make_node(1, [[2, 0]], _CONVEYOR_ID),
    ]
    assert _build_traversable_clusters(nodes, []) == [[0, 1]]


def test_clusters_two_non_corridor_adjacent_no_door_stay_separate() -> None:
    # Generic parts touching each other without a door are NOT merged.
    nodes = [
        make_node(0, [[0, 0]], _GENERIC_ID),
        make_node(1, [[2, 0]], _GENERIC_ID),
    ]
    assert _build_traversable_clusters(nodes, []) == [[0], [1]]


def test_clusters_corridor_adjacent_to_non_corridor_no_door_stay_separate() -> None:
    # Mere coordinate adjacency is not enough without structural contact.
    nodes = [
        make_node(0, [[0, 0]], _CORRIDOR_ID),
        make_node(1, [[2, 0]], _GENERIC_ID),
    ]
    assert _build_traversable_clusters(nodes, []) == [[0], [1]]


def test_clusters_touching_corridor_and_room_with_adjacent_walkable_cells_stay_separate() -> None:
    nodes = [
        make_node(0, [[0, 0]], _CORRIDOR_ID),
        make_node(1, [[2, 0], [2, 2]], _CREW_QUARTERS_ID),
    ]
    edges = [{"source": 0, "target": 1, "kind": "touching", "shared_sides": 1}]
    assert _build_traversable_clusters(nodes, edges) == [[0], [1]]


def test_clusters_two_corridors_sharing_same_cell_merge() -> None:
    nodes = [
        make_node(0, [[4, 4]], _CORRIDOR_ID),
        make_node(1, [[4, 4]], _CORRIDOR_ID),
    ]
    assert _build_traversable_clusters(nodes, []) == [[0, 1]]


def test_clusters_corridor_diagonal_cells_not_adjacent() -> None:
    # Cells differing by 2 in both axes are diagonal and should not merge.
    nodes = [
        make_node(0, [[0, 0]], _CORRIDOR_ID),
        make_node(1, [[2, 2]], _CORRIDOR_ID),
    ]
    assert _build_traversable_clusters(nodes, []) == [[0], [1]]


def test_clusters_three_parts_two_corridors_connected_one_generic_isolated() -> None:
    nodes = [
        make_node(0, [[0, 0]], _CORRIDOR_ID),
        make_node(1, [[2, 0]], _CORRIDOR_ID),   # adjacent corridor → merges with 0
        make_node(2, [[4, 0]], _GENERIC_ID),    # generic, no door → isolated
    ]
    assert _build_traversable_clusters(nodes, []) == [[0, 1], [2]]


def test_clusters_generic_parts_merged_via_door_despite_no_adjacency() -> None:
    # Non-corridor parts far apart but joined by a door still cluster together.
    nodes = [
        make_node(0, [[0, 0]], _GENERIC_ID),
        make_node(1, [[200, 200]], _GENERIC_ID),
    ]
    assert _build_traversable_clusters(nodes, [make_door_edge(0, 1)]) == [[0, 1]]


# ---------------------------------------------------------------------------
# _build_traversable_clusters — ordering / determinism
# ---------------------------------------------------------------------------


def test_clusters_member_ids_sorted_within_cluster() -> None:
    nodes = [
        make_node(5, [[0, 0]], _CORRIDOR_ID),
        make_node(2, [[0, 0]], _CORRIDOR_ID),
        make_node(9, [[0, 0]], _CORRIDOR_ID),
    ]
    assert _build_traversable_clusters(nodes, []) == [[2, 5, 9]]


def test_clusters_outer_list_sorted_by_first_member() -> None:
    nodes = [make_node(3, [[10, 0]], _GENERIC_ID), make_node(1, [[0, 0]], _GENERIC_ID)]
    assert _build_traversable_clusters(nodes, []) == [[1], [3]]


def test_clusters_ordering_deterministic_regardless_of_input_order() -> None:
    nodes = [make_node(i, [[i * 100, 0]], _GENERIC_ID) for i in range(5)]
    assert _build_traversable_clusters(nodes, []) == _build_traversable_clusters(
        list(reversed(nodes)), []
    )


def test_clusters_part_without_walkable_cells_excluded() -> None:
    nodes = [make_node(0, [[0, 0]], _CORRIDOR_ID), make_node(1, part_id=_CORRIDOR_ID)]
    result = _build_traversable_clusters(nodes, [])
    assert result == [[0]]
    assert 1 not in [m for cluster in result for m in cluster]


# ---------------------------------------------------------------------------
# _enrich_graph
# ---------------------------------------------------------------------------


def test_enrich_graph_adds_expansion_graph() -> None:
    result = _enrich_graph(make_graph_data([make_node(0, [[0, 0]], _GENERIC_ID)]))
    assert "X_expansion_structural" in result["graphs"]


def test_enrich_graph_preserves_existing_graph_and_extra_keys() -> None:
    graph_data = make_graph_data([make_node(0, part_id=_GENERIC_ID)])
    graph_data["custom_key"] = "hello"
    result = _enrich_graph(graph_data)
    assert result["custom_key"] == "hello"
    assert "A_structural_part_graph" in result["graphs"]


def test_enrich_graph_adds_global_ship_node() -> None:
    result = _enrich_graph(make_graph_data([make_node(0, part_id=_GENERIC_ID)]))
    exp_nodes = result["graphs"]["X_expansion_structural"]["nodes"]
    global_nodes = [n for n in exp_nodes if n["kind"] == "global_ship_info"]
    assert len(global_nodes) == 1
    assert global_nodes[0]["id"] == "global_ship"


def test_enrich_graph_global_ship_node_carries_ship_metadata() -> None:
    ship = {"Name": "TestShip", "crew": 3}
    result = _enrich_graph(make_graph_data([make_node(0, part_id=_GENERIC_ID)], ship=ship))
    exp_nodes = result["graphs"]["X_expansion_structural"]["nodes"]
    global_node = next(n for n in exp_nodes if n["id"] == "global_ship")
    assert global_node["ship"] == ship


def test_enrich_graph_global_member_edges_connect_to_all_parts() -> None:
    result = _enrich_graph(
        make_graph_data([make_node(0, part_id=_GENERIC_ID), make_node(1, part_id=_GENERIC_ID)])
    )
    cross_edges = result["graphs"]["X_expansion_structural"]["cross_edges"]
    global_edges = [e for e in cross_edges if e["kind"] == "global_member"]
    assert {e["target"] for e in global_edges} == {0, 1}


def test_enrich_graph_cluster_nodes_for_each_isolated_walkable_part() -> None:
    # Two groups of adjacent corridor parts with no cross-group connection → two clusters.
    # Each group has 9 cells (18 total) and 2 members so both survive filtering.
    nodes = [
        make_node(0, [[i * 2, 0] for i in range(9)], _CORRIDOR_ID),
        make_node(1, [[18 + i * 2, 0] for i in range(9)], _CORRIDOR_ID),
        make_node(2, [[i * 2, 200] for i in range(9)], _CORRIDOR_ID),
        make_node(3, [[18 + i * 2, 200] for i in range(9)], _CORRIDOR_ID),
    ]
    result = _enrich_graph(make_graph_data(nodes))
    exp_nodes = result["graphs"]["X_expansion_structural"]["nodes"]
    cluster_nodes = [n for n in exp_nodes if n["kind"] == "traversable_cluster"]
    assert len(cluster_nodes) == 2


def test_enrich_graph_door_edge_merges_clusters() -> None:
    # Two generic parts with a door edge → one cluster.
    nodes = [make_node(0, [[0, 0]], _GENERIC_ID), make_node(1, [[100, 100]], _GENERIC_ID)]
    edges = [make_door_edge(0, 1)]
    result = _enrich_graph(make_graph_data(nodes, edges=edges))
    exp_nodes = result["graphs"]["X_expansion_structural"]["nodes"]
    cluster_nodes = [n for n in exp_nodes if n["kind"] == "traversable_cluster"]
    assert len(cluster_nodes) == 1


def test_enrich_graph_corridor_adjacency_merges_clusters() -> None:
    # Two adjacent corridor parts → one cluster (no door required).
    # Each node has 9 2x cells (total 18 > 16) to survive small-cluster filtering;
    # the last cell of node 0 ([16,0]) is adjacent to the first cell of node 1 ([18,0]).
    nodes = [
        make_node(0, [[i * 2, 0] for i in range(9)], _CORRIDOR_ID),
        make_node(1, [[18 + i * 2, 0] for i in range(9)], _CORRIDOR_ID),
    ]
    result = _enrich_graph(make_graph_data(nodes))
    exp_nodes = result["graphs"]["X_expansion_structural"]["nodes"]
    cluster_nodes = [n for n in exp_nodes if n["kind"] == "traversable_cluster"]
    assert len(cluster_nodes) == 1


def test_enrich_graph_no_cluster_nodes_when_no_walkable_cells() -> None:
    result = _enrich_graph(
        make_graph_data([make_node(0, part_id=_GENERIC_ID), make_node(1, part_id=_GENERIC_ID)])
    )
    exp_nodes = result["graphs"]["X_expansion_structural"]["nodes"]
    assert [n for n in exp_nodes if n["kind"] == "traversable_cluster"] == []


def test_enrich_graph_super_member_edges_connect_cluster_to_members() -> None:
    # Two adjacent corridor parts form one cluster; both should appear as super_member targets.
    # Each node has 9 2x cells (total 18 > 16) to survive small-cluster filtering.
    nodes = [
        make_node(0, [[i * 2, 0] for i in range(9)], _CORRIDOR_ID),
        make_node(1, [[18 + i * 2, 0] for i in range(9)], _CORRIDOR_ID),
    ]
    result = _enrich_graph(make_graph_data(nodes))
    cross_edges = result["graphs"]["X_expansion_structural"]["cross_edges"]
    super_edges = [e for e in cross_edges if e["kind"] == "super_member"]
    assert {e["target"] for e in super_edges} == {0, 1}
    assert len({e["source"] for e in super_edges}) == 1


def test_enrich_graph_summary_counts_are_consistent() -> None:
    # Two groups of adjacent corridors (2 members, 18 cells each) plus a non-walkable part.
    # Each walkable group survives both the single-part and small-footprint filters.
    nodes = [
        make_node(0, [[i * 2, 0] for i in range(9)], _CORRIDOR_ID),
        make_node(1, [[18 + i * 2, 0] for i in range(9)], _CORRIDOR_ID),
        make_node(2, [[i * 2, 200] for i in range(9)], _CORRIDOR_ID),
        make_node(3, [[18 + i * 2, 200] for i in range(9)], _CORRIDOR_ID),
        make_node(4, part_id=_GENERIC_ID),
    ]
    result = _enrich_graph(make_graph_data(nodes))
    summary = result["graphs"]["X_expansion_structural"]["summary"]
    assert summary["global_ship_nodes"] == 1
    assert summary["traversable_clusters"] == 2
    assert summary["global_member_edges"] == 5
    assert summary["super_member_edges"] == 4


def test_enrich_graph_traversable_tester_regression_counts(tmp_path: Path) -> None:
    graph_data = _load_traversable_tester_graph(tmp_path)
    result = _enrich_graph(graph_data)

    expansion_graph = result["graphs"]["X_expansion_structural"]
    node_kinds = [node["kind"] for node in expansion_graph["nodes"]]
    cross_edges = expansion_graph["cross_edges"]

    # One cluster: the isolated crew_quarters_med (4 2x cells, no doors) is filtered.
    assert node_kinds.count("traversable_cluster") == 1
    expected_crew_access_edges = 3 if _ENABLE_CREW_ROOM_PROXY_FALLBACK else 2
    assert sum(edge["kind"] == "crew_access_factory" for edge in cross_edges) == expected_crew_access_edges
    assert sum(edge["kind"] == "crew_access_reactor" for edge in cross_edges) == expected_crew_access_edges
    assert sum(edge["kind"] == "reactor_supports_engine_room" for edge in cross_edges) == 1
    assert sum(edge["kind"] == "reactor_supports_shield" for edge in cross_edges) == 1
    assert sum(edge["kind"] == "reactor_supports_thruster" for edge in cross_edges) == 1
    assert sum(edge["kind"] == "factory_supports_ammo_weapon" for edge in cross_edges) == 2
    assert sum(edge["kind"] == "factory_supports_storage" for edge in cross_edges) == 1


def test_enrich_graph_thermal_tester_regression_counts(tmp_path: Path) -> None:
    """Regression guard: thermal_tester.ship.png must produce exactly 4 thermal networks.

    Ship layout (25 structural parts, 22 in thermal networks):
    - thermal_network_0: 1 member  — isolated thermal_dilation_pump backbone; its only
      port-matched neighbour is an OC power_storage that preferred the larger network
    - thermal_network_1: 17 members — main conduit spine, OC attachments, and
      heat-exchanger radius pulls; includes the OC power_storage that chose this
      network over the smaller thermal_dilation_pump cluster (largest-wins rule)
    - thermal_network_2: 2 members  — isolated heat_pipe_adaptive + thermal_dilation_pump
    - thermal_network_3: 2 members  — thermal_battery + OC laser_blaster_large, isolated
      by two-phase clustering
    """

    graph_data = _load_thermal_tester_graph(tmp_path)
    result = _enrich_graph(graph_data)

    expansion_graph = result["graphs"]["X_expansion_structural"]
    node_kinds = [node["kind"] for node in expansion_graph["nodes"]]
    cross_edges = expansion_graph["cross_edges"]

    assert node_kinds.count("thermal_network") == 4
    # 24 edges: railgun parts connected to two bottom networks each get two thermal_member
    # edges (multi-network leaf membership), adding 2 extra edges vs the pre-feature count.
    assert sum(e["kind"] == "thermal_member" for e in cross_edges) == 24

    thermal_nodes = [n for n in expansion_graph["nodes"] if n["kind"] == "thermal_network"]
    sizes = sorted(
        sum(1 for e in cross_edges if e["kind"] == "thermal_member" and e["source"] == tn["id"])
        for tn in thermal_nodes
    )
    assert sizes == [2, 2, 3, 17]


def test_enrich_graph_expansion_metadata() -> None:
    result = _enrich_graph(make_graph_data([make_node(0, part_id=_GENERIC_ID)]))
    assert result["expansion"]["backend"] == "structural"
    assert result["expansion"]["version"] == EXPANSION_VERSION
    assert "X_expansion_structural" in result["expansion"]["graphs_added"]
    pass_names = [p["name"] for p in result["expansion"]["passes"]]
    assert "base_indexes" in pass_names
    assert "global_ship_info" in pass_names
    assert "traversable_clusters" in pass_names
    assert "crew_access_layer1" in pass_names
    assert "core_support_layer2" in pass_names


def test_enrich_graph_does_not_mutate_input() -> None:
    graph_data = make_graph_data([make_node(0, [[0, 0]], _CORRIDOR_ID)])
    original_top_keys = set(graph_data)
    original_graph_keys = set(graph_data["graphs"])
    _enrich_graph(graph_data)
    assert set(graph_data) == original_top_keys
    assert set(graph_data["graphs"]) == original_graph_keys


# ---------------------------------------------------------------------------
# expand_dir
# ---------------------------------------------------------------------------


def test_expand_dir_empty_directory(tmp_path: Path) -> None:
    result = expand_dir(input_dir=tmp_path, output_dir=tmp_path / "out")
    assert result["files_expanded"] == 0


def test_expand_dir_ignores_manifest_json(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"ships_processed": 3}', encoding="utf-8")
    result = expand_dir(input_dir=tmp_path, output_dir=tmp_path / "out")
    assert result["files_expanded"] == 0


def test_expand_dir_enriches_json_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    write_graph_json(input_dir / "ship_a.json", [make_node(0, [[0, 0]], _CORRIDOR_ID)])
    write_graph_json(input_dir / "ship_b.json", [make_node(0, part_id=_GENERIC_ID)])

    result = expand_dir(input_dir=input_dir, output_dir=output_dir, workers=1, executor="thread")

    assert result["files_expanded"] == 2
    for name in ("ship_a.json", "ship_b.json"):
        enriched = json.loads((output_dir / name).read_text(encoding="utf-8"))
        assert "X_expansion_structural" in enriched["graphs"]


def test_expand_dir_in_place_when_output_equals_input(tmp_path: Path) -> None:
    write_graph_json(tmp_path / "ship.json", [make_node(0, part_id=_GENERIC_ID)])
    expand_dir(input_dir=tmp_path, output_dir=tmp_path, workers=1, executor="thread")
    enriched = json.loads((tmp_path / "ship.json").read_text(encoding="utf-8"))
    assert "expansion" in enriched


def test_expand_dir_in_place_second_run_skips_all(tmp_path: Path) -> None:
    """A second in-place run with the same version should skip all files.

    Before the fix, _needs_regen compared a file's mtime to itself, which
    always returned False — correct by accident. After the fix the early-
    return is explicit and intentional, so this test guards the semantics.
    """

    write_graph_json(tmp_path / "ship.json", [make_node(0, part_id=_GENERIC_ID)])
    first = expand_dir(input_dir=tmp_path, output_dir=tmp_path, workers=1, executor="thread")
    assert first["files_expanded"] == 1

    second = expand_dir(input_dir=tmp_path, output_dir=tmp_path, workers=1, executor="thread")
    assert second["files_expanded"] == 0


def test_expand_dir_skips_bad_files_and_continues(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    write_graph_json(input_dir / "good.json", [make_node(0, part_id=_GENERIC_ID)])
    (input_dir / "bad.json").write_text("{not valid json", encoding="utf-8")

    result = expand_dir(input_dir=input_dir, output_dir=output_dir, workers=1, executor="thread")

    assert result["files_expanded"] == 1
    assert (output_dir / "good.json").exists()
    assert not (output_dir / "bad.json").exists()


def test_expand_dir_traversable_clusters_total_is_sum_across_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    # ship_a: two isolated generic parts with a door → 1 cluster (door merges them)
    write_graph_json(
        input_dir / "ship_a.json",
        nodes=[make_node(0, [[0, 0]], _GENERIC_ID), make_node(1, [[100, 100]], _GENERIC_ID)],
        edges=[make_door_edge(0, 1)],
    )
    # ship_b: two adjacent corridor parts (18 total 2x cells) → 1 cluster
    write_graph_json(
        input_dir / "ship_b.json",
        nodes=[
            make_node(0, [[i * 2, 0] for i in range(9)], _CORRIDOR_ID),
            make_node(1, [[18 + i * 2, 0] for i in range(9)], _CORRIDOR_ID),
        ],
    )

    result = expand_dir(input_dir=input_dir, output_dir=output_dir, workers=1, executor="thread")

    assert result["traversable_clusters_total"] == 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_build_parser_succeeds() -> None:
    assert build_parser() is not None


def test_cli_expand_structural_enriches_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    write_graph_json(input_dir / "ship.json", [make_node(0, [[0, 0]], _CORRIDOR_ID)])

    exit_code = graph_expansion_main([
        "expand", "structural",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--workers", "1",
        "--executor", "thread",
    ])

    assert exit_code == 0
    enriched = json.loads((output_dir / "ship.json").read_text(encoding="utf-8"))
    assert "X_expansion_structural" in enriched["graphs"]


def test_cli_expand_structural_empty_dir_exits_zero(tmp_path: Path) -> None:
    exit_code = graph_expansion_main([
        "expand", "structural",
        "--input-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert exit_code == 0


def test_cli_expand_rejects_unknown_legacy_pipeline_name(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        graph_expansion_main([
            "expand", "nonexistent",
            "--input-dir", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
        ])
