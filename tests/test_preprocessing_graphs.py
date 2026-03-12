"""Regression tests for preprocessing cell-graph and structural-edge rules."""

from collections import defaultdict

from common.geometry import infer_meta, load_vanilla_part_geometry
from preprocessing.graphs import cell_graph, part_cells, structural_edges


def _build_cell_to_parts(part_records: list[dict]) -> dict:
    """Derive cell_to_parts from part_records using the real geometry, mirroring
    what process_ship() does so tests stay consistent with footprint changes."""

    cell_to_parts: dict = defaultdict(set)
    for record in part_records:
        meta, _ = infer_meta(record["part_id"], record["rotation"])
        part_dict = {"Location": record["location"], "Rotation": record["rotation"]}
        for cell in part_cells(part_dict, meta):
            cell_to_parts[cell].add(record["index"])
    return dict(cell_to_parts)


def make_part_record(index: int, cells: set[tuple[int, int]], walkable_cells: set[tuple[int, int]]) -> dict:
    """Build the minimal part record payload used by `cell_graph` tests."""

    return {
        "index": index,
        "cells": cells,
        "walkable_cells": walkable_cells,
    }


def test_door_edge_is_skipped_when_an_endpoint_cell_is_blocked() -> None:
    """Blocked endpoint cells should prevent door-edge creation."""

    # Model a door between two occupied cells where only the lower cell is
    # crew-walkable. The door record should not reintroduce the blocked cell
    # into the traversable graph.
    part_records = [
        make_part_record(0, {(0, 0)}, set()),
        make_part_record(1, {(0, 1)}, {(0, 1)}),
    ]
    cell_to_parts = {
        (0, 0): {0},
        (0, 1): {1},
    }
    doors = [{"Cell": [0, 1], "Orientation": 0}]

    graph = cell_graph(part_records, cell_to_parts, doors)

    assert [edge for edge in graph["edges"] if edge["kind"] == "door"] == []
    assert graph["summary"]["valid_door_edges"] == 0
    assert graph["summary"]["blocked_door_records"] == 1
    assert graph["summary"]["dangling_door_records"] == 0


def test_door_edge_is_preserved_when_both_endpoint_cells_are_walkable() -> None:
    """Walkable endpoint cells should preserve traversable door edges."""

    # When both occupied endpoint cells are crew-walkable, the door should
    # still be represented as a traversable graph edge.
    part_records = [
        make_part_record(0, {(0, 0)}, {(0, 0)}),
        make_part_record(1, {(0, 1)}, {(0, 1)}),
    ]
    cell_to_parts = {
        (0, 0): {0},
        (0, 1): {1},
    }
    doors = [{"Cell": [0, 1], "Orientation": 0}]

    graph = cell_graph(part_records, cell_to_parts, doors)

    door_edges = [edge for edge in graph["edges"] if edge["kind"] == "door"]
    assert len(door_edges) == 1
    assert door_edges[0]["traversable"] is True
    assert graph["summary"]["valid_door_edges"] == 1
    assert graph["summary"]["blocked_door_records"] == 0


def test_structural_edges_filter_out_wedge_air_contacts() -> None:
    """Structural edge generation should ignore wedge empty-space contacts."""

    geometry_cache = load_vanilla_part_geometry()
    part_records = [
        {
            "index": 0,
            "part_id": "cosmoteer.armor_wedge",
            "location": [0, 0],
            "rotation": 0,
        },
        {
            "index": 1,
            "part_id": "cosmoteer.armor",
            "location": [0, -1],
            "rotation": 0,
        },
    ]
    # Derive cell_to_parts from real geometry so the map stays consistent with
    # footprint changes.  The armor directly above the wedge's air side should
    # produce no structural edge.
    cell_to_parts = _build_cell_to_parts(part_records)

    assert structural_edges(part_records, cell_to_parts, geometry_cache) == []


def test_structural_edges_accept_r_wedge_alias_ids() -> None:
    """Structural edge generation should support mirrored `_R` wedge aliases."""

    geometry_cache = load_vanilla_part_geometry()
    part_records = [
        {
            "index": 0,
            "part_id": "cosmoteer.armor_1x2_wedge_R",
            "location": [0, 0],
            "rotation": 0,
        },
        {
            "index": 1,
            "part_id": "cosmoteer.armor",
            "location": [1, 0],
            "rotation": 0,
        },
    ]
    # Derive cell_to_parts from real geometry so the test stays consistent with
    # footprint changes and does not rely on a manually guessed cell layout.
    cell_to_parts = _build_cell_to_parts(part_records)

    edges = structural_edges(part_records, cell_to_parts, geometry_cache)
    assert len(edges) == 1
    assert edges[0]["source"] == 0
    assert edges[0]["target"] == 1
