"""Regression tests for preprocessing cell-graph door traversal rules."""

from preprocessing.graphs import cell_graph


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
