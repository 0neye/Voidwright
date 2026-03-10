"""Regression tests for preprocessing cell-graph door traversal rules."""

import unittest

from preprocessing.graphs import cell_graph


def make_part_record(index: int, cells: set[tuple[int, int]], walkable_cells: set[tuple[int, int]]) -> dict:
    """Build the minimal part record payload used by `cell_graph` tests."""

    return {
        "index": index,
        "cells": cells,
        "walkable_cells": walkable_cells,
    }


class CellGraphDoorTraversabilityTests(unittest.TestCase):
    def test_door_edge_is_skipped_when_an_endpoint_cell_is_blocked(self) -> None:
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

        self.assertEqual(
            [edge for edge in graph["edges"] if edge["kind"] == "door"],
            [],
        )
        self.assertEqual(graph["summary"]["valid_door_edges"], 0)
        self.assertEqual(graph["summary"]["blocked_door_records"], 1)
        self.assertEqual(graph["summary"]["dangling_door_records"], 0)

    def test_door_edge_is_preserved_when_both_endpoint_cells_are_walkable(self) -> None:
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
        self.assertEqual(len(door_edges), 1)
        self.assertTrue(door_edges[0]["traversable"])
        self.assertEqual(graph["summary"]["valid_door_edges"], 1)
        self.assertEqual(graph["summary"]["blocked_door_records"], 0)


if __name__ == "__main__":
    unittest.main()
