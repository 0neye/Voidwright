from __future__ import annotations

from markov.order import order_ship_parts_from_graph
from markov.types import ShipPart


def test_order_ship_parts_from_graph_ignores_door_edges_for_bfs_order() -> None:
    """Only touching edges should drive graph-derived BFS ordering."""

    parts = [
        ShipPart(part_id="cosmoteer.corridor", rotation=0, x=0, y=0),
        ShipPart(part_id="cosmoteer.corridor", rotation=0, x=1, y=0),
        ShipPart(part_id="cosmoteer.corridor", rotation=0, x=2, y=0),
        ShipPart(part_id="cosmoteer.corridor", rotation=0, x=20, y=0),
    ]
    node_id_to_idx = {index: index for index in range(len(parts))}
    touching_edges = [
        {"source": 2, "target": 1, "kind": "touching"},
        {"source": 1, "target": 0, "kind": "touching"},
    ]

    ordered_without_door = order_ship_parts_from_graph(parts, node_id_to_idx, touching_edges)
    ordered_with_door = order_ship_parts_from_graph(
        parts,
        node_id_to_idx,
        touching_edges + [{"source": 2, "target": 3, "kind": "door"}],
    )

    assert ordered_with_door == ordered_without_door
    assert [part.x for part, _anchor in ordered_with_door] == [2, 1, 0, 20]
