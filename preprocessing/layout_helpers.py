"""Shared preprocessing helpers for ship layout coordinate semantics."""

from __future__ import annotations

from typing import Optional

from common.geometry import Coord

__all__ = ["DOOR_CELL_DELTAS", "door_adjacent_cells"]

# Cosmoteer ship Door.Cell names the right/bottom occupied cell of the doorway
# span. These deltas therefore point from the earlier occupied cell to the
# stored Door.Cell value.
DOOR_CELL_DELTAS: dict[int, Coord] = {
    0: (0, 1),
    1: (1, 0),
}


def door_adjacent_cells(cell: Coord, orientation: int) -> Optional[tuple[Coord, Coord]]:
    """Return the two occupied cells connected by one stored door record.

    Args:
        cell: The stored `Door.Cell` coordinate from extracted ship JSON
        orientation: Cosmoteer door orientation index

    Returns:
        The `(previous_cell, stored_cell)` pair when the orientation is known,
        otherwise `None`
    """

    delta = DOOR_CELL_DELTAS.get(orientation)
    if delta is None:
        return None

    return (cell[0] - delta[0], cell[1] - delta[1]), cell
