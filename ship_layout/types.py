"""Shared ship-layout types used outside Markov-specific code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

__all__ = ["Coord", "Coord2x", "PlacedPart", "Segment2x"]

Coord = Tuple[int, int]
Coord2x = Tuple[int, int]
Segment2x = Tuple[Coord2x, Coord2x]


@dataclass(frozen=True)
class PlacedPart:
    """Normalized part placement record for shared layout checks."""

    part_id: str
    rotation: int
    x: int
    y: int
    flip_x: bool = False
    flip_y: bool = False

    @classmethod
    def from_object(cls, part: Any) -> "PlacedPart":
        """Build a shared placement record from dict-like or object-like inputs.

        Args:
            part: Source placement object with `part_id`, `rotation`, `x`, and `y`

        Returns:
            A normalized `PlacedPart` with integer coordinates
        """

        if isinstance(part, cls):
            return part
        if isinstance(part, dict):
            return cls(
                part_id=str(part["part_id"]),
                rotation=int(part.get("rotation", 0)) % 4,
                x=int(part["x"]),
                y=int(part["y"]),
                flip_x=bool(part.get("flip_x", part.get("FlipX", False))),
                flip_y=bool(part.get("flip_y", part.get("FlipY", False))),
            )
        return cls(
            part_id=str(getattr(part, "part_id")),
            rotation=int(getattr(part, "rotation")) % 4,
            x=int(getattr(part, "x")),
            y=int(getattr(part, "y")),
            flip_x=bool(getattr(part, "flip_x", False)),
            flip_y=bool(getattr(part, "flip_y", False)),
        )


