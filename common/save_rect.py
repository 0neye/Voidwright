"""Helpers for live-game SaveRect anchor semantics.

Cosmoteer rule files may define ``SaveRect = [x, y, w, h]`` for a part.
That suggests ship-file ``Part.Location`` can be serialized relative to a
sub-rectangle instead of the full local footprint origin. The current repo
uses normalized top-left footprint origins internally, so parser/encoder
boundaries need to translate to and from the ship-file coordinate system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from common.geometry import load_vanilla_part_geometry


_ID_RE = re.compile(r"\bID\s*=\s*([^\s/]+)")
_SAVE_RECT_RE = re.compile(
    r"\bSaveRect\s*=\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
)


@dataclass(frozen=True)
class SaveRect:
    part_id: str
    x: int
    y: int
    width: int
    height: int
    source_file: str

    def offset_for_rotation(self, rotation: int, base_width: int, base_height: int) -> tuple[int, int]:
        """Return the rotated SaveRect top-left offset for one CW rotation."""

        cells = [
            (cell_x, cell_y)
            for cell_x in range(self.x, self.x + self.width)
            for cell_y in range(self.y, self.y + self.height)
        ]
        rotated = [_rotate_cell(cell, rotation % 4, base_width, base_height) for cell in cells]
        min_x = min(cell_x for cell_x, _cell_y in rotated)
        min_y = min(cell_y for _cell_x, cell_y in rotated)
        return min_x, min_y


def _rotate_cell(cell: tuple[int, int], rotation: int, width: int, height: int) -> tuple[int, int]:
    """Rotate one local tile inside a ``width x height`` box clockwise."""

    x, y = cell
    normalized = rotation % 4
    if normalized == 0:
        return x, y
    if normalized == 1:
        return height - 1 - y, x
    if normalized == 2:
        return width - 1 - x, height - 1 - y
    return y, width - 1 - x


# Curated runtime SaveRect overrides derived from the live vanilla game files.
# These are safe to use without requiring a local game install.
KNOWN_SAVE_RECTS: dict[str, SaveRect] = {
    "cosmoteer.shield_gen_small": SaveRect(
        part_id="cosmoteer.shield_gen_small",
        x=0,
        y=1,
        width=2,
        height=2,
        source_file="Data/ships/terran/shield_gen_small/shield_gen_small.rules",
    ),
}


def known_save_rects() -> dict[str, SaveRect]:
    """Return the curated runtime SaveRect table."""

    return KNOWN_SAVE_RECTS


def _offset_for_part(part_id: str, rotation: int, save_rects: dict[str, SaveRect] | None = None) -> tuple[int, int]:
    save_rect_map = save_rects if save_rects is not None else KNOWN_SAVE_RECTS
    save_rect = save_rect_map.get(part_id)
    if save_rect is None:
        return 0, 0

    geometry = load_vanilla_part_geometry().get(part_id)
    if geometry is None:
        return 0, 0
    base_geometry = geometry.rotations.get(0)
    if base_geometry is None:
        base_geometry = next(iter(geometry.rotations.values()))
    return save_rect.offset_for_rotation(rotation, base_geometry.width, base_geometry.height)


def stored_location_to_origin(
    part_id: str,
    rotation: int,
    location: tuple[int, int] | list[int],
    save_rects: dict[str, SaveRect] | None = None,
) -> tuple[int, int]:
    """Convert ship-file stored ``Location`` into normalized footprint origin."""

    offset_x, offset_y = _offset_for_part(part_id, rotation, save_rects)
    return int(location[0]) - offset_x, int(location[1]) - offset_y


def origin_to_stored_location(
    part_id: str,
    rotation: int,
    origin: tuple[int, int] | list[int],
    save_rects: dict[str, SaveRect] | None = None,
) -> tuple[int, int]:
    """Convert normalized footprint origin into ship-file stored ``Location``."""

    offset_x, offset_y = _offset_for_part(part_id, rotation, save_rects)
    return int(origin[0]) + offset_x, int(origin[1]) + offset_y


def load_live_save_rects(game_root: str | Path) -> dict[str, SaveRect]:
    """Scan a live Cosmoteer install for Terran part ``SaveRect`` definitions."""

    terran_root = Path(game_root) / "Data" / "ships" / "terran"
    results: dict[str, SaveRect] = {}

    for rules_path in terran_root.rglob("*.rules"):
        try:
            text = rules_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = rules_path.read_text(encoding="utf-8-sig")

        id_match = _ID_RE.search(text)
        save_rect_match = _SAVE_RECT_RE.search(text)
        if id_match is None or save_rect_match is None:
            continue

        part_id = id_match.group(1)
        results[part_id] = SaveRect(
            part_id=part_id,
            x=int(save_rect_match.group(1)),
            y=int(save_rect_match.group(2)),
            width=int(save_rect_match.group(3)),
            height=int(save_rect_match.group(4)),
            source_file=str(rules_path),
        )

    return results


__all__ = [
    "KNOWN_SAVE_RECTS",
    "SaveRect",
    "known_save_rects",
    "load_live_save_rects",
    "origin_to_stored_location",
    "stored_location_to_origin",
]
