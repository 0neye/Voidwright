"""Helpers for SaveRect-style ship location anchor semantics.

Cosmoteer ship files may serialize ``Part.Location`` relative to a sub-rect of
the full local footprint instead of the top-left footprint origin. The repo
stores normalized footprint origins internally, so parser and encoder
boundaries need to translate to and from the ship-file coordinate system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from common.geometry import (
    PartRect,
    load_vanilla_part_geometry,
    resolve_geometry_part_id_and_rotation,
)

__all__ = [
    "KNOWN_SAVE_RECTS",
    "SaveRect",
    "known_save_rects",
    "load_live_save_rects",
    "origin_to_stored_location",
    "stored_location_to_origin",
]


_ID_RE = re.compile(r"\bID\s*=\s*([^\s/]+)")
_SAVE_RECT_RE = re.compile(
    r"\bSaveRect\s*=\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
)


@dataclass(frozen=True)
class SaveRect:
    """Stored-location anchor rect used by Cosmoteer ship save files."""

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


def _save_rect_from_part_rect(part_id: str, rect: PartRect) -> SaveRect:
    """Convert shared geometry rect metadata into the save-rect helper shape."""

    return SaveRect(
        part_id=part_id,
        x=rect.x,
        y=rect.y,
        width=rect.width,
        height=rect.height,
        source_file=rect.source,
    )


def _canonical_part_id(part_id: str) -> str:
    """Return the canonical vanilla part ID used by geometry-backed rect data."""

    resolved_part_id, _resolved_rotation = resolve_geometry_part_id_and_rotation(part_id, 0)
    return resolved_part_id


def _effective_rect_from_geometry(part_id: str) -> SaveRect | None:
    """Resolve the stored-location rect for one vanilla part.

    Ship save files only use explicit ``save_rect`` metadata. When the export
    leaves that field null, the stored location remains anchored to the full
    footprint origin.
    """

    canonical_part_id = _canonical_part_id(part_id)
    geometry = load_vanilla_part_geometry().get(canonical_part_id)
    if geometry is None:
        return None

    rect = geometry.save_rect
    if rect is None:
        return None
    return _save_rect_from_part_rect(canonical_part_id, rect)


@lru_cache(maxsize=1)
def _geometry_save_rects() -> dict[str, SaveRect]:
    """Build the runtime save-rect table from shared geometry metadata."""

    result: dict[str, SaveRect] = {}

    # Centralize the effective-rect selection here so parser, encoder, and
    # audit helpers all agree on which vanilla parts need location offsets.
    for part_id in load_vanilla_part_geometry():
        save_rect = _effective_rect_from_geometry(part_id)
        if save_rect is not None:
            result[part_id] = save_rect
    return result


KNOWN_SAVE_RECTS: dict[str, SaveRect] = _geometry_save_rects()


def known_save_rects() -> dict[str, SaveRect]:
    """Return the repo-backed runtime SaveRect table."""

    return dict(_geometry_save_rects())


def _offset_for_part(part_id: str, rotation: int, save_rects: dict[str, SaveRect] | None = None) -> tuple[int, int]:
    """Resolve the rotated stored-location offset for one part placement."""

    canonical_part_id = _canonical_part_id(part_id)
    save_rect_map = save_rects if save_rects is not None else _geometry_save_rects()
    save_rect = save_rect_map.get(canonical_part_id) or save_rect_map.get(part_id)
    if save_rect is None:
        return 0, 0

    geom = load_vanilla_part_geometry()
    geometry = geom.get(canonical_part_id) or geom.get(part_id)
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

