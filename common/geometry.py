"""Shared Cosmoteer geometry and ship-layout helpers.

This module is the canonical owner of geometry metadata that is shared across
preprocessing, training, and generation. It intentionally avoids importing
package-specific logic so other modules can depend on it without creating
layering inversions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

Coord = Tuple[int, int]
DATA_DIR = Path(__file__).resolve().parent / "data"
VANILLA_PARTS_PATH = DATA_DIR / "vanilla-parts-from-game-files.json"
VANILLA_NAMESPACE = "cosmoteer."
PART_ID_ALIASES = {
    "cosmoteer.electro_bolter": "cosmoteer.disruptor",
}

TRAVERSABLE_HINTS = (
    "corridor",
    "conveyor",
    "crew_quarters",
    "quarters",
    "storage",
    "reactor",
    "engine_room",
    "control_room",
    "heat_pipe",
    "radiator",
    "heat_exchanger",
    "thermal_",
    "power_storage",
    "factory",
    "ammo_factory",
    "ammo_storage",
    "airlock",
    "hyperdrive",
    "ftl_drive",
    "shield_gen",
    "sensor_array",
    "fire_extinguisher",
)

NON_TRAVERSABLE_HINTS = (
    "armor",
    "structure",
    "thruster",
    "point_defense",
    "cannon",
    "chaingun",
    "laser_blaster",
    "disruptor",
    "railgun",
    "ion_beam",
    "tractor_beam",
    "manipulator_beam",
    "mining_laser",
    "resonance_beam",
    "electro_bolter",
    "explosive_charge",
    "missile_launcher",
    "missile_silo",
    "missile_storage",
    "roof_",
)


@dataclass(frozen=True)
class RotationGeometry:
    """Geometry for one rotated vanilla part footprint."""

    rotation: int
    width: int
    height: int
    footprint_tiles: frozenset[Coord]
    unblocked_tiles: frozenset[Coord]
    blocked_travel_cells: frozenset[Coord]
    allowed_door_locations: Tuple[Coord, ...]


@dataclass(frozen=True)
class VanillaPartGeometry:
    """Per-rotation geometry table for one vanilla part ID."""

    part_id: str
    rotations: Dict[int, RotationGeometry]
    note: str = "game-file geometry"


@dataclass(frozen=True)
class PartMeta:
    """Shared footprint and traversability metadata for one part placement."""

    width: int
    height: int
    traversable: bool = False
    note: str = ""
    footprint_tiles: frozenset[Coord] = frozenset()
    unblocked_tiles: frozenset[Coord] = frozenset()
    blocked_travel_cells: frozenset[Coord] = frozenset()
    allowed_door_locations: Tuple[Coord, ...] = ()


def is_vanilla_part_id(part_id: str) -> bool:
    """Return True when a part ID belongs to the vanilla Cosmoteer namespace."""

    return part_id.startswith(VANILLA_NAMESPACE)


@lru_cache(maxsize=1)
def load_vanilla_part_geometry() -> Dict[str, VanillaPartGeometry]:
    """Load shared vanilla part geometry exported from game files.

    Returns:
        Mapping of part ID to per-rotation geometry metadata
    """

    with VANILLA_PARTS_PATH.open(encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    result: Dict[str, VanillaPartGeometry] = {}

    # Normalize the game-file export into small immutable records that can be
    # safely shared across preprocessing, training, and generation.
    for part in payload.get("parts", []):
        part_id = part.get("id")
        per_rotation = ((part.get("geometry") or {}).get("per_rotation") or {})
        rotations: Dict[int, RotationGeometry] = {}

        for key, rotation_payload in per_rotation.items():
            rotation = int(rotation_payload.get("rotation_index", key))
            size = rotation_payload.get("size") or [0, 0]
            rotations[rotation] = RotationGeometry(
                rotation=rotation,
                width=int(size[0]),
                height=int(size[1]),
                footprint_tiles=frozenset(
                    tuple(map(int, cell))
                    for cell in rotation_payload.get("footprint_tiles", [])
                ),
                unblocked_tiles=frozenset(
                    tuple(map(int, cell))
                    for cell in rotation_payload.get("unblocked_footprint_tiles", [])
                ),
                blocked_travel_cells=frozenset(
                    tuple(map(int, cell))
                    for cell in rotation_payload.get("blocked_travel_cells", [])
                ),
                allowed_door_locations=tuple(
                    tuple(map(int, cell))
                    for cell in rotation_payload.get("allowed_door_locations", [])
                ),
            )

        if part_id and rotations:
            result[part_id] = VanillaPartGeometry(part_id=part_id, rotations=rotations)

    return result


def normalize_part_id(part: dict) -> Optional[str]:
    """Return the canonical part ID used by extracted ship JSON.

    This function also applies preprocessing aliases so equivalent parts can be
    treated as one canonical vanilla part throughout the pipeline.
    """

    # Prefer explicit ID first, then fallback to legacy IDString field
    part_id = part.get("ID") or part.get("IDString")
    if not part_id:
        return None

    # Canonicalize equivalent parts to one ID for downstream preprocessing
    return PART_ID_ALIASES.get(part_id, part_id)


def infer_meta(part_id: str, rotation: int) -> Tuple[PartMeta, bool]:
    """Return `(PartMeta, is_inferred)` for one part ID and rotation.

    Args:
        part_id: Raw part identifier from ship JSON
        rotation: Part rotation index in Cosmoteer's 0-3 convention

    Returns:
        A metadata record plus a flag indicating whether the result fell back to
        regex/name-hint inference instead of exact vanilla geometry
    """

    vanilla_geometry = load_vanilla_part_geometry().get(part_id)
    if vanilla_geometry is not None:
        rotation_geometry = (
            vanilla_geometry.rotations.get(rotation)
            or vanilla_geometry.rotations.get(rotation % 4)
            or next(iter(vanilla_geometry.rotations.values()))
        )
        traversable = bool(rotation_geometry.unblocked_tiles)
        return (
            PartMeta(
                width=rotation_geometry.width,
                height=rotation_geometry.height,
                traversable=traversable,
                note="game-file geometry",
                footprint_tiles=rotation_geometry.footprint_tiles,
                unblocked_tiles=rotation_geometry.unblocked_tiles,
                blocked_travel_cells=rotation_geometry.blocked_travel_cells,
                allowed_door_locations=rotation_geometry.allowed_door_locations,
            ),
            False,
        )

    # Fall back to conservative rectangular inference for non-vanilla or
    # unknown parts so graph tooling can still reason about them approximately.
    match = re.search(r"_(\d+)x(\d+)(?:_|$)", part_id)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
    else:
        width, height = 1, 1

    lower_part_id = part_id.lower()
    traversable = any(token in lower_part_id for token in TRAVERSABLE_HINTS)
    if any(token in lower_part_id for token in NON_TRAVERSABLE_HINTS):
        traversable = False

    footprint_tiles = frozenset((x, y) for x in range(width) for y in range(height))
    return (
        PartMeta(
            width=width,
            height=height,
            traversable=traversable,
            note="regex/fallback inferred",
            footprint_tiles=footprint_tiles,
            unblocked_tiles=footprint_tiles if traversable else frozenset(),
        ),
        True,
    )


def iter_ship_files(input_dir: Path) -> Iterator[Path]:
    """Yield ship JSON files in deterministic order for corpus processing."""

    yield from sorted(input_dir.glob("*.json"))


__all__ = [
    "Coord",
    "DATA_DIR",
    "NON_TRAVERSABLE_HINTS",
    "PART_ID_ALIASES",
    "PartMeta",
    "RotationGeometry",
    "TRAVERSABLE_HINTS",
    "VANILLA_NAMESPACE",
    "VANILLA_PARTS_PATH",
    "VanillaPartGeometry",
    "infer_meta",
    "is_vanilla_part_id",
    "iter_ship_files",
    "load_vanilla_part_geometry",
    "normalize_part_id",
]
