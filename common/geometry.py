"""Shared Cosmoteer geometry and ship-layout helpers.

This module is the canonical owner of geometry metadata that is shared across
preprocessing, training, and generation. It intentionally avoids importing
package-specific logic so other modules can depend on it without creating
layering inversions.
"""

from __future__ import annotations

import orjson
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional, Tuple

Coord = Tuple[int, int]
Coord2x = Tuple[int, int]
DATA_DIR = Path(__file__).resolve().parent / "data"
VANILLA_PARTS_PATH = DATA_DIR / "vanilla_parts_full_geometry.json"
VANILLA_NAMESPACE = "cosmoteer."
PART_ID_ALIASES = {
    # Legacy / alternate vanilla IDs declared via OtherIDs in terran rules.
    "ammo_factory": "cosmoteer.factory_ammo",
    "ammo_storage": "cosmoteer.storage_2x2",
    "ammo_supply": "cosmoteer.factory_ammo",
    "armor": "cosmoteer.armor",
    "armor2": "cosmoteer.armor_2x1",
    "armor_tri": "cosmoteer.armor_tri",
    "armor_wedge": "cosmoteer.armor_wedge",
    "aux_cockpit": "cosmoteer.control_room_small",
    "big_cannon": "cosmoteer.cannon_large",
    "big_thruster": "cosmoteer.thruster_large",
    "bunk": "cosmoteer.crew_quarters_small",
    "cockpit": "cosmoteer.control_room_small",
    "conveyor": "cosmoteer.conveyor",
    "corridor": "cosmoteer.corridor",
    "cosmoteer.ammo_factory": "cosmoteer.factory_ammo",
    "cosmoteer.ammo_storage": "cosmoteer.storage_2x2",
    "cosmoteer.crew_quarters_small_a": "cosmoteer.crew_quarters_small",
    "cosmoteer.crew_quarters_small_b": "cosmoteer.crew_quarters_small",
    "cosmoteer.crew_quarters_small_c": "cosmoteer.crew_quarters_small",
    "cosmoteer.deck_cannon": "cosmoteer.cannon_deck",
    "cosmoteer.electro_bolter": "cosmoteer.disruptor",
    "cosmoteer.ftl_drive": "cosmoteer.hyperdrive_small",
    "cosmoteer.heat_pipe": "cosmoteer.heat_pipe_adaptive",
    "cosmoteer.heat_pipe_3way_junction": "cosmoteer.heat_pipe_adaptive",
    "cosmoteer.heat_pipe_4way_junction": "cosmoteer.heat_pipe_adaptive",
    "cosmoteer.heat_pipe_corner": "cosmoteer.heat_pipe_adaptive",
    "cosmoteer.heat_pipe_junction": "cosmoteer.heat_pipe_adaptive",
    "cosmoteer.heat_sink": "cosmoteer.heat_exchanger",
    "cosmoteer.ion_beam_prism_45": "cosmoteer.ion_beam_prism",
    "cosmoteer.laser_blaster": "cosmoteer.laser_blaster_small",
    # Legacy / alternate factory IDs declared via OtherIDs in vanilla rules.
    "cosmoteer.missile_factory": "cosmoteer.factory_he",
    "missile_factory": "cosmoteer.factory_he",
    "cosmoteer.missile_factory_high_explosive": "cosmoteer.factory_he",
    "cosmoteer.missile_factory_he": "cosmoteer.factory_he",
    "cosmoteer.missile_factory_emp": "cosmoteer.factory_emp",
    "cosmoteer.missile_factory_nuke": "cosmoteer.factory_nuke",
    "cosmoteer.missile_storage": "cosmoteer.storage_3x2",
    "cosmoteer.mine_factory": "cosmoteer.factory_mine",
    "cosmoteer.resource_collector": "cosmoteer.manipulator_beam_emitter",
    "cosmoteer.thermal_tank": "cosmoteer.thermal_battery",
    "electro_bolt": "cosmoteer.disruptor",
    "explosive_charge": "cosmoteer.explosive_charge",
    "fire_extinguisher": "cosmoteer.fire_extinguisher",
    "ftl_drive": "cosmoteer.hyperdrive_small",
    "general_jobs": "cosmoteer.crew_quarters_med",
    "ion_beam": "cosmoteer.ion_beam_emitter",
    "med_cannon": "cosmoteer.cannon_med",
    "med_thruster": "cosmoteer.thruster_med",
    "missile_launcher": "cosmoteer.missile_launcher",
    "missile_storage": "cosmoteer.storage_3x2",
    "point_defense": "cosmoteer.point_defense",
    "power_storage": "cosmoteer.power_storage",
    "power_supply": "cosmoteer.power_storage",
    "quarters": "cosmoteer.crew_quarters_med",
    "reactor": "cosmoteer.reactor_small",
    "sensor_array": "cosmoteer.sensor_array",
    "shield_generator": "cosmoteer.shield_gen_small",
    "small_laser": "cosmoteer.laser_blaster_small",
    "small_thruster": "cosmoteer.thruster_small",
    "structure": "cosmoteer.structure",
    "structure_wedge": "cosmoteer.structure_wedge",
    # The _L variant of each flippable wedge is identical to the base part.
    # The handed _R variant must stay out of PART_ID_ALIASES because
    # normalize_part_id() cannot also remap rotation. Those go in
    # FLIP_H_PART_IDS so geometry lookup preserves the current wedge behavior.
    "cosmoteer.armor_1x2_wedge_L": "cosmoteer.armor_1x2_wedge",
    "armor2_wedge_L": "cosmoteer.armor_1x2_wedge",
    "Kroom.Armor_1x3_Wedge_L": "cosmoteer.armor_1x3_wedge",
    "cosmoteer.structure_1x2_wedge_L": "cosmoteer.structure_1x2_wedge",
    "structure2_wedge_L": "cosmoteer.structure_1x2_wedge",
    "Kroom.Structure_1x3_Wedge_L": "cosmoteer.structure_1x3_wedge",
}

# Part IDs that map to a base part with a horizontal flip (mirror) applied on
# load.  The _R wedge variants are stored under the base part ID in the geometry
# database, but the game mirrors them when deserialising ship files.
#
# FlipH also remaps the rotation index.  The rules files specify:
#   FlipHRotate = [0, 3, 2, 1]
# meaning a part saved at rotation r is equivalent to the base part at rotation
# FLIP_H_ROTATE[r] after mirroring.
FLIP_H_PART_IDS: Dict[str, str] = {
    "armor2_wedge_R": "cosmoteer.armor_1x2_wedge",
    "cosmoteer.armor_1x2_wedge_R": "cosmoteer.armor_1x2_wedge",
    "Kroom.Armor_1x3_Wedge_R": "cosmoteer.armor_1x3_wedge",
    "structure2_wedge_R": "cosmoteer.structure_1x2_wedge",
    "cosmoteer.structure_1x2_wedge_R": "cosmoteer.structure_1x2_wedge",
    "Kroom.Structure_1x3_Wedge_R": "cosmoteer.structure_1x3_wedge",
}
# Rotation remapping applied together with a horizontal flip (index = saved
# rotation, value = equivalent base-part rotation after mirroring).
FLIP_H_ROTATE: Tuple[int, int, int, int] = (0, 3, 2, 1)

# Canonical base part IDs that support a horizontal flip in real ship data.
# These are exactly the target IDs from FLIP_H_PART_IDS (the wedge types whose
# right-handed _R variants appear there).  When a part in this set has
# flip_x=True in graph data, it should be encoded as a virtual training token
# via encode_flipped_part_id() so the model learns the flipped geometry as a
# distinct part type rather than via a separate imbalanced binary head.
FLIPPABLE_PART_IDS: frozenset[str] = frozenset(FLIP_H_PART_IDS.values())

# Suffix appended to a base part ID to form the virtual training token for a
# horizontally-flipped wedge.  Double-underscore prevents collision with any
# real game ID (which uses single underscores and namespace dots only).
FLIPPED_PART_ID_SUFFIX: str = "__flipped"

# These fallback substrings are only used for unknown or non-vanilla parts.
# They are intentionally conservative and derived from the vanilla corpus:
# categories with positive crew_speed_factor go here, while known zero-speed
# categories stay in NON_TRAVERSABLE_HINTS below.
TRAVERSABLE_HINTS = (
    "airlock",
    "cannon",
    "chaingun",
    "control_room",
    "conveyor",
    "corridor",
    "crew_quarters",
    "quarters",
    "disruptor",
    "engine_room",
    "factory",
    "fire_extinguisher",
    "ftl_drive",
    "hyperdrive",
    "ion_beam_emitter",
    "laser_blaster",
    "manipulator_beam",
    "mining_laser",
    "missile_launcher",
    "point_defense",
    "power_storage",
    "railgun",
    "reactor",
    "resonance_beam",
    "sensor_array",
    "shield_gen",
    "storage",
    "storage_",
    "thermal_amplification_pump",
    "thermal_dilation_pump",
    "thruster",
    "tractor_beam",
)

NON_TRAVERSABLE_HINTS = (
    "armor",
    "explosive_charge",
    "heat_exchanger",
    "heat_pipe_adaptive",
    "ion_beam_prism",
    "radiator",
    "roof_",
    "structure",
    "thermal_battery",
)

__all__ = [
    "Coord",
    "Coord2x",
    "DATA_DIR",
    "FLIP_H_PART_IDS",
    "FLIP_H_ROTATE",
    "FLIPPABLE_PART_IDS",
    "FLIPPED_PART_ID_SUFFIX",
    "NON_TRAVERSABLE_HINTS",
    "PART_ID_ALIASES",
    "PartMeta",
    "PartRect",
    "RotationGeometry",
    "ThermalPort",
    "TRAVERSABLE_HINTS",
    "UIToggleDef",
    "VANILLA_NAMESPACE",
    "VANILLA_PARTS_PATH",
    "VanillaPartGeometry",
    "decode_flipped_part_id",
    "encode_flipped_part_id",
    "infer_meta",
    "is_vanilla_part_id",
    "iter_ship_files",
    "load_vanilla_part_geometry",
    "normalize_part_id",
    "parse_polygon_vertex",
    "part_rect_to_2x_bounds",
    "polygon_vertices_to_2x",
    "resolve_geometry_part_id_and_rotation",
]


@dataclass(frozen=True)
class PartRect:
    """Rectangular local-tile region used by save and collision metadata."""

    x: int
    y: int
    width: int
    height: int
    source: str


@dataclass(frozen=True)
class ThermalPort:
    """One thermal network port on a vanilla part at a specific rotation."""

    location: Coord  # part-local tile coord (NOT 2x-scaled)
    direction: str   # "Up", "Down", "Left", or "Right"
    overclock_conditional: bool  # True = only active when part is overclocked


@dataclass(frozen=True)
class RotationGeometry:
    """Geometry for one rotated vanilla part footprint.

    Note:
        This record carries richer per-rotation travel metadata sourced from
        ``common/data/vanilla_parts_full_geometry.json``. Preprocessing graph
        JSON intentionally does not inline these fields for now; later stages
        that need detailed movement semantics should load them from the shared
        geometry cache on demand.
    """

    rotation: int
    width: int
    height: int
    footprint_tiles: frozenset[Coord]
    unblocked_tiles: frozenset[Coord]
    blocked_travel_cells: frozenset[Coord]
    allowed_door_locations: Tuple[Coord, ...]
    polygon_vertices: Tuple[Tuple[float, float], ...] = ()
    blocked_travel_cell_directions: Mapping[Coord, frozenset[str]] = field(default_factory=dict)
    force_manhattan_path: bool | None = None
    crew_speed_factor: float | None = None
    crew_speed_by_direction: Mapping[str, float] | None = None
    crew_congested_speed_factor: float | None = None
    crew_congested_speed_by_direction: Mapping[str, float] | None = None
    thermal_ports: Tuple[ThermalPort, ...] = ()

    def crew_speed_for_direction(self, direction: str, default: float | None = None) -> float | None:
        """Return the travel speed for a world-cardinal movement direction."""

        if self.crew_speed_by_direction:
            return self.crew_speed_by_direction.get(direction, default)
        if self.crew_speed_factor is not None:
            return self.crew_speed_factor
        return default

    def crew_congested_speed_for_direction(
        self,
        direction: str,
        default: float | None = None,
    ) -> float | None:
        """Return the congested travel speed for a world-cardinal direction."""

        if self.crew_congested_speed_by_direction:
            return self.crew_congested_speed_by_direction.get(direction, default)
        if self.crew_congested_speed_factor is not None:
            return self.crew_congested_speed_factor
        return default

    def is_direction_blocked(self, tile: Coord, direction: str) -> bool:
        """Return True when movement from *tile* in *direction* is blocked."""

        return direction in self.blocked_travel_cell_directions.get(tile, frozenset())


@dataclass(frozen=True)
class VanillaPartGeometry:
    """Per-rotation geometry table for one vanilla part ID."""

    part_id: str
    rotations: Dict[int, RotationGeometry]
    save_rect: Optional[PartRect] = None
    physical_rect: Optional[PartRect] = None
    crew_speed_factor: float | None = None
    crew_speed_by_direction: Mapping[str, float] | None = None
    crew_congested_speed_factor: float | None = None
    crew_congested_speed_by_direction: Mapping[str, float] | None = None
    cell_occupancy_factor: Optional[float] = None
    ui_toggles: Tuple[UIToggleDef, ...] = ()
    note: str = "game-file geometry"

    def rotation_geometry(self, rotation: int) -> RotationGeometry:
        """Return the best matching rotation geometry for *rotation*."""

        return (
            self.rotations.get(rotation)
            or self.rotations.get(rotation % 4)
            or next(iter(self.rotations.values()))
        )

    def crew_speed_for_direction(
        self,
        rotation: int,
        direction: str,
        default: float | None = None,
    ) -> float | None:
        """Return travel speed for a rotated part moving in a world direction."""

        rotation_geometry = self.rotation_geometry(rotation)
        per_rotation_speed = rotation_geometry.crew_speed_for_direction(direction)
        if per_rotation_speed is not None:
            return per_rotation_speed
        if self.crew_speed_by_direction:
            return self.crew_speed_by_direction.get(direction, default)
        if self.crew_speed_factor is not None:
            return self.crew_speed_factor
        return default


@dataclass(frozen=True)
class UIToggleDef:
    """Definition of one UIToggle component exposed in PartUIToggleStates.

    Attributes:
        component: Rules-file component name (e.g. ``"MissileType"``).
        toggle_id: ToggleID string used as the key in ``PartUIToggleStates``
            (e.g. ``"missile_type"``).
        default: Default integer value when the toggle is not set in a ship
            file.
        choices: Sorted tuple of all valid integer mode values.  Binary
            toggles have ``(0, 1)``; multi-mode toggles list every mode.
        always_allow_in_build_mode: ``True`` when the toggle can be changed
            in the ship editor; ``None`` when not specified in the rules file.
    """

    component: str
    toggle_id: str
    default: int
    choices: Tuple[int, ...]
    always_allow_in_build_mode: bool | None = None


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
    ui_toggles: Tuple[UIToggleDef, ...] = ()


def parse_coord(cell: object) -> Coord:
    """Convert a raw two-element coordinate payload into a typed Coord."""

    if not isinstance(cell, (list, tuple)) or len(cell) != 2:
        raise ValueError(f"Expected a two-element coordinate, got {cell!r}")
    return int(cell[0]), int(cell[1])


def parse_polygon_vertex(vertex: object) -> Tuple[float, float]:
    """Convert a raw polygon vertex payload into an `(x, y)` tuple."""

    if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
        raise ValueError(f"Expected a two-element polygon vertex, got {vertex!r}")
    return float(vertex[0]), float(vertex[1])


def parse_part_rect(raw_rect: object, *, source: str) -> Optional[PartRect]:
    """Convert a raw four-element rectangle payload into a typed PartRect.

    Args:
        raw_rect: Raw JSON payload from the exported vanilla geometry file
        source: Short label describing where this rect came from

    Returns:
        A parsed PartRect, or None when the payload is null
    """

    if raw_rect is None:
        return None
    if not isinstance(raw_rect, (list, tuple)) or len(raw_rect) != 4:
        raise ValueError(f"Expected a four-element rect, got {raw_rect!r}")
    return PartRect(
        x=int(raw_rect[0]),
        y=int(raw_rect[1]),
        width=int(raw_rect[2]),
        height=int(raw_rect[3]),
        source=source,
    )


def polygon_vertices_to_2x(
    vertices: Tuple[Tuple[float, float], ...],
) -> Tuple[Coord2x, ...]:
    """Convert local tile-space polygon vertices into exact integer 2x points."""

    local_vertices_2x: list[Coord2x] = []
    for vertex_x, vertex_y in vertices:
        local_vertices_2x.append((int(round(vertex_x * 2)), int(round(vertex_y * 2))))
    return tuple(local_vertices_2x)


def part_rect_to_2x_bounds(
    rect: Optional[PartRect],
) -> Optional[Tuple[int, int, int, int]]:
    """Convert a PartRect into `(x2, y2, width2, height2)` 2x-space bounds."""

    if rect is None:
        return None
    return (rect.x * 2, rect.y * 2, rect.width * 2, rect.height * 2)


def _normalize_directional_speed_map(raw_value: object) -> Dict[str, float] | None:
    """Normalize a scalar-or-mapping travel speed payload into a direction map."""

    if not isinstance(raw_value, Mapping):
        return None
    normalized: Dict[str, float] = {}
    for direction, value in raw_value.items():
        if not isinstance(direction, str):
            continue
        try:
            normalized[direction] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized or None


def _normalize_blocked_travel_directions(raw_value: object) -> Dict[Coord, frozenset[str]]:
    """Normalize per-cell blocked travel directions from exported JSON."""

    if not isinstance(raw_value, list):
        return {}
    normalized: Dict[Coord, frozenset[str]] = {}
    for entry in raw_value:
        if not isinstance(entry, Mapping):
            continue
        tile = entry.get("tile")
        if not isinstance(tile, (list, tuple)) or len(tile) != 2:
            continue
        directions = entry.get("value")
        if not isinstance(directions, list):
            continue
        direction_names = frozenset(str(direction) for direction in directions if isinstance(direction, str))
        if not direction_names:
            continue
        normalized[(int(tile[0]), int(tile[1]))] = direction_names
    return normalized


def _has_positive_travel_speed(
    scalar_speed: float | None,
    directional_speed: Mapping[str, float] | None,
) -> bool:
    """Return True when either scalar or directional travel speed is positive."""

    if scalar_speed is not None and scalar_speed > 0:
        return True
    if directional_speed:
        return any(speed > 0 for speed in directional_speed.values())
    return False


def is_vanilla_part_id(part_id: str) -> bool:
    """Return True when a part ID belongs to the vanilla Cosmoteer namespace."""

    return part_id.startswith(VANILLA_NAMESPACE)


@lru_cache(maxsize=1)
def load_vanilla_part_geometry() -> Dict[str, VanillaPartGeometry]:
    """Load shared vanilla part geometry exported from game files.

    Returns:
        Mapping of part ID to per-rotation geometry metadata
    """

    payload = orjson.loads(VANILLA_PARTS_PATH.read_bytes())

    result: Dict[str, VanillaPartGeometry] = {}

    # Normalize the game-file export into small immutable records that can be
    # safely shared across preprocessing, training, and generation.
    for part in payload.get("parts", []):
        part_id = part.get("id")
        per_rotation = ((part.get("geometry") or {}).get("per_rotation") or {})
        travel_payload = part.get("travel") or {}
        top_level_speed = part.get("crew_speed_factor", travel_payload.get("crew_speed_factor"))
        top_level_congested_speed = part.get(
            "crew_congested_speed_factor",
            travel_payload.get("crew_congested_speed_factor"),
        )
        rotations: Dict[int, RotationGeometry] = {}
        source_file = str(part.get("source_file") or "repo geometry export")

        for key, rotation_payload in per_rotation.items():
            rotation = int(rotation_payload.get("rotation_index", key))
            size = rotation_payload.get("size") or [0, 0]
            rotation_speed = rotation_payload.get("crew_speed_factor")
            rotation_congested_speed = rotation_payload.get("crew_congested_speed_factor")
            force_manhattan = rotation_payload.get("force_manhattan_path")
            rotations[rotation] = RotationGeometry(
                rotation=rotation,
                width=int(size[0]),
                height=int(size[1]),
                footprint_tiles=frozenset(
                    parse_coord(cell)
                    for cell in rotation_payload.get("footprint_tiles", [])
                ),
                unblocked_tiles=frozenset(
                    parse_coord(cell)
                    for cell in rotation_payload.get("unblocked_footprint_tiles", [])
                ),
                blocked_travel_cells=frozenset(
                    parse_coord(cell)
                    for cell in rotation_payload.get("blocked_travel_cells", [])
                ),
                allowed_door_locations=tuple(
                    parse_coord(cell)
                    for cell in rotation_payload.get("allowed_door_locations", [])
                ),
                polygon_vertices=tuple(
                    parse_polygon_vertex(vertex)
                    for vertex in rotation_payload.get("polygon_vertices", [])
                ),
                blocked_travel_cell_directions=_normalize_blocked_travel_directions(
                    rotation_payload.get("blocked_travel_cell_directions", [])
                ),
                force_manhattan_path=(bool(force_manhattan) if force_manhattan is not None else None),
                crew_speed_factor=(
                    float(rotation_speed)
                    if isinstance(rotation_speed, (int, float))
                    else None
                ),
                crew_speed_by_direction=_normalize_directional_speed_map(rotation_speed),
                crew_congested_speed_factor=(
                    float(rotation_congested_speed)
                    if isinstance(rotation_congested_speed, (int, float))
                    else None
                ),
                crew_congested_speed_by_direction=_normalize_directional_speed_map(
                    rotation_congested_speed
                ),
                thermal_ports=tuple(
                    ThermalPort(
                        location=(int(tp["location"][0]), int(tp["location"][1])),
                        direction=str(tp["direction"]),
                        overclock_conditional=bool(tp.get("overclock_conditional", False)),
                    )
                    for tp in rotation_payload.get("thermal_ports", [])
                    if isinstance(tp, dict)
                       and isinstance(tp.get("location"), list)
                       and len(tp["location"]) == 2
                       and isinstance(tp.get("direction"), str)
                ),
            )

        if part_id and rotations:
            ui_toggles_raw = part.get("ui_toggles") or []
            ui_toggles = tuple(
                UIToggleDef(
                    component=str(t["component"]),
                    toggle_id=str(t["toggle_id"]),
                    default=int(t.get("default", 0)),
                    choices=tuple(int(c) for c in (t.get("choices") or [])),
                    always_allow_in_build_mode=(
                        bool(t["always_allow_in_build_mode"])
                        if t.get("always_allow_in_build_mode") is not None
                        else None
                    ),
                )
                for t in ui_toggles_raw
                if isinstance(t, dict) and t.get("toggle_id")
            )
            result[part_id] = VanillaPartGeometry(
                part_id=part_id,
                rotations=rotations,
                save_rect=parse_part_rect(
                    part.get("save_rect"),
                    source=f"{source_file}:save_rect",
                ),
                physical_rect=parse_part_rect(
                    part.get("physical_rect"),
                    source=f"{source_file}:physical_rect",
                ),
                crew_speed_factor=(
                    float(top_level_speed)
                    if isinstance(top_level_speed, (int, float))
                    else None
                ),
                crew_speed_by_direction=_normalize_directional_speed_map(top_level_speed),
                crew_congested_speed_factor=(
                    float(top_level_congested_speed)
                    if isinstance(top_level_congested_speed, (int, float))
                    else None
                ),
                crew_congested_speed_by_direction=_normalize_directional_speed_map(
                    top_level_congested_speed
                ),
                cell_occupancy_factor=travel_payload.get("cell_occupancy_factor"),
                ui_toggles=ui_toggles,
            )

    return result


def encode_flipped_part_id(part_id: str) -> str:
    """Return the virtual training token for a horizontally-flipped wedge.

    The returned ID uses the lowercased *part_id* so it matches the lowercase
    vocabulary entries built by ``VocabRegistry.build_from_corpus``.  The
    caller is responsible for checking that *part_id* belongs to
    ``FLIPPABLE_PART_IDS`` before calling this function.
    """
    return part_id.lower() + FLIPPED_PART_ID_SUFFIX


def decode_flipped_part_id(virtual_id: str) -> Optional[str]:
    """Return the base part ID when *virtual_id* is a flipped virtual token.

    Returns ``None`` when *virtual_id* is not a virtual flipped token.  The
    returned base ID is lowercased and matches entries in ``FLIPPABLE_PART_IDS``.
    """
    if virtual_id.endswith(FLIPPED_PART_ID_SUFFIX):
        return virtual_id[: -len(FLIPPED_PART_ID_SUFFIX)]
    return None


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


def resolve_geometry_part_id_and_rotation(
    part_id: str,
    rotation: int,
) -> Tuple[str, int]:
    """Resolve aliases and mirrored IDs to geometry-cache lookup coordinates.

    Args:
        part_id: Raw or normalized part ID from ship data
        rotation: Rotation index in Cosmoteer's 0-3 convention

    Returns:
        Tuple of `(geometry_part_id, geometry_rotation)` that can be used to
        index `load_vanilla_part_geometry()`
    """

    resolved_part_id = PART_ID_ALIASES.get(part_id, part_id)
    resolved_rotation = int(rotation) % 4

    # `_R` wedge IDs are mirrored aliases that must map to the base part ID
    # plus the FlipH rotation remap before indexing geometry caches.
    if resolved_part_id in FLIP_H_PART_IDS:
        resolved_part_id = FLIP_H_PART_IDS[resolved_part_id]
        resolved_rotation = FLIP_H_ROTATE[resolved_rotation]

    return resolved_part_id, resolved_rotation


def infer_meta(part_id: str, rotation: int) -> Tuple[PartMeta, bool]:
    """Return `(PartMeta, is_inferred)` for one part ID and rotation.

    Args:
        part_id: Raw part identifier from ship JSON
        rotation: Part rotation index in Cosmoteer's 0-3 convention

    Returns:
        A metadata record plus a flag indicating whether the result fell back to
        regex/name-hint inference instead of exact vanilla geometry
    """

    part_id, rotation = resolve_geometry_part_id_and_rotation(part_id, rotation)

    vanilla_geometry = load_vanilla_part_geometry().get(part_id)
    if vanilla_geometry is not None:
        rotation_geometry = vanilla_geometry.rotation_geometry(rotation)

        # Vanilla traversability is gated by travel speed, while unblocked tiles
        # still describe which cells inside the footprint are actually walkable
        # once movement is allowed. Some parts, such as conveyors, now expose
        # direction-specific travel speed maps instead of one scalar factor.
        traversable = _has_positive_travel_speed(
            vanilla_geometry.crew_speed_factor,
            vanilla_geometry.crew_speed_by_direction,
        ) and bool(rotation_geometry.unblocked_tiles)
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
                ui_toggles=vanilla_geometry.ui_toggles,
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

    yield from sorted(p for p in input_dir.glob("*.json") if not p.name.startswith("."))

