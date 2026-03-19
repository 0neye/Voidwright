"""Ship-level relative coordinate transforms for preprocessing artifacts."""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence, Tuple

from common.geometry import infer_meta, normalize_part_id

Coord = Tuple[int, int]

__all__ = [
    "apply_relative_coords_transform",
    "canonicalize_for_translation_invariant_hash",
]


def _coerce_coord_pair(value: object) -> Coord | None:
    """Return an integer `(x, y)` tuple when *value* looks like a coordinate pair."""

    if not isinstance(value, list) or len(value) != 2:
        return None
    return int(value[0]), int(value[1])


def _iter_part_occupied_cells(parts: object) -> Iterator[Coord]:
    """Yield occupied world cells for every valid part in *parts*."""

    if not isinstance(parts, list):
        return

    for raw_part in parts:
        if not isinstance(raw_part, dict):
            continue

        normalized_part_id = normalize_part_id(raw_part)
        location = _coerce_coord_pair(raw_part.get("Location"))
        if not normalized_part_id or location is None:
            continue

        rotation = int(raw_part.get("Rotation", 0)) % 4
        part_meta, _inferred = infer_meta(normalized_part_id, rotation)
        origin_x, origin_y = location

        # Prefer exact game-file footprint tiles when available, then fall back
        # to the metadata rectangle so every part still contributes to the bbox.
        if part_meta.footprint_tiles:
            for local_x, local_y in part_meta.footprint_tiles:
                yield origin_x + local_x, origin_y + local_y
            continue

        for local_x in range(part_meta.width):
            for local_y in range(part_meta.height):
                yield origin_x + local_x, origin_y + local_y


def _resolve_bbox_center_2x(parts: object) -> tuple[list[int], dict]:
    """Compute occupied-cell bbox metadata and its integer center in 2x space."""

    occupied_cells = list(_iter_part_occupied_cells(parts))
    if not occupied_cells:
        return [0, 0], {"min": [0, 0], "max": [0, 0]}

    min_x = min(cell_x for cell_x, _cell_y in occupied_cells)
    max_x = max(cell_x for cell_x, _cell_y in occupied_cells)
    min_y = min(cell_y for _cell_x, cell_y in occupied_cells)
    max_y = max(cell_y for _cell_x, cell_y in occupied_cells)
    return [min_x + max_x, min_y + max_y], {"min": [min_x, min_y], "max": [max_x, max_y]}


def _to_local_2x(point: Sequence[int], center_2x: Sequence[int]) -> list[int]:
    """Convert one grid coordinate pair into the ship-local centered 2x frame."""

    return [
        int(point[0]) * 2 - int(center_2x[0]),
        int(point[1]) * 2 - int(center_2x[1]),
    ]


def _with_location_2x(parts: object, center_2x: Sequence[int]) -> list[dict]:
    """Copy part records and replace world `Location` with centered `Location2x`."""

    transformed_parts: list[dict] = []
    for raw_part in parts if isinstance(parts, list) else []:
        if not isinstance(raw_part, dict):
            continue

        transformed_part = dict(raw_part)
        location = _coerce_coord_pair(raw_part.get("Location"))
        if location is not None:
            transformed_part["Location2x"] = _to_local_2x(location, center_2x)
            transformed_part.pop("Location", None)
        transformed_parts.append(transformed_part)
    return transformed_parts


def _with_cell_2x(doors: object, center_2x: Sequence[int]) -> list[dict]:
    """Copy door records and replace world `Cell` with centered `Cell2x`."""

    transformed_doors: list[dict] = []
    for raw_door in doors if isinstance(doors, list) else []:
        if not isinstance(raw_door, dict):
            continue

        transformed_door = dict(raw_door)
        door_cell = _coerce_coord_pair(raw_door.get("Cell"))
        if door_cell is not None:
            transformed_door["Cell2x"] = _to_local_2x(door_cell, center_2x)
            transformed_door.pop("Cell", None)
        transformed_doors.append(transformed_door)
    return transformed_doors


def apply_relative_coords_transform(ship_data: dict) -> dict:
    """Return *ship_data* rewritten into the centered `2x` local coordinate frame."""

    transformed_payload = dict(ship_data)
    center_2x, occupied_bbox = _resolve_bbox_center_2x(ship_data.get("Parts"))

    transformed_payload["Parts"] = _with_location_2x(ship_data.get("Parts"), center_2x)
    transformed_payload["Doors"] = _with_cell_2x(ship_data.get("Doors", []), center_2x)
    transformed_payload["coord_transform"] = {
        "version": 1,
        "frame": "bbox_center_2x",
        "scale": 2,
        "center_2x": center_2x,
        "occupied_bbox": occupied_bbox,
    }
    return transformed_payload


def canonicalize_for_translation_invariant_hash(ship_data: object) -> object:
    """Return a dedupe-safe projection where global translation does not matter."""

    if not isinstance(ship_data, dict):
        return ship_data

    projected_payload = dict(ship_data)
    projected_parts: list[dict] = []
    for raw_part in ship_data.get("Parts", []) if isinstance(ship_data.get("Parts"), list) else []:
        if not isinstance(raw_part, dict):
            continue

        projected_parts.append(dict(raw_part))
    projected_payload["Parts"] = projected_parts

    projected_doors: list[dict] = []
    for raw_door in ship_data.get("Doors", []) if isinstance(ship_data.get("Doors"), list) else []:
        if not isinstance(raw_door, dict):
            continue

        projected_doors.append(dict(raw_door))
    projected_payload["Doors"] = projected_doors

    coord_transform = ship_data.get("coord_transform")
    center_2x: list[int] | None = None
    if isinstance(coord_transform, dict):
        raw_center = coord_transform.get("center_2x")
        if isinstance(raw_center, list) and len(raw_center) == 2:
            center_2x = [int(raw_center[0]), int(raw_center[1])]
        projected_payload["coord_transform"] = {
            "version": int(coord_transform.get("version", 1)),
            "frame": str(coord_transform.get("frame", "bbox_center_2x")),
            "scale": int(coord_transform.get("scale", 2)),
        }

    # PartUIToggleStates entries carry Key[0].Location in world grid coordinates.
    # Without canonicalization, translated copies of the same overclocked ship
    # produce different hashes even though their normalised layouts are identical.
    # Replace Key[0].Location with a centered-2x equivalent so the hash is
    # invariant to pure translation.
    raw_toggle_states = ship_data.get("PartUIToggleStates")
    if isinstance(raw_toggle_states, list):
        projected_toggle_states: list[dict] = []
        for entry in raw_toggle_states:
            if not isinstance(entry, dict):
                continue
            key = entry.get("Key")
            if not isinstance(key, list) or not key or not isinstance(key[0], dict):
                projected_toggle_states.append(dict(entry))
                continue
            projected_key_0 = dict(key[0])
            if center_2x is not None:
                loc = _coerce_coord_pair(projected_key_0.get("Location"))
                if loc is not None:
                    projected_key_0["Location2x"] = _to_local_2x(loc, center_2x)
                    projected_key_0.pop("Location", None)
            projected_entry = dict(entry)
            projected_entry["Key"] = [projected_key_0, *key[1:]]
            projected_toggle_states.append(projected_entry)
        projected_payload["PartUIToggleStates"] = projected_toggle_states

    return projected_payload
