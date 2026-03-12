"""Markov backend CLI input parsing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from common.cosmoteer import parse_ship_png
from preprocessing.relative_coords import apply_relative_coords_transform

__all__ = [
    "load_allowlist",
    "load_requirements",
    "load_seed_parts_from_json",
    "load_seed_parts_from_png",
]


def _coerce_coord_pair(value: object) -> list[int] | None:
    """Return an integer coordinate pair when *value* looks like `[x, y]`."""

    if not isinstance(value, list) or len(value) != 2:
        return None
    return [int(value[0]), int(value[1])]


def _is_preprocessed_relative_payload(ship_data: object) -> bool:
    """Return True when payload already has centered-`2x` preprocessing metadata.

    Args:
        ship_data: Parsed ship payload from PNG extraction

    Returns:
        True when `coord_transform.center_2x` and at least one `Parts[*].Location2x`
        coordinate are present
    """

    if not isinstance(ship_data, dict):
        return False
    coord_transform = ship_data.get("coord_transform")
    center_2x = (
        _coerce_coord_pair(coord_transform.get("center_2x"))
        if isinstance(coord_transform, dict)
        else None
    )
    if center_2x is None:
        return False
    for raw_part in ship_data.get("Parts", []) if isinstance(ship_data.get("Parts"), list) else []:
        if isinstance(raw_part, dict) and _coerce_coord_pair(raw_part.get("Location2x")) is not None:
            return True
    return False


def _contains_world_location_parts(ship_data: object) -> bool:
    """Return True when payload still uses world-grid `Parts[*].Location` fields."""

    if not isinstance(ship_data, dict):
        return False
    for raw_part in ship_data.get("Parts", []) if isinstance(ship_data.get("Parts"), list) else []:
        if isinstance(raw_part, dict) and _coerce_coord_pair(raw_part.get("Location")) is not None:
            return True
    return False


def _prepare_seed_ship_for_markov(ship_data: dict) -> dict:
    """Normalize parsed seed payload into preprocessing's centered `2x` frame.

    This mirrors the preprocessing extract stage behavior so `--seed-png` ships
    with only world `Location` fields still flow through the same coordinate
    normalization before Markov seed loading.
    """

    # Keep already-normalized payloads untouched so existing `Location2x` +
    # `coord_transform` metadata is preserved exactly.
    if _is_preprocessed_relative_payload(ship_data):
        return ship_data
    if _contains_world_location_parts(ship_data):
        return apply_relative_coords_transform(ship_data)
    return ship_data


def load_allowlist(
    allowlist_values: Optional[list[str]],
    allowlist_file_path: Optional[Path],
) -> Optional[frozenset[str]]:
    """Combine inline and file-based allowlist values.

    Args:
        allowlist_values: Values supplied directly on the command line
        allowlist_file_path: Optional file containing one value per line or a JSON array

    Returns:
        A frozenset of allowed part IDs, or None when no allowlist is provided
    """

    allowed_part_ids: set[str] = set()
    if allowlist_values:
        allowed_part_ids.update(value.strip() for value in allowlist_values if value.strip())
    if allowlist_file_path is not None:
        file_text = allowlist_file_path.read_text(encoding="utf-8")
        stripped_file_text = file_text.strip()
        if stripped_file_text.startswith("["):
            allowed_part_ids.update(json.loads(stripped_file_text))
        else:
            for raw_line in file_text.splitlines():
                line = raw_line.strip()
                if line and not line.startswith("#"):
                    allowed_part_ids.add(line)
    return frozenset(allowed_part_ids) if allowed_part_ids else None


def load_requirements(
    requirement_pairs: Optional[list[list[str]]],
    requirements_file_path: Optional[Path],
) -> Optional[dict[str, int]]:
    """Parse generation requirements from CLI arguments and an optional file.

    Args:
        requirement_pairs: Repeated `PART_ID COUNT` pairs from the CLI
        requirements_file_path: Optional JSON or line-based requirements file

    Returns:
        Mapping of part IDs to minimum required counts, or None when empty
    """

    part_requirements: dict[str, int] = {}
    if requirement_pairs:
        for part_id, count_text in requirement_pairs:
            count = int(count_text)
            if count <= 0:
                raise ValueError(f"Requirement count must be > 0, got {count} for {part_id}")
            normalized_part_id = part_id.strip()
            part_requirements[normalized_part_id] = max(
                part_requirements.get(normalized_part_id, 0),
                count,
            )
    if requirements_file_path is not None:
        file_text = requirements_file_path.read_text(encoding="utf-8").strip()
        if file_text.startswith("{"):
            parsed_requirements = json.loads(file_text)
            for part_id, count in parsed_requirements.items():
                normalized_part_id = part_id.strip()
                part_requirements[normalized_part_id] = max(
                    part_requirements.get(normalized_part_id, 0),
                    int(count),
                )
        else:
            for raw_line in file_text.splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                pieces = line.split()
                if len(pieces) != 2:
                    raise ValueError(f"Expected 'PART_ID COUNT' in requirements file, got: {line!r}")
                part_id, count_text = pieces
                normalized_part_id = part_id.strip()
                part_requirements[normalized_part_id] = max(
                    part_requirements.get(normalized_part_id, 0),
                    int(count_text),
                )
    return part_requirements if part_requirements else None


def load_seed_parts_from_json(seed_json_path: Path) -> list[dict]:
    """Load seed parts from generated JSON or extracted Cosmoteer ship JSON."""

    data = json.loads(seed_json_path.read_text(encoding="utf-8"))
    if "parts" in data and isinstance(data["parts"], list):
        raw_parts = data["parts"]
        if raw_parts and isinstance(raw_parts[0], dict) and "part_id" in raw_parts[0]:
            return raw_parts
        return [
            {
                "part_id": part["ID"],
                "rotation": int(part.get("Rotation", 0)),
                "x": int(part["Location"][0]),
                "y": int(part["Location"][1]),
                "flip_x": bool(part.get("FlipX", False)),
                "flip_y": bool(part.get("FlipY", False)),
            }
            for part in raw_parts
            if isinstance(part, dict) and "ID" in part and "Location" in part
        ]
    if "Parts" in data:
        return [
            {
                "part_id": part["ID"],
                "rotation": int(part.get("Rotation", 0)),
                "x": int(part["Location"][0]),
                "y": int(part["Location"][1]),
                "flip_x": bool(part.get("FlipX", False)),
                "flip_y": bool(part.get("FlipY", False)),
            }
            for part in data["Parts"]
            if isinstance(part, dict)
            and "ID" in part
            and isinstance(part.get("Location"), list)
            and len(part["Location"]) == 2
        ]
    raise ValueError(
        f"Could not parse seed parts from {seed_json_path}: expected 'parts' or 'Parts' key"
    )


def load_seed_parts_from_png(seed_png_path: Path, vanilla_part_loader) -> list[dict]:
    """Load vanilla seed parts from an encoded ship PNG.

    Args:
        seed_png_path: `.ship.png` file to parse
        vanilla_part_loader: Callable that converts extracted ship JSON into ShipPart records
    """

    ship_data = _prepare_seed_ship_for_markov(parse_ship_png(seed_png_path))
    return [
        {
            "part_id": ship_part.part_id,
            "rotation": ship_part.rotation,
            "x": ship_part.x,
            "y": ship_part.y,
            "flip_x": ship_part.flip_x,
            "flip_y": ship_part.flip_y,
        }
        for ship_part in vanilla_part_loader(ship_data)
    ]
