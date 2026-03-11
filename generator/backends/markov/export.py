"""Export generated Markov ship JSON to Cosmoteer ``.ship.png`` files.

This module converts the generator's output format (list of ShipPart dicts with
``part_id / rotation / x / y``) into the Cosmoteer binary format and embeds it
into a carrier PNG, ready to be loaded in-game.

Limitations (current export):
- Door records are preserved when the input payload already contains them, but
  stochastic generation still does not synthesise new doors yet.
- Color fields use game-default values; the ship will appear with default
  roof / crew colours.
- Decals, PartControlGroups, PartUIToggleStates and WeaponSelfTargets are
  omitted from the minimal ship dict.  The game falls back to defaults.
- Roundtrip validation re-extracts the PNG and compares parts list; it does
  NOT verify in-game gameplay validity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from common.cosmoteer import create_ship_png_bytes, write_ship_png

__all__ = [
    "generated_parts_to_cosmoteer_parts",
    "generated_doors_to_cosmoteer_doors",
    "make_minimal_ship_dict",
    "graph_to_generated_parts_payload",
    "roundtrip_validate",
    "export_ship_png",
    "export_batch",
]


# ── default field values ──────────────────────────────────────────────────────

# Default RGBA color quads (float32 channels encoded as 4-byte hex, LE IEEE 754)
_DEFAULT_CREW_COLOR = ["0000803F", "00000000", "00000000", "0000803F"]
_DEFAULT_ROOF_COLOR = ["907F083F", "907F083F", "907F083F", "0000403F"]
_DEFAULT_DECAL_COLOR1 = ["9A99193E", "9A99193E", "9A99193E", "0000803F"]
_DEFAULT_DECAL_COLOR2 = ["0000803F", "0000803F", "0000803F", "0000803F"]


# ── ship dict construction ────────────────────────────────────────────────────


def generated_parts_to_cosmoteer_parts(parts: list[dict]) -> list[dict]:
    """Convert Markov generator part dicts to Cosmoteer ``Parts`` format.

    Generator format: ``{part_id, rotation, x, y}``
    Cosmoteer format: ``{ID, Location: [x, y], Rotation}``
    """
    result = []
    for part in parts:
        exported = {
            "ID": str(part["part_id"]),
            "Location": [int(part["x"]), int(part["y"])],
            "Rotation": int(part["rotation"]),
        }
        if "flip_x" in part:
            exported["FlipX"] = bool(part["flip_x"])
        if "flip_y" in part:
            exported["FlipY"] = bool(part["flip_y"])
        result.append(exported)
    return result


def _recover_legacy_coord_from_2x(local_2x: object, center_2x: object) -> list[int] | None:
    """Recover one legacy grid coordinate pair from centered 2x coordinates."""

    if not isinstance(local_2x, list) or len(local_2x) != 2:
        return None
    if not isinstance(center_2x, list) or len(center_2x) != 2:
        return None

    local_x = int(local_2x[0])
    local_y = int(local_2x[1])
    center_x = int(center_2x[0])
    center_y = int(center_2x[1])
    if (local_x + center_x) % 2 != 0 or (local_y + center_y) % 2 != 0:
        return None
    return [(local_x + center_x) // 2, (local_y + center_y) // 2]


def generated_doors_to_cosmoteer_doors(
    doors: list[dict],
    *,
    center_2x: list[int] | None = None,
) -> list[dict]:
    """Convert generator door dicts to Cosmoteer ``Doors`` format.

    The generator currently uses the same field names as the Cosmoteer payload
    so graph replay can preserve normalized door records losslessly. During the
    relative-coordinates migration this also accepts `Cell2x` when `center_2x`
    is available.
    """

    normalized_doors = []
    for door in doors:
        if not isinstance(door, dict):
            continue

        cell = door.get("Cell")
        if (not isinstance(cell, list) or len(cell) != 2) and center_2x is not None:
            cell = _recover_legacy_coord_from_2x(door.get("Cell2x"), center_2x)
        if not isinstance(cell, list) or len(cell) != 2:
            continue
        if "Orientation" not in door:
            continue

        normalized_doors.append(
            {
                "Cell": [int(cell[0]), int(cell[1])],
                "Orientation": int(door["Orientation"]),
            }
        )

    return normalized_doors


def make_minimal_ship_dict(
    parts: list[dict],
    doors: list[dict] | None = None,
    name: str = "generated",
    flight_direction: int = 1,
    ship_rules_id: str = "cosmoteer.terran",
) -> dict:
    """Build a minimal Cosmoteer ship dict from a list of Cosmoteer-format parts.

    Only the fields essential for in-game loading are included.  The game treats
    absent optional fields as defaults, so the ship should load and be flyable.
    """
    return {
        "Version": 1,
        "Name": name,
        "FlightDirection": flight_direction,
        "FormationOrder": 0,
        "ShipRulesID": ship_rules_id,
        "RoofBaseTexture": "scratched1",
        "CrewUniformColor": _DEFAULT_CREW_COLOR,
        "RoofBaseColor": _DEFAULT_ROOF_COLOR,
        "RoofDecalColor1": _DEFAULT_DECAL_COLOR1,
        "RoofDecalColor2": _DEFAULT_DECAL_COLOR2,
        "Parts": parts,
        "Doors": doors or [],
    }


def graph_to_generated_parts_payload(
    graph_data: dict,
    *,
    name: str | None = None,
) -> dict:
    """Convert a structural graph payload into deterministic generator output.

    Args:
        graph_data: Graph JSON payload produced by preprocessing
        name: Optional ship name override for the returned generator payload

    Returns:
        Generator-style JSON payload containing exact replayable part placements

    Notes:
        Graph nodes are expected to carry legacy normalized `location` fields.
        During phased coord-transform migration this loader can also recover
        legacy coordinates from `location_2x` plus `coord_transform.center_2x`.
    """
    nodes = graph_data.get("graphs", {}).get("A_structural_part_graph", {}).get("nodes", [])
    coord_transform = graph_data.get("coord_transform", {})
    center_2x = (
        coord_transform.get("center_2x")
        if isinstance(coord_transform, dict) and isinstance(coord_transform.get("center_2x"), list)
        else None
    )
    graph_doors = generated_doors_to_cosmoteer_doors(
        graph_data.get("doors", []),
        center_2x=center_2x if isinstance(center_2x, list) and len(center_2x) == 2 else None,
    )
    generated_parts = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        part_id = node.get("part_id")
        location = node.get("location")
        if (
            (not isinstance(location, list) or len(location) != 2)
            and isinstance(node.get("location_2x"), list)
            and len(node.get("location_2x")) == 2
            and isinstance(center_2x, list)
            and len(center_2x) == 2
        ):
            # Allow phased graph payloads that only carry centered 2x node
            # coordinates by reconstructing legacy normalized grid locations.
            local_2x = [int(node["location_2x"][0]), int(node["location_2x"][1])]
            cx = int(center_2x[0])
            cy = int(center_2x[1])
            if (local_2x[0] + cx) % 2 == 0 and (local_2x[1] + cy) % 2 == 0:
                location = [(local_2x[0] + cx) // 2, (local_2x[1] + cy) // 2]

        if not part_id or not isinstance(location, list) or len(location) != 2:
            continue

        generated_part = {
            "part_id": str(part_id),
            "rotation": int(node.get("rotation", 0)) % 4,
            "x": int(location[0]),
            "y": int(location[1]),
        }
        if "flip_x" in node or "FlipX" in node:
            generated_part["flip_x"] = bool(node.get("flip_x", node.get("FlipX", False)))
        if "flip_y" in node or "FlipY" in node:
            generated_part["flip_y"] = bool(node.get("flip_y", node.get("FlipY", False)))
        generated_parts.append(generated_part)

    # Reuse the existing generator export schema so graph replay can flow through
    # the same PNG export and validation path as stochastic Markov samples
    return {
        "name": name or graph_data.get("ship", {}).get("name") or "graph-replay",
        "generator": "graph_replay",
        "stats": {
            "parts_generated": len(generated_parts),
            "doors_preserved": len(graph_doors),
            "stop_reason": "graph_replay",
        },
        "parts": generated_parts,
        "doors": graph_doors,
        "notes": [
            "Deterministic replay of normalized part placements from preprocessing graph JSON.",
            "Normalized door records are preserved explicitly when present in the graph payload.",
            "This payload intentionally bypasses stochastic Markov sampling.",
        ],
    }


def _normalize_door_for_comparison(door: dict) -> tuple:
    """Return a stable comparable tuple for one Cosmoteer-format door."""

    cell = door.get("Cell", [])
    return (
        int(cell[0]) if isinstance(cell, list) and len(cell) == 2 else 0,
        int(cell[1]) if isinstance(cell, list) and len(cell) == 2 else 0,
        int(door.get("Orientation", 0)),
    )


# ── roundtrip validation ──────────────────────────────────────────────────────


def roundtrip_validate(generated_json: dict) -> dict:
    """Encode the generated ship to PNG bytes, re-extract it, and compare.

    Returns a validation report dict with keys:
        ok           – bool, True if parts round-tripped exactly
        parts_in     – int
        parts_out    – int
        parts_match  – bool
        mismatches   – list of dicts describing any part discrepancies
        doors_in     – int
        doors_out    – int
        doors_match  – bool
        door_mismatches – list of dicts describing any door discrepancies
        warnings     – list of str
        png_bytes    – int (size of generated PNG)
    """
    cosmoteer_parts = generated_parts_to_cosmoteer_parts(generated_json.get("parts", []))
    cosmoteer_doors = generated_doors_to_cosmoteer_doors(generated_json.get("doors", []))
    ship_dict = make_minimal_ship_dict(
        cosmoteer_parts,
        doors=cosmoteer_doors,
        name=generated_json.get("name", "generated"),
    )

    png_bytes = create_ship_png_bytes(ship_dict)

    # Re-extract using the parser
    from common.cosmoteer.parser import _decode_ship_data, _extract_embedded_payload

    try:
        payload = _extract_embedded_payload(png_bytes)
        extracted = _decode_ship_data(payload)
    except Exception as exc:
        return {
            "ok": False,
            "parts_in": len(cosmoteer_parts),
            "parts_out": 0,
            "parts_match": False,
            "doors_in": len(cosmoteer_doors),
            "doors_out": 0,
            "doors_match": False,
            "mismatches": [],
            "door_mismatches": [],
            "warnings": [f"extraction failed: {exc}"],
            "png_bytes": len(png_bytes),
        }

    extracted_parts = extracted.get("Parts", [])
    extracted_doors = extracted.get("Doors", [])
    parts_in = len(cosmoteer_parts)
    parts_out = len(extracted_parts)
    doors_in = len(cosmoteer_doors)
    doors_out = len(extracted_doors)
    mismatches = []
    door_mismatches = []
    warnings = []

    if parts_in != parts_out:
        warnings.append(f"part count mismatch: encoded {parts_in}, extracted {parts_out}")

    for i, (orig, ext) in enumerate(zip(cosmoteer_parts, extracted_parts)):
        orig_id = orig.get("ID", "")
        ext_id = ext.get("ID", "") if isinstance(ext, dict) else ""
        orig_loc = orig.get("Location", [])
        ext_loc = ext.get("Location", []) if isinstance(ext, dict) else []
        orig_rot = orig.get("Rotation", 0)
        ext_rot = ext.get("Rotation", 0) if isinstance(ext, dict) else -1
        orig_flip_x = orig.get("FlipX", False)
        ext_flip_x = ext.get("FlipX", False) if isinstance(ext, dict) else False
        orig_flip_y = orig.get("FlipY", False)
        ext_flip_y = ext.get("FlipY", False) if isinstance(ext, dict) else False

        if (
            orig_id != ext_id
            or orig_loc != ext_loc
            or orig_rot != ext_rot
            or orig_flip_x != ext_flip_x
            or orig_flip_y != ext_flip_y
        ):
            mismatches.append({
                "index": i,
                "orig": {
                    "ID": orig_id,
                    "Location": orig_loc,
                    "Rotation": orig_rot,
                    "FlipX": orig_flip_x,
                    "FlipY": orig_flip_y,
                },
                "extracted": {
                    "ID": ext_id,
                    "Location": ext_loc,
                    "Rotation": ext_rot,
                    "FlipX": ext_flip_x,
                    "FlipY": ext_flip_y,
                },
            })

    if doors_in != doors_out:
        warnings.append(f"door count mismatch: encoded {doors_in}, extracted {doors_out}")

    original_sorted_doors = sorted(_normalize_door_for_comparison(door) for door in cosmoteer_doors)
    extracted_sorted_doors = sorted(
        _normalize_door_for_comparison(door)
        for door in extracted_doors
        if isinstance(door, dict)
    )
    if original_sorted_doors != extracted_sorted_doors:
        # Compare sorted normalized tuples so replay validation remains stable
        # even if a future encoder/parser changes door record ordering.
        max_door_count = max(len(original_sorted_doors), len(extracted_sorted_doors))
        for index in range(max_door_count):
            original_door = original_sorted_doors[index] if index < len(original_sorted_doors) else None
            extracted_door = extracted_sorted_doors[index] if index < len(extracted_sorted_doors) else None
            if original_door == extracted_door:
                continue
            door_mismatches.append(
                {
                    "index": index,
                    "orig": (
                        {
                            "Cell": [original_door[0], original_door[1]],
                            "Orientation": original_door[2],
                        }
                        if original_door is not None
                        else None
                    ),
                    "extracted": (
                        {
                            "Cell": [extracted_door[0], extracted_door[1]],
                            "Orientation": extracted_door[2],
                        }
                        if extracted_door is not None
                        else None
                    ),
                }
            )

    parts_match = parts_in == parts_out and len(mismatches) == 0
    doors_match = doors_in == doors_out and len(door_mismatches) == 0
    ok = parts_match and doors_match and len(warnings) == 0

    # Sanity-check version and basic fields
    if extracted.get("Version") != ship_dict.get("Version"):
        warnings.append(
            f"Version mismatch: encoded {ship_dict.get('Version')}, "
            f"extracted {extracted.get('Version')}"
        )
        ok = False
    if extracted.get("ShipRulesID") != ship_dict.get("ShipRulesID"):
        warnings.append(
            f"ShipRulesID mismatch: encoded {ship_dict.get('ShipRulesID')!r}, "
            f"extracted {extracted.get('ShipRulesID')!r}"
        )
        ok = False

    return {
        "ok": ok,
        "parts_in": parts_in,
        "parts_out": parts_out,
        "parts_match": parts_match,
        "doors_in": doors_in,
        "doors_out": doors_out,
        "doors_match": doors_match,
        "mismatches": mismatches[:20],
        "door_mismatches": door_mismatches[:20],
        "warnings": warnings,
        "png_bytes": len(png_bytes),
    }


# ── export entry-point ────────────────────────────────────────────────────────


def export_ship_png(
    generated_json: dict,
    output_path: str | Path,
    *,
    name: Optional[str] = None,
    validate: bool = True,
) -> dict:
    """Export a single generated ship JSON to a ``.ship.png`` file.

    Args:
        generated_json: Output dict from ``RelativeMarkovModel.generate()``.
        output_path: Destination path for the ``.ship.png`` file.
        name: Ship name embedded in the file.  Defaults to the output filename stem.
        validate: If True, performs a roundtrip encode→decode check and includes
                  the validation report in the return value.

    Returns:
        A dict with export metadata and (optionally) the validation report.
    """
    output_path = Path(output_path)
    if name is None:
        name = output_path.stem.replace(".ship", "")

    cosmoteer_parts = generated_parts_to_cosmoteer_parts(generated_json.get("parts", []))
    cosmoteer_doors = generated_doors_to_cosmoteer_doors(generated_json.get("doors", []))
    ship_dict = make_minimal_ship_dict(cosmoteer_parts, doors=cosmoteer_doors, name=name)
    write_ship_png(ship_dict, output_path)

    result: dict[str, Any] = {
        "output_path": str(output_path),
        "parts_exported": len(cosmoteer_parts),
        "doors_exported": len(cosmoteer_doors),
        "ship_name": name,
        "generator_stop_reason": generated_json.get("stats", {}).get("stop_reason", "unknown"),
    }

    if validate:
        result["roundtrip"] = roundtrip_validate(generated_json)
        result["valid"] = result["roundtrip"]["ok"]
    else:
        result["valid"] = None

    return result


def export_batch(
    sample_dir: str | Path,
    output_dir: str | Path,
    *,
    validate: bool = True,
    name_prefix: str = "gen",
) -> list[dict]:
    """Export all ``.json`` samples in *sample_dir* to ``.ship.png`` files in *output_dir*.

    Returns a list of export result dicts (one per file).
    """
    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for json_path in sorted(sample_dir.glob("*.json")):
        try:
            generated_json = json.loads(json_path.read_text())
        except Exception as exc:
            results.append({"source": str(json_path), "error": str(exc)})
            continue

        stem = json_path.stem  # e.g. "sample-000"
        out_path = output_dir / f"{stem}.ship.png"
        ship_name = f"{name_prefix}-{stem}"
        try:
            result = export_ship_png(
                generated_json,
                out_path,
                name=ship_name,
                validate=validate,
            )
            result["source"] = str(json_path)
            results.append(result)
        except Exception as exc:
            results.append({"source": str(json_path), "output_path": str(out_path), "error": str(exc)})

    return results
