"""Export generated Markov ship JSON to Cosmoteer ``.ship.png`` files.

This module converts the generator's output format (list of ShipPart dicts with
``part_id / rotation / x / y``) into the Cosmoteer binary format and embeds it
into a carrier PNG, ready to be loaded in-game.

Limitations (first-pass export):
- No doors are synthesised.  Generated ships will load but crews have no paths
  to many rooms.  Door synthesis is a second-pass concern.
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

from common.cosmoteer import create_ship_png_bytes, parse_ship_png, write_ship_png


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
        result.append({
            "ID": str(part["part_id"]),
            "Location": [int(part["x"]), int(part["y"])],
            "Rotation": int(part["rotation"]),
        })
    return result


def make_minimal_ship_dict(
    parts: list[dict],
    name: str = "generated",
    flight_direction: int = 1,
    ship_rules_id: str = "cosmoteer.terran",
) -> dict:
    """Build a minimal Cosmoteer ship dict from a list of Cosmoteer-format parts.

    Only the fields essential for in-game loading are included.  The game treats
    absent optional fields as defaults, so the ship should load and be flyable
    (without doors / accessibility cleanup).
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
        "Doors": [],
    }


# ── roundtrip validation ──────────────────────────────────────────────────────


def roundtrip_validate(generated_json: dict) -> dict:
    """Encode the generated ship to PNG bytes, re-extract it, and compare.

    Returns a validation report dict with keys:
        ok           – bool, True if parts round-tripped exactly
        parts_in     – int
        parts_out    – int
        parts_match  – bool
        mismatches   – list of dicts describing any discrepancies
        warnings     – list of str
        png_bytes    – int (size of generated PNG)
    """
    cosmoteer_parts = generated_parts_to_cosmoteer_parts(generated_json.get("parts", []))
    ship_dict = make_minimal_ship_dict(cosmoteer_parts, name=generated_json.get("name", "generated"))

    png_bytes = create_ship_png_bytes(ship_dict)

    # Re-extract using the parser
    import io as _io
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
            "mismatches": [],
            "warnings": [f"extraction failed: {exc}"],
            "png_bytes": len(png_bytes),
        }

    extracted_parts = extracted.get("Parts", [])
    parts_in = len(cosmoteer_parts)
    parts_out = len(extracted_parts)
    mismatches = []
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

        if orig_id != ext_id or orig_loc != ext_loc or orig_rot != ext_rot:
            mismatches.append({
                "index": i,
                "orig": {"ID": orig_id, "Location": orig_loc, "Rotation": orig_rot},
                "extracted": {"ID": ext_id, "Location": ext_loc, "Rotation": ext_rot},
            })

    parts_match = parts_in == parts_out and len(mismatches) == 0
    ok = parts_match and len(warnings) == 0

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
        "mismatches": mismatches[:20],
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
    ship_dict = make_minimal_ship_dict(cosmoteer_parts, name=name)
    write_ship_png(ship_dict, output_path)

    result: dict[str, Any] = {
        "output_path": str(output_path),
        "parts_exported": len(cosmoteer_parts),
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
