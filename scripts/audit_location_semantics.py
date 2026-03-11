"""Audit live-game SaveRect semantics against extracted ship JSON files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.geometry import load_vanilla_part_geometry, normalize_part_id
from common.save_rect import known_save_rects, load_live_save_rects


STORED_LOCATION_FRAME = "stored_location"
NORMALIZED_ORIGIN_FRAME = "normalized_origin"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the location-semantics audit script."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare current top-left footprint semantics against repo-backed "
            "effective rect offsets and report ships whose occupancy or "
            "connectivity would change."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="extracted_ship_data_canonical",
        help="Directory of extracted or canonical ship JSON files to audit.",
    )
    parser.add_argument(
        "--game-root",
        default=None,
        help=(
            "Optional path to a local Cosmoteer install root. When provided, "
            "live SaveRect values override the repo-backed geometry export."
        ),
    )
    parser.add_argument(
        "--output",
        default="out/debug/location-semantics-audit.json",
        help="Where to write the JSON audit report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of matching ships to inspect.",
    )
    return parser


def resolve_save_rects(game_root: str | Path | None) -> dict:
    """Resolve the effective save-rect table for audit comparisons.

    Args:
        game_root: Optional path to a live Cosmoteer install

    Returns:
        Repo-backed rect metadata, optionally overridden by live SaveRect scans
    """

    save_rects = known_save_rects()
    if game_root is None:
        return save_rects

    # Live rule files remain useful as an override source when validating the
    # repo export against an installed game version.
    save_rects.update(load_live_save_rects(game_root))
    return save_rects


def _occupied_cells(parts: list[dict], geometry_cache: dict, save_rects: dict, *, corrected: bool) -> tuple[Counter, list[dict]]:
    """Resolve occupied tiles for part placements with optional SaveRect correction."""

    counter: Counter = Counter()
    placements: list[dict] = []

    for index, part in enumerate(parts):
        geometry = geometry_cache.get(part["part_id"])
        if geometry is None:
            continue
        rotation = part["rotation"] % 4
        rotation_geometry = geometry.rotations.get(rotation)
        if rotation_geometry is None:
            continue

        origin_x = part["x"]
        origin_y = part["y"]
        save_rect = save_rects.get(part["part_id"])
        coordinate_frame = part.get("coordinate_frame", STORED_LOCATION_FRAME)
        if corrected and save_rect is not None and coordinate_frame == STORED_LOCATION_FRAME:
            # Ship-file "Parts" payloads store SaveRect-relative locations that need
            # to be translated back to normalized footprint origins before auditing
            base_geometry = geometry.rotations.get(0) or rotation_geometry
            offset_x, offset_y = save_rect.offset_for_rotation(
                rotation=rotation,
                base_width=base_geometry.width,
                base_height=base_geometry.height,
            )
            origin_x -= offset_x
            origin_y -= offset_y

        cells = sorted(
            (origin_x + dx, origin_y + dy)
            for dx, dy in rotation_geometry.footprint_tiles
        )
        for cell in cells:
            counter[cell] += 1

        placements.append(
            {
                "index": index,
                "part_id": part["part_id"],
                "rotation": rotation,
                "coordinate_frame": coordinate_frame,
                "stored_location": [part["x"], part["y"]],
                "resolved_origin": [origin_x, origin_y],
                "cells": [list(cell) for cell in cells],
            }
        )

    return counter, placements


def _component_count(counter: Counter) -> int:
    cells = set(counter.keys())
    if not cells:
        return 0

    remaining = set(cells)
    components = 0
    while remaining:
        components += 1
        start = remaining.pop()
        stack = [start]
        while stack:
            x, y = stack.pop()
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def _overlap_cells(counter: Counter) -> list[list[int]]:
    return [list(cell) for cell, count in sorted(counter.items()) if count > 1]


def _extract_parts(data: dict, geometry_cache: dict) -> list[dict]:
    """Extract parts from extracted ship JSON and generator output payloads."""

    parts: list[dict] = []

    raw_parts = data.get("Parts")
    if isinstance(raw_parts, list):
        for raw in raw_parts:
            if not isinstance(raw, dict):
                continue
            part_id = normalize_part_id(raw)
            location = raw.get("Location")
            if not part_id or not isinstance(location, list) or len(location) != 2:
                continue
            if part_id not in geometry_cache:
                continue
            parts.append(
                {
                    "part_id": part_id,
                    "rotation": int(raw.get("Rotation", 0)) % 4,
                    "x": int(location[0]),
                    "y": int(location[1]),
                    "coordinate_frame": STORED_LOCATION_FRAME,
                }
            )

    raw_generated_parts = data.get("parts")
    if isinstance(raw_generated_parts, list):
        for raw in raw_generated_parts:
            if not isinstance(raw, dict):
                continue
            part_id = raw.get("part_id")
            if part_id not in geometry_cache:
                continue
            parts.append(
                {
                    "part_id": part_id,
                    "rotation": int(raw.get("rotation", 0)) % 4,
                    "x": int(raw.get("x", 0)),
                    "y": int(raw.get("y", 0)),
                    "coordinate_frame": NORMALIZED_ORIGIN_FRAME,
                }
            )

    return parts


def run_audit(
    input_dir: str | Path,
    game_root: str | Path | None,
    output_path: str | Path,
    limit: int | None = None,
) -> dict:
    """Audit occupancy changes caused by stored-location rect corrections."""

    geometry_cache = load_vanilla_part_geometry()
    save_rects = resolve_save_rects(game_root)

    input_path = Path(input_dir)
    reports: list[dict] = []
    scanned = 0
    candidate_ships = 0
    usage_counter: Counter = Counter()
    save_rect_part_ids = tuple(sorted(save_rects))

    for ship_path in sorted(input_path.glob("*.json")):
        scanned += 1
        raw_text = ship_path.read_text(encoding="utf-8")
        if not any(part_id in raw_text for part_id in save_rect_part_ids):
            continue

        data = json.loads(raw_text)
        parts = _extract_parts(data, geometry_cache)
        if not parts:
            continue

        save_rect_part_counts: Counter = Counter()
        for part in parts:
            if part["part_id"] in save_rects:
                save_rect_part_counts[part["part_id"]] += 1

        if not save_rect_part_counts:
            continue

        candidate_ships += 1
        usage_counter.update(save_rect_part_counts)

        naive_counter, naive_placements = _occupied_cells(parts, geometry_cache, save_rects, corrected=False)
        corrected_counter, corrected_placements = _occupied_cells(parts, geometry_cache, save_rects, corrected=True)
        naive_components = _component_count(naive_counter)
        corrected_components = _component_count(corrected_counter)
        naive_overlaps = _overlap_cells(naive_counter)
        corrected_overlaps = _overlap_cells(corrected_counter)

        changed = (
            naive_components != corrected_components
            or naive_overlaps != corrected_overlaps
        )
        if changed:
            reports.append(
                {
                    "ship": ship_path.name,
                    "save_rect_parts": dict(save_rect_part_counts),
                    "naive": {
                        "occupied_cells": len(naive_counter),
                        "component_count": naive_components,
                        "overlap_cells": naive_overlaps,
                    },
                    "save_rect_corrected": {
                        "occupied_cells": len(corrected_counter),
                        "component_count": corrected_components,
                        "overlap_cells": corrected_overlaps,
                    },
                    "placements": {
                        "naive": naive_placements,
                        "save_rect_corrected": corrected_placements,
                    },
                }
            )
            if limit is not None and len(reports) >= limit:
                break

    output_file = Path(output_path)
    summary = {
        "game_root": str(Path(game_root)) if game_root is not None else None,
        "input_dir": str(input_path),
        "output_path": str(output_file),
        "ships_scanned": scanned,
        "ships_with_save_rect_parts": candidate_ships,
        "ships_with_changed_occupancy_or_connectivity": len(reports),
        "save_rect_parts": {
            part_id: {
                "offset": [save_rect.x, save_rect.y],
                "size": [save_rect.width, save_rect.height],
                "source_file": save_rect.source_file,
                "instances_in_scanned_ships": usage_counter.get(part_id, 0),
            }
            for part_id, save_rect in sorted(save_rects.items())
        },
        "reports": reports,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_audit(
        input_dir=args.input_dir,
        game_root=args.game_root,
        output_path=args.output,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "ships_with_save_rect_parts": summary["ships_with_save_rect_parts"],
                "ships_with_changed_occupancy_or_connectivity": summary[
                    "ships_with_changed_occupancy_or_connectivity"
                ],
                "output": summary["output_path"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
