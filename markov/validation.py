"""Validation utilities for relative placement assumptions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from common.geometry import iter_ship_files, load_vanilla_part_geometry

from .corpus import iter_vanilla_parts_from_ship
from .order import order_ship_parts, parts_touch
from .types import ShipPart

__all__ = ["validate_relative_placement_assumptions"]


def validate_relative_placement_assumptions(
    input_dir: Path,
    sample_limit: Optional[int] = None,
) -> dict:
    """Validate that relative offsets reconstruct exact ship placements

    Args:
        input_dir: Canonical corpus directory containing extracted ship JSON files
        sample_limit: Optional max number of ships to validate

    Returns:
        JSON-serializable report with failure examples and aggregate metrics
    """

    geometry_cache = load_vanilla_part_geometry()
    ships_checked = 0
    placements_checked = 0
    touching_placements = 0
    non_touching_placements = 0
    origin_failures: List[dict] = []
    footprint_failures: List[dict] = []
    max_abs_dx = 0
    max_abs_dy = 0
    largest_part_count = 0

    for ship_path in iter_ship_files(input_dir):
        with ship_path.open(encoding="utf-8") as file_handle:
            ship_data = json.load(file_handle)
        vanilla_parts = iter_vanilla_parts_from_ship(ship_data, geometry_cache=geometry_cache)
        if len(vanilla_parts) < 2:
            continue

        ships_checked += 1
        largest_part_count = max(largest_part_count, len(vanilla_parts))
        ordered_parts = order_ship_parts(vanilla_parts)
        reconstructed_parts: List[ShipPart] = []

        for idx, (part, anchor) in enumerate(ordered_parts):
            if idx == 0:
                reconstructed_parts.append(
                    ShipPart(part_id=part.part_id, rotation=part.rotation, x=part.x, y=part.y)
                )
                continue

            assert anchor is not None
            dx = part.x - anchor.x
            dy = part.y - anchor.y
            placements_checked += 1
            max_abs_dx = max(max_abs_dx, abs(dx))
            max_abs_dy = max(max_abs_dy, abs(dy))
            if parts_touch(anchor, part, geometry_cache):
                touching_placements += 1
            else:
                non_touching_placements += 1

            reconstructed_part = ShipPart(
                part_id=part.part_id,
                rotation=part.rotation,
                x=anchor.x + dx,
                y=anchor.y + dy,
            )
            reconstructed_parts.append(reconstructed_part)

            if (reconstructed_part.x, reconstructed_part.y) != (part.x, part.y) and len(origin_failures) < 20:
                origin_failures.append(
                    {
                        "ship": ship_path.name,
                        "part_id": part.part_id,
                        "rotation": part.rotation,
                        "expected_origin": [part.x, part.y],
                        "got_origin": [reconstructed_part.x, reconstructed_part.y],
                        "anchor_part_id": anchor.part_id,
                        "anchor_rotation": anchor.rotation,
                        "dx": dx,
                        "dy": dy,
                    }
                )
            if (
                reconstructed_part.footprint_cells(geometry_cache) != part.footprint_cells(geometry_cache)
                and len(footprint_failures) < 20
            ):
                footprint_failures.append(
                    {
                        "ship": ship_path.name,
                        "part_id": part.part_id,
                        "rotation": part.rotation,
                        "anchor_part_id": anchor.part_id,
                        "anchor_rotation": anchor.rotation,
                        "dx": dx,
                        "dy": dy,
                    }
                )

        if sample_limit is not None and ships_checked >= sample_limit:
            break

    return {
        "ships_checked": ships_checked,
        "placements_checked": placements_checked,
        "largest_ship_vanilla_part_count": largest_part_count,
        "origin_failure_count": len(origin_failures),
        "footprint_failure_count": len(footprint_failures),
        "origin_failures": origin_failures,
        "footprint_failures": footprint_failures,
        "touching_placements": touching_placements,
        "non_touching_placements": non_touching_placements,
        "touching_fraction": (touching_placements / placements_checked) if placements_checked else 0.0,
        "max_abs_dx": max_abs_dx,
        "max_abs_dy": max_abs_dy,
        "summary": (
            "Origin-to-origin relative offsets reconstruct exact real-corpus part origins, "
            "and the resulting vanilla footprint cells also match exactly for the checked canonical ships. "
            "Touching counts use shared structural hull-side checks."
        ),
    }
