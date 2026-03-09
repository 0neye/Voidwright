#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

# Ensure project root is importable so we can reuse canonical game-file geometry.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from generators.markov.door_rules import load_vanilla_part_geometry  # noqa: E402

Coord = Tuple[int, int]


@dataclass(frozen=True)
class PartMeta:
    width: int
    height: int
    traversable: bool = False
    note: str = ""
    footprint_tiles: frozenset = frozenset()  # local (dx, dy) offsets relative to part origin


# Kept as last-resort hints for unknown/non-vanilla part IDs.
TRAVERSABLE_HINTS = (
    "corridor", "conveyor", "crew_quarters", "quarters", "storage", "reactor",
    "engine_room", "control_room", "heat_pipe", "radiator", "heat_exchanger",
    "thermal_", "power_storage", "factory", "ammo_factory", "ammo_storage",
    "airlock", "hyperdrive", "ftl_drive", "shield_gen", "sensor_array",
    "fire_extinguisher",
)

NON_TRAVERSABLE_HINTS = (
    "armor", "structure", "thruster", "point_defense", "cannon", "chaingun",
    "laser_blaster", "disruptor", "railgun", "ion_beam", "tractor_beam",
    "manipulator_beam", "mining_laser", "resonance_beam", "electro_bolter",
    "explosive_charge", "missile_launcher", "missile_silo", "missile_storage",
    "roof_",
)

DOOR_ORIENTATION_DELTAS = {
    0: (1, 0),
    1: (0, 1),
}


def normalize_part_id(part: dict) -> Optional[str]:
    return part.get("ID") or part.get("IDString")


def normalize_parts(parts) -> List[dict]:
    normalized: List[dict] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        pid = normalize_part_id(part)
        if not pid or "Location" not in part:
            continue
        normalized.append(
            {
                "ID": pid,
                "Location": part["Location"],
                "Rotation": int(part.get("Rotation", 0)),
            }
        )
    return normalized


def normalize_doors(value) -> List[dict]:
    if not isinstance(value, list):
        return []
    return [d for d in value if isinstance(d, dict) and "Cell" in d and "Orientation" in d]


def infer_meta(part_id: str, rotation: int) -> Tuple[PartMeta, bool]:
    """Return (PartMeta, is_inferred).

    Uses canonical game-file geometry for vanilla parts (exact footprint tiles,
    exact size, correct traversability). Falls back to regex/name heuristics for
    unknown or non-vanilla parts.
    """
    geometry_cache = load_vanilla_part_geometry()
    vanilla = geometry_cache.get(part_id)
    if vanilla is not None:
        rot = vanilla.rotations.get(rotation % 4) or next(iter(vanilla.rotations.values()))
        traversable = bool(rot.unblocked_tiles)
        return PartMeta(
            width=rot.width,
            height=rot.height,
            traversable=traversable,
            note="game-file geometry",
            footprint_tiles=rot.footprint_tiles,
        ), False

    # Unknown/non-vanilla fallback: regex dimension parse + name hints.
    match = re.search(r"_(\d+)x(\d+)(?:_|$)", part_id)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
    else:
        width, height = 1, 1
    lower = part_id.lower()
    traversable = any(token in lower for token in TRAVERSABLE_HINTS)
    if any(token in lower for token in NON_TRAVERSABLE_HINTS):
        traversable = False
    tiles = frozenset((dx, dy) for dx in range(width) for dy in range(height))
    note = "regex/fallback inferred"
    return PartMeta(width, height, traversable, note, tiles), True


def part_cells(part: dict, meta: PartMeta) -> Set[Coord]:
    x0, y0 = map(int, part["Location"])
    if meta.footprint_tiles:
        return {(x0 + dx, y0 + dy) for dx, dy in meta.footprint_tiles}
    # Fallback for any part where footprint_tiles was not populated.
    rotation = int(part.get("Rotation", 0))
    w, h = (meta.height, meta.width) if rotation % 2 else (meta.width, meta.height)
    return {(x0 + dx, y0 + dy) for dx in range(w) for dy in range(h)}


def structural_edges(part_records: List[dict], cell_to_parts: Dict[Coord, Set[int]]) -> List[dict]:
    adjacency: Dict[Tuple[int, int], dict] = {}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for cell, owners in cell_to_parts.items():
        x, y = cell
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if neighbor not in cell_to_parts:
                continue
            for a in owners:
                for b in cell_to_parts[neighbor]:
                    if a == b:
                        continue
                    key = (a, b) if a < b else (b, a)
                    edge = adjacency.setdefault(
                        key,
                        {
                            "source": key[0],
                            "target": key[1],
                            "kind": "touching",
                            "shared_sides": 0,
                        },
                    )
                    edge["shared_sides"] += 1

    return sorted(adjacency.values(), key=lambda e: (e["source"], e["target"]))


def cell_graph(
    part_records: List[dict],
    cell_to_parts: Dict[Coord, Set[int]],
    doors: List[dict],
) -> dict:
    nodes = []
    traversable_cells: Set[Coord] = set()
    part_by_index = {record["index"]: record for record in part_records}

    for (x, y), owners in sorted(cell_to_parts.items()):
        owner_records = [part_by_index[i] for i in sorted(owners)]
        traversable = any(record["traversable"] for record in owner_records)
        if traversable:
            traversable_cells.add((x, y))
        nodes.append(
            {
                "id": f"{x},{y}",
                "x": x,
                "y": y,
                "occupied": True,
                "traversable": traversable,
                "part_indices": [record["index"] for record in owner_records],
            }
        )

    edges: Dict[Tuple[str, str, str], dict] = {}

    for record in part_records:
        if not record["traversable"]:
            continue
        cells = record["cells"]
        for x, y in cells:
            for dx, dy in ((1, 0), (0, 1)):
                neighbor = (x + dx, y + dy)
                if neighbor not in cells:
                    continue
                a = f"{x},{y}"
                b = f"{neighbor[0]},{neighbor[1]}"
                if a > b:
                    a, b = b, a
                edges[(a, b, "intra_part")] = {
                    "source": a,
                    "target": b,
                    "kind": "intra_part",
                    "traversable": True,
                    "part_index": record["index"],
                }

    valid_doors = 0
    dangling_doors = 0
    for idx, door in enumerate(doors):
        x, y = map(int, door["Cell"])
        orientation = int(door.get("Orientation", 0))
        dx, dy = DOOR_ORIENTATION_DELTAS.get(orientation, (1, 0))
        a_coord = (x, y)
        b_coord = (x + dx, y + dy)
        if a_coord not in cell_to_parts or b_coord not in cell_to_parts:
            dangling_doors += 1
            continue
        valid_doors += 1
        a = f"{a_coord[0]},{a_coord[1]}"
        b = f"{b_coord[0]},{b_coord[1]}"
        if a > b:
            a, b = b, a
        edges[(a, b, f"door:{idx}")] = {
            "source": a,
            "target": b,
            "kind": "door",
            "traversable": True,
            "door_index": idx,
            "orientation": orientation,
        }

    return {
        "nodes": nodes,
        "edges": sorted(edges.values(), key=lambda e: (e["source"], e["target"], e["kind"])),
        "summary": {
            "occupied_cells": len(nodes),
            "traversable_cells": len(traversable_cells),
            "door_records": len(doors),
            "valid_door_edges": valid_doors,
            "dangling_door_records": dangling_doors,
        },
    }


def process_ship(ship_path: Path) -> dict:
    with ship_path.open() as fh:
        data = json.load(fh)

    raw_parts = data.get("Parts", [])
    parts = normalize_parts(raw_parts)
    doors = normalize_doors(data.get("Doors", []))

    part_records = []
    cell_to_parts: Dict[Coord, Set[int]] = defaultdict(set)
    unknown_ids: Counter = Counter()

    for index, part in enumerate(parts):
        rotation = int(part.get("Rotation", 0)) % 4
        meta, inferred = infer_meta(part["ID"], rotation)
        if inferred:
            unknown_ids[part["ID"]] += 1
        cells = part_cells(part, meta)
        record = {
            "index": index,
            "part_id": part["ID"],
            "location": list(map(int, part["Location"])),
            "rotation": rotation,
            "width": meta.width,
            "height": meta.height,
            "traversable": meta.traversable,
            "meta_note": meta.note,
            "cells": cells,
        }
        part_records.append(record)
        for cell in cells:
            cell_to_parts[cell].add(index)

    structure_nodes = [
        {
            "id": record["index"],
            "part_id": record["part_id"],
            "location": record["location"],
            "rotation": record["rotation"],
            "footprint": {
                "cell_count": len(record["cells"]),
                "width": record["width"],
                "height": record["height"],
            },
            "traversable": record["traversable"],
            "meta_note": record["meta_note"],
        }
        for record in part_records
    ]

    structure_edges = structural_edges(part_records, cell_to_parts)
    cells = cell_graph(part_records, cell_to_parts, doors)

    return {
        "ship": {
            "name": data.get("Name"),
            "author": data.get("Author"),
            "source_file": ship_path.name,
            "version": data.get("Version"),
            "flight_direction": data.get("FlightDirection"),
        },
        "schema_version": 2,
        "assumptions": {
            "geometry_model": "Vanilla part footprints use exact game-file tile data via load_vanilla_part_geometry(). Non-vanilla/unknown parts use regex dimension inference with rectangular approximation.",
            "door_model": "Door orientation 0 is east-west, orientation 1 is north-south.",
            "traversability_model": "Cell traversability uses game-file unblocked_footprint_tiles for vanilla parts; name-hint heuristics for others.",
        },
        "graphs": {
            "A_structural_part_graph": {
                "nodes": structure_nodes,
                "edges": structure_edges,
                "summary": {
                    "parts": len(structure_nodes),
                    "touching_edges": len(structure_edges),
                },
            },
            "C_cell_graph": cells,
        },
        "validation": {
            "normalized_part_count": len(parts),
            "raw_part_count": len(raw_parts) if isinstance(raw_parts, list) else 0,
            "unknown_part_ids": dict(sorted(unknown_ids.items())),
        },
    }


def generate_all(input_dir: Path, output_dir: Path, limit: Optional[int] = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.json"))
    if limit is not None:
        files = files[:limit]

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "schema_version": 2,
        "geometry_source": "game-file canonical (load_vanilla_part_geometry)",
        "ships_processed": 0,
        "ships_with_unknown_part_ids": 0,
        "total_unknown_part_instances": 0,
        "unknown_part_ids": Counter(),
        "door_stats": Counter(),
        "sample_outputs": [],
    }

    for ship_path in files:
        graph_data = process_ship(ship_path)
        out_path = output_dir / ship_path.name
        with out_path.open("w") as fh:
            json.dump(graph_data, fh, separators=(",", ":"))
            fh.write("\n")

        unknown_map = graph_data["validation"]["unknown_part_ids"]
        if unknown_map:
            manifest["ships_with_unknown_part_ids"] += 1
            manifest["unknown_part_ids"].update(unknown_map)
            manifest["total_unknown_part_instances"] += sum(unknown_map.values())

        cell_summary = graph_data["graphs"]["C_cell_graph"]["summary"]
        manifest["door_stats"].update(
            {
                "door_records": cell_summary["door_records"],
                "valid_door_edges": cell_summary["valid_door_edges"],
                "dangling_door_records": cell_summary["dangling_door_records"],
                "occupied_cells": cell_summary["occupied_cells"],
                "traversable_cells": cell_summary["traversable_cells"],
            }
        )
        manifest["ships_processed"] += 1
        if len(manifest["sample_outputs"]) < 10:
            manifest["sample_outputs"].append(out_path.name)

    manifest["unknown_part_ids"] = dict(manifest["unknown_part_ids"].most_common())
    manifest["door_stats"] = dict(manifest["door_stats"])

    with (output_dir / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate structural and cell graphs for extracted Cosmoteer ship JSONs. "
                    "Uses canonical game-file geometry for vanilla parts."
    )
    parser.add_argument("--input-dir", default="extracted_ship_data", help="Directory with extracted *.json ship files")
    parser.add_argument("--output-dir", default="generated_ship_graphs", help="Directory to write per-ship graph JSON files")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for partial validation runs")
    args = parser.parse_args()

    manifest = generate_all(Path(args.input_dir), Path(args.output_dir), args.limit)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
