"""Generate graph-oriented preprocessing artifacts from ship JSON files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from common.geometry import PartMeta, infer_meta, load_vanilla_part_geometry, normalize_part_id
from .layout_helpers import door_adjacent_cells

Coord = Tuple[int, int]


def normalize_parts(parts: object) -> List[dict]:
    """Filter and normalize raw ship `Parts` records."""

    normalized_parts: List[dict] = []
    for part in parts if isinstance(parts, list) else []:
        if not isinstance(part, dict):
            continue
        part_id = normalize_part_id(part)
        if not part_id or "Location" not in part:
            continue
        normalized_parts.append(
            {
                "ID": part_id,
                "Location": part["Location"],
                "Rotation": int(part.get("Rotation", 0)),
            }
        )
    return normalized_parts


def normalize_doors(value: object) -> List[dict]:
    """Filter and normalize raw ship `Doors` records."""

    if not isinstance(value, list):
        return []
    return [
        door
        for door in value
        if isinstance(door, dict) and "Cell" in door and "Orientation" in door
    ]


def part_cells(part: dict, meta: PartMeta) -> Set[Coord]:
    """Return the occupied world cells for one normalized part."""

    origin_x, origin_y = map(int, part["Location"])
    if meta.footprint_tiles:
        return {(origin_x + dx, origin_y + dy) for dx, dy in meta.footprint_tiles}

    # This rectangular fallback is only used when detailed footprint tiles are unavailable.
    rotation = int(part.get("Rotation", 0))
    width, height = (meta.height, meta.width) if rotation % 2 else (meta.width, meta.height)
    return {
        (origin_x + dx, origin_y + dy)
        for dx in range(width)
        for dy in range(height)
    }


def structural_edges(part_records: List[dict], cell_to_parts: Dict[Coord, Set[int]]) -> List[dict]:
    """Build conservative touching edges between distinct parts."""

    adjacency: Dict[Tuple[int, int], dict] = {}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for cell, owners in cell_to_parts.items():
        cell_x, cell_y = cell
        for delta_x, delta_y in directions:
            neighbor = (cell_x + delta_x, cell_y + delta_y)
            if neighbor not in cell_to_parts:
                continue
            for owner_a in owners:
                for owner_b in cell_to_parts[neighbor]:
                    if owner_a == owner_b:
                        continue
                    key = (owner_a, owner_b) if owner_a < owner_b else (owner_b, owner_a)
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

    return sorted(adjacency.values(), key=lambda edge: (edge["source"], edge["target"]))


def cell_graph(
    part_records: List[dict],
    cell_to_parts: Dict[Coord, Set[int]],
    doors: List[dict],
) -> dict:
    """Build the conservative occupied-cell graph for one ship."""

    nodes = []
    traversable_cells: Set[Coord] = set()
    part_by_index = {record["index"]: record for record in part_records}

    for (cell_x, cell_y), owners in sorted(cell_to_parts.items()):
        owner_records = [part_by_index[index] for index in sorted(owners)]
        traversable = any(record["traversable"] for record in owner_records)
        if traversable:
            traversable_cells.add((cell_x, cell_y))
        nodes.append(
            {
                "id": f"{cell_x},{cell_y}",
                "x": cell_x,
                "y": cell_y,
                "occupied": True,
                "traversable": traversable,
                "part_indices": [record["index"] for record in owner_records],
            }
        )

    edges: Dict[Tuple[str, str, str], dict] = {}

    # Add intra-part traversal edges for traversable parts only.
    for record in part_records:
        if not record["traversable"]:
            continue
        cells = record["cells"]
        for cell_x, cell_y in cells:
            for delta_x, delta_y in ((1, 0), (0, 1)):
                neighbor = (cell_x + delta_x, cell_y + delta_y)
                if neighbor not in cells:
                    continue
                source = f"{cell_x},{cell_y}"
                target = f"{neighbor[0]},{neighbor[1]}"
                if source > target:
                    source, target = target, source
                edges[(source, target, "intra_part")] = {
                    "source": source,
                    "target": target,
                    "kind": "intra_part",
                    "traversable": True,
                    "part_index": record["index"],
                }

    valid_door_count = 0
    dangling_door_count = 0
    for door_index, door in enumerate(doors):
        cell_x, cell_y = map(int, door["Cell"])
        orientation = int(door.get("Orientation", 0))
        adjacent_cells = door_adjacent_cells((cell_x, cell_y), orientation)
        if adjacent_cells is None:
            dangling_door_count += 1
            continue
        source_coord, target_coord = adjacent_cells
        if source_coord not in cell_to_parts or target_coord not in cell_to_parts:
            dangling_door_count += 1
            continue
        valid_door_count += 1
        source = f"{source_coord[0]},{source_coord[1]}"
        target = f"{target_coord[0]},{target_coord[1]}"
        if source > target:
            source, target = target, source
        edges[(source, target, f"door:{door_index}")] = {
            "source": source,
            "target": target,
            "kind": "door",
            "traversable": True,
            "door_index": door_index,
            "orientation": orientation,
        }

    return {
        "nodes": nodes,
        "edges": sorted(edges.values(), key=lambda edge: (edge["source"], edge["target"], edge["kind"])),
        "summary": {
            "occupied_cells": len(nodes),
            "traversable_cells": len(traversable_cells),
            "door_records": len(doors),
            "valid_door_edges": valid_door_count,
            "dangling_door_records": dangling_door_count,
        },
    }


def process_ship(ship_path: Path) -> dict:
    """Generate graph artifacts for one extracted or canonical ship JSON file."""

    with ship_path.open(encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    raw_parts = data.get("Parts", [])
    parts = normalize_parts(raw_parts)
    doors = normalize_doors(data.get("Doors", []))

    part_records = []
    cell_to_parts: Dict[Coord, Set[int]] = defaultdict(set)
    unknown_part_ids: Counter = Counter()

    for index, part in enumerate(parts):
        rotation = int(part.get("Rotation", 0)) % 4
        meta, inferred = infer_meta(part["ID"], rotation)
        if inferred:
            unknown_part_ids[part["ID"]] += 1
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
            "geometry_model": (
                "Vanilla part footprints use exact game-file tile data via "
                "load_vanilla_part_geometry(). Non-vanilla and unknown parts use regex "
                "dimension inference with rectangular approximation."
            ),
            "door_model": (
                "Door.Cell names the right or bottom occupied cell of the doorway span. "
                "Orientation 0 joins (x,y-1)<->(x,y); orientation 1 joins (x-1,y)<->(x,y)."
            ),
            "traversability_model": (
                "Cell traversability uses game-file unblocked_footprint_tiles for vanilla parts; "
                "name-hint heuristics for others."
            ),
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
            "unknown_part_ids": dict(sorted(unknown_part_ids.items())),
        },
    }


def generate_all(input_dir: Path, output_dir: Path, limit: Optional[int] = None) -> dict:
    """Generate graph JSON files for an input corpus directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in input_dir.glob("*.json") if path.name != "manifest.json")
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
        output_file_path = output_dir / ship_path.name
        with output_file_path.open("w", encoding="utf-8") as file_handle:
            json.dump(graph_data, file_handle, separators=(",", ":"))
            file_handle.write("\n")

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
            manifest["sample_outputs"].append(output_file_path.name)

    manifest["unknown_part_ids"] = dict(manifest["unknown_part_ids"].most_common())
    manifest["door_stats"] = dict(manifest["door_stats"])

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2)
        file_handle.write("\n")

    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for graph generation."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate structural and cell graphs for extracted Cosmoteer ship JSONs. "
            "Uses canonical game-file geometry for vanilla parts."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="extracted_ship_data",
        help="Directory with extracted *.json ship files",
    )
    parser.add_argument(
        "--output-dir",
        default="generated_ship_graphs",
        help="Directory to write per-ship graph JSON files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for partial validation runs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the graph generation CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = generate_all(Path(args.input_dir), Path(args.output_dir), args.limit)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
