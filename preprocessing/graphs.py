"""Generate graph-oriented preprocessing artifacts from ship JSON files."""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from common.geometry import PartMeta, infer_meta, load_vanilla_part_geometry, normalize_part_id
from .concurrency import add_concurrency_arguments, run_auto_parallel_work, resolve_worker_count
from .layout_helpers import door_adjacent_cells

__all__ = [
    "normalize_parts",
    "normalize_doors",
    "part_cells",
    "part_walkable_cells",
    "structural_edges",
    "cell_graph",
    "process_ship",
    "generate_all",
    "build_parser",
    "main",
]

Coord = Tuple[int, int]


def _coerce_coord_pair(value: object) -> list[int] | None:
    """Return an integer coordinate pair when *value* has two numeric entries."""

    if not isinstance(value, list) or len(value) != 2:
        return None
    return [int(value[0]), int(value[1])]


def _recover_legacy_coord_from_2x(local_2x: object, center_2x: object) -> list[int] | None:
    """Recover a legacy grid coordinate from centered `2x` coordinates."""

    local_pair = _coerce_coord_pair(local_2x)
    center_pair = _coerce_coord_pair(center_2x)
    if local_pair is None or center_pair is None:
        return None

    summed_x = local_pair[0] + center_pair[0]
    summed_y = local_pair[1] + center_pair[1]
    if summed_x % 2 != 0 or summed_y % 2 != 0:
        return None
    return [summed_x // 2, summed_y // 2]


def _legacy_to_local_2x(location: Sequence[int], center_2x: Sequence[int] | None) -> list[int] | None:
    """Map one legacy grid coordinate into centered `2x` coordinates."""

    if center_2x is None:
        return None
    return [int(location[0]) * 2 - int(center_2x[0]), int(location[1]) * 2 - int(center_2x[1])]


def normalize_parts(parts: object, *, center_2x: Sequence[int] | None = None) -> List[dict]:
    """Filter and normalize raw ship `Parts` records."""

    normalized_parts: List[dict] = []
    for part in parts if isinstance(parts, list) else []:
        if not isinstance(part, dict):
            continue
        part_id = normalize_part_id(part)
        if not part_id:
            continue

        location = _coerce_coord_pair(part.get("Location"))
        location_2x = _coerce_coord_pair(part.get("Location2x"))

        # Support phased migration payloads where legacy locations may be absent.
        if location is None and location_2x is not None and center_2x is not None:
            location = _recover_legacy_coord_from_2x(location_2x, center_2x)
        if location is None:
            continue
        if location_2x is None:
            location_2x = _legacy_to_local_2x(location, center_2x)

        normalized_parts.append(
            {
                "ID": part_id,
                "Location": location,
                "Location2x": location_2x,
                "Rotation": int(part.get("Rotation", 0)),
            }
        )
    return normalized_parts


def normalize_doors(value: object, *, center_2x: Sequence[int] | None = None) -> List[dict]:
    """Filter and normalize raw ship `Doors` records."""

    if not isinstance(value, list):
        return []
    normalized_doors: List[dict] = []
    for door in value:
        if not isinstance(door, dict):
            continue

        cell = _coerce_coord_pair(door.get("Cell"))
        cell_2x = _coerce_coord_pair(door.get("Cell2x"))
        if cell is None and cell_2x is not None and center_2x is not None:
            cell = _recover_legacy_coord_from_2x(cell_2x, center_2x)
        if cell is None:
            continue
        if cell_2x is None:
            cell_2x = _legacy_to_local_2x(cell, center_2x)
        if "Orientation" not in door:
            continue

        # Preserve normalized door records explicitly so downstream replay and
        # future generation passes can emit real ship doors instead of only
        # inferring traversable edges from the cell graph summary.
        normalized_doors.append(
            {
                "Cell": cell,
                "Cell2x": cell_2x,
                "Orientation": int(door["Orientation"]),
            }
        )

    return normalized_doors


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


def part_walkable_cells(part: dict, meta: PartMeta) -> Set[Coord]:
    """Return the crew-walkable world cells for one normalized part.

    Args:
        part: Normalized part placement record
        meta: Shared geometry and traversability metadata for the part

    Returns:
        World cells that crew can move through for this placement
    """

    if not meta.traversable:
        return set()

    origin_x, origin_y = map(int, part["Location"])

    # Keep the per-tile walkability information from the canonical game data so
    # partially blocked vanilla parts do not become fully traversable.
    local_walkable_tiles = meta.unblocked_tiles - meta.blocked_travel_cells
    if local_walkable_tiles:
        return {(origin_x + dx, origin_y + dy) for dx, dy in local_walkable_tiles}

    return set()


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
    *,
    center_2x: Sequence[int] | None = None,
) -> dict:
    """Build the conservative occupied-cell graph for one ship."""

    nodes = []
    traversable_cells: Set[Coord] = set()
    part_by_index = {record["index"]: record for record in part_records}

    for (cell_x, cell_y), owners in sorted(cell_to_parts.items()):
        owner_records = [part_by_index[index] for index in sorted(owners)]
        traversable = any((cell_x, cell_y) in record["walkable_cells"] for record in owner_records)
        if traversable:
            traversable_cells.add((cell_x, cell_y))
        nodes.append(
            {
                "id": f"{cell_x},{cell_y}",
                "x": cell_x,
                "y": cell_y,
                "center_2x": _legacy_to_local_2x([cell_x, cell_y], center_2x),
                "occupied": True,
                "traversable": traversable,
                "part_indices": [record["index"] for record in owner_records],
            }
        )

    edges: Dict[Tuple[str, str, str], dict] = {}

    # Add intra-part traversal edges for traversable parts only.
    for record in part_records:
        if not record["walkable_cells"]:
            continue
        cells = record["walkable_cells"]
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
    blocked_door_count = 0
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

        # Door edges are only traversable when both occupied endpoint cells are
        # themselves crew-walkable. This keeps partially blocked vanilla parts
        # from leaking blocked cells back into the reachable graph.
        if source_coord not in traversable_cells or target_coord not in traversable_cells:
            blocked_door_count += 1
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
            "blocked_door_records": blocked_door_count,
        },
    }


def process_ship(ship_path: Path) -> dict:
    """Generate graph artifacts for one extracted or canonical ship JSON file."""

    with ship_path.open(encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    coord_transform = data.get("coord_transform", {})
    center_2x = (
        _coerce_coord_pair(coord_transform.get("center_2x")) if isinstance(coord_transform, dict) else None
    )
    raw_parts = data.get("Parts", [])
    parts = normalize_parts(raw_parts, center_2x=center_2x)
    doors = normalize_doors(data.get("Doors", []), center_2x=center_2x)

    part_records = []
    cell_to_parts: Dict[Coord, Set[int]] = defaultdict(set)
    unknown_part_ids: Counter = Counter()

    for index, part in enumerate(parts):
        rotation = int(part.get("Rotation", 0)) % 4
        meta, inferred = infer_meta(part["ID"], rotation)
        if inferred:
            unknown_part_ids[part["ID"]] += 1
        cells = part_cells(part, meta)
        walkable_cells = part_walkable_cells(part, meta)
        record = {
            "index": index,
            "part_id": part["ID"],
            "location": list(map(int, part["Location"])),
            "location_2x": part.get("Location2x") or _legacy_to_local_2x(part["Location"], center_2x),
            "rotation": rotation,
            "width": meta.width,
            "height": meta.height,
            "traversable": meta.traversable,
            "meta_note": meta.note,
            "cells": cells,
            "walkable_cells": walkable_cells,
        }
        part_records.append(record)
        for cell in cells:
            cell_to_parts[cell].add(index)

    structure_nodes = [
        {
            "id": record["index"],
            "part_id": record["part_id"],
            "location": record["location"],
            "location_2x": record["location_2x"],
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
    cells = cell_graph(part_records, cell_to_parts, doors, center_2x=center_2x)

    return {
        "ship": {
            "name": data.get("Name"),
            "author": data.get("Author"),
            "source_file": ship_path.name,
            "version": data.get("Version"),
            "flight_direction": data.get("FlightDirection"),
        },
        "schema_version": 4,
        "coord_transform": {
            "version": int(coord_transform.get("version", 1)) if isinstance(coord_transform, dict) else 1,
            "frame": (
                str(coord_transform.get("frame", "bbox_center_2x"))
                if isinstance(coord_transform, dict)
                else "bbox_center_2x"
            ),
            "scale": int(coord_transform.get("scale", 2)) if isinstance(coord_transform, dict) else 2,
            "legacy_frame": (
                str(coord_transform.get("legacy_frame", "normalized_origin_grid"))
                if isinstance(coord_transform, dict)
                else "normalized_origin_grid"
            ),
            "center_2x": center_2x or [0, 0],
        },
        # Keep normalized door records alongside the derived graph so callers can
        # later replay or synthesize doors without reverse-engineering them from
        # graph edges and summary counters.
        "doors": doors,
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
                "Vanilla part traversability requires positive crew_speed_factor, while "
                "walkable cells come from game-file unblocked_footprint_tiles. "
                "Non-vanilla parts still use name-hint heuristics."
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
            "normalized_door_count": len(doors),
            "raw_part_count": len(raw_parts) if isinstance(raw_parts, list) else 0,
            "unknown_part_ids": dict(sorted(unknown_part_ids.items())),
        },
    }


def _generate_single_graph(source_json_path: str, output_dir: str) -> dict:
    """Process and write graph artifacts for one ship JSON file.

    Args:
        source_json_path: Canonical or extracted ship JSON input path
        output_dir: Directory that should receive the graph JSON output

    Returns:
        A compact summary payload used to build the final manifest
    """

    ship_path = Path(source_json_path)
    graph_output_dir = Path(output_dir)
    graph_data = process_ship(ship_path)
    output_file_path = graph_output_dir / ship_path.name

    with output_file_path.open("w", encoding="utf-8") as file_handle:
        json.dump(graph_data, file_handle, separators=(",", ":"))
        file_handle.write("\n")

    cell_summary = graph_data["graphs"]["C_cell_graph"]["summary"]
    return {
        "output_name": output_file_path.name,
        "unknown_part_ids": graph_data["validation"]["unknown_part_ids"],
        "door_stats": {
            "door_records": cell_summary["door_records"],
            "valid_door_edges": cell_summary["valid_door_edges"],
            "dangling_door_records": cell_summary["dangling_door_records"],
            "blocked_door_records": cell_summary["blocked_door_records"],
            "occupied_cells": cell_summary["occupied_cells"],
            "traversable_cells": cell_summary["traversable_cells"],
        },
    }


def generate_all(
    input_dir: Path,
    output_dir: Path,
    limit: Optional[int] = None,
    workers: int | None = None,
    executor: str = "auto",
) -> dict:
    """Generate graph JSON files for an input corpus directory.

    Args:
        input_dir: Directory containing canonical or extracted ship JSON files
        output_dir: Destination directory for graph JSON files
        limit: Optional subset size for validation runs
        workers: Optional worker-count override for graph generation
        executor: Executor mode override: `auto`, `thread`, or `process`

    Returns:
        A manifest describing the generated graph corpus
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in input_dir.glob("*.json") if path.name != "manifest.json")
    if limit is not None:
        files = files[:limit]

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "schema_version": 4,
        "geometry_source": "game-file canonical (load_vanilla_part_geometry)",
        "coord_frame": "legacy grid coordinates with centered 2x companion fields",
        "ships_processed": 0,
        "ships_with_unknown_part_ids": 0,
        "total_unknown_part_instances": 0,
        "unknown_part_ids": Counter(),
        "door_stats": Counter(),
        "sample_outputs": [],
    }

    graph_results: List[dict] = []
    if files:
        worker_count = resolve_worker_count(
            task_count=len(files),
            stage_name="graphs",
            requested_workers=workers,
            requested_mode=executor,
        )

        def submit_graph_work(executor_factory: type) -> List[dict]:
            """Submit graph-generation work with one executor implementation."""

            results: List[dict] = []
            with executor_factory(max_workers=worker_count) as graph_executor:
                future_to_path = {
                    graph_executor.submit(_generate_single_graph, str(ship_path), str(output_dir)): ship_path
                    for ship_path in files
                }
                for index, future in enumerate(as_completed(future_to_path), start=1):
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        failed_path = future_to_path[future]
                        print(
                            f"Warning: skipping {failed_path.name} — graph generation failed: {exc}",
                            flush=True,
                        )
                    if index % 1000 == 0:
                        print(
                            f"Generated {index}/{len(files)} graph files with {worker_count} worker(s)...",
                            flush=True,
                        )
            return results

        graph_results, _ = run_auto_parallel_work(
            stage_name="graphs",
            requested_mode=executor,
            worker_count=worker_count,
            submit_work=submit_graph_work,
        )

    # Reduce the worker summaries in filename order so manifest counters and
    # sample-output lists remain stable no matter which worker finished first.
    for graph_result in sorted(graph_results, key=lambda result: result["output_name"]):
        unknown_map = graph_result["unknown_part_ids"]
        if unknown_map:
            manifest["ships_with_unknown_part_ids"] += 1
            manifest["unknown_part_ids"].update(unknown_map)
            manifest["total_unknown_part_instances"] += sum(unknown_map.values())

        manifest["door_stats"].update(graph_result["door_stats"])
        manifest["ships_processed"] += 1
        if len(manifest["sample_outputs"]) < 10:
            manifest["sample_outputs"].append(graph_result["output_name"])

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
    add_concurrency_arguments(
        parser,
        help_prefix="graph generation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the graph generation CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = generate_all(
        Path(args.input_dir),
        Path(args.output_dir),
        args.limit,
        workers=args.workers,
        executor=args.executor,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
