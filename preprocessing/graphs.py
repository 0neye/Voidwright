"""Generate graph-oriented preprocessing artifacts from ship JSON files."""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from common.files import inputs_needing_regeneration, prune_stale_json_outputs, write_output_version
from common.geometry import PartMeta, infer_meta, load_vanilla_part_geometry, normalize_part_id
from ship_layout.connectivity import shared_attachment_sides
from .concurrency import add_concurrency_arguments, run_auto_parallel_work, resolve_worker_count
from .layout_helpers import door_adjacent_cells

_GRAPH_SCHEMA_VERSION = 5

__all__ = [
    "normalize_parts",
    "normalize_doors",
    "part_cells",
    "part_walkable_cells",
    "structural_edges",
    "structural_door_edges",
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


def _local_2x_to_global_grid(local_2x: object, center_2x: object) -> list[int] | None:
    """Convert one centered `2x` coordinate pair into global grid coordinates."""

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


def _sorted_local_2x_cells(cells: Set[Coord], center_2x: Sequence[int] | None) -> List[list[int]]:
    """Return deterministic centered `2x` coordinates for a set of world cells."""

    local_cells = []
    for cell_x, cell_y in sorted(cells):
        local_2x = _legacy_to_local_2x((cell_x, cell_y), center_2x)
        if local_2x is not None:
            local_cells.append(local_2x)
    return local_cells


def normalize_parts(parts: object, *, center_2x: Sequence[int] | None = None) -> List[dict]:
    """Filter and normalize centered-`2x` ship `Parts` records."""

    normalized_parts: List[dict] = []
    for part in parts if isinstance(parts, list) else []:
        if not isinstance(part, dict):
            continue
        part_id = normalize_part_id(part)
        if not part_id:
            continue

        location_2x = _coerce_coord_pair(part.get("Location2x"))
        if location_2x is None or center_2x is None:
            continue
        location = _local_2x_to_global_grid(location_2x, center_2x)
        if location is None:
            continue

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
    """Filter and normalize centered-`2x` ship `Doors` records."""

    if not isinstance(value, list):
        return []
    normalized_doors: List[dict] = []
    for door in value:
        if not isinstance(door, dict):
            continue

        cell_2x = _coerce_coord_pair(door.get("Cell2x"))
        if cell_2x is None or center_2x is None:
            continue
        cell = _local_2x_to_global_grid(cell_2x, center_2x)
        if cell is None:
            continue
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


def _part_from_record(record: dict) -> dict:
    """Convert one graph part record into a shared placement input payload."""

    return {
        "part_id": str(record["part_id"]),
        "rotation": int(record["rotation"]) % 4,
        "x": int(record["location"][0]),
        "y": int(record["location"][1]),
    }


def structural_edges(
    part_records: List[dict],
    cell_to_parts: Dict[Coord, Set[int]],
    geometry_cache: Dict[str, object],
) -> List[dict]:
    """Build structural-touching edges between distinct parts."""

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
                    if key in adjacency:
                        continue
                    source_part = _part_from_record(part_records[key[0]])
                    target_part = _part_from_record(part_records[key[1]])
                    shared_sides = shared_attachment_sides(source_part, target_part, geometry_cache)
                    if not shared_sides:
                        continue
                    adjacency[key] = {
                        "source": key[0],
                        "target": key[1],
                        "kind": "touching",
                        "shared_sides": len(shared_sides),
                    }

    return sorted(adjacency.values(), key=lambda edge: (edge["source"], edge["target"]))


def structural_door_edges(
    part_records: List[dict],
    doors: List[dict],
    cell_to_parts: Dict[Coord, Set[int]],
    geometry_cache: Dict[str, object],
) -> Tuple[List[dict], dict]:
    """Build door edges between distinct parts for the structural part graph.

    Each door that connects occupied cells of two distinct parts becomes an
    edge with ``kind = "door"``. Doors where both occupied cells belong to the
    same part are counted as internal and skipped. Doors whose occupied cells
    cannot be resolved to known ship cells are counted as dangling and skipped.

    Each edge carries ``source_cell_2x`` and ``target_cell_2x``: the centered
    local 2x coordinates of the occupied cells on the source and target parts
    respectively. These are derived from ``door["Cell2x"]`` when present, and
    are ``None`` otherwise. Consumers can use these to validate or constrain
    which cells on each part a door may legally attach to.

    Args:
        part_records: Normalized part-placement records indexed by graph node id.
        doors: Normalized door records from :func:`normalize_doors`.
        cell_to_parts: Mapping from grid cell coordinates to part index sets.
        geometry_cache: Shared vanilla geometry cache used for attachment checks.

    Returns:
        A ``(edges, stats)`` pair. *edges* is a sorted list of door-edge dicts.
        *stats* has ``door_records``, ``door_edges``, ``dangling_door_records``,
        ``internal_door_records``, and ``non_structural_door_records`` counts.
    """

    # In 2x space, DOOR_CELL_DELTAS {0: (0,1), 1: (1,0)} double to these.
    _DOOR_DELTA_2X: Dict[int, Tuple[int, int]] = {0: (0, 2), 1: (2, 0)}

    edges: List[dict] = []
    dangling = 0
    internal = 0
    non_structural = 0

    for door_index, door in enumerate(doors):
        cell_x, cell_y = map(int, door["Cell"])
        orientation = int(door.get("Orientation", 0))
        adjacent = door_adjacent_cells((cell_x, cell_y), orientation)
        if adjacent is None:
            dangling += 1
            continue
        source_coord, target_coord = adjacent
        if source_coord not in cell_to_parts or target_coord not in cell_to_parts:
            dangling += 1
            continue

        # Compute 2x coordinates for both adjacent cells.
        # door["Cell2x"] is the stored (right/bottom) cell == target_coord in 2x.
        # The other (previous/left/top) cell is offset by the 2x delta.
        raw_cell_2x = _coerce_coord_pair(door.get("Cell2x"))
        if raw_cell_2x is not None:
            dx, dy = _DOOR_DELTA_2X.get(orientation, (0, 0))
            stored_cell_2x: list[int] | None = raw_cell_2x
            prev_cell_2x: list[int] | None = [raw_cell_2x[0] - dx, raw_cell_2x[1] - dy]
        else:
            stored_cell_2x = None
            prev_cell_2x = None

        # source_coord = previous/left/top cell; target_coord = stored/right/bottom cell.
        # cell_2x for source_coord = prev_cell_2x; for target_coord = stored_cell_2x.
        coord_to_2x = {source_coord: prev_cell_2x, target_coord: stored_cell_2x}

        cross_pairs = {
            (min(a, b), max(a, b))
            for a in cell_to_parts[source_coord]
            for b in cell_to_parts[target_coord]
            if a != b
        }
        if not cross_pairs:
            internal += 1
            continue

        # For each cross-pair, determine which cell (2x) belongs to which endpoint.
        # source_parts owns source_coord; target_parts owns target_coord.
        source_parts = cell_to_parts[source_coord]
        for part_a, part_b in sorted(cross_pairs):
            source_part = _part_from_record(part_records[part_a])
            target_part = _part_from_record(part_records[part_b])
            if not shared_attachment_sides(source_part, target_part, geometry_cache):
                non_structural += 1
                continue

            # part_a < part_b by construction. Determine cell ownership.
            if part_a in source_parts:
                cell_2x_a, cell_2x_b = coord_to_2x[source_coord], coord_to_2x[target_coord]
            else:
                cell_2x_a, cell_2x_b = coord_to_2x[target_coord], coord_to_2x[source_coord]
            edges.append(
                {
                    "source": part_a,
                    "target": part_b,
                    "kind": "door",
                    "door_index": door_index,
                    "orientation": orientation,
                    "source_cell_2x": cell_2x_a,
                    "target_cell_2x": cell_2x_b,
                }
            )

    return (
        sorted(edges, key=lambda e: (e["source"], e["target"], e["door_index"])),
        {
            "door_records": len(doors),
            "door_edges": len(edges),
            "dangling_door_records": dangling,
            "internal_door_records": internal,
            "non_structural_door_records": non_structural,
        },
    )


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
            "location_2x": part["Location2x"],
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
            "location_2x": record["location_2x"],
            "rotation": record["rotation"],
            "footprint": {
                "cell_count": len(record["cells"]),
                "width": record["width"],
                "height": record["height"],
            },
            "traversable": record["traversable"],
            "walkable_cells_2x": _sorted_local_2x_cells(record["walkable_cells"], center_2x),
            "meta_note": record["meta_note"],
        }
        for record in part_records
    ]

    geometry_cache = load_vanilla_part_geometry()
    touch_edges = structural_edges(part_records, cell_to_parts, geometry_cache)
    door_edges_list, door_stats = structural_door_edges(
        part_records,
        doors,
        dict(cell_to_parts),
        geometry_cache,
    )
    all_edges = sorted(
        touch_edges + door_edges_list,
        key=lambda e: (e["source"], e["target"], e["kind"]),
    )

    traversable_cell_count = len({cell for record in part_records for cell in record["walkable_cells"]})

    return {
        "ship": {
            "name": data.get("Name"),
            "author": data.get("Author"),
            "source_file": ship_path.name,
            "version": data.get("Version"),
            "flight_direction": data.get("FlightDirection"),
        },
        "schema_version": _GRAPH_SCHEMA_VERSION,
        "coord_transform": {
            "version": int(coord_transform.get("version", 1)) if isinstance(coord_transform, dict) else 1,
            "frame": (
                str(coord_transform.get("frame", "bbox_center_2x"))
                if isinstance(coord_transform, dict)
                else "bbox_center_2x"
            ),
            "scale": int(coord_transform.get("scale", 2)) if isinstance(coord_transform, dict) else 2,
            "center_2x": center_2x or [0, 0],
        },
        # Keep normalized door records alongside the derived graph so callers can
        # later replay or synthesize doors without reverse-engineering them from
        # graph edges and summary counters.
        "doors": [
            {
                "Cell2x": door["Cell2x"],
                "Orientation": door["Orientation"],
            }
            for door in doors
        ],
        "assumptions": {
            "geometry_model": (
                "Vanilla part footprints use exact game-file tile data via "
                "load_vanilla_part_geometry(). Non-vanilla and unknown parts use regex "
                "dimension inference with rectangular approximation."
            ),
            "structural_touch_model": (
                "Structural edges require a shared attachable hull side. Polygon parts "
                "(wedge/tri) use axis-aligned polygon edges, parts with physical_rect use "
                "their core body rect, and remaining parts fall back to footprint boundaries. "
                "shared_sides counts unit 2x attachment segments, not footprint-cell adjacencies."
            ),
            "door_model": (
                "Door.Cell names the right or bottom occupied cell of the doorway span. "
                "Orientation 0 joins (x,y-1)<->(x,y); orientation 1 joins (x-1,y)<->(x,y). "
                "Doors appear as edges with kind='door' in A_structural_part_graph, linking the "
                "two distinct parts whose occupied cells the door connects, but only when those "
                "parts also share an attachable structural wall. Doors whose cells cannot be "
                "resolved to known ship cells are counted as dangling and omitted. Doors "
                "connecting two cells of the same part are counted as internal and omitted."
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
                "edges": all_edges,
                "summary": {
                    "parts": len(structure_nodes),
                    "touching_edges": len(touch_edges),
                    "door_edges": door_stats["door_edges"],
                    "door_records": door_stats["door_records"],
                    "dangling_door_records": door_stats["dangling_door_records"],
                    "internal_door_records": door_stats["internal_door_records"],
                    "non_structural_door_records": door_stats["non_structural_door_records"],
                },
            },
        },
        "validation": {
            "normalized_part_count": len(parts),
            "normalized_door_count": len(doors),
            "raw_part_count": len(raw_parts) if isinstance(raw_parts, list) else 0,
            "occupied_cells": len(cell_to_parts),
            "traversable_cells": traversable_cell_count,
            "unknown_part_ids": dict(sorted(unknown_part_ids.items())),
        },
    }


def _read_existing_graph_summary(output_path: Path) -> dict | None:
    """Read a compact summary from an already-generated graph JSON file.

    Returns the same shape as the dict returned by _generate_single_graph, or
    None if the file cannot be read or is structurally incomplete.
    """
    try:
        with output_path.open(encoding="utf-8") as fh:
            graph_data = json.load(fh)
        struct_summary = graph_data["graphs"]["A_structural_part_graph"]["summary"]
        validation = graph_data["validation"]
        return {
            "output_name": output_path.name,
            "unknown_part_ids": validation["unknown_part_ids"],
            "door_stats": {
                "door_records": struct_summary["door_records"],
                "door_edges": struct_summary["door_edges"],
                "dangling_door_records": struct_summary["dangling_door_records"],
                "internal_door_records": struct_summary["internal_door_records"],
                "non_structural_door_records": struct_summary["non_structural_door_records"],
                "occupied_cells": validation["occupied_cells"],
                "traversable_cells": validation["traversable_cells"],
            },
        }
    except Exception:
        return None


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

    struct_summary = graph_data["graphs"]["A_structural_part_graph"]["summary"]
    validation = graph_data["validation"]
    return {
        "output_name": output_file_path.name,
        "unknown_part_ids": validation["unknown_part_ids"],
        "door_stats": {
            "door_records": struct_summary["door_records"],
            "door_edges": struct_summary["door_edges"],
            "dangling_door_records": struct_summary["dangling_door_records"],
            "internal_door_records": struct_summary["internal_door_records"],
            "non_structural_door_records": struct_summary["non_structural_door_records"],
            "occupied_cells": validation["occupied_cells"],
            "traversable_cells": validation["traversable_cells"],
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

    files_to_process, skipped_files = inputs_needing_regeneration(
        files,
        output_dir,
        current_version=_GRAPH_SCHEMA_VERSION,
        version_key="schema_version",
    )
    ships_skipped = len(skipped_files)
    if ships_skipped:
        print(f"Skipping {ships_skipped} up-to-date graph file(s) in {output_dir}", flush=True)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "schema_version": _GRAPH_SCHEMA_VERSION,
        "geometry_source": "game-file canonical (load_vanilla_part_geometry)",
        "coord_frame": "centered 2x local coordinates with global replay metadata",
        "ships_processed": 0,
        "ships_skipped": ships_skipped,
        "ships_with_unknown_part_ids": 0,
        "total_unknown_part_instances": 0,
        "unknown_part_ids": Counter(),
        "door_stats": Counter(),
        "sample_outputs": [],
    }

    graph_results: List[dict] = []
    if files_to_process:
        worker_count = resolve_worker_count(
            task_count=len(files_to_process),
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
                    for ship_path in files_to_process
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
                            f"Generated {index}/{len(files_to_process)} graph files with {worker_count} worker(s)...",
                            flush=True,
                        )
            return results

        graph_results, _ = run_auto_parallel_work(
            stage_name="graphs",
            requested_mode=executor,
            worker_count=worker_count,
            submit_work=submit_graph_work,
        )

    # Prune stale outputs left over from previous runs, but only when processing
    # the full input set.  A limited run is a non-destructive validation subset,
    # so its truncated keep-list must never be used to delete unrelated outputs.
    if limit is None:
        pruned_count = prune_stale_json_outputs(
            output_dir, (f.name for f in files), exclude=["manifest.json"]
        )
        if pruned_count:
            print(f"Pruned {pruned_count} stale graph file(s) from {output_dir}", flush=True)

    # Collect summaries from skipped (up-to-date) output files so the manifest
    # reflects the full corpus, not just the incremental delta.
    for skipped_path in skipped_files:
        summary = _read_existing_graph_summary(output_dir / skipped_path.name)
        if summary is not None:
            graph_results.append(summary)

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

    if limit is None:
        write_output_version(output_dir, "schema_version", _GRAPH_SCHEMA_VERSION)

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
        default="extracted_ship_data_canonical",
        help="Directory with extracted *.json ship files",
    )
    parser.add_argument(
        "--output-dir",
        default="generated_ship_graphs_canonical",
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
