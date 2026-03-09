#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


Coord = Tuple[int, int]


@dataclass(frozen=True)
class PartMeta:
    width: int
    height: int
    traversable: bool = False
    note: str = ""


# Compact explicit metadata for common/validated IDs seen in the corpus.
PART_META: Dict[str, PartMeta] = {
    "cosmoteer.armor": PartMeta(1, 1, False),
    "cosmoteer.armor_2x1": PartMeta(2, 1, False),
    "cosmoteer.armor_wedge": PartMeta(1, 1, False, "wedge approximated as full 1x1 tile"),
    "cosmoteer.armor_tri": PartMeta(1, 1, False, "triangle approximated as full 1x1 tile"),
    "cosmoteer.armor_1x2_wedge": PartMeta(1, 2, False, "wedge approximated as full rectangle"),
    "cosmoteer.armor_1x2_wedge_L": PartMeta(1, 2, False, "wedge approximated as full rectangle"),
    "cosmoteer.armor_1x2_wedge_R": PartMeta(1, 2, False, "wedge approximated as full rectangle"),
    "cosmoteer.armor_1x3_wedge": PartMeta(1, 3, False, "wedge approximated as full rectangle"),
    "cosmoteer.armor_structure_hybrid_1x1": PartMeta(1, 1, False),
    "cosmoteer.armor_structure_hybrid_1x2": PartMeta(1, 2, False),
    "cosmoteer.armor_structure_hybrid_1x3": PartMeta(1, 3, False),
    "cosmoteer.armor_structure_hybrid_tri": PartMeta(1, 1, False, "triangle approximated as full 1x1 tile"),
    "cosmoteer.structure": PartMeta(1, 1, False),
    "cosmoteer.structure_wedge": PartMeta(1, 1, False, "wedge approximated as full 1x1 tile"),
    "cosmoteer.structure_tri": PartMeta(1, 1, False, "triangle approximated as full 1x1 tile"),
    "cosmoteer.structure_1x2_wedge": PartMeta(1, 2, False, "wedge approximated as full rectangle"),
    "cosmoteer.structure_1x2_wedge_L": PartMeta(1, 2, False, "wedge approximated as full rectangle"),
    "cosmoteer.structure_1x2_wedge_R": PartMeta(1, 2, False, "wedge approximated as full rectangle"),
    "cosmoteer.structure_1x3_wedge": PartMeta(1, 3, False, "wedge approximated as full rectangle"),
    "cosmoteer.corridor": PartMeta(1, 1, True),
    "cosmoteer.conveyor": PartMeta(1, 1, True),
    "cosmoteer.airlock": PartMeta(1, 2, True),
    "cosmoteer.fire_extinguisher": PartMeta(1, 1, True),
    "cosmoteer.heat_pipe_adaptive": PartMeta(1, 1, True),
    "cosmoteer.heat_pipe_adaptive_structure": PartMeta(1, 1, False),
    "cosmoteer.heat_pipe_crossing": PartMeta(1, 1, True),
    "cosmoteer.radiator": PartMeta(2, 1, True),
    "cosmoteer.heat_exchanger": PartMeta(2, 2, True),
    "cosmoteer.thermal_battery": PartMeta(2, 2, True),
    "cosmoteer.thermal_amplification_pump": PartMeta(2, 2, True),
    "cosmoteer.thermal_dilation_pump": PartMeta(2, 2, True),
    "cosmoteer.power_storage": PartMeta(2, 2, True),
    "cosmoteer.reactor_small": PartMeta(2, 2, True),
    "cosmoteer.reactor_med": PartMeta(3, 3, True),
    "cosmoteer.reactor_large": PartMeta(5, 5, True),
    "cosmoteer.engine_room": PartMeta(3, 3, True),
    "cosmoteer.control_room_small": PartMeta(2, 2, True),
    "cosmoteer.control_room_med": PartMeta(3, 3, True),
    "cosmoteer.control_room_large": PartMeta(4, 4, True),
    "cosmoteer.crew_quarters_small": PartMeta(2, 1, True),
    "cosmoteer.crew_quarters_med": PartMeta(3, 2, True),
    "cosmoteer.crew_quarters_large": PartMeta(4, 3, True),
    "ultranova.crew_quarters_walkthrough": PartMeta(2, 1, True),
    "janiTNT.1x1quarters": PartMeta(1, 1, True),
    "janiTNT.crew_quarters_4x1": PartMeta(4, 1, True),
    "cosmoteer.storage_2x2": PartMeta(2, 2, True),
    "cosmoteer.storage_3x2": PartMeta(3, 2, True),
    "cosmoteer.storage_3x3": PartMeta(3, 3, True),
    "cosmoteer.storage_4x3": PartMeta(4, 3, True),
    "cosmoteer.storage_4x4": PartMeta(4, 4, True),
    "cosmoteer.ammo_storage": PartMeta(2, 2, True),
    "cosmoteer.ammo_factory": PartMeta(3, 2, True),
    "cosmoteer.factory_ammo": PartMeta(3, 2, True),
    "cosmoteer.factory_emp": PartMeta(3, 2, True),
    "cosmoteer.factory_he": PartMeta(3, 2, True),
    "cosmoteer.factory_mine": PartMeta(3, 2, True),
    "cosmoteer.factory_nuke": PartMeta(3, 2, True),
    "cosmoteer.factory_thermal": PartMeta(3, 2, True),
    "cosmoteer.mine_factory": PartMeta(3, 2, True),
    "cosmoteer.missile_factory": PartMeta(3, 2, True),
    "cosmoteer.missile_factory_emp": PartMeta(3, 2, True),
    "cosmoteer.missile_factory_he": PartMeta(3, 2, True),
    "cosmoteer.missile_factory_nuke": PartMeta(3, 2, True),
    "cosmoteer.hyperdrive_small": PartMeta(3, 3, True),
    "cosmoteer.hyperdrive_med": PartMeta(5, 5, True),
    "cosmoteer.ftl_drive": PartMeta(5, 5, True),
    "cosmoteer.sensor_array": PartMeta(3, 3, True),
    "cosmoteer.shield_gen_small": PartMeta(2, 2, True),
    "cosmoteer.shield_gen_large": PartMeta(3, 3, True),
    "Bonible.HardlightShield2": PartMeta(2, 2, True),
    "swefpifh.Kebechet_GenSmall_ModularShield": PartMeta(2, 2, True),
    "cosmoteer.point_defense": PartMeta(1, 1, False),
    "cosmoteer.cannon_deck": PartMeta(2, 1, False),
    "cosmoteer.cannon_med": PartMeta(2, 2, False),
    "cosmoteer.cannon_large": PartMeta(3, 3, False),
    "cosmoteer.flak_cannon_large": PartMeta(3, 3, False),
    "cosmoteer.chaingun": PartMeta(2, 2, False),
    "cosmoteer.chaingun_magazine": PartMeta(1, 1, False),
    "cosmoteer.disruptor": PartMeta(2, 2, False),
    "danger.autocannon": PartMeta(2, 2, False),
    "cosmoteer.electro_bolter": PartMeta(2, 2, False),
    "cosmoteer.laser_blaster_small": PartMeta(2, 2, False),
    "cosmoteer.laser_blaster_large": PartMeta(3, 3, False),
    "cosmoteer.ion_beam_emitter": PartMeta(3, 3, False),
    "cosmoteer.ion_beam_prism": PartMeta(1, 1, False),
    "cosmoteer.ion_beam_prism_45": PartMeta(1, 1, False),
    "cosmoteer.railgun_accelerator": PartMeta(1, 1, False),
    "cosmoteer.railgun_loader": PartMeta(2, 2, False),
    "cosmoteer.railgun_launcher": PartMeta(3, 3, False),
    "cosmoteer.missile_launcher": PartMeta(2, 3, False),
    "cosmoteer.missile_storage": PartMeta(2, 2, True),
    "cosmoteer.mining_laser_small": PartMeta(1, 2, False),
    "cosmoteer.tractor_beam_emitter": PartMeta(2, 2, False),
    "cosmoteer.manipulator_beam_emitter": PartMeta(2, 2, False),
    "cosmoteer.resonance_beam_turret": PartMeta(3, 3, False),
    "rustydios.missile_silo": PartMeta(2, 3, False),
    "cosmoteer.roof_headlight": PartMeta(1, 1, False),
    "cosmoteer.roof_light": PartMeta(1, 1, False),
    "cosmoteer.explosive_charge": PartMeta(1, 1, False),
    "jfjohnny5.armor_heavy_2x1": PartMeta(2, 1, False),
    "cosmoteer.thruster_small": PartMeta(1, 1, False),
    "cosmoteer.thruster_small_2way": PartMeta(1, 1, False),
    "cosmoteer.thruster_med": PartMeta(1, 2, False),
    "cosmoteer.thruster_large": PartMeta(2, 3, False),
    "cosmoteer.thruster_huge": PartMeta(3, 5, False),
    "janiTNT.pulse_thruster_huge": PartMeta(3, 5, False),
    "cosmoteer.thruster_boost": PartMeta(1, 2, False),
    "cosmoteer.thruster_rocket_battery": PartMeta(2, 2, False),
    "cosmoteer.thruster_rocket_extender": PartMeta(1, 1, False),
    "cosmoteer.thruster_rocket_nozzle": PartMeta(2, 1, False),
}


TRAVERSABLE_HINTS = (
    "corridor",
    "conveyor",
    "crew_quarters",
    "quarters",
    "storage",
    "reactor",
    "engine_room",
    "control_room",
    "heat_pipe",
    "radiator",
    "heat_exchanger",
    "thermal_",
    "power_storage",
    "factory",
    "ammo_factory",
    "ammo_storage",
    "airlock",
    "hyperdrive",
    "ftl_drive",
    "shield_gen",
    "sensor_array",
    "fire_extinguisher",
)

NON_TRAVERSABLE_HINTS = (
    "armor",
    "structure",
    "thruster",
    "point_defense",
    "cannon",
    "chaingun",
    "laser_blaster",
    "disruptor",
    "railgun",
    "ion_beam",
    "tractor_beam",
    "manipulator_beam",
    "mining_laser",
    "resonance_beam",
    "electro_bolter",
    "explosive_charge",
    "missile_launcher",
    "missile_silo",
    "missile_storage",
    "roof_",
)


DOOR_ORIENTATION_DELTAS = {
    0: (1, 0),  # assumed east-west door between (x,y) and (x+1,y)
    1: (0, 1),  # assumed north-south door between (x,y) and (x,y+1)
}


def normalize_part_id(part: dict) -> Optional[str]:
    return part.get("ID") or part.get("IDString")


def normalize_parts(parts: Sequence[dict]) -> List[dict]:
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
    doors: List[dict] = []
    for door in value:
        if isinstance(door, dict) and "Cell" in door and "Orientation" in door:
            doors.append(door)
    return doors


def infer_meta(part_id: str) -> Tuple[PartMeta, bool]:
    if part_id in PART_META:
        return PART_META[part_id], False

    match = re.search(r"_(\d+)x(\d+)(?:_|$)", part_id)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
    else:
        width, height = 1, 1

    lower = part_id.lower()
    traversable = any(token in lower for token in TRAVERSABLE_HINTS)
    if any(token in lower for token in NON_TRAVERSABLE_HINTS):
        traversable = False

    note = "regex/fallback inferred"
    if width == height == 1:
        note = "defaulted to 1x1 fallback"
    return PartMeta(width, height, traversable, note), True


def rotate_dims(width: int, height: int, rotation: int) -> Tuple[int, int]:
    return (height, width) if rotation % 2 else (width, height)


def part_cells(part: dict, meta: PartMeta) -> Set[Coord]:
    x0, y0 = map(int, part["Location"])
    width, height = rotate_dims(meta.width, meta.height, int(part.get("Rotation", 0)))
    return {(x0 + dx, y0 + dy) for dx in range(width) for dy in range(height)}


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
    traversable_cells = set()

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

    # Conservative intra-part traversability: adjacent cells inside the same traversable part.
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
                edges[(a, b, "intra_part")]= {
                    "source": a,
                    "target": b,
                    "kind": "intra_part",
                    "traversable": True,
                    "part_index": record["index"],
                }

    # Door links between adjacent occupied cells.
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
    unknown_ids = Counter()

    for index, part in enumerate(parts):
        meta, inferred = infer_meta(part["ID"])
        if inferred:
            unknown_ids[part["ID"]] += 1
        cells = part_cells(part, meta)
        record = {
            "index": index,
            "part_id": part["ID"],
            "location": list(map(int, part["Location"])),
            "rotation": int(part.get("Rotation", 0)),
            "width": meta.width,
            "height": meta.height,
            "rotated_width": rotate_dims(meta.width, meta.height, int(part.get("Rotation", 0)))[0],
            "rotated_height": rotate_dims(meta.width, meta.height, int(part.get("Rotation", 0)))[1],
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
                "base_width": record["width"],
                "base_height": record["height"],
                "rotated_width": record["rotated_width"],
                "rotated_height": record["rotated_height"],
                "cell_count": len(record["cells"]),
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
        "schema_version": 1,
        "assumptions": {
            "rotation_model": "Location treated as top-left anchor of the unrotated footprint; odd rotations swap width/height.",
            "wedge_model": "Wedge/triangle parts are approximated by their bounding rectangles for occupancy/touch analysis.",
            "door_model": "Door orientation 0 is assumed east-west and 1 north-south, based on validation against occupied-cell adjacency.",
            "traversability_model": "Cell traversability is conservative and metadata-driven; it should be treated as an analysis heuristic, not exact in-game pathing.",
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
    parser = argparse.ArgumentParser(description="Generate structural and cell graphs for extracted Cosmoteer ship JSONs.")
    parser.add_argument("--input-dir", default="extracted_ship_data", help="Directory containing extracted *.json ship files")
    parser.add_argument("--output-dir", default="generated_ship_graphs", help="Directory to write per-ship graph JSON files")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for partial validation runs")
    args = parser.parse_args()

    manifest = generate_all(Path(args.input_dir), Path(args.output_dir), args.limit)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
