from __future__ import annotations

import json
from pathlib import Path

from common.cosmoteer import create_ship_png_bytes, parse_ship_png
from common.files import output_name_for_ship_png
from common.geometry import load_vanilla_part_geometry
from common.cosmoteer import create_ship_png_bytes
from common.cosmoteer.parser import _decode_ship_data, _extract_embedded_payload
from generator.backends.markov.export import (
    export_ship_png,
    generated_parts_to_cosmoteer_parts,
    generated_parts_to_toggle_states,
    graph_to_generated_parts_payload,
    make_minimal_ship_dict,
)
from markov.model import ShipPart
from preprocessing.pipeline import run_pipeline


def _build_normalized_ship(parts: list[dict], *, name: str, doors: list[dict] | None = None) -> dict:
    """Build a minimal normalized ship payload for end-to-end roundtrip tests."""

    return {
        "Version": 1,
        "Name": name,
        "FlightDirection": 1,
        "FormationOrder": 0,
        "ShipRulesID": "cosmoteer.terran",
        "RoofBaseTexture": "scratched1",
        "CrewUniformColor": ["0000803F", "00000000", "00000000", "0000803F"],
        "RoofBaseColor": ["907F083F", "907F083F", "907F083F", "0000403F"],
        "RoofDecalColor1": ["9A99193E", "9A99193E", "9A99193E", "0000803F"],
        "RoofDecalColor2": ["0000803F", "0000803F", "0000803F", "0000803F"],
        "Parts": parts,
        "Doors": doors or [],
    }


def _normalize_part_for_comparison(part: dict) -> tuple:
    """Return a stable comparable tuple for one normalized ship part."""

    location = part.get("Location", [0, 0])
    return (
        str(part["ID"]),
        int(location[0]),
        int(location[1]),
        int(part.get("Rotation", 0)),
        bool(part.get("FlipX", False)),
        bool(part.get("FlipY", False)),
    )


def _sorted_parts(parts: list[dict]) -> list[tuple]:
    """Sort normalized ship parts into a stable canonical order."""

    return sorted(_normalize_part_for_comparison(part) for part in parts)


def _normalize_door_for_comparison(door: dict) -> tuple:
    """Return a stable comparable tuple for one normalized ship door."""

    cell = door.get("Cell", [0, 0])
    return (
        int(cell[0]),
        int(cell[1]),
        int(door.get("Orientation", 0)),
    )


def _sorted_doors(doors: list[dict]) -> list[tuple]:
    """Sort normalized ship doors into a stable canonical order."""

    return sorted(_normalize_door_for_comparison(door) for door in doors)


def _grid_from_local_2x(point: list[int], center_2x: list[int]) -> list[int]:
    """Convert one centered `2x` coordinate into global grid coordinates."""

    return [
        (int(point[0]) + int(center_2x[0])) // 2,
        (int(point[1]) + int(center_2x[1])) // 2,
    ]


def _sorted_parts_from_relative_payload(payload: dict) -> list[tuple]:
    """Normalize a centered-`2x` payload into stable world-grid part tuples."""

    center_2x = payload.get("coord_transform", {}).get("center_2x", [0, 0])
    normalized_parts = []
    for part in payload.get("Parts", []):
        location = _grid_from_local_2x(part["Location2x"], center_2x)
        normalized_parts.append(
            {
                "ID": part["ID"],
                "Location": location,
                "Rotation": part.get("Rotation", 0),
                "FlipX": part.get("FlipX", False),
                "FlipY": part.get("FlipY", False),
            }
        )
    return _sorted_parts(normalized_parts)


def _sorted_doors_from_relative_payload(payload: dict) -> list[tuple]:
    """Normalize a centered-`2x` payload into stable world-grid door tuples."""

    center_2x = payload.get("coord_transform", {}).get("center_2x", [0, 0])
    normalized_doors = []
    for door in payload.get("Doors", []):
        normalized_doors.append(
            {
                "Cell": _grid_from_local_2x(door["Cell2x"], center_2x),
                "Orientation": door.get("Orientation", 0),
            }
        )
    return _sorted_doors(normalized_doors)


def _place_part_on_rightmost_frontier(
    *,
    part_id: str,
    occupied_cells: set[tuple[int, int]],
) -> ShipPart:
    """Place one part so it touches the current ship frontier without overlap.

    Args:
        part_id: Vanilla part ID to place
        occupied_cells: Current occupied ship cells

    Returns:
        A connected, non-overlapping `ShipPart` placement
    """

    geometry_cache = load_vanilla_part_geometry()
    footprint_tiles = geometry_cache[part_id].rotations[0].footprint_tiles
    if not footprint_tiles:
        raise ValueError(f"Expected footprint tiles for {part_id}")

    rightmost_x = max(cell_x for cell_x, _cell_y in occupied_cells)
    frontier_cells = sorted(cell for cell in occupied_cells if cell[0] == rightmost_x)
    anchor_x, anchor_y = frontier_cells[0]

    # Use the new part's left edge so every candidate cell lands strictly to the
    # right of the existing ship while still touching the current frontier
    leftmost_local_x = min(cell_x for cell_x, _cell_y in footprint_tiles)
    left_edge_tiles = sorted(tile for tile in footprint_tiles if tile[0] == leftmost_local_x)
    touch_tile_x, touch_tile_y = left_edge_tiles[0]

    return ShipPart(
        part_id=part_id,
        rotation=0,
        x=anchor_x + 1 - touch_tile_x,
        y=anchor_y - touch_tile_y,
    )


def _build_connected_all_vanilla_parts_ship(*, name: str) -> dict:
    """Build one connected normalized ship containing every vanilla part once."""

    geometry_cache = load_vanilla_part_geometry()
    placed_parts: list[ShipPart] = []
    occupied_cells: set[tuple[int, int]] = set()

    for index, part_id in enumerate(sorted(geometry_cache)):
        if index == 0:
            part = ShipPart(part_id=part_id, rotation=0, x=0, y=0)
        else:
            part = _place_part_on_rightmost_frontier(
                part_id=part_id,
                occupied_cells=occupied_cells,
            )

        part_cells = part.footprint_cells(geometry_cache)
        if part_cells & occupied_cells:
            raise AssertionError(f"Unexpected overlap while placing {part_id}")

        # Every part after the first must share at least one touching side with
        # the existing chain so the structural graph remains fully connected
        if placed_parts:
            is_connected = any(
                (cell_x - 1, cell_y) in occupied_cells
                or (cell_x + 1, cell_y) in occupied_cells
                or (cell_x, cell_y - 1) in occupied_cells
                or (cell_x, cell_y + 1) in occupied_cells
                for cell_x, cell_y in part_cells
            )
            if not is_connected:
                raise AssertionError(f"Expected connected placement for {part_id}")

        placed_parts.append(part)
        occupied_cells.update(part_cells)

    normalized_parts = [
        {
            "ID": part.part_id,
            "Location": [part.x, part.y],
            "Rotation": part.rotation,
            "FlipX": part.flip_x,
            "FlipY": part.flip_y,
        }
        for part in placed_parts
    ]
    return _build_normalized_ship(normalized_parts, name=name)


def _build_two_corridors_with_shared_door(*, name: str) -> dict:
    """Build a minimal traversable ship with one preserved vertical door."""

    # Use two adjacent corridors so the door sits on a simple, fully walkable
    # occupied-cell boundary that the graph stage can preserve exactly.
    return _build_normalized_ship(
        [
            {"ID": "cosmoteer.corridor", "Location": [0, 0], "Rotation": 0},
            {"ID": "cosmoteer.corridor", "Location": [0, 1], "Rotation": 0},
        ],
        doors=[{"Cell": [0, 1], "Orientation": 0}],
        name=name,
    )


def test_pipeline_roundtrip_replays_all_vanilla_parts_from_graph(tmp_path: Path) -> None:
    """The full preprocessing pipeline should replay all vanilla parts exactly."""

    normalized_ship = _build_connected_all_vanilla_parts_ship(name="all-vanilla-roundtrip")
    source_path = tmp_path / "all-vanilla.ship.png"
    extracted_dir = tmp_path / "extracted"
    canonical_dir = tmp_path / "canonical"
    graph_dir = tmp_path / "graphs"
    exported_path = tmp_path / "replayed.ship.png"

    source_path.write_bytes(create_ship_png_bytes(normalized_ship))

    # Persist every stage so the test can assert the real pipeline artifacts,
    # not just the final returned manifest
    pipeline_payload = run_pipeline(
        input_paths=[source_path],
        output_dir=graph_dir,
        extracted_dir=extracted_dir,
        canonical_dir=canonical_dir,
        opt_in_csv=tmp_path / "no-opt-in.csv",
        extract_workers=1,
        extract_executor="thread",
        canonicalize_workers=1,
        canonicalize_executor="thread",
        graph_workers=1,
        graph_executor="thread",
    )

    extracted_path = extracted_dir / output_name_for_ship_png(source_path)
    canonical_path = canonical_dir / output_name_for_ship_png(source_path)
    graph_path = graph_dir / output_name_for_ship_png(source_path)

    assert extracted_path.exists()
    assert canonical_path.exists()
    assert graph_path.exists()

    extracted_payload = json.loads(extracted_path.read_text(encoding="utf-8"))
    canonical_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_nodes = graph_payload["graphs"]["A_structural_part_graph"]["nodes"]

    assert _sorted_parts_from_relative_payload(extracted_payload) == _sorted_parts(normalized_ship["Parts"])
    assert _sorted_parts_from_relative_payload(canonical_payload) == _sorted_parts(normalized_ship["Parts"])

    assert pipeline_payload["extract_exit_code"] == 0
    assert pipeline_payload["canonicalization"]["parsed_input_json_files"] == 1
    assert pipeline_payload["canonicalization"]["unique_content_groups"] == 1
    assert pipeline_payload["graphs"]["ships_processed"] == 1
    assert all("walkable_cells_2x" in node for node in graph_nodes)
    assert any(node["walkable_cells_2x"] for node in graph_nodes)
    assert graph_payload["graphs"]["A_structural_part_graph"]["summary"]["parts"] == len(
        normalized_ship["Parts"]
    )
    assert graph_payload["doors"] == []

    generated_payload = graph_to_generated_parts_payload(graph_payload, name="graph-replay")
    export_result = export_ship_png(generated_payload, exported_path, validate=True)
    reparsed_payload = parse_ship_png(exported_path)

    assert export_result["valid"] is True
    assert generated_payload["stats"]["stop_reason"] == "graph_replay"
    assert len(generated_payload["parts"]) == len(normalized_ship["Parts"])
    assert generated_payload["doors"] == []
    assert _sorted_parts(reparsed_payload["Parts"]) == _sorted_parts(normalized_ship["Parts"])


def test_pipeline_roundtrip_replays_doors_from_graph(tmp_path: Path) -> None:
    """The full preprocessing pipeline should preserve and replay doors exactly."""

    normalized_ship = _build_two_corridors_with_shared_door(name="door-roundtrip")
    source_path = tmp_path / "door-roundtrip.ship.png"
    extracted_dir = tmp_path / "extracted"
    canonical_dir = tmp_path / "canonical"
    graph_dir = tmp_path / "graphs"
    exported_path = tmp_path / "door-replayed.ship.png"

    source_path.write_bytes(create_ship_png_bytes(normalized_ship))

    # Persist every stage so the test can verify that explicit door records
    # survive the whole preprocessing and replay pipeline without inference.
    pipeline_payload = run_pipeline(
        input_paths=[source_path],
        output_dir=graph_dir,
        extracted_dir=extracted_dir,
        canonical_dir=canonical_dir,
        opt_in_csv=tmp_path / "no-opt-in.csv",
        extract_workers=1,
        extract_executor="thread",
        canonicalize_workers=1,
        canonicalize_executor="thread",
        graph_workers=1,
        graph_executor="thread",
    )

    extracted_path = extracted_dir / output_name_for_ship_png(source_path)
    canonical_path = canonical_dir / output_name_for_ship_png(source_path)
    graph_path = graph_dir / output_name_for_ship_png(source_path)

    extracted_payload = json.loads(extracted_path.read_text(encoding="utf-8"))
    canonical_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_nodes = graph_payload["graphs"]["A_structural_part_graph"]["nodes"]
    generated_payload = graph_to_generated_parts_payload(graph_payload, name="door-graph-replay")

    assert pipeline_payload["graphs"]["ships_processed"] == 1
    assert _sorted_doors_from_relative_payload(extracted_payload) == _sorted_doors(normalized_ship["Doors"])
    assert _sorted_doors_from_relative_payload(canonical_payload) == _sorted_doors(normalized_ship["Doors"])
    assert _sorted_doors(generated_payload["doors"]) == _sorted_doors(normalized_ship["Doors"])
    assert graph_payload["validation"]["normalized_door_count"] == len(normalized_ship["Doors"])

    struct_summary = graph_payload["graphs"]["A_structural_part_graph"]["summary"]
    assert struct_summary["door_records"] == 1
    assert struct_summary["door_edges"] == 1
    assert struct_summary["dangling_door_records"] == 0
    assert struct_summary["internal_door_records"] == 0
    assert struct_summary["non_structural_door_records"] == 0
    assert all(node["walkable_cells_2x"] for node in graph_nodes)

    export_result = export_ship_png(generated_payload, exported_path, validate=True)
    reparsed_payload = parse_ship_png(exported_path)

    assert generated_payload["stats"]["doors_preserved"] == len(normalized_ship["Doors"])
    assert export_result["doors_exported"] == len(normalized_ship["Doors"])
    assert export_result["valid"] is True
    assert export_result["roundtrip"]["doors_match"] is True
    assert _sorted_doors(reparsed_payload["Doors"]) == _sorted_doors(normalized_ship["Doors"])
    assert _sorted_parts(reparsed_payload["Parts"]) == _sorted_parts(normalized_ship["Parts"])


def test_graph_replay_payload_preserves_unknown_part_nodes() -> None:
    """Graph replay should preserve non-vanilla nodes instead of dropping them."""

    graph_payload = {
        "ship": {"name": "mixed-graph"},
        "graphs": {
            "A_structural_part_graph": {
                "nodes": [
                    {
                        "id": 0,
                        "part_id": "cosmoteer.corridor",
                        "location_2x": [0, 0],
                        "rotation": 0,
                    },
                    {
                        "id": 1,
                        "part_id": "mod.custom_corridor",
                        "location_2x": [4, 0],
                        "rotation": 3,
                    },
                ],
            }
        },
        "coord_transform": {"version": 1, "frame": "bbox_center_2x", "scale": 2, "center_2x": [0, 0]},
    }

    generated_payload = graph_to_generated_parts_payload(graph_payload)

    assert generated_payload["name"] == "mixed-graph"
    assert generated_payload["stats"]["parts_generated"] == 2
    assert generated_payload["stats"]["doors_preserved"] == 0
    assert generated_payload["doors"] == []
    assert generated_payload["parts"] == [
        {
            "part_id": "cosmoteer.corridor",
            "rotation": 0,
            "x": 0,
            "y": 0,
        },
        {
            "part_id": "mod.custom_corridor",
            "rotation": 3,
            "x": 2,
            "y": 0,
        },
    ]


def test_graph_replay_carries_overclocked_flag_and_exports_toggle_states() -> None:
    """Graph replay should carry overclocked=True and export PartUIToggleStates."""

    # corridor has no save_rect offset, so stored location == footprint origin
    graph_payload = {
        "ship": {"name": "overclock-replay"},
        "graphs": {
            "A_structural_part_graph": {
                "nodes": [
                    {
                        "id": 0,
                        "part_id": "cosmoteer.corridor",
                        "location_2x": [0, 0],
                        "rotation": 0,
                        "overclocked": True,
                    },
                    {
                        "id": 1,
                        "part_id": "cosmoteer.corridor",
                        "location_2x": [4, 0],
                        "rotation": 0,
                    },
                ],
            }
        },
        "coord_transform": {"version": 1, "frame": "bbox_center_2x", "scale": 2, "center_2x": [0, 0]},
    }

    generated_payload = graph_to_generated_parts_payload(graph_payload)

    # graph_to_generated_parts_payload should propagate overclocked
    assert generated_payload["parts"][0].get("overclocked") is True
    assert "overclocked" not in generated_payload["parts"][1]

    # generated_parts_to_toggle_states should produce exactly one entry
    toggle_states = generated_parts_to_toggle_states(generated_payload["parts"])
    assert len(toggle_states) == 1
    assert toggle_states[0]["Key"][1] == "thermal_overclock"
    assert toggle_states[0]["Value"] == 1
    key_ref = toggle_states[0]["Key"][0]
    assert key_ref["ID"] == "cosmoteer.corridor"
    assert key_ref["Rotation"] == 0
    # corridor has no save_rect offset → stored location == footprint origin == (0, 0)
    assert key_ref["Location"] == [0, 0]

    # export_ship_png should embed PartUIToggleStates in the PNG
    cosmoteer_parts = generated_parts_to_cosmoteer_parts(generated_payload["parts"])
    ship_dict = make_minimal_ship_dict(cosmoteer_parts, toggle_states=toggle_states)
    png_bytes = create_ship_png_bytes(ship_dict)
    extracted = _decode_ship_data(_extract_embedded_payload(png_bytes))
    assert "PartUIToggleStates" in extracted
    assert len(extracted["PartUIToggleStates"]) == 1
    extracted_toggle = extracted["PartUIToggleStates"][0]
    assert extracted_toggle["Key"][1] == "thermal_overclock"
    assert extracted_toggle["Value"] == 1
    assert extracted_toggle["Key"][0]["ID"] == "cosmoteer.corridor"
