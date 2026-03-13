from __future__ import annotations

import json
from pathlib import Path

from generator.backends.markov.export import graph_to_generated_parts_payload
from preprocessing.canonicalize import canonicalize_json_text, run_canonicalize
from preprocessing.graphs import process_ship
from preprocessing.relative_coords import apply_relative_coords_transform


def _base_ship_payload(parts: list[dict], *, doors: list[dict] | None = None, name: str = "test-ship") -> dict:
    """Build a minimal normalized ship payload for relative-coords tests."""

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


def _write_json(path: Path, payload: object) -> None:
    """Write one JSON fixture with deterministic formatting."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_apply_relative_coords_transform_uses_centered_integer_2x_frame() -> None:
    """Relative transform should center coordinates in an integer-only 2x frame."""

    transformed = apply_relative_coords_transform(
        _base_ship_payload(
            parts=[
                {"ID": "cosmoteer.corridor", "Location": [0, 0], "Rotation": 0},
                {"ID": "cosmoteer.corridor", "Location": [1, 1], "Rotation": 0},
            ],
            doors=[{"Cell": [1, 1], "Orientation": 0}],
            name="even-center",
        )
    )

    assert transformed["coord_transform"]["frame"] == "bbox_center_2x"
    assert transformed["coord_transform"]["scale"] == 2
    assert transformed["coord_transform"]["center_2x"] == [1, 1]
    assert "Location" not in transformed["Parts"][0]
    assert transformed["Parts"][0]["Location2x"] == [-1, -1]
    assert transformed["Parts"][1]["Location2x"] == [1, 1]
    assert "Cell" not in transformed["Doors"][0]
    assert transformed["Doors"][0]["Cell2x"] == [1, 1]


def test_canonicalize_hash_is_translation_invariant_when_location2x_exists() -> None:
    """Canonical hash should collapse payloads that only differ by world offset."""

    ship_a = apply_relative_coords_transform(
        _base_ship_payload(
            parts=[
                {"ID": "cosmoteer.corridor", "Location": [0, 0], "Rotation": 0},
                {"ID": "cosmoteer.corridor", "Location": [2, 0], "Rotation": 0},
            ],
            name="same-layout",
        )
    )
    ship_b = apply_relative_coords_transform(
        _base_ship_payload(
            parts=[
                {"ID": "cosmoteer.corridor", "Location": [10, 7], "Rotation": 0},
                {"ID": "cosmoteer.corridor", "Location": [12, 7], "Rotation": 0},
            ],
            name="same-layout",
        )
    )

    ship_a_text = json.dumps(ship_a, sort_keys=True)
    ship_b_text = json.dumps(ship_b, sort_keys=True)

    _, legacy_hash_a = canonicalize_json_text(ship_a_text, translation_invariant=False)
    _, legacy_hash_b = canonicalize_json_text(ship_b_text, translation_invariant=False)
    _, relative_hash_a = canonicalize_json_text(ship_a_text, translation_invariant=True)
    _, relative_hash_b = canonicalize_json_text(ship_b_text, translation_invariant=True)

    assert legacy_hash_a != legacy_hash_b
    assert relative_hash_a == relative_hash_b


def test_run_canonicalize_dedupes_shifted_layouts_and_keeps_transform_metadata(tmp_path: Path) -> None:
    """Canonicalization should dedupe shifted layouts and preserve transform fields."""

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    ship_a = apply_relative_coords_transform(
        _base_ship_payload(
            parts=[{"ID": "cosmoteer.corridor", "Location": [0, 0], "Rotation": 0}],
            doors=[{"Cell": [0, 0], "Orientation": 0}],
            name="dedupe-layout",
        )
    )
    ship_b = apply_relative_coords_transform(
        _base_ship_payload(
            parts=[{"ID": "cosmoteer.corridor", "Location": [11, -9], "Rotation": 0}],
            doors=[{"Cell": [11, -9], "Orientation": 0}],
            name="dedupe-layout",
        )
    )
    _write_json(input_dir / "ship-a.json", ship_a)
    _write_json(input_dir / "ship-b.json", ship_b)

    manifest = run_canonicalize(
        input_dir=input_dir,
        output_dir=output_dir,
        report_json=tmp_path / "report.json",
        workers=1,
        executor="thread",
    )

    assert manifest["parsed_input_json_files"] == 2
    assert manifest["unique_content_groups"] == 1

    written_json_files = sorted(path for path in output_dir.glob("*.json") if path.name != "manifest.json")
    assert len(written_json_files) == 1
    canonical_payload = json.loads(written_json_files[0].read_text(encoding="utf-8"))
    assert "coord_transform" in canonical_payload
    assert "center_2x" in canonical_payload["coord_transform"]
    assert "Location" not in canonical_payload["Parts"][0]
    assert "Location2x" in canonical_payload["Parts"][0]
    assert "Cell" not in canonical_payload["Doors"][0]
    assert "Cell2x" in canonical_payload["Doors"][0]


def test_process_ship_uses_centered_2x_payloads(tmp_path: Path) -> None:
    """Graph processing should accept centered-`2x` payloads directly."""

    source_path = tmp_path / "ship.json"
    payload = _base_ship_payload(
        parts=[{"ID": "cosmoteer.corridor", "Location2x": [0, 0], "Rotation": 0}],
        doors=[{"Cell2x": [0, 2], "Orientation": 0}],
        name="location2x-only",
    )
    payload["coord_transform"] = {
        "version": 1,
        "frame": "bbox_center_2x",
        "scale": 2,
        "center_2x": [4, 6],
    }
    _write_json(source_path, payload)

    graph_payload = process_ship(source_path)
    node = graph_payload["graphs"]["A_structural_part_graph"]["nodes"][0]
    door = graph_payload["doors"][0]

    assert graph_payload["schema_version"] == 5
    assert "location" not in node
    assert node["location_2x"] == [0, 0]
    assert node["walkable_cells_2x"] == [[0, 0]]
    assert "Cell" not in door
    assert door["Cell2x"] == [0, 2]


def test_graph_replay_converts_location_2x_nodes_for_export() -> None:
    """Graph replay payload conversion should denormalize `2x` coordinates for export."""

    graph_payload = {
        "coord_transform": {
            "version": 1,
            "frame": "bbox_center_2x",
            "scale": 2,
            "center_2x": [8, -2],
        },
        "doors": [{"Cell2x": [0, 0], "Orientation": 1}],
        "graphs": {
            "A_structural_part_graph": {
                "nodes": [
                    {
                        "id": 0,
                        "part_id": "cosmoteer.corridor",
                        "location_2x": [0, 0],
                        "rotation": 0,
                    }
                ]
            }
        },
    }

    generated_payload = graph_to_generated_parts_payload(graph_payload, name="location2x-replay")

    assert generated_payload["parts"] == [
        {
            "part_id": "cosmoteer.corridor",
            "rotation": 0,
            "x": 4,
            "y": -1,
        }
    ]
    assert generated_payload["doors"] == [{"Cell": [4, -1], "Orientation": 1}]
