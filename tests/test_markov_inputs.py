"""Regression tests for Markov seed input normalization helpers."""

from __future__ import annotations

import json
from pathlib import Path

from markov.inputs import load_seed_parts_from_json, load_seed_parts_from_png
from markov.model import iter_vanilla_parts_from_ship
from markov.types import ShipPart


def test_load_seed_parts_from_png_applies_relative_transform_for_world_locations(
    monkeypatch,
) -> None:
    """Seed PNG loading should run preprocessing transform for world `Location` payloads."""

    parsed_ship_payload = {
        "Parts": [
            {"ID": "cosmoteer.corridor", "Location": [4, 6], "Rotation": 0},
            {"ID": "cosmoteer.corridor", "Location": [5, 6], "Rotation": 0},
        ],
        "Doors": [{"Cell": [5, 6], "Orientation": 0}],
    }
    monkeypatch.setattr("markov.inputs.parse_ship_png", lambda _path: parsed_ship_payload)
    seen_ship_payload: dict = {}

    def _capture_loader(ship_data: dict) -> list[ShipPart]:
        """Capture loader input so test can assert preprocessing fields."""

        seen_ship_payload.update(ship_data)
        return [
            ShipPart(
                part_id="cosmoteer.corridor",
                rotation=0,
                x=4,
                y=6,
            )
        ]

    loaded_seed_parts = load_seed_parts_from_png(Path("seed.ship.png"), _capture_loader)

    assert seen_ship_payload["coord_transform"]["frame"] == "bbox_center_2x"
    assert "Location2x" in seen_ship_payload["Parts"][0]
    assert "Location" not in seen_ship_payload["Parts"][0]
    assert loaded_seed_parts == [
        {
            "part_id": "cosmoteer.corridor",
            "rotation": 0,
            "x": 4,
            "y": 6,
            "flip_x": False,
            "flip_y": False,
        }
    ]


def test_load_seed_parts_from_png_keeps_existing_relative_payload(monkeypatch) -> None:
    """Seed PNG loading should preserve already-preprocessed centered `2x` payloads."""

    parsed_ship_payload = {
        "coord_transform": {"version": 1, "frame": "bbox_center_2x", "scale": 2, "center_2x": [8, 10]},
        "Parts": [{"ID": "cosmoteer.corridor", "Location2x": [0, 0], "Rotation": 0}],
        "Doors": [{"Cell2x": [0, 2], "Orientation": 0}],
    }
    monkeypatch.setattr("markov.inputs.parse_ship_png", lambda _path: parsed_ship_payload)
    seen_ship_payload: dict = {}

    def _capture_loader(ship_data: dict) -> list[ShipPart]:
        """Capture loader input so test can verify no coordinate-frame rewrite."""

        seen_ship_payload.update(ship_data)
        return []

    _ = load_seed_parts_from_png(Path("seed.ship.png"), _capture_loader)

    assert seen_ship_payload["coord_transform"]["center_2x"] == [8, 10]
    assert seen_ship_payload["Parts"][0]["Location2x"] == [0, 0]
    assert "Location" not in seen_ship_payload["Parts"][0]


def test_load_seed_parts_from_real_seed_fixture() -> None:
    """Seed PNG loader should parse the checked-in real seed fixture."""

    fixture_png_path = Path(__file__).resolve().parent / "data" / "seed_ship.ship.png"
    loaded_seed_parts = load_seed_parts_from_png(fixture_png_path, iter_vanilla_parts_from_ship)

    assert loaded_seed_parts
    required_keys = {"part_id", "rotation", "x", "y", "flip_x", "flip_y"}
    for seed_part in loaded_seed_parts:
        assert required_keys <= seed_part.keys(), (
            f"Seed part missing keys {required_keys - seed_part.keys()}: {seed_part}"
        )
        assert isinstance(seed_part["x"], int), f"x should be int, got {type(seed_part['x'])}"
        assert isinstance(seed_part["y"], int), f"y should be int, got {type(seed_part['y'])}"
        assert isinstance(seed_part["rotation"], int), f"rotation should be int"
        assert isinstance(seed_part["flip_x"], bool)
        assert isinstance(seed_part["flip_y"], bool)


def test_load_seed_parts_from_json_applies_relative_transform_for_world_locations(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Seed JSON loading should normalize world `Location` payloads through preprocessing."""

    seed_json_path = tmp_path / "seed.json"
    seed_json_path.write_text(
        json.dumps({"Parts": [{"ID": "cosmoteer.corridor", "Location": [4, 6], "Rotation": 1}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "markov.inputs.apply_relative_coords_transform",
        lambda _ship_data: {
            "coord_transform": {"version": 1, "frame": "bbox_center_2x", "scale": 2, "center_2x": [8, 12]},
            "Parts": [{"ID": "cosmoteer.corridor", "Location2x": [10, 2], "Rotation": 1}],
        },
    )

    loaded_seed_parts = load_seed_parts_from_json(seed_json_path)

    assert loaded_seed_parts == [
        {
            "part_id": "cosmoteer.corridor",
            "rotation": 1,
            "x": 9,
            "y": 7,
            "flip_x": False,
            "flip_y": False,
        }
    ]


def test_load_seed_parts_from_json_supports_preprocessed_relative_payload(tmp_path: Path) -> None:
    """Seed JSON loading should read extracted payloads that only include `Location2x`."""

    seed_json_path = tmp_path / "seed-relative.json"
    seed_json_path.write_text(
        json.dumps(
            {
                "coord_transform": {
                    "version": 1,
                    "frame": "bbox_center_2x",
                    "scale": 2,
                    "center_2x": [8, 12],
                },
                "Parts": [{"ID": "cosmoteer.corridor", "Location2x": [10, 2], "Rotation": 1}],
            }
        ),
        encoding="utf-8",
    )

    loaded_seed_parts = load_seed_parts_from_json(seed_json_path)

    assert loaded_seed_parts == [
        {
            "part_id": "cosmoteer.corridor",
            "rotation": 1,
            "x": 9,
            "y": 7,
            "flip_x": False,
            "flip_y": False,
        }
    ]
