"""Regression tests for Markov seed input normalization helpers."""

from __future__ import annotations

from pathlib import Path

from markov.inputs import load_seed_parts_from_png
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
    assert all("part_id" in seed_part for seed_part in loaded_seed_parts)
    assert all("rotation" in seed_part for seed_part in loaded_seed_parts)
