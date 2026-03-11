from __future__ import annotations

from pathlib import Path
import tempfile

from common.cosmoteer import create_ship_png_bytes, parse_ship_png
from common.cosmoteer.encoder import _denormalize_ship_part_locations
from common.geometry import load_vanilla_part_geometry
from common.save_rect import known_save_rects, origin_to_stored_location, stored_location_to_origin


EXPECTED_EFFECTIVE_RECT_PART_IDS = {
    "cosmoteer.cannon_large",
    "cosmoteer.cannon_med",
    "cosmoteer.chaingun",
    "cosmoteer.disruptor",
    "cosmoteer.flak_cannon_large",
    "cosmoteer.ion_beam_emitter",
    "cosmoteer.laser_blaster_large",
    "cosmoteer.laser_blaster_small",
    "cosmoteer.missile_launcher",
    "cosmoteer.point_defense",
    "cosmoteer.radiator",
    "cosmoteer.railgun_launcher",
    "cosmoteer.resonance_beam_turret",
    "cosmoteer.shield_gen_small",
    "cosmoteer.thruster_boost",
    "cosmoteer.thruster_huge",
    "cosmoteer.thruster_large",
    "cosmoteer.thruster_med",
    "cosmoteer.thruster_rocket_nozzle",
    "cosmoteer.thruster_small",
    "cosmoteer.thruster_small_2way",
    "cosmoteer.thruster_small_3way",
}


def _build_normalized_ship(parts: list[dict], *, name: str) -> dict:
    """Build a minimal normalized ship payload for location roundtrip tests."""

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
        "Doors": [],
    }


def _part_payload(part_id: str, x: int, y: int, rotation: int) -> dict:
    """Build one normalized part payload entry for parser and encoder tests."""

    return {
        "ID": part_id,
        "Location": [x, y],
        "Rotation": rotation,
        "FlipX": False,
        "FlipY": False,
    }


def test_effective_rect_helpers_are_inverse_for_all_repo_backed_parts() -> None:
    """Stored/origin helpers should roundtrip for every repo-backed effective rect."""

    geometry_cache = load_vanilla_part_geometry()
    save_rects = known_save_rects()

    assert set(save_rects) == EXPECTED_EFFECTIVE_RECT_PART_IDS

    # Sweep every affected vanilla part across all four rotations so the new
    # geometry-backed rect table cannot silently miss a case.
    for index, part_id in enumerate(sorted(EXPECTED_EFFECTIVE_RECT_PART_IDS)):
        for rotation in range(4):
            origin_location = (index * 11 - 50, rotation * 7 - 10)
            stored_location = origin_to_stored_location(part_id, rotation, origin_location)
            restored_origin = stored_location_to_origin(part_id, rotation, stored_location)

            assert restored_origin == origin_location, (
                f"Unexpected restored origin for {part_id} rot={rotation}"
            )

            # Compare against the direct rotated rect offset so this test checks
            # the geometry-driven shift, not only inverse helper behavior.
            base_geometry = geometry_cache[part_id].rotations[0]
            expected_offset = save_rects[part_id].offset_for_rotation(
                rotation,
                base_geometry.width,
                base_geometry.height,
            )
            assert stored_location == (
                origin_location[0] + expected_offset[0],
                origin_location[1] + expected_offset[1],
            )


def test_all_effective_rect_parts_roundtrip_through_png_parser() -> None:
    """PNG encode/decode should preserve normalized locations for all affected parts."""

    save_rects = known_save_rects()
    parts: list[dict] = []
    expected_stored_locations: list[list[int]] = []

    # Space parts far apart so the payload stays easy to inspect if a future
    # regression changes one stored-location transform.
    for index, part_id in enumerate(sorted(EXPECTED_EFFECTIVE_RECT_PART_IDS)):
        rotation = index % 4
        origin_x = index * 8 - 80
        origin_y = index * 5 - 40
        parts.append(_part_payload(part_id, origin_x, origin_y, rotation))
        stored_x, stored_y = origin_to_stored_location(part_id, rotation, (origin_x, origin_y))
        expected_stored_locations.append([stored_x, stored_y])

    normalized_ship = _build_normalized_ship(parts, name="all-effective-rect-roundtrip")
    stored_ship = _denormalize_ship_part_locations(normalized_ship)
    assert [part["Location"] for part in stored_ship["Parts"]] == expected_stored_locations

    parsed = parse_ship_png_bytes(create_ship_png_bytes(normalized_ship))
    assert [part["Location"] for part in parsed["Parts"]] == [
        part["Location"] for part in normalized_ship["Parts"]
    ]


def test_parts_without_save_rect_roundtrip_without_location_shift() -> None:
    """Physical-rect-only parts should keep full-footprint ship locations."""

    normalized_ship = _build_normalized_ship(
        [_part_payload("cosmoteer.shield_gen_large", -4, 0, 0)],
        name="no-save-rect-roundtrip",
    )

    stored_ship = _denormalize_ship_part_locations(normalized_ship)
    assert stored_ship["Parts"][0]["Location"] == [-4, 0]

    parsed = parse_ship_png_bytes(create_ship_png_bytes(normalized_ship))
    assert parsed["Parts"][0]["Location"] == [-4, 0]


def test_legacy_electro_bolter_alias_uses_disruptor_save_rect() -> None:
    """Legacy electro-bolter IDs should use the same stored-location offset as disruptor."""

    origin_location = (10, 20)

    assert origin_to_stored_location("cosmoteer.disruptor", 0, origin_location) == (10, 21)
    assert origin_to_stored_location("cosmoteer.electro_bolter", 0, origin_location) == (10, 21)
    assert stored_location_to_origin("cosmoteer.electro_bolter", 0, (10, 21)) == origin_location


def parse_ship_png_bytes(png_bytes: bytes) -> dict:
    """Parse in-memory ship PNG bytes by writing a temporary ship file."""

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "ship.ship.png"
        path.write_bytes(png_bytes)
        parsed = parse_ship_png(path)
    assert isinstance(parsed, dict)
    return parsed
