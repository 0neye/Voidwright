from common.cosmoteer import create_ship_png_bytes, parse_ship_png
from common.cosmoteer.encoder import _denormalize_ship_part_locations
from common.save_rect import origin_to_stored_location, stored_location_to_origin


def test_shield_generator_save_rect_location_roundtrips() -> None:
    """Shield generator locations should roundtrip via denormalize and parser."""

    normalized_ship = {
        "Version": 1,
        "Name": "save-rect-check",
        "FlightDirection": 1,
        "FormationOrder": 0,
        "ShipRulesID": "cosmoteer.terran",
        "RoofBaseTexture": "scratched1",
        "CrewUniformColor": ["0000803F", "00000000", "00000000", "0000803F"],
        "RoofBaseColor": ["907F083F", "907F083F", "907F083F", "0000403F"],
        "RoofDecalColor1": ["9A99193E", "9A99193E", "9A99193E", "0000803F"],
        "RoofDecalColor2": ["0000803F", "0000803F", "0000803F", "0000803F"],
        "Parts": [
            {
                "ID": "cosmoteer.shield_gen_small",
                "Location": [-4, 0],
                "Rotation": 0,
                "FlipX": False,
                "FlipY": False,
            }
        ],
        "Doors": [],
    }

    stored_ship = _denormalize_ship_part_locations(normalized_ship)
    assert stored_ship["Parts"][0]["Location"] == [-4, 1]

    png_bytes = create_ship_png_bytes(normalized_ship)
    parsed = parse_ship_png_bytes(png_bytes)
    assert parsed["Parts"][0]["Location"] == [-4, 0]


def test_save_rect_helpers_are_inverse_for_shield_generator() -> None:
    """Stored/origin helper conversions should be inverse for shield generator."""

    stored = origin_to_stored_location("cosmoteer.shield_gen_small", 0, (-4, 0))
    origin = stored_location_to_origin("cosmoteer.shield_gen_small", 0, stored)
    assert stored == (-4, 1)
    assert origin == (-4, 0)


def test_save_rect_helpers_are_inverse_for_multiple_parts_and_rotations() -> None:
    """Stored/origin helpers should remain inverses across parts and rotations."""

    # These cases intentionally mix SaveRect and non-SaveRect parts so both
    # offset and no-offset code paths are covered by one assertion loop.
    part_cases = [
        ("cosmoteer.shield_gen_small", 0, (-4, 0), (-4, 1)),
        ("cosmoteer.shield_gen_small", 1, (-4, 0), (-4, 0)),
        ("cosmoteer.shield_gen_small", 2, (-4, 0), (-4, 0)),
        ("cosmoteer.shield_gen_small", 3, (-4, 0), (-3, 0)),
        ("cosmoteer.corridor", 0, (7, -3), (7, -3)),
        ("cosmoteer.corridor", 1, (7, -3), (7, -3)),
        ("cosmoteer.reactor_large", 2, (2, 5), (2, 5)),
    ]

    for part_id, rotation, origin_location, expected_stored in part_cases:
        stored_location = origin_to_stored_location(part_id, rotation, origin_location)
        restored_origin = stored_location_to_origin(part_id, rotation, stored_location)
        assert stored_location == expected_stored, (
            f"Unexpected stored location for {part_id} rot={rotation}"
        )
        assert restored_origin == origin_location, (
            f"Unexpected restored origin for {part_id} rot={rotation}"
        )


def test_mixed_part_locations_roundtrip_through_png_parser() -> None:
    """PNG encode/decode should preserve normalized locations for mixed parts."""

    normalized_ship = {
        "Version": 1,
        "Name": "mixed-location-roundtrip",
        "FlightDirection": 1,
        "FormationOrder": 0,
        "ShipRulesID": "cosmoteer.terran",
        "RoofBaseTexture": "scratched1",
        "CrewUniformColor": ["0000803F", "00000000", "00000000", "0000803F"],
        "RoofBaseColor": ["907F083F", "907F083F", "907F083F", "0000403F"],
        "RoofDecalColor1": ["9A99193E", "9A99193E", "9A99193E", "0000803F"],
        "RoofDecalColor2": ["0000803F", "0000803F", "0000803F", "0000803F"],
        "Parts": [
            {
                "ID": "cosmoteer.shield_gen_small",
                "Location": [-4, 0],
                "Rotation": 0,
                "FlipX": False,
                "FlipY": False,
            },
            {
                "ID": "cosmoteer.corridor",
                "Location": [2, -1],
                "Rotation": 3,
                "FlipX": False,
                "FlipY": False,
            },
            {
                "ID": "cosmoteer.reactor_large",
                "Location": [6, 2],
                "Rotation": 1,
                "FlipX": False,
                "FlipY": False,
            },
        ],
        "Doors": [],
    }

    # First verify denormalization only shifts parts that need SaveRect offsets.
    stored_ship = _denormalize_ship_part_locations(normalized_ship)
    assert stored_ship["Parts"][0]["Location"] == [-4, 1]
    assert stored_ship["Parts"][1]["Location"] == [2, -1]
    assert stored_ship["Parts"][2]["Location"] == [6, 2]

    # Then verify parser normalization restores the original per-part locations.
    png_bytes = create_ship_png_bytes(normalized_ship)
    parsed = parse_ship_png_bytes(png_bytes)
    assert [part["Location"] for part in parsed["Parts"]] == [
        part["Location"] for part in normalized_ship["Parts"]
    ]


def parse_ship_png_bytes(png_bytes: bytes) -> dict:
    """Parse in-memory ship PNG bytes by writing a temporary ship file."""

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "ship.ship.png"
        path.write_bytes(png_bytes)
        parsed = parse_ship_png(path)
    assert isinstance(parsed, dict)
    return parsed
