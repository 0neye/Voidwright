import unittest

from common.cosmoteer import create_ship_png_bytes, parse_ship_png
from common.cosmoteer.encoder import _denormalize_ship_part_locations
from common.save_rect import origin_to_stored_location, stored_location_to_origin


class PartLocationSemanticsTests(unittest.TestCase):
    def test_shield_generator_save_rect_location_roundtrips(self) -> None:
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
        self.assertEqual(stored_ship["Parts"][0]["Location"], [-4, 1])

        png_bytes = create_ship_png_bytes(normalized_ship)
        parsed = parse_ship_png_bytes(png_bytes)
        self.assertEqual(parsed["Parts"][0]["Location"], [-4, 0])

    def test_save_rect_helpers_are_inverse_for_shield_generator(self) -> None:
        stored = origin_to_stored_location("cosmoteer.shield_gen_small", 0, (-4, 0))
        origin = stored_location_to_origin("cosmoteer.shield_gen_small", 0, stored)
        self.assertEqual(stored, (-4, 1))
        self.assertEqual(origin, (-4, 0))


def parse_ship_png_bytes(png_bytes: bytes) -> dict:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "ship.ship.png"
        path.write_bytes(png_bytes)
        parsed = parse_ship_png(path)
    assert isinstance(parsed, dict)
    return parsed


if __name__ == "__main__":
    unittest.main()
