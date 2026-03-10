import unittest

from common.geometry import load_vanilla_part_geometry
from markov.model import ShipPart
from markov.symmetry import mirror_rotation
from generator.backends.markov.export import generated_parts_to_cosmoteer_parts, roundtrip_validate


class MirrorRotationTests(unittest.TestCase):
    def test_default_horizontal_mirror_rule_is_preserved(self) -> None:
        self.assertEqual(mirror_rotation(0), 0)
        self.assertEqual(mirror_rotation(1), 3)
        self.assertEqual(mirror_rotation(2), 2)
        self.assertEqual(mirror_rotation(3), 1)

    def test_directional_wedges_use_corpus_backed_override(self) -> None:
        self.assertEqual(mirror_rotation(0, "cosmoteer.armor_wedge"), 1)
        self.assertEqual(mirror_rotation(1, "cosmoteer.armor_wedge"), 0)
        self.assertEqual(mirror_rotation(2, "cosmoteer.armor_wedge"), 3)
        self.assertEqual(mirror_rotation(3, "cosmoteer.armor_wedge"), 2)
        self.assertEqual(mirror_rotation(0, "cosmoteer.structure_wedge"), 1)
        self.assertEqual(mirror_rotation(2, "cosmoteer.structure_wedge"), 3)
        self.assertEqual(mirror_rotation(0, "cosmoteer.armor_structure_hybrid_1x1"), 1)

    def test_mirror_part_toggles_flip_x_for_mirrored_parts(self) -> None:
        from markov.symmetry import mirror_part

        geometry_cache = load_vanilla_part_geometry()
        original = ShipPart(
            part_id="cosmoteer.armor_1x3_wedge",
            rotation=3,
            x=-9,
            y=-13,
            flip_x=False,
        )
        mirrored = mirror_part(original, geometry_cache)
        self.assertIsNotNone(mirrored)
        assert mirrored is not None
        self.assertEqual(mirrored.rotation, 1)
        self.assertTrue(mirrored.flip_x)

        baseline = ShipPart(
            part_id="cosmoteer.reactor_large",
            rotation=0,
            x=-4,
            y=0,
            flip_x=False,
        )
        mirrored_baseline = mirror_part(baseline, geometry_cache)
        self.assertIsNotNone(mirrored_baseline)
        assert mirrored_baseline is not None
        self.assertTrue(mirrored_baseline.flip_x)

    def test_export_preserves_flip_flags(self) -> None:
        exported = generated_parts_to_cosmoteer_parts(
            [
                {
                    "part_id": "cosmoteer.armor_1x2_wedge",
                    "rotation": 3,
                    "x": 4,
                    "y": -2,
                    "flip_x": True,
                    "flip_y": False,
                }
            ]
        )
        self.assertEqual(
            exported,
            [
                {
                    "ID": "cosmoteer.armor_1x2_wedge",
                    "Location": [4, -2],
                    "Rotation": 3,
                    "FlipX": True,
                    "FlipY": False,
                }
            ],
        )

    def test_roundtrip_validation_checks_flip_flags(self) -> None:
        report = roundtrip_validate(
            {
                "name": "flip-check",
                "parts": [
                    {
                        "part_id": "cosmoteer.armor_1x2_wedge",
                        "rotation": 3,
                        "x": 4,
                        "y": -2,
                        "flip_x": True,
                        "flip_y": False,
                    }
                ],
            }
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["mismatches"], [])


if __name__ == "__main__":
    unittest.main()
