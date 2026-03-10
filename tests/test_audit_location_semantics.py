import importlib.util
import unittest
from dataclasses import dataclass
from pathlib import Path

from common.save_rect import SaveRect


@dataclass
class _RotationGeometry:
    width: int
    height: int
    footprint_tiles: tuple[tuple[int, int], ...]


@dataclass
class _PartGeometry:
    rotations: dict[int, _RotationGeometry]


def _load_audit_module():
    """Load the audit script as a module for direct function testing."""

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_location_semantics.py"
    spec = importlib.util.spec_from_file_location("audit_location_semantics", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load audit_location_semantics module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AuditLocationSemanticsTests(unittest.TestCase):
    def test_corrected_occupancy_respects_part_coordinate_frame(self) -> None:
        """Use frame metadata to avoid double-shifting normalized generator parts."""

        audit_module = _load_audit_module()

        geometry_cache = {
            "cosmoteer.shield_gen_small": _PartGeometry(
                rotations={
                    0: _RotationGeometry(
                        width=2,
                        height=3,
                        footprint_tiles=((0, 0), (1, 0), (0, 1), (1, 1)),
                    )
                }
            )
        }
        save_rects = {
            "cosmoteer.shield_gen_small": SaveRect(
                part_id="cosmoteer.shield_gen_small",
                x=0,
                y=1,
                width=2,
                height=2,
                source_file="shield_gen_small.rules",
            )
        }

        extracted_payload_parts = [
            {
                "part_id": "cosmoteer.shield_gen_small",
                "rotation": 0,
                "x": -4,
                "y": 1,
                "coordinate_frame": audit_module.STORED_LOCATION_FRAME,
            }
        ]
        generated_payload_parts = [
            {
                "part_id": "cosmoteer.shield_gen_small",
                "rotation": 0,
                "x": -4,
                "y": 0,
                "coordinate_frame": audit_module.NORMALIZED_ORIGIN_FRAME,
            }
        ]

        extracted_counter, _ = audit_module._occupied_cells(
            extracted_payload_parts, geometry_cache, save_rects, corrected=True
        )
        generated_counter, _ = audit_module._occupied_cells(
            generated_payload_parts, geometry_cache, save_rects, corrected=True
        )

        self.assertEqual(extracted_counter, generated_counter)


if __name__ == "__main__":
    unittest.main()
