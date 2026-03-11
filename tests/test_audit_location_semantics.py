import importlib.util
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


def test_corrected_occupancy_respects_part_coordinate_frame() -> None:
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

    assert extracted_counter == generated_counter


def test_corrected_occupancy_matches_for_mixed_payload_frames() -> None:
    """Mixed payloads should honor coordinate_frame per part consistently.

    Covers all four shield_gen_small rotations to exercise every distinct
    SaveRect offset:
      rot=0 → offset (0, 1)  — y-shift
      rot=1 → offset (0, 0)  — no shift
      rot=2 → offset (0, 0)  — no shift
      rot=3 → offset (1, 0)  — x-shift
    """

    audit_module = _load_audit_module()

    # Geometry matches the live vanilla-parts data for shield_gen_small.
    # Rotation 0 dimensions (2 wide × 3 tall) are also used by _occupied_cells
    # when computing the rotated SaveRect offset for all other rotations.
    geometry_cache = {
        "cosmoteer.shield_gen_small": _PartGeometry(
            rotations={
                0: _RotationGeometry(
                    width=2,
                    height=3,
                    footprint_tiles=((0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)),
                ),
                1: _RotationGeometry(
                    width=3,
                    height=2,
                    footprint_tiles=((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)),
                ),
                2: _RotationGeometry(
                    width=2,
                    height=3,
                    footprint_tiles=((0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)),
                ),
                3: _RotationGeometry(
                    width=3,
                    height=2,
                    footprint_tiles=((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)),
                ),
            }
        ),
        "cosmoteer.corridor": _PartGeometry(
            rotations={
                0: _RotationGeometry(
                    width=1,
                    height=1,
                    footprint_tiles=((0, 0),),
                )
            }
        ),
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

    # Stored payload: x/y are SaveRect-shifted for shield_gen_small, unshifted
    # for corridor (no SaveRect in the game files).
    #
    # Part positions are chosen to be non-overlapping across all rotations so
    # that the cell counters form a clean one-to-one map:
    #   rot=0 origin (-4,  0) → stored (-4,  1)  [y += 1]
    #   rot=1 origin ( 5, -5) → stored ( 5, -5)  [no offset]
    #   rot=2 origin ( 0,  8) → stored ( 0,  8)  [no offset]
    #   rot=3 origin (-10, 3) → stored (-9,  3)  [x += 1]
    #   corridor origin (-1, 0)                  [no SaveRect]
    extracted_payload_parts = [
        {
            "part_id": "cosmoteer.shield_gen_small",
            "rotation": 0,
            "x": -4,
            "y": 1,
            "coordinate_frame": audit_module.STORED_LOCATION_FRAME,
        },
        {
            "part_id": "cosmoteer.shield_gen_small",
            "rotation": 1,
            "x": 5,
            "y": -5,
            "coordinate_frame": audit_module.STORED_LOCATION_FRAME,
        },
        {
            "part_id": "cosmoteer.shield_gen_small",
            "rotation": 2,
            "x": 0,
            "y": 8,
            "coordinate_frame": audit_module.STORED_LOCATION_FRAME,
        },
        {
            "part_id": "cosmoteer.shield_gen_small",
            "rotation": 3,
            "x": -9,
            "y": 3,
            "coordinate_frame": audit_module.STORED_LOCATION_FRAME,
        },
        {
            "part_id": "cosmoteer.corridor",
            "rotation": 0,
            "x": -1,
            "y": 0,
            "coordinate_frame": audit_module.STORED_LOCATION_FRAME,
        },
    ]
    # Generator payload: x/y are already normalized footprint origins and must
    # not be SaveRect-adjusted by the corrected path.
    generated_payload_parts = [
        {
            "part_id": "cosmoteer.shield_gen_small",
            "rotation": 0,
            "x": -4,
            "y": 0,
            "coordinate_frame": audit_module.NORMALIZED_ORIGIN_FRAME,
        },
        {
            "part_id": "cosmoteer.shield_gen_small",
            "rotation": 1,
            "x": 5,
            "y": -5,
            "coordinate_frame": audit_module.NORMALIZED_ORIGIN_FRAME,
        },
        {
            "part_id": "cosmoteer.shield_gen_small",
            "rotation": 2,
            "x": 0,
            "y": 8,
            "coordinate_frame": audit_module.NORMALIZED_ORIGIN_FRAME,
        },
        {
            "part_id": "cosmoteer.shield_gen_small",
            "rotation": 3,
            "x": -10,
            "y": 3,
            "coordinate_frame": audit_module.NORMALIZED_ORIGIN_FRAME,
        },
        {
            "part_id": "cosmoteer.corridor",
            "rotation": 0,
            "x": -1,
            "y": 0,
            "coordinate_frame": audit_module.NORMALIZED_ORIGIN_FRAME,
        },
    ]

    extracted_counter, _ = audit_module._occupied_cells(
        extracted_payload_parts, geometry_cache, save_rects, corrected=True
    )
    generated_counter, _ = audit_module._occupied_cells(
        generated_payload_parts, geometry_cache, save_rects, corrected=True
    )

    assert extracted_counter == generated_counter


def test_resolve_save_rects_defaults_to_repo_geometry() -> None:
    """Audit save rect resolution should work without a live game install."""

    audit_module = _load_audit_module()
    save_rects = audit_module.resolve_save_rects(None)

    assert "cosmoteer.shield_gen_small" in save_rects
    assert "cosmoteer.shield_gen_large" not in save_rects
