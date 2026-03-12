from common.geometry import load_vanilla_part_geometry
from generator.backends.markov.export import generated_parts_to_cosmoteer_parts, roundtrip_validate
from markov.model import ShipPart
from markov.symmetry import mirror_rotation
from ship_layout.validation import (
    footprint_is_mirror_balanced,
    is_anchor_eligible_mirror_primary,
    mirror_cells_x,
    occupied_cells_are_mirror_balanced,
)


def test_default_horizontal_mirror_rule_is_preserved() -> None:
    """Baseline mirror behavior should keep the default rotation mapping."""

    assert mirror_rotation(0) == 0
    assert mirror_rotation(1) == 3
    assert mirror_rotation(2) == 2
    assert mirror_rotation(3) == 1


def test_directional_wedges_use_corpus_backed_override() -> None:
    """Directional wedge parts should continue to use the corpus-backed override."""

    assert mirror_rotation(0, "cosmoteer.armor_wedge") == 1
    assert mirror_rotation(1, "cosmoteer.armor_wedge") == 0
    assert mirror_rotation(2, "cosmoteer.armor_wedge") == 3
    assert mirror_rotation(3, "cosmoteer.armor_wedge") == 2
    assert mirror_rotation(0, "cosmoteer.structure_wedge") == 1
    assert mirror_rotation(2, "cosmoteer.structure_wedge") == 3
    assert mirror_rotation(0, "cosmoteer.armor_structure_hybrid_1x1") == 1


def test_half_cell_triangles_keep_rotation_when_mirrored_horizontally() -> None:
    """Triangle half-cells should mirror via FlipX without a vertical-looking rotation swap."""

    assert mirror_rotation(0, "cosmoteer.armor_tri") == 0
    assert mirror_rotation(1, "cosmoteer.armor_tri") == 1
    assert mirror_rotation(2, "cosmoteer.armor_tri") == 2
    assert mirror_rotation(3, "cosmoteer.armor_tri") == 3
    assert mirror_rotation(1, "cosmoteer.structure_tri") == 1
    assert mirror_rotation(3, "cosmoteer.armor_structure_hybrid_tri") == 3


def test_mirror_part_toggles_flip_x_for_mirrored_parts() -> None:
    """Mirroring should toggle FlipX while preserving valid mirrored placement."""

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
    assert mirrored is not None
    assert mirrored.rotation == 1
    assert mirrored.flip_x is True

    baseline = ShipPart(
        part_id="cosmoteer.reactor_large",
        rotation=0,
        x=-4,
        y=0,
        flip_x=False,
    )
    mirrored_baseline = mirror_part(baseline, geometry_cache)
    assert mirrored_baseline is not None
    assert mirrored_baseline.flip_x is True


def test_mirror_part_keeps_triangle_rotation_and_toggles_flip_x() -> None:
    """Triangle parts should mirror with the same rotation plus FlipX."""

    from markov.symmetry import mirror_part

    geometry_cache = load_vanilla_part_geometry()
    original = ShipPart(
        part_id="cosmoteer.armor_tri",
        rotation=3,
        x=-2,
        y=7,
        flip_x=False,
    )
    mirrored = mirror_part(original, geometry_cache)
    assert mirrored is not None
    assert mirrored.rotation == 3
    assert mirrored.flip_x is True


def test_mirror_cell_helpers_detect_balanced_centerline_footprints() -> None:
    """Mirror-cell helpers should treat centerline-straddling sets as balanced."""

    centered_cells = frozenset({(-1, 0), (0, 0), (-1, 1), (0, 1)})
    assert mirror_cells_x(centered_cells) == centered_cells
    assert occupied_cells_are_mirror_balanced(centered_cells) is True

    off_center_cells = frozenset({(-2, 0), (-1, 0), (0, 0)})
    assert occupied_cells_are_mirror_balanced(off_center_cells) is False


def test_anchor_eligibility_accepts_balanced_straddlers() -> None:
    """Mirror primary anchors should include balanced centerline straddlers."""

    geometry_cache = load_vanilla_part_geometry()
    centered_two_wide = ShipPart(part_id="cosmoteer.armor_2x1", rotation=0, x=-1, y=4)
    left_only = ShipPart(part_id="cosmoteer.armor_2x1", rotation=0, x=-3, y=4)
    right_only = ShipPart(part_id="cosmoteer.armor_2x1", rotation=0, x=1, y=4)

    assert footprint_is_mirror_balanced(centered_two_wide, geometry_cache) is True
    assert is_anchor_eligible_mirror_primary(centered_two_wide, geometry_cache) is True
    assert is_anchor_eligible_mirror_primary(left_only, geometry_cache) is True
    assert is_anchor_eligible_mirror_primary(right_only, geometry_cache) is False


def test_export_preserves_flip_flags() -> None:
    """Export conversion should preserve both flip flags in output payloads."""

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
    assert exported == [
        {
            "ID": "cosmoteer.armor_1x2_wedge",
            "Location": [4, -2],
            "Rotation": 3,
            "FlipX": True,
            "FlipY": False,
        }
    ]


def test_roundtrip_validation_checks_flip_flags() -> None:
    """Roundtrip validation should not report mismatches for valid flip fields."""

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
    assert report["ok"] is True
    assert report["mismatches"] == []
