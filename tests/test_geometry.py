"""Regression tests for shared geometry inference helpers."""

from __future__ import annotations

from common.geometry import VANILLA_PARTS_PATH, infer_meta, load_vanilla_part_geometry


def test_unknown_part_fallback_preserves_generic_traversable_aliases() -> None:
    """Common modded IDs should keep their legacy traversability hints."""

    traversable_ids = (
        "quarters",
        "BrutaTralb.storageomni",
        "Kroom.CrewQuarters_1x1",
        "ftl_drive",
    )

    for part_id in traversable_ids:
        meta, inferred = infer_meta(part_id, 0)
        assert inferred is True
        assert meta.traversable is True


def test_vanilla_geometry_still_overrides_fallback_name_matches() -> None:
    """Vanilla exact geometry should win even if a generic token appears in the name."""

    meta, inferred = infer_meta("cosmoteer.thermal_battery", 0)

    assert inferred is False
    assert meta.traversable is False


def test_full_geometry_loader_exposes_rect_metadata() -> None:
    """The full-geometry export should expose save and physical rect metadata."""

    geometry_cache = load_vanilla_part_geometry()
    shield_small = geometry_cache["cosmoteer.shield_gen_small"]
    shield_large = geometry_cache["cosmoteer.shield_gen_large"]

    # This migration should read the richer repo-backed geometry export.
    assert VANILLA_PARTS_PATH.name == "vanilla_parts_full_geometry.json"

    # Small shield generator has an explicit save rect but no physical rect.
    assert shield_small.save_rect is not None
    assert (shield_small.save_rect.x, shield_small.save_rect.y) == (0, 1)
    assert (shield_small.save_rect.width, shield_small.save_rect.height) == (2, 2)
    assert shield_small.physical_rect is None

    # Large shield generator exposes physics metadata without a save rect.
    assert shield_large.save_rect is None
    assert shield_large.physical_rect is not None
    assert (shield_large.physical_rect.x, shield_large.physical_rect.y) == (0, 2)
    assert (shield_large.physical_rect.width, shield_large.physical_rect.height) == (3, 4)
