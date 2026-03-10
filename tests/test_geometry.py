"""Regression tests for shared geometry inference helpers."""

from __future__ import annotations

from common.geometry import infer_meta


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
