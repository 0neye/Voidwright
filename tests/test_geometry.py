"""Regression tests for shared geometry inference helpers."""

from __future__ import annotations

from common.geometry import (
    VANILLA_PARTS_PATH,
    infer_meta,
    load_vanilla_part_geometry,
    normalize_part_id,
    polygon_vertices_to_2x,
    resolve_geometry_part_id_and_rotation,
)


def test_unknown_part_fallback_preserves_generic_traversable_aliases() -> None:
    """Common modded IDs should keep their legacy traversability hints."""

    traversable_ids = (
        "BrutaTralb.storageomni",
        "Kroom.CrewQuarters_1x1",
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


def test_legacy_factory_aliases_resolve_to_exact_vanilla_geometry() -> None:
    """Old factory IDs should map onto the canonical vanilla part geometry."""

    alias_expectations = {
        "cosmoteer.missile_factory_nuke": (4, 4),
        "cosmoteer.missile_factory_emp": (3, 4),
        "cosmoteer.missile_factory": (3, 3),
        "missile_factory": (3, 3),
        "cosmoteer.missile_factory_high_explosive": (3, 3),
        "cosmoteer.missile_factory_he": (3, 3),
        "cosmoteer.mine_factory": (4, 3),
    }

    for part_id, expected_size in alias_expectations.items():
        meta, inferred = infer_meta(part_id, 0)
        assert inferred is False, part_id
        assert (meta.width, meta.height) == expected_size, part_id


def test_legacy_other_id_aliases_resolve_to_exact_vanilla_geometry() -> None:
    """Legacy OtherIDs that do not encode handedness should normalize safely."""

    alias_expectations = {
        "big_thruster": (2, 3),
        "med_thruster": (1, 3),
        "small_thruster": (1, 2),
        "small_laser": (1, 3),
        "ftl_drive": (2, 2),
        "shield_generator": (2, 3),
        "cosmoteer.heat_pipe_corner": (1, 1),
        "cosmoteer.resource_collector": (2, 2),
    }

    for part_id, expected_size in alias_expectations.items():
        meta, inferred = infer_meta(part_id, 0)
        assert inferred is False, part_id
        assert (meta.width, meta.height) == expected_size, part_id


def test_flipped_wedge_aliases_preserve_handedness_during_normalization() -> None:
    """Normalize-only preprocessing must not erase handed wedge orientation."""

    raw_part = {"ID": "armor2_wedge_R"}

    assert normalize_part_id(raw_part) == "armor2_wedge_R"
    assert resolve_geometry_part_id_and_rotation("armor2_wedge_R", 1) == (
        "cosmoteer.armor_1x2_wedge",
        3,
    )


def test_legacy_flipped_wedge_aliases_resolve_through_flip_h_geometry() -> None:
    """Legacy handed wedge aliases should match canonical geometry after flip remap."""

    alias_expectations = {
        "armor2_wedge_R": ("cosmoteer.armor_1x2_wedge", 0, (1, 2)),
        "structure2_wedge_R": ("cosmoteer.structure_1x2_wedge", 0, (1, 2)),
        "Kroom.Armor_1x3_Wedge_R": ("cosmoteer.armor_1x3_wedge", 0, (1, 3)),
        "Kroom.Structure_1x3_Wedge_R": ("cosmoteer.structure_1x3_wedge", 0, (1, 3)),
    }

    for part_id, (canonical_part_id, rotation, expected_size) in alias_expectations.items():
        meta, inferred = infer_meta(part_id, rotation)
        assert inferred is False, part_id
        assert (meta.width, meta.height) == expected_size, part_id
        assert resolve_geometry_part_id_and_rotation(part_id, rotation) == (canonical_part_id, rotation)


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


def test_rotation_geometry_exposes_polygon_vertices_for_wedges_and_triangles(
) -> None:
    """Wedge and tri parts should keep per-rotation polygon vertices."""

    geometry_cache = load_vanilla_part_geometry()
    wedge = geometry_cache["cosmoteer.armor_1x2_wedge"].rotations[0]
    tri = geometry_cache["cosmoteer.armor_tri"].rotations[0]

    assert wedge.polygon_vertices == ((1.0, 0.0), (1.0, 2.0), (0.0, 2.0))
    assert tri.polygon_vertices == ((0.5, 0.5), (1.0, 1.0), (0.0, 1.0))


def test_triangle_polygon_vertices_convert_cleanly_to_integer_2x_coordinates() -> None:
    """Half-tile triangle points should become integer points in 2x space."""

    geometry_cache = load_vanilla_part_geometry()
    tri_vertices = geometry_cache["cosmoteer.armor_tri"].rotations[0].polygon_vertices

    assert polygon_vertices_to_2x(tri_vertices) == ((1, 1), (2, 2), (0, 2))


def test_conveyor_directional_speed_maps_load_and_rotate() -> None:
    """Conveyors should preserve direction-dependent travel speeds per rotation."""

    geometry_cache = load_vanilla_part_geometry()
    conveyor = geometry_cache["cosmoteer.conveyor"]

    assert conveyor.crew_speed_factor is None
    assert conveyor.crew_speed_by_direction == {
        "Up": 2.0,
        "Right": 0.75,
        "Down": 0.25,
        "Left": 0.75,
    }
    assert conveyor.rotation_geometry(0).crew_speed_for_direction("Up") == 2.0
    assert conveyor.rotation_geometry(1).crew_speed_for_direction("Right") == 2.0
    assert conveyor.rotation_geometry(1).crew_speed_for_direction("Up") == 0.75
    assert conveyor.crew_speed_for_direction(1, "Right") == 2.0


def test_rotation_geometry_exposes_blocked_travel_directions_and_manhattan_flag() -> None:
    """Per-rotation blocked travel directions should rotate with the part geometry."""

    geometry_cache = load_vanilla_part_geometry()
    control_room = geometry_cache["cosmoteer.control_room_med"]

    rotation_0 = control_room.rotation_geometry(0)
    rotation_1 = control_room.rotation_geometry(1)

    assert rotation_0.force_manhattan_path is True
    assert rotation_0.is_direction_blocked((1, 1), "Down") is True
    assert rotation_0.is_direction_blocked((1, 2), "Up") is True

    # Rotating 3x3 clockwise maps (1,1)->(1,1) and Down->Left,
    # while (1,2)->(0,1) and Up->Right.
    assert rotation_1.force_manhattan_path is True
    assert rotation_1.is_direction_blocked((1, 1), "Left") is True
    assert rotation_1.is_direction_blocked((0, 1), "Right") is True
