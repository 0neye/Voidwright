"""Regression tests for shared structural connectivity semantics."""

from __future__ import annotations

from common.geometry import load_vanilla_part_geometry
from ship_layout.connectivity import part_attachment_cells, parts_structurally_touch


def test_wedge_flat_side_contact_counts_as_structural_touch() -> None:
    """A wedge should connect when another part touches its flat side."""

    geometry_cache = load_vanilla_part_geometry()
    wedge = {"part_id": "cosmoteer.armor_wedge", "rotation": 0, "x": 0, "y": 0}
    armor = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 1, "y": 0}

    assert parts_structurally_touch(wedge, armor, geometry_cache) is True


def test_wedge_air_side_contact_is_not_structural_touch() -> None:
    """A wedge should not connect through the empty side above the diagonal."""

    geometry_cache = load_vanilla_part_geometry()
    wedge = {"part_id": "cosmoteer.armor_wedge", "rotation": 0, "x": 0, "y": 0}
    armor = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 0, "y": -1}

    assert parts_structurally_touch(wedge, armor, geometry_cache) is False


def test_physical_rect_overhang_cells_do_not_count_as_attachment_cells() -> None:
    """Parts with physical_rect should only expose core hull cells as attachment."""

    geometry_cache = load_vanilla_part_geometry()
    thruster = {"part_id": "cosmoteer.thruster_small", "rotation": 0, "x": 0, "y": 0}

    assert part_attachment_cells(thruster, geometry_cache) == {(0, 0)}


def test_physical_rect_overhang_contact_does_not_create_structural_touch() -> None:
    """Contact against thruster nozzle overhang should not count as structural."""

    geometry_cache = load_vanilla_part_geometry()
    thruster = {"part_id": "cosmoteer.thruster_small", "rotation": 0, "x": 0, "y": 0}
    armor_below_nozzle = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 0, "y": 2}

    assert parts_structurally_touch(thruster, armor_below_nozzle, geometry_cache) is False


def test_rotated_physical_rect_attachment_cells_are_rotated_with_part() -> None:
    """Physical-rect body cells should rotate with the part orientation."""

    geometry_cache = load_vanilla_part_geometry()
    thruster_rotated = {"part_id": "cosmoteer.thruster_small", "rotation": 1, "x": 0, "y": 0}

    assert part_attachment_cells(thruster_rotated, geometry_cache) == {(1, 0)}


def test_r_wedge_alias_resolves_without_geometry_key_errors() -> None:
    """Shared connectivity should resolve mirrored `_R` wedge IDs safely."""

    geometry_cache = load_vanilla_part_geometry()
    right_wedge_alias = {"part_id": "cosmoteer.armor_1x2_wedge_R", "rotation": 0, "x": 0, "y": 0}
    armor = {"part_id": "cosmoteer.armor", "rotation": 0, "x": 1, "y": 0}

    assert parts_structurally_touch(right_wedge_alias, armor, geometry_cache) is True
