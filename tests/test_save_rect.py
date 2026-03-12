from __future__ import annotations

import pytest

from common.save_rect import SaveRect, known_save_rects


def test_shield_generator_save_rect_rotates_with_part() -> None:
    """SaveRect offsets should rotate with the part footprint orientation."""

    save_rect = SaveRect(
        part_id="cosmoteer.shield_gen_small",
        x=0,
        y=1,
        width=2,
        height=2,
        source_file="game.rules",
    )

    assert save_rect.offset_for_rotation(0, 2, 3) == (0, 1)
    assert save_rect.offset_for_rotation(1, 2, 3) == (0, 0)
    assert save_rect.offset_for_rotation(2, 2, 3) == (0, 0)
    assert save_rect.offset_for_rotation(3, 2, 3) == (1, 0)


def test_known_save_rects_include_explicit_save_rect_parts() -> None:
    """Repo-backed save rects should cover every explicit stored-location rect.

    This is an exhaustive snapshot: it must match the exact set of parts that
    carry save-rect metadata in vanilla_parts_full_geometry.json.  When a new
    vanilla part with a stored-location rect is added to the game data, add it
    to `expected_part_ids` here as well.
    """

    save_rects = known_save_rects()
    expected_part_ids = {
        "cosmoteer.cannon_large",
        "cosmoteer.cannon_med",
        "cosmoteer.chaingun",
        "cosmoteer.disruptor",
        "cosmoteer.flak_cannon_large",
        "cosmoteer.ion_beam_emitter",
        "cosmoteer.laser_blaster_large",
        "cosmoteer.laser_blaster_small",
        "cosmoteer.missile_launcher",
        "cosmoteer.point_defense",
        "cosmoteer.radiator",
        "cosmoteer.railgun_launcher",
        "cosmoteer.resonance_beam_turret",
        "cosmoteer.shield_gen_small",
        "cosmoteer.thruster_boost",
        "cosmoteer.thruster_huge",
        "cosmoteer.thruster_large",
        "cosmoteer.thruster_med",
        "cosmoteer.thruster_rocket_nozzle",
        "cosmoteer.thruster_small",
        "cosmoteer.thruster_small_2way",
        "cosmoteer.thruster_small_3way",
    }

    assert set(save_rects) == expected_part_ids
    assert save_rects["cosmoteer.shield_gen_small"].source_file.endswith(":save_rect")


@pytest.mark.parametrize(
    ("part_id", "base_width", "base_height", "expected_offsets"),
    [
        (
            "cosmoteer.shield_gen_small",
            2,
            3,
            {0: (0, 1), 1: (0, 0), 2: (0, 0), 3: (1, 0)},
        ),
        (
            "cosmoteer.thruster_small_3way",
            3,
            2,
            {0: (1, 0), 1: (1, 1), 2: (1, 1), 3: (0, 1)},
        ),
    ],
)
def test_repo_backed_save_rect_offsets_rotate_as_expected(
    part_id: str,
    base_width: int,
    base_height: int,
    expected_offsets: dict[int, tuple[int, int]],
) -> None:
    """Repository geometry should drive the same rotated offset semantics."""

    save_rect = known_save_rects()[part_id]

    for rotation, expected_offset in expected_offsets.items():
        assert save_rect.offset_for_rotation(rotation, base_width, base_height) == expected_offset


def test_parts_without_save_rect_metadata_are_excluded_from_repo_table() -> None:
    """Parts with only physical-rect metadata should not shift ship locations."""

    save_rects = known_save_rects()

    assert "cosmoteer.shield_gen_large" not in save_rects
