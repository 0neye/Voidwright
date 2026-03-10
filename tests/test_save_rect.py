from common.save_rect import SaveRect


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
