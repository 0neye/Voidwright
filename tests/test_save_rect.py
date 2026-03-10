import unittest

from common.save_rect import SaveRect


class SaveRectTests(unittest.TestCase):
    def test_shield_generator_save_rect_rotates_with_part(self) -> None:
        save_rect = SaveRect(
            part_id="cosmoteer.shield_gen_small",
            x=0,
            y=1,
            width=2,
            height=2,
            source_file="game.rules",
        )

        self.assertEqual(save_rect.offset_for_rotation(0, 2, 3), (0, 1))
        self.assertEqual(save_rect.offset_for_rotation(1, 2, 3), (0, 0))
        self.assertEqual(save_rect.offset_for_rotation(2, 2, 3), (0, 0))
        self.assertEqual(save_rect.offset_for_rotation(3, 2, 3), (1, 0))


if __name__ == "__main__":
    unittest.main()
