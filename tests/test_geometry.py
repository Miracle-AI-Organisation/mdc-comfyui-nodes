"""Unit tests for the pure crop geometry (no cv2 / torch needed)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_detect import expand_and_clamp  # noqa: E402


class TestExpandAndClamp(unittest.TestCase):
    def test_returns_the_same_box_when_crop_factor_is_one(self):
        self.assertEqual(
            expand_and_clamp((100, 200, 50, 60), 800, 1200, 1.0),
            (100, 200, 150, 260),
        )

    def test_doubles_the_box_around_its_center_when_crop_factor_is_two(self):
        # centre (125, 230); 50x60 -> 100x120 => x 75..175, y 170..290
        self.assertEqual(
            expand_and_clamp((100, 200, 50, 60), 800, 1200, 2.0),
            (75, 170, 175, 290),
        )

    def test_clamps_to_the_image_when_the_expanded_box_runs_off_the_top_left(self):
        # centre (30, 30), half-extent 60 => -30..90; only the negative side is cut.
        self.assertEqual(expand_and_clamp((10, 10, 40, 40), 800, 1200, 3.0), (0, 0, 90, 90))

    def test_clamps_to_the_image_when_the_expanded_box_runs_off_the_bottom_right(self):
        self.assertEqual(
            expand_and_clamp((760, 1160, 40, 40), 800, 1200, 3.0),
            (720, 1120, 800, 1200),
        )

    def test_never_exceeds_the_image_bounds_for_an_absurd_crop_factor(self):
        x1, y1, x2, y2 = expand_and_clamp((400, 600, 100, 100), 800, 1200, 50.0)
        self.assertEqual((x1, y1, x2, y2), (0, 0, 800, 1200))

    def test_keeps_at_least_one_pixel_for_a_degenerate_zero_size_box(self):
        x1, y1, x2, y2 = expand_and_clamp((500, 500, 0, 0), 800, 1200, 1.0)
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)

    def test_accepts_a_box_whose_origin_is_negative(self):
        # YuNet can report slightly out-of-frame boxes.
        self.assertEqual(expand_and_clamp((-20, -30, 100, 100), 800, 1200, 1.0), (0, 0, 80, 70))


if __name__ == "__main__":
    unittest.main(verbosity=2)
