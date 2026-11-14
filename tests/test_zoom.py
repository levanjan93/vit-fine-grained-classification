import unittest

import torch

from cs604.engine.zoom import center_boxes, crop_zoom, select_boxes


class SelectBoxesTest(unittest.TestCase):
    def test_single_hot_patch_gets_smallest_box_around_it(self):
        saliency = torch.zeros(1, 21 * 21)
        saliency[0, 20 * 21 + 20] = 1.0  # bottom-right corner patch
        top, left, side = select_boxes(saliency, 21, 0.7, 8)[0].tolist()
        self.assertEqual(side, 8)
        self.assertTrue(top <= 20 < top + side and left <= 20 < left + side)
        self.assertEqual((top + side, left + side), (21, 21))  # stays inside the grid

    def test_uniform_map_needs_enough_area(self):
        saliency = torch.ones(1, 21 * 21)
        _top, _left, side = select_boxes(saliency, 21, 0.7, 8)[0].tolist()
        self.assertEqual(side, 18)  # 18^2 / 21^2 = 0.735 >= 0.7 > 17^2 / 21^2

    def test_center_boxes_keep_size(self):
        boxes = torch.tensor([[0, 13, 8], [3, 0, 18]])
        self.assertEqual(center_boxes(boxes, 21).tolist(), [[6, 6, 8], [1, 1, 18]])

    def test_crop_zoom_shape(self):
        big = torch.randn(2, 3, 64, 64)
        boxes = torch.tensor([[0, 0, 2], [1, 1, 3]])
        self.assertEqual(tuple(crop_zoom(big, boxes, 16, 32).shape), (2, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
